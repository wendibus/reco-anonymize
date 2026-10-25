//! `reco-anon` — anonymize every human head in a video, locally.
//!
//! Workflow: `analyze` → (`status`, `zone`, `render --preview`) → `approve` → `render`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anon_detect::rfdetr::RfDetrSeg;
use anon_io::{Codec, EncodeSettings};
use anon_model::{AnalysisSettings, ModelRef, ProjectDir, TileSettings, Zone};
use anon_pipeline::progress::{Eta, Throttle, format_remaining};
use anon_pipeline::{AnalyzeOptions, RenderOptions};
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "reco-anon",
    version,
    about = "Opaque head anonymization for reco videos, 100% local"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Probe a video (resolution, timing, colour, rotation, audio).
    Info { video: PathBuf },
    /// Detect every person, derive heads, link, flag. Writes `<video>.anon/`.
    Analyze {
        video: PathBuf,
        /// ONNX model; a manifest `<model>.json` must sit next to it.
        #[arg(long)]
        model: PathBuf,
        /// Project directory (default: `<video>.anon`).
        #[arg(long)]
        project: Option<PathBuf>,
        /// Discard an existing project that belongs to another file.
        #[arg(long)]
        fresh: bool,
        /// Continue an interrupted analysis from its last checkpoint.
        #[arg(long)]
        resume: bool,
        /// Tile edge in source pixels (default: model input size).
        #[arg(long, default_value_t = 0)]
        tile: u32,
        #[arg(long, default_value_t = 0.25)]
        overlap: f32,
        #[arg(long)]
        no_flip: bool,
        #[arg(long)]
        no_full_frame: bool,
        /// Perspective tiling: from this fraction of the frame height down
        /// (0..1), use coarse tiles, e.g. 0.6 for a camera looking down a field.
        #[arg(long)]
        coarse_below: Option<f32>,
        /// Coarse tile edge in source pixels (default: twice the tile).
        #[arg(long, default_value_t = 0)]
        coarse_tile: u32,
        #[arg(long, default_value_t = 0.15)]
        threshold: f32,
        #[arg(long, default_value_t = 1.5)]
        head_expand: f32,
        #[arg(long, default_value_t = 3)]
        dilation: u32,
        #[arg(long, default_value_t = 30)]
        gap: u32,
        /// Also mask jersey numbers / text on players with this PP-OCR det
        /// model (manifest kind "text-db", see scripts/fetch_ppocr.py).
        #[arg(long)]
        text_model: Option<PathBuf>,
    },
    /// Show analysis state, flagged frames and approval validity.
    Status {
        project: PathBuf,
        #[arg(long, default_value_t = 30)]
        flags: usize,
        #[arg(long)]
        json: bool,
    },
    /// Manage static zones (stands, scoreboard) that are always masked.
    Zone {
        #[command(subcommand)]
        cmd: ZoneCmd,
    },
    /// Approve the current masks after review. Required before `render`.
    Approve {
        project: PathBuf,
        /// Confirms that the preview was checked frame by frame where flagged.
        #[arg(long)]
        i_reviewed_the_preview: bool,
        /// Accept a clip in which no person was detected at all.
        #[arg(long)]
        no_people_expected: bool,
        #[arg(long)]
        note: Option<String>,
    },
    /// Re-detect on the rendered video and check every head is covered.
    Audit {
        project: PathBuf,
        /// The rendered file to audit.
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 20)]
        show: usize,
    },
    /// Acknowledge audit hits that are not leaks (requires a reason).
    AuditAck {
        project: PathBuf,
        /// Frames to acknowledge; omit together with --all.
        #[arg(long = "frame")]
        frames: Vec<u64>,
        #[arg(long)]
        all: bool,
        #[arg(long)]
        reason: String,
    },
    /// Export reviewed head labels (COCO, RF-DETR / Reco Trainer layout) to
    /// train a dedicated head detector. Contains UN-anonymized frames.
    ExportDataset {
        /// One or more videos or .anon projects.
        #[arg(required = true)]
        projects: Vec<PathBuf>,
        #[arg(short, long)]
        out: PathBuf,
        /// Take every n-th frame.
        #[arg(long, default_value_t = 15)]
        every: u64,
        /// Also export projects without a valid approval.
        #[arg(long)]
        allow_unreviewed: bool,
    },
    /// Check for a new release and install it.
    Update {
        /// Only check, do not install.
        #[arg(long)]
        check: bool,
    },
    /// Render the anonymized video.
    Render {
        project: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Unapproved render into `<project>/preview.<ext>` for review.
        #[arg(long)]
        preview: bool,
        #[arg(long, value_enum, default_value_t = CodecArg::Hevc)]
        codec: CodecArg,
        /// Force an FFmpeg encoder, e.g. libx264.
        #[arg(long)]
        encoder: Option<String>,
        /// 0..100 (86 ≈ CRF 16).
        #[arg(long, default_value_t = 86)]
        quality: u8,
    },
}

#[derive(Subcommand)]
enum ZoneCmd {
    /// Add a polygon zone: --points "x,y x,y x,y ..." in source pixels.
    Add {
        project: PathBuf,
        #[arg(long)]
        label: String,
        #[arg(long)]
        points: String,
        #[arg(long)]
        from: Option<u64>,
        #[arg(long)]
        to: Option<u64>,
    },
    List {
        project: PathBuf,
    },
    Remove {
        project: PathBuf,
        id: u32,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum CodecArg {
    Hevc,
    H264,
}

fn project_dir(p: &Path) -> ProjectDir {
    if p.extension().is_some_and(|e| e == "anon") || p.join("project.json").exists() {
        ProjectDir::open(p)
    } else {
        ProjectDir::for_source(p)
    }
}

fn load_model(path: &Path) -> Result<(RfDetrSeg, ModelRef)> {
    Ok(anon_detect::rfdetr::load_verified(path)?)
}

/// Progress on stderr with frame rate and remaining time, at most twice a second.
fn progress_line() -> impl FnMut(u64, Option<u64>) {
    let mut eta = Eta::new();
    let mut throttle = Throttle::new(Duration::from_millis(500));
    move |done, total| {
        let est = eta.update(done, total);
        if !throttle.ready(done, total) {
            return;
        }
        let mut line = match total {
            Some(t) if t > 0 => {
                format!("frame {done}/{t} ({:.0}%)", done as f64 * 100.0 / t as f64)
            }
            _ => format!("frame {done}"),
        };
        if let Some(e) = est {
            line.push_str(&format!("  {}", e.rate_text("s/frame")));
            if let Some(r) = e.remaining {
                line.push_str(&format!("  {} left", format_remaining(r)));
            }
        }
        eprint!("\r  {line}   ");
        let _ = std::io::stderr().flush();
    }
}

fn parse_points(s: &str) -> Result<Vec<[f32; 2]>> {
    let pts = s
        .split_whitespace()
        .map(|p| {
            let (x, y) = p.split_once(',').context("point must be x,y")?;
            Ok([x.trim().parse()?, y.trim().parse()?])
        })
        .collect::<Result<Vec<_>>>()?;
    if pts.len() < 3 {
        bail!("a zone needs at least 3 points");
    }
    Ok(pts)
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    anon_update::cleanup();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Info { video } => {
            anon_io::init()?;
            let p = anon_io::probe(&video)?;
            println!("{p:#?}");
        }
        Cmd::Analyze {
            video,
            model,
            project,
            fresh,
            resume,
            tile,
            overlap,
            no_flip,
            no_full_frame,
            coarse_below,
            coarse_tile,
            threshold,
            head_expand,
            dilation,
            gap,
            text_model,
        } => {
            let dir = project.map_or_else(|| ProjectDir::for_source(&video), ProjectDir::open);
            let (model, model_ref) = load_model(&model)?;
            let settings = AnalysisSettings {
                person_threshold: threshold,
                flag_band: (threshold, threshold.max(0.35)),
                tile: TileSettings {
                    tile,
                    overlap,
                    full_frame_pass: !no_full_frame,
                    flip_pass: !no_flip,
                    coarse_below,
                    coarse_tile,
                },
                head_expand,
                gap_fill_frames: gap,
                dilation_frames: dilation,
                mask_jersey_numbers: text_model.is_some(),
                ..AnalysisSettings::default()
            };
            eprintln!("analyzing {} -> {}", video.display(), dir.root.display());
            let s = anon_pipeline::analyze(
                &video,
                &dir,
                Box::new(model),
                AnalyzeOptions {
                    settings,
                    model_ref,
                    fresh,
                    resume,
                    text: match &text_model {
                        Some(p) => {
                            let (d, r) = anon_detect::ppocr::load_verified(p)?;
                            Some((Box::new(d) as Box<dyn anon_detect::text::TextDetector>, r))
                        }
                        None => None,
                    },
                },
                progress_line(),
            )?;
            eprintln!();
            println!(
                "frames {} (resumed at {})  head tracks {}  text tracks {}  max persons/frame {}  flagged {}  errors {}",
                s.frames,
                s.resumed_from,
                s.tracks,
                s.text_tracks,
                s.max_persons,
                s.flagged,
                s.errors
            );
            println!(
                "next: reco-anon render {} --preview, check it, then approve",
                dir.root.display()
            );
        }
        Cmd::Status {
            project,
            flags,
            json,
        } => {
            let s = anon_pipeline::status(&project_dir(&project), flags)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&s)?);
            } else {
                println!(
                    "source        {} ({})",
                    s.source.display(),
                    if s.source_matches {
                        "unchanged"
                    } else {
                        "CHANGED/MISSING"
                    }
                );
                println!(
                    "frames        {} analysed / {} total, {} errors",
                    s.analyzed, s.frames, s.errors
                );
                println!(
                    "tracks        {} auto, {} manual, {} suppressed",
                    s.auto_tracks, s.manual_tracks, s.suppressed_tracks
                );
                println!("zones         {}", s.zones);
                println!("flagged       {} frame(s)", s.flagged_frames);
                for (f, why) in &s.flagged {
                    println!("  frame {f:>7}  {}", why.join(", "));
                }
                println!(
                    "export        {} ({})",
                    if s.export_clean { "CLEAN" } else { "not clean" },
                    s.export_note
                );
                println!(
                    "approval      {}",
                    match (s.approved, s.approval_valid) {
                        (true, true) => "valid",
                        (true, false) => "VOID (masks changed after approval)",
                        _ => "none",
                    }
                );
            }
        }
        Cmd::Zone { cmd } => match cmd {
            ZoneCmd::Add {
                project,
                label,
                points,
                from,
                to,
            } => {
                let dir = project_dir(&project);
                let mut p = dir.load_project()?;
                let id = p.zones.iter().map(|z| z.id).max().unwrap_or(0) + 1;
                let range = match (from, to) {
                    (None, None) => None,
                    (a, b) => Some((a.unwrap_or(0), b.unwrap_or(u64::MAX))),
                };
                p.zones.push(Zone {
                    id,
                    label,
                    polygon: parse_points(&points)?,
                    range,
                });
                dir.save_project(&p)?;
                println!("zone {id} added (approval, if any, is now void)");
            }
            ZoneCmd::List { project } => {
                for z in project_dir(&project).load_project()?.zones {
                    println!("{:>3}  {:<16} {:?} {:?}", z.id, z.label, z.range, z.polygon);
                }
            }
            ZoneCmd::Remove { project, id } => {
                let dir = project_dir(&project);
                let mut p = dir.load_project()?;
                let n = p.zones.len();
                p.zones.retain(|z| z.id != id);
                if p.zones.len() == n {
                    bail!("no zone {id}");
                }
                dir.save_project(&p)?;
                println!("zone {id} removed");
            }
        },
        Cmd::Approve {
            project,
            i_reviewed_the_preview,
            no_people_expected,
            note,
        } => {
            if !i_reviewed_the_preview {
                bail!(
                    "automatic detection is never 100% reliable. Render a preview, check every flagged frame \
                     (`reco-anon status`), then re-run with --i-reviewed-the-preview"
                );
            }
            let hash = anon_pipeline::approve(&project_dir(&project), note, no_people_expected)?;
            println!("approved ({})", &hash[..16]);
        }
        Cmd::Audit {
            project,
            output,
            model,
            show,
        } => {
            let dir = project_dir(&project);
            let (model, model_ref) = load_model(&model)?;
            eprintln!("auditing {}", output.display());
            let total = dir.load_project().ok().map(|p| p.source.frame_count);
            let mut progress = progress_line();
            let r = anon_pipeline::audit(&dir, &output, Box::new(model), &model_ref.sha256, |n| {
                progress(n, total)
            })?;
            eprintln!();
            for h in r.hits.iter().take(show) {
                println!(
                    "  frame {:>7}  head {:?}  coverage {:.0}%  score {:.2}",
                    h.frame,
                    h.head.map(|v| v.round() as i32),
                    h.coverage * 100.0,
                    h.score
                );
            }
            if r.passed() {
                println!("audit PASSED: {} frames, no uncovered head found", r.frames);
            } else {
                println!(
                    "audit FAILED: {} uncovered head(s) in {} frames. Fix the masks (zones, re-analysis) \
                     or acknowledge reviewed false hits with `audit-ack`",
                    r.open_hits(),
                    r.frames
                );
                std::process::exit(2);
            }
        }
        Cmd::AuditAck {
            project,
            frames,
            all,
            reason,
        } => {
            if frames.is_empty() && !all {
                bail!("give --frame N (repeatable) or --all");
            }
            let dir = project_dir(&project);
            let sel = (!all).then_some(frames.as_slice());
            let n = anon_pipeline::acknowledge(&dir, sel, &reason)?;
            let (clean, note) = anon_pipeline::export_clean(&dir)?;
            println!(
                "{n} hit(s) acknowledged; export {}: {note}",
                if clean { "CLEAN" } else { "NOT clean" }
            );
        }
        Cmd::ExportDataset {
            projects,
            out,
            every,
            allow_unreviewed,
        } => {
            let dirs: Vec<ProjectDir> = projects.iter().map(|p| project_dir(p)).collect();
            let opts = anon_pipeline::ExportOptions {
                every,
                allow_unreviewed,
                ..Default::default()
            };
            let s = anon_pipeline::export_dataset(&dirs, &out, &opts, |name, f| {
                if f.is_multiple_of(250) {
                    eprint!("\r  {name}: frame {f}   ");
                }
            })?;
            eprintln!();
            println!(
                "{} project(s) -> {}: train {} images / {} heads, valid {} / {}, test {} / {}",
                s.projects,
                out.display(),
                s.images[0],
                s.boxes[0],
                s.images[1],
                s.boxes[1],
                s.images[2],
                s.boxes[2]
            );
            println!("contains UN-anonymized frames: keep it local");
        }
        Cmd::Update { check } => {
            let current = env!("CARGO_PKG_VERSION");
            match anon_update::check(current)? {
                None => println!("reco-anonymize {current} is up to date"),
                Some(r) if check => println!("update available: {} ({})", r.version, r.page),
                Some(r) => {
                    println!("installing {} …", r.version);
                    let report = anon_update::install(&r, |done, total| {
                        if let Some(t) = total {
                            eprint!("\r  {} / {} KiB   ", done / 1024, t / 1024);
                        }
                    })?;
                    eprintln!();
                    println!(
                        "updated to {} ({} file(s) replaced)",
                        r.version, report.files
                    );
                }
            }
        }
        Cmd::Render {
            project,
            output,
            preview,
            codec,
            encoder,
            quality,
        } => {
            let dir = project_dir(&project);
            let out = match (output, preview) {
                (Some(o), _) => o,
                (None, true) => PathBuf::from("preview.mp4"),
                (None, false) => bail!("--output is required"),
            };
            let opts = RenderOptions {
                encode: EncodeSettings {
                    codec: match codec {
                        CodecArg::Hevc => Codec::Hevc,
                        CodecArg::H264 => Codec::H264,
                    },
                    encoder,
                    quality,
                },
                preview,
            };
            let s = anon_pipeline::render(&dir, &out, &opts)?;
            println!(
                "{} {} frames ({} fully opaque), {} via {} [{}], {} audio stream(s) copied, timing verified",
                if s.preview { "preview" } else { "rendered" },
                s.frames,
                s.fully_opaque_frames,
                s.output.display(),
                s.encoder,
                s.pixel_format,
                s.audio_streams
            );
        }
    }
    Ok(())
}
