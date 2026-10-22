//! `reco-anon` — anonymize every human head in a video, locally.
//!
//! Workflow: `analyze` → (`status`, `zone`, `render --preview`) → `approve` → `render`.

use std::io::Write;
use std::path::{Path, PathBuf};

use anon_detect::rfdetr::RfDetrSeg;
use anon_io::{Codec, EncodeSettings};
use anon_model::{AnalysisSettings, ModelRef, ProjectDir, TileSettings, Zone};
use anon_pipeline::{AnalyzeOptions, RenderOptions};
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Deserialize;
use sha2::{Digest, Sha256};

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
        /// Tile edge in source pixels (default: model input size).
        #[arg(long, default_value_t = 0)]
        tile: u32,
        #[arg(long, default_value_t = 0.25)]
        overlap: f32,
        #[arg(long)]
        no_flip: bool,
        #[arg(long)]
        no_full_frame: bool,
        #[arg(long, default_value_t = 0.15)]
        threshold: f32,
        #[arg(long, default_value_t = 1.5)]
        head_expand: f32,
        #[arg(long, default_value_t = 3)]
        dilation: u32,
        #[arg(long, default_value_t = 30)]
        gap: u32,
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

/// `<model>.json`, written by `scripts/export_rfdetr.py`.
#[derive(Deserialize)]
struct ModelManifest {
    id: String,
    sha256: String,
    license: String,
    #[serde(default = "default_person_class")]
    person_class: usize,
}

fn default_person_class() -> usize {
    1
}

fn project_dir(p: &Path) -> ProjectDir {
    if p.extension().is_some_and(|e| e == "anon") || p.join("project.json").exists() {
        ProjectDir::open(p)
    } else {
        ProjectDir::for_source(p)
    }
}

fn load_model(path: &Path) -> Result<(RfDetrSeg, ModelRef)> {
    let mut manifest_path = path.as_os_str().to_owned();
    manifest_path.push(".json");
    let manifest_path = PathBuf::from(manifest_path);
    let manifest: ModelManifest = serde_json::from_slice(
        &std::fs::read(&manifest_path)
            .with_context(|| format!("model manifest {} missing", manifest_path.display()))?,
    )
    .context("invalid model manifest")?;
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let sha = hex::encode(Sha256::digest(&bytes));
    if sha != manifest.sha256 {
        bail!(
            "model hash {sha} does not match manifest {}; refusing unknown model",
            manifest.sha256
        );
    }
    let model = RfDetrSeg::load(path, manifest.person_class)?;
    Ok((
        model,
        ModelRef {
            id: manifest.id,
            sha256: sha,
            license: manifest.license,
        },
    ))
}

fn progress_line(done: u64, total: Option<u64>) {
    if !done.is_multiple_of(25) {
        return;
    }
    match total {
        Some(t) if t > 0 => eprint!(
            "\r  frame {done}/{t} ({:.0}%)   ",
            done as f64 * 100.0 / t as f64
        ),
        _ => eprint!("\r  frame {done}   "),
    }
    let _ = std::io::stderr().flush();
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
            tile,
            overlap,
            no_flip,
            no_full_frame,
            threshold,
            head_expand,
            dilation,
            gap,
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
                },
                head_expand,
                gap_fill_frames: gap,
                dilation_frames: dilation,
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
                },
                progress_line,
            )?;
            eprintln!();
            println!(
                "frames {}  tracks {}  max persons/frame {}  flagged {}  errors {}",
                s.frames, s.tracks, s.max_persons, s.flagged, s.errors
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
