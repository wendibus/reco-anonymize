//! Long-running work on worker threads: analyze, preview, render, audit,
//! training-data export, update install.

use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use std::time::Duration;

use anon_io::{Codec, EncodeSettings};
use anon_model::{AnalysisSettings, ProjectDir};
use anon_pipeline::progress::{Eta, Throttle, format_remaining};
use anon_pipeline::{AnalyzeOptions, RenderOptions};

use crate::i18n::{tr, trf};

pub enum Job {
    Analyze {
        src: PathBuf,
        model: PathBuf,
        /// Mask jersey numbers with this text model.
        text: Option<PathBuf>,
        resume: bool,
        /// Also run every tile mirrored (slower, slightly higher recall).
        flip: bool,
        /// Perspective tiling from this fraction of the frame height down.
        coarse_below: Option<f32>,
        /// Detect only every this many milliseconds, track in between.
        detect_interval_ms: Option<u32>,
    },
    Preview {
        dir: ProjectDir,
    },
    Render {
        dir: ProjectDir,
        out: PathBuf,
        /// When set, the export is audited right after rendering.
        model: Option<PathBuf>,
    },
    Audit {
        dir: ProjectDir,
        out: PathBuf,
        model: PathBuf,
    },
    Export {
        dir: ProjectDir,
        out: PathBuf,
    },
    Update {
        release: anon_update::Release,
    },
    /// Background update check (no UI lock).
    CheckUpdate,
}

impl Job {
    pub fn title(&self) -> &'static str {
        match self {
            Job::Analyze { .. } => tr("analyzing…"),
            Job::Preview { .. } => tr("rendering preview…"),
            Job::Render { .. } => tr("rendering…"),
            Job::Audit { .. } => tr("auditing…"),
            Job::Export { .. } => tr("exporting training data…"),
            Job::Update { .. } => tr("downloading update…"),
            Job::CheckUpdate => "",
        }
    }
}

pub enum JobMsg {
    /// `fraction < 0` = indeterminate.
    Progress {
        fraction: f32,
        text: String,
    },
    /// `reload`: project path to reopen afterwards.
    Done {
        text: String,
        ok: bool,
        reload: Option<PathBuf>,
    },
    UpdateAvailable(anon_update::Release),
}

/// Output of the last approved render of this project.
pub fn last_render(dir: &ProjectDir) -> Option<PathBuf> {
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path("render-report.json")).ok()?).ok()?;
    v.get("output")?.as_str().map(PathBuf::from)
}

fn open_externally(p: &Path) {
    #[cfg(target_os = "macos")]
    let cmd = "open";
    #[cfg(target_os = "windows")]
    let cmd = "explorer";
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    let cmd = "xdg-open";
    let _ = std::process::Command::new(cmd).arg(p).spawn();
}

fn frames_progress(tx: &Sender<JobMsg>, label: &'static str) -> impl FnMut(u64, Option<u64>) {
    let tx = tx.clone();
    let mut eta = Eta::new();
    let mut throttle = Throttle::new(Duration::from_millis(500));
    move |done, total| {
        let est = eta.update(done, total);
        if !throttle.ready(done, total) {
            return;
        }
        let (fraction, mut text) = match total {
            Some(t) if t > 0 => (
                (done as f32 / t as f32).min(1.0),
                trf("{}: frame {} / {}", &[&tr(label), &done, &t]),
            ),
            _ => (-1.0, trf("{}: frame {}", &[&tr(label), &done])),
        };
        if let Some(e) = est {
            text.push_str(" · ");
            text.push_str(&e.rate_text(tr("s/frame")));
            if let Some(r) = e.remaining {
                text.push_str(" · ");
                text.push_str(&trf("{} left", &[&format_remaining(r)]));
            }
        }
        let _ = tx.send(JobMsg::Progress { fraction, text });
    }
}

fn encode() -> EncodeSettings {
    EncodeSettings {
        codec: Codec::Hevc,
        encoder: None,
        quality: 86,
    }
}

fn audit(
    dir: &ProjectDir,
    out: &Path,
    model: &Path,
    tx: &Sender<JobMsg>,
) -> Result<(String, bool), String> {
    let (m, mref) = anon_detect::rfdetr::load_verified(model).map_err(|e| e.to_string())?;
    let mut p = frames_progress(tx, "audit");
    let total = dir.load_project().ok().map(|p| p.source.frame_count);
    let r = anon_pipeline::audit(dir, out, Box::new(m), &mref.sha256, |n| p(n, total))
        .map_err(|e| e.to_string())?;
    Ok(if r.passed() {
        (
            trf("audit PASSED: {} frames, no uncovered head", &[&r.frames]),
            true,
        )
    } else {
        (
            trf(
                "audit FAILED: {} uncovered head(s) — see audit-report.json, fix masks and render again",
                &[&r.open_hits()],
            ),
            false,
        )
    })
}

/// Ok((text, ok, reload))
fn run(job: Job, tx: &Sender<JobMsg>) -> Result<(String, bool, Option<PathBuf>), String> {
    match job {
        Job::Analyze {
            src,
            model,
            text,
            resume,
            flip,
            coarse_below,
            detect_interval_ms,
        } => {
            let (m, model_ref) =
                anon_detect::rfdetr::load_verified(&model).map_err(|e| e.to_string())?;
            let text = match &text {
                Some(p) => {
                    let (d, r) = anon_detect::ppocr::load_verified(p).map_err(|e| e.to_string())?;
                    Some((Box::new(d) as Box<dyn anon_detect::text::TextDetector>, r))
                }
                None => None,
            };
            let dir = ProjectDir::for_source(&src);
            let mut settings = dir
                .load_project()
                .map(|p| p.settings)
                .unwrap_or_else(|_| AnalysisSettings::default());
            settings.mask_jersey_numbers = text.is_some();
            settings.tile.flip_pass = flip;
            settings.tile.coarse_below = coarse_below;
            settings.detect_every = 1;
            settings.detect_interval_ms = detect_interval_ms;
            // projects from before 0.7 kept the dilation in frames only
            settings.dilation_ms.get_or_insert(100);
            let s = anon_pipeline::analyze(
                &src,
                &dir,
                Box::new(m),
                AnalyzeOptions {
                    settings,
                    model_ref,
                    fresh: false,
                    resume,
                    text,
                },
                frames_progress(tx, "analyze"),
            )
            .map_err(|e| e.to_string())?;
            Ok((
                trf(
                    "analysis done: {} frames, {} head tracks, {} text tracks, {} flagged, {} errors",
                    &[&s.frames, &s.tracks, &s.text_tracks, &s.flagged, &s.errors],
                ),
                true,
                Some(dir.root),
            ))
        }
        Job::Preview { dir } => {
            let r = anon_pipeline::render(
                &dir,
                Path::new("preview.mp4"),
                &RenderOptions {
                    encode: encode(),
                    preview: true,
                },
            )
            .map_err(|e| e.to_string())?;
            open_externally(&r.output);
            Ok((
                trf("preview written: {}", &[&r.output.display()]),
                true,
                None,
            ))
        }
        Job::Render { dir, out, model } => {
            let r = anon_pipeline::render(
                &dir,
                &out,
                &RenderOptions {
                    encode: encode(),
                    preview: false,
                },
            )
            .map_err(|e| e.to_string())?;
            let mut text = trf(
                "rendered {} frames via {}, timing verified",
                &[&r.frames, &r.encoder],
            );
            let mut ok = true;
            if let Some(model) = model {
                let _ = tx.send(JobMsg::Progress {
                    fraction: -1.0,
                    text: tr("auditing the export…").into(),
                });
                let (t, passed) = audit(&dir, &r.output, &model, tx)?;
                text += "\n";
                text += &t;
                ok = passed;
            }
            Ok((text, ok, Some(dir.root)))
        }
        Job::Audit { dir, out, model } => {
            let (t, ok) = audit(&dir, &out, &model, tx)?;
            Ok((t, ok, Some(dir.root)))
        }
        Job::Export { dir, out } => {
            let mut p = frames_progress(tx, "export");
            let s = anon_pipeline::export_dataset(
                &[dir],
                &out,
                &anon_pipeline::ExportOptions::default(),
                |_, f| p(f, None),
            )
            .map_err(|e| e.to_string())?;
            open_externally(&out);
            Ok((
                trf(
                    "training data: {} images, {} heads → {} (UN-anonymized, keep it local)",
                    &[
                        &s.images.iter().sum::<usize>(),
                        &s.boxes.iter().sum::<usize>(),
                        &out.display(),
                    ],
                ),
                true,
                None,
            ))
        }
        Job::Update { release } => {
            let txp = tx.clone();
            anon_update::install(&release, move |done, total| {
                let fraction = total.map_or(-1.0, |t| done as f32 / t.max(1) as f32);
                let _ = txp.send(JobMsg::Progress {
                    fraction,
                    text: format!("{} KiB", done / 1024),
                });
            })
            .map_err(|e| e.to_string())?;
            Ok((
                trf(
                    "updated to {} — restart the app to finish",
                    &[&release.version],
                ),
                true,
                None,
            ))
        }
        Job::CheckUpdate => {
            if let Ok(Some(r)) = anon_update::check(env!("CARGO_PKG_VERSION")) {
                let _ = tx.send(JobMsg::UpdateAvailable(r));
            }
            Ok((String::new(), true, None))
        }
    }
}

pub fn spawn(job: Job, tx: Sender<JobMsg>) {
    std::thread::spawn(move || {
        let silent = matches!(job, Job::CheckUpdate);
        let msg = match run(job, &tx) {
            Ok((text, ok, reload)) => JobMsg::Done { text, ok, reload },
            Err(text) => JobMsg::Done {
                text,
                ok: false,
                reload: None,
            },
        };
        if !silent {
            let _ = tx.send(msg);
        }
    });
}
