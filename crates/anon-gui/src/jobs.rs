//! Long-running work on worker threads: analyze, preview, render, audit.

use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;

use anon_io::{Codec, EncodeSettings};
use anon_model::{AnalysisSettings, ProjectDir};
use anon_pipeline::{AnalyzeOptions, RenderOptions};

pub enum Job {
    Analyze {
        src: PathBuf,
        model: PathBuf,
        /// Mask jersey numbers with this text model.
        text: Option<PathBuf>,
        resume: bool,
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
}

impl Job {
    pub fn title(&self) -> &'static str {
        match self {
            Job::Analyze { .. } => "analyzing…",
            Job::Preview { .. } => "rendering preview…",
            Job::Render { .. } => "rendering…",
            Job::Audit { .. } => "auditing…",
        }
    }
}

pub enum JobMsg {
    /// `fraction < 0` = indeterminate.
    Progress { fraction: f32, text: String },
    /// `reload`: project path to reopen afterwards.
    Done {
        text: String,
        ok: bool,
        reload: Option<PathBuf>,
    },
}

fn config_file(name: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let dir = PathBuf::from(home).join(".reco-anonymize");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join(name))
}

pub fn last_model() -> Option<PathBuf> {
    let p = std::fs::read_to_string(config_file("last-model")?).ok()?;
    let p = PathBuf::from(p.trim());
    p.exists().then_some(p)
}

pub fn last_text_model() -> Option<PathBuf> {
    let p = std::fs::read_to_string(config_file("last-text-model")?).ok()?;
    let p = PathBuf::from(p.trim());
    p.exists().then_some(p)
}

pub fn remember_text_model(p: &Path) {
    if let Some(f) = config_file("last-text-model") {
        let _ = std::fs::write(f, p.display().to_string());
    }
}

pub fn remember_model(p: &Path) {
    if let Some(f) = config_file("last-model") {
        let _ = std::fs::write(f, p.display().to_string());
    }
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

fn frames_progress(tx: &Sender<JobMsg>, label: &str) -> impl FnMut(u64, Option<u64>) {
    let tx = tx.clone();
    let label = label.to_string();
    move |done, total| {
        if done % 10 != 0 {
            return;
        }
        let (fraction, text) = match total {
            Some(t) if t > 0 => (
                (done as f32 / t as f32).min(1.0),
                format!("{label}: frame {done} / {t}"),
            ),
            _ => (-1.0, format!("{label}: frame {done}")),
        };
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
) -> Result<String, String> {
    let (m, mref) = anon_detect::rfdetr::load_verified(model).map_err(|e| e.to_string())?;
    let mut p = frames_progress(tx, "audit");
    let r = anon_pipeline::audit(dir, out, Box::new(m), &mref.sha256, |n| p(n, None))
        .map_err(|e| e.to_string())?;
    Ok(if r.passed() {
        format!("audit PASSED: {} frames, no uncovered head", r.frames)
    } else {
        format!(
            "audit FAILED: {} uncovered head(s) — see audit-report.json, fix masks and render again",
            r.open_hits()
        )
    })
}

fn run(job: Job, tx: &Sender<JobMsg>) -> Result<(String, Option<PathBuf>), String> {
    match job {
        Job::Analyze {
            src,
            model,
            text,
            resume,
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
                format!(
                    "analysis done: {} frames, {} head tracks, {} text tracks, {} flagged, {} errors",
                    s.frames, s.tracks, s.text_tracks, s.flagged, s.errors
                ),
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
            Ok((format!("preview written: {}", r.output.display()), None))
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
            let mut text = format!(
                "rendered {} frames via {}, timing verified",
                r.frames, r.encoder
            );
            if let Some(model) = model {
                let _ = tx.send(JobMsg::Progress {
                    fraction: -1.0,
                    text: "auditing the export…".into(),
                });
                text += "\n";
                text += &audit(&dir, &r.output, &model, tx)?;
            }
            Ok((text, Some(dir.root)))
        }
        Job::Audit { dir, out, model } => Ok((audit(&dir, &out, &model, tx)?, Some(dir.root))),
    }
}

pub fn spawn(job: Job, tx: Sender<JobMsg>) {
    std::thread::spawn(move || {
        let msg = match run(job, &tx) {
            Ok((text, reload)) => JobMsg::Done {
                ok: !text.contains("audit FAILED"),
                text,
                reload,
            },
            Err(text) => JobMsg::Done {
                text,
                ok: false,
                reload: None,
            },
        };
        let _ = tx.send(msg);
    });
}
