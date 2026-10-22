//! Post-render audit: run the detector again on the *rendered* video and
//! check, from the actual output pixels, that every head it finds is
//! covered by the opaque fill. Independent of the mask data that produced
//! the render, so it also catches bugs in tracking, resolving or painting.

use std::path::{Path, PathBuf};

use anon_detect::PersonSegmenter;
use anon_detect::RgbImage;
use anon_detect::analyzer::{Analyzer, AnalyzerConfig};
use anon_detect::image::unrotate_point;
use anon_io::{FrameReader, RgbConverter};
use anon_model::{BBox, Fingerprint, ProjectDir};
use serde::{Deserialize, Serialize};

use crate::{PipelineError, now_unix};

/// Detection threshold used by the audit (more sensitive than analysis).
pub const AUDIT_THRESHOLD: f32 = 0.10;
/// A head counts as masked when this fraction of its box is fill-black.
pub const MIN_COVERAGE: f32 = 0.95;
/// Max channel value still considered fill-black after lossy encoding.
const BLACK_MAX: u8 = 24;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuditHit {
    pub frame: u64,
    pub pts: i64,
    pub head: [f32; 4],
    pub person: [f32; 4],
    pub score: f32,
    pub coverage: f32,
    /// Reviewer's reason when acknowledged as not a leak.
    pub acknowledged: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    pub output: PathBuf,
    pub output_fingerprint: Fingerprint,
    pub model_sha256: String,
    pub at: u64,
    pub frames: u64,
    pub hits: Vec<AuditHit>,
}

impl AuditReport {
    pub fn open_hits(&self) -> usize {
        self.hits
            .iter()
            .filter(|h| h.acknowledged.is_none())
            .count()
    }

    pub fn passed(&self) -> bool {
        self.open_hits() == 0
    }
}

const REPORT: &str = "audit-report.json";

pub fn load_report(dir: &ProjectDir) -> Option<AuditReport> {
    serde_json::from_slice(&std::fs::read(dir.path(REPORT)).ok()?).ok()
}

fn save_report(dir: &ProjectDir, r: &AuditReport) -> Result<(), PipelineError> {
    std::fs::write(
        dir.path(REPORT),
        serde_json::to_vec_pretty(r).expect("serializable"),
    )?;
    Ok(())
}

/// Fraction of pixels in `b` (clipped to the image) that are fill-black.
pub fn black_fraction(img: &RgbImage, b: &BBox) -> f32 {
    let x0 = b.x0.floor().max(0.0) as u32;
    let y0 = b.y0.floor().max(0.0) as u32;
    let x1 = (b.x1.ceil().max(0.0) as u32).min(img.width);
    let y1 = (b.y1.ceil().max(0.0) as u32).min(img.height);
    if x0 >= x1 || y0 >= y1 {
        return 1.0;
    }
    let mut black = 0u64;
    for y in y0..y1 {
        for x in x0..x1 {
            if img.px(x, y).iter().all(|&c| c <= BLACK_MAX) {
                black += 1;
            }
        }
    }
    black as f32 / (u64::from(x1 - x0) * u64::from(y1 - y0)) as f32
}

fn unrotate_box(b: &BBox, deg: i32, w: f32, h: f32) -> BBox {
    let (x0, y0) = unrotate_point(deg, w, h, b.x0, b.y0);
    let (x1, y1) = unrotate_point(deg, w, h, b.x1, b.y1);
    BBox::new(x0, y0, x1, y1)
}

/// Audit `rendered` (the output of `render` for this project).
pub fn audit(
    dir: &ProjectDir,
    rendered: &Path,
    model: Box<dyn PersonSegmenter>,
    model_sha256: &str,
    mut progress: impl FnMut(u64),
) -> Result<AuditReport, PipelineError> {
    anon_io::init()?;
    let p = dir.load_project()?;
    let probe = anon_io::probe(rendered)?;
    if (probe.width, probe.height) != (p.source.width, p.source.height) {
        return Err(PipelineError::Verify(format!(
            "{} is {}x{}, project source is {}x{}",
            rendered.display(),
            probe.width,
            probe.height,
            p.source.width,
            p.source.height
        )));
    }
    let cfg = AnalyzerConfig {
        tile: p.settings.tile.clone(),
        person_threshold: AUDIT_THRESHOLD.min(p.settings.person_threshold),
        flag_below: 1.0,
        head_expand: 1.0,
    };
    let mut analyzer = Analyzer::new(model, cfg);
    let rot = probe.rotation_deg;
    let (w, h) = (probe.width as f32, probe.height as f32);
    let (dw, dh) = if rot % 180 == 0 { (w, h) } else { (h, w) };

    let mut reader = FrameReader::open(rendered)?;
    let mut rgb = RgbConverter::default();
    let mut hits = Vec::new();
    let mut index = 0u64;
    while let Some((frame, pts)) = reader.next_frame()? {
        let img = rgb.convert(&frame)?;
        let det = analyzer.analyze(&img.rotate_cw(rot)).map_err(|e| {
            PipelineError::Verify(format!("audit inference failed at frame {index}: {e}"))
        })?;
        for hd in det.heads {
            let head = unrotate_box(&hd.head, rot, dw, dh);
            let person = unrotate_box(&hd.person, rot, dw, dh);
            // If the head is masked, a segmenter may leave the black blob out
            // of the silhouette; then the band above the person is the head.
            let band = {
                let up = rot.rem_euclid(360);
                let ph = 0.3 * person.height().max(person.width());
                match up {
                    // display "up" in stored coordinates
                    90 => BBox::new(person.x0 - ph, person.y0, person.x0, person.y1),
                    180 => BBox::new(person.x0, person.y1, person.x1, person.y1 + ph),
                    270 => BBox::new(person.x1, person.y0, person.x1 + ph, person.y1),
                    _ => BBox::new(person.x0, person.y0 - ph, person.x1, person.y0),
                }
            };
            let coverage = black_fraction(&img, &head).max(black_fraction(&img, &band));
            if coverage < MIN_COVERAGE {
                hits.push(AuditHit {
                    frame: index,
                    pts,
                    head: [head.x0, head.y0, head.x1, head.y1],
                    person: [person.x0, person.y0, person.x1, person.y1],
                    score: hd.score,
                    coverage,
                    acknowledged: None,
                });
            }
        }
        index += 1;
        progress(index);
    }
    let report = AuditReport {
        output: rendered
            .canonicalize()
            .unwrap_or_else(|_| rendered.to_path_buf()),
        output_fingerprint: anon_io::fingerprint(rendered)?,
        model_sha256: model_sha256.to_string(),
        at: now_unix(),
        frames: index,
        hits,
    };
    save_report(dir, &report)?;
    Ok(report)
}

/// Mark audit hits as reviewed and not a leak (e.g. a poster, a mannequin,
/// a head already fully hidden). `frames = None` acknowledges all.
pub fn acknowledge(
    dir: &ProjectDir,
    frames: Option<&[u64]>,
    reason: &str,
) -> Result<usize, PipelineError> {
    if reason.trim().is_empty() {
        return Err(PipelineError::Refused("a reason is required".into()));
    }
    let mut r = load_report(dir).ok_or_else(|| PipelineError::Refused("no audit report".into()))?;
    let mut n = 0;
    for h in &mut r.hits {
        if h.acknowledged.is_none() && frames.is_none_or(|f| f.contains(&h.frame)) {
            h.acknowledged = Some(reason.to_string());
            n += 1;
        }
    }
    save_report(dir, &r)?;
    Ok(n)
}

/// Why an export is (not) clean.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportNote {
    NotAudited,
    OutputChanged,
    Passed,
    PassedAcknowledged(usize),
    OpenHits(usize),
}

impl std::fmt::Display for ExportNote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportNote::NotAudited => write!(f, "not audited"),
            ExportNote::OutputChanged => write!(f, "output changed or missing since audit"),
            ExportNote::Passed => write!(f, "audit passed"),
            ExportNote::PassedAcknowledged(n) => write!(f, "audit passed, {n} hit(s) acknowledged"),
            ExportNote::OpenHits(n) => write!(f, "{n} open audit hit(s)"),
        }
    }
}

/// Is the rendered file a clean export: audited in its current state with
/// no open hits.
pub fn export_clean(dir: &ProjectDir) -> Result<(bool, ExportNote), PipelineError> {
    let Some(r) = load_report(dir) else {
        return Ok((false, ExportNote::NotAudited));
    };
    match anon_io::fingerprint(&r.output) {
        Ok(fp) if fp == r.output_fingerprint => {}
        _ => return Ok((false, ExportNote::OutputChanged)),
    }
    Ok(if !r.passed() {
        (false, ExportNote::OpenHits(r.open_hits()))
    } else if r.hits.is_empty() {
        (true, ExportNote::Passed)
    } else {
        (true, ExportNote::PassedAcknowledged(r.hits.len()))
    })
}
