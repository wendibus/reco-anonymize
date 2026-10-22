//! Export reviewed head labels as a training dataset for a dedicated head
//! detector (milestone M5).
//!
//! Layout matches RF-DETR / Reco Trainer: `train/`, `valid/`, `test/`, each
//! with images and `_annotations.coco.json`, category ids starting at 1.
//!
//! Labels come from the review: automatic head detections (tight head
//! boxes from `detections.jsonl`) that belong to non-suppressed tracks, plus
//! keyframes of manual tracks. Only approved projects are exported by
//! default, so every label has passed a human review.
//!
//! The dataset contains *un-anonymized* frames. It is meant for local
//! training only; the export writes a README saying so.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{BufRead, BufReader, BufWriter};
use std::path::{Path, PathBuf};

use anon_io::{FrameReader, RgbConverter};
use anon_model::hash::approval_hash;
use anon_model::{BBox, FrameStatus, Origin, ProjectDir, RegionKind, Shape};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::PipelineError;

#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Take every n-th analysed frame.
    pub every: u64,
    /// Fractions of blocks going to valid and test (rest: train).
    pub valid: f32,
    pub test: f32,
    /// Frames per split block; whole blocks go to one split so near-
    /// identical neighbouring frames never end up in train *and* test.
    pub block: u64,
    /// Also export projects without a valid approval.
    pub allow_unreviewed: bool,
    pub jpeg_quality: u8,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            every: 15,
            valid: 0.15,
            test: 0.10,
            block: 300,
            allow_unreviewed: false,
            jpeg_quality: 95,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ExportSummary {
    pub projects: usize,
    pub images: [usize; 3],
    pub boxes: [usize; 3],
}

const SPLITS: [&str; 3] = ["train", "valid", "test"];

#[derive(Deserialize)]
struct DetLine {
    frame: u64,
    head: [f32; 4],
    #[serde(default)]
    kind: Option<String>,
}

/// Deterministic split of a block index (stable across runs).
fn split_of(block: u64, valid: f32, test: f32) -> usize {
    // spread blocks evenly: golden-ratio sequence in [0, 1)
    let u = (block as f64 * 0.618_033_988_749_895).fract() as f32;
    if u < test {
        2
    } else if u < test + valid {
        1
    } else {
        0
    }
}

fn manual_box(s: &Shape) -> Option<BBox> {
    match s {
        // manual heads are stored as ellipse_around(box, 1.0)
        Shape::Ellipse { cx, cy, rx, ry } => {
            let hw = rx / std::f32::consts::SQRT_2;
            let hh = ry / std::f32::consts::SQRT_2;
            Some(BBox::new(cx - hw, cy - hh, cx + hw, cy + hh))
        }
        Shape::Polygon { .. } => None,
    }
}

struct Labels {
    /// frame -> head boxes
    by_frame: BTreeMap<u64, Vec<BBox>>,
}

/// Collect reviewed head boxes of one project.
fn labels(dir: &ProjectDir) -> Result<Labels, PipelineError> {
    let tracks = dir.load_tracks()?;
    // centres of suppressed auto head tracks per frame: the detections
    // behind them are false positives and must not become labels
    let mut suppressed: HashMap<u64, Vec<(f32, f32)>> = HashMap::new();
    let mut by_frame: BTreeMap<u64, Vec<BBox>> = BTreeMap::new();
    for t in &tracks.tracks {
        if t.kind != RegionKind::Head {
            continue;
        }
        match (&t.origin, &t.suppressed) {
            (Origin::Manual, None) => {
                for k in t.keyframes.iter().filter(|k| !k.interpolated) {
                    if let Some(b) = manual_box(&k.shape) {
                        by_frame.entry(k.frame).or_default().push(b);
                    }
                }
            }
            (Origin::Auto { .. }, Some(_)) => {
                for k in t.keyframes.iter().filter(|k| !k.interpolated) {
                    suppressed
                        .entry(k.frame)
                        .or_default()
                        .push(k.shape.bounds().center());
                }
            }
            _ => {}
        }
    }
    let path = dir.path("detections.jsonl");
    if path.exists() {
        for line in BufReader::new(fs::File::open(&path)?).lines() {
            let Ok(dl) = serde_json::from_str::<DetLine>(&line?) else {
                continue;
            };
            if dl.kind.as_deref().is_some_and(|k| k != "head") {
                continue;
            }
            let b = BBox::new(dl.head[0], dl.head[1], dl.head[2], dl.head[3]);
            // a detection belongs to a suppressed track if that track's
            // ellipse centre lies inside the head box at the same frame
            let hit = suppressed.get(&dl.frame).is_some_and(|cs| {
                cs.iter().any(|&(x, y)| {
                    x >= b.x0 - 1.0 && x <= b.x1 + 1.0 && y >= b.y0 - 1.0 && y <= b.y1 + 1.0
                })
            });
            if !hit {
                by_frame.entry(dl.frame).or_default().push(b);
            }
        }
    }
    Ok(Labels { by_frame })
}

fn clip_box(b: &BBox, w: u32, h: u32) -> Option<BBox> {
    let c = BBox::new(
        b.x0.clamp(0.0, w as f32),
        b.y0.clamp(0.0, h as f32),
        b.x1.clamp(0.0, w as f32),
        b.y1.clamp(0.0, h as f32),
    );
    (c.width() >= 2.0 && c.height() >= 2.0).then_some(c)
}

/// Export the given projects into `out`.
pub fn export_dataset(
    projects: &[ProjectDir],
    out: &Path,
    opts: &ExportOptions,
    mut progress: impl FnMut(&str, u64),
) -> Result<ExportSummary, PipelineError> {
    anon_io::init()?;
    if opts.every == 0 || opts.block == 0 || opts.valid + opts.test >= 1.0 {
        return Err(PipelineError::Refused("invalid export options".into()));
    }
    let mut summary = ExportSummary::default();
    let mut coco: Vec<(Vec<serde_json::Value>, Vec<serde_json::Value>)> =
        vec![(Vec::new(), Vec::new()); 3];
    for s in SPLITS {
        fs::create_dir_all(out.join(s))?;
    }
    let mut sources = Vec::new();

    for dir in projects {
        let p = dir.load_project()?;
        let tracks = dir.load_tracks()?;
        let frames = dir.load_frames()?;
        let approved = p.review.approved
            && p.review.approved_hash.as_deref()
                == Some(approval_hash(&p.source.fingerprint, &tracks, &p.zones, &frames).as_str());
        if !approved && !opts.allow_unreviewed {
            return Err(PipelineError::Refused(format!(
                "{} is not approved; review it first or allow unreviewed labels",
                dir.root.display()
            )));
        }
        if anon_io::fingerprint(&p.source.path)? != p.source.fingerprint {
            return Err(PipelineError::Refused(format!(
                "{} changed since analysis",
                p.source.path.display()
            )));
        }
        let labels = labels(dir)?;
        let stem: String = p
            .source
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "clip".into())
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let prefix = format!("{stem}_{}", &p.source.fingerprint.head_tail_sha256[..8]);
        sources.push(json!({
            "file": p.source.path.file_name().map(|n| n.to_string_lossy().into_owned()),
            "fingerprint": p.source.fingerprint.head_tail_sha256,
            "approved": approved,
        }));

        let mut reader = FrameReader::open(&p.source.path)?;
        let mut rgb = RgbConverter::default();
        let mut index = 0u64;
        while let Some((frame, pts)) = reader.next_frame()? {
            let i = index;
            index += 1;
            progress(&prefix, i);
            if !i.is_multiple_of(opts.every) {
                continue;
            }
            match frames.get(i) {
                Some(r) if r.status == FrameStatus::Analyzed && r.pts == pts => {}
                _ => continue,
            }
            let img = rgb.convert(&frame)?;
            let boxes: Vec<BBox> = labels
                .by_frame
                .get(&i)
                .map(|v| {
                    v.iter()
                        .filter_map(|b| clip_box(b, img.width, img.height))
                        .collect()
                })
                .unwrap_or_default();
            let split = split_of(i / opts.block, opts.valid, opts.test);
            let file_name = format!("{prefix}_{i:07}.jpg");
            let path = out.join(SPLITS[split]).join(&file_name);
            let buf = image::RgbImage::from_raw(img.width, img.height, img.data)
                .ok_or_else(|| PipelineError::Refused("frame buffer size".into()))?;
            let mut w = BufWriter::new(fs::File::create(&path)?);
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut w, opts.jpeg_quality)
                .encode_image(&buf)
                .map_err(|e| PipelineError::Refused(format!("jpeg: {e}")))?;
            let (images, anns) = &mut coco[split];
            let image_id = images.len() + 1;
            images.push(json!({
                "id": image_id,
                "file_name": file_name,
                "width": img.width,
                "height": img.height,
            }));
            for b in &boxes {
                anns.push(json!({
                    "id": anns.len() + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [b.x0, b.y0, b.width(), b.height()],
                    "area": b.area(),
                    "iscrowd": 0,
                }));
            }
            summary.images[split] += 1;
            summary.boxes[split] += boxes.len();
        }
        summary.projects += 1;
    }

    for (k, s) in SPLITS.iter().enumerate() {
        let (images, annotations) = &coco[k];
        let doc = json!({
            "info": {
                "description": "reco-anonymize head dataset (reviewed labels)",
                "version": "1",
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "head", "supercategory": "person"}],
        });
        fs::write(
            out.join(s).join("_annotations.coco.json"),
            serde_json::to_vec_pretty(&doc).expect("serializable"),
        )?;
    }
    fs::write(
        out.join("dataset-info.json"),
        serde_json::to_vec_pretty(&json!({
            "generator": format!("reco-anonymize {}", env!("CARGO_PKG_VERSION")),
            "category": "head",
            "every_nth_frame": opts.every,
            "split_block_frames": opts.block,
            "sources": sources,
            "images": summary.images,
            "boxes": summary.boxes,
        }))
        .expect("serializable"),
    )?;
    fs::write(
        out.join("README.txt"),
        "This folder contains UN-ANONYMIZED video frames with head labels for training a local \
         head detector (RF-DETR / Reco Trainer COCO layout). Keep it on this computer. Do not \
         share or upload it.\n",
    )?;
    Ok(summary)
}

/// Default dataset folder next to the first project's source video.
pub fn default_out(first: &ProjectDir) -> Result<PathBuf, PipelineError> {
    let p = first.load_project()?;
    Ok(p.source
        .path
        .parent()
        .unwrap_or(Path::new("."))
        .join("head-dataset"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_is_deterministic_and_proportional() {
        let mut n = [0usize; 3];
        for b in 0..1000 {
            n[split_of(b, 0.15, 0.10)] += 1;
            assert_eq!(split_of(b, 0.15, 0.10), split_of(b, 0.15, 0.10));
        }
        assert!((730..=770).contains(&n[0]), "{n:?}");
        assert!((130..=170).contains(&n[1]), "{n:?}");
        assert!((80..=120).contains(&n[2]), "{n:?}");
    }

    #[test]
    fn manual_ellipse_back_to_box() {
        let b = BBox::new(10.0, 20.0, 30.0, 60.0);
        let back = manual_box(&Shape::ellipse_around(&b, 1.0)).unwrap();
        assert!((back.x0 - 10.0).abs() < 1e-3 && (back.y1 - 60.0).abs() < 1e-3);
    }
}
