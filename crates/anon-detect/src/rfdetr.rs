//! RF-DETR(-Seg) backend on ONNX Runtime.
//!
//! On macOS the session uses CoreML in the MLProgram format (see
//! `create_session`); elsewhere it is built by `reco_detect::create_ort_session`,
//! so execution provider selection (CUDA / TensorRT / DirectML / CPU) matches
//! the reco engine.
//!
//! Expected ONNX contract (as produced by `rfdetr`'s `model.export()`):
//!
//! | name     | shape            | content                                  |
//! |----------|------------------|------------------------------------------|
//! | `input`  | `[1, 3, S, S]`   | RGB, 0..1, ImageNet mean/std normalized  |
//! | `dets`   | `[1, Q, 4]`      | boxes, cx cy w h, normalized to 0..1     |
//! | `labels` | `[1, Q, C]`      | class logits (sigmoid)                   |
//! | `masks`  | `[1, Q, Hm, Wm]` | mask logits over the whole input (seg)   |
//!
//! Outputs are matched by name first, then by rank/shape, so a re-export
//! with different names still works. `masks` is optional (detection-only
//! models fall back to box-based head derivation).

use std::path::Path;

use anon_model::BBox;
use ort::session::Session;
use ort::value::Tensor;

use crate::image::RgbImage;
use crate::{DetectError, InstanceMask, PersonInstance, PersonSegmenter};

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
/// Queries below this score are not even decoded (cheap pre-filter; the
/// analyzer applies the real threshold).
const PREFILTER: f32 = 0.05;

pub struct RfDetrSeg {
    session: Session,
    size: u32,
    person_class: usize,
    names: Names,
}

#[derive(Debug, Clone)]
struct Names {
    boxes: String,
    logits: String,
    masks: Option<String>,
}

fn err(e: impl std::fmt::Display) -> DetectError {
    DetectError::Model(e.to_string())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// CoreML in the MLProgram format, on the GPU. reco-detect's session uses the
/// older NeuralNetwork format, which splits RF-DETR into ~100 CoreML/CPU
/// partitions and on Apple silicon is slower than the CPU alone (M4 Pro,
/// seg-medium at 432: NeuralNetwork 364 ms, CPU 115 ms, MLProgram 42 ms). The
/// Neural Engine was 2-3x slower than the GPU for every RF-DETR size.
#[cfg(feature = "coreml")]
fn create_session(path: &Path) -> Result<(Session, u32), DetectError> {
    use ort::ep::coreml::{ComputeUnits, ModelFormat};
    use ort::session::builder::GraphOptimizationLevel;

    reco_detect::ort_session::ort_runtime_available().map_err(DetectError::Model)?;
    let cache = reco_detect::ort_session::reco_cache_dir("coreml-mlprogram-cache");
    let builder = Session::builder()
        .map_err(err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(err)?;
    let mut builder = match builder.with_execution_providers([ort::ep::CoreML::default()
        .with_model_format(ModelFormat::MLProgram)
        .with_compute_units(ComputeUnits::CPUAndGPU)
        .with_static_input_shapes(true)
        .with_model_cache_dir(cache.to_string_lossy())
        .build()])
    {
        Ok(b) => {
            log::info!("ORT: CoreML (MLProgram, GPU) execution provider enabled");
            b
        }
        Err(e) => {
            log::warn!("ORT: CoreML EP failed ({e}), falling back to CPU");
            e.recover()
        }
    };
    let session = builder.commit_from_file(path).map_err(err)?;
    let size = match session.inputs()[0].dtype() {
        ort::value::ValueType::Tensor { shape, .. } if shape.len() == 4 && shape[2] > 0 => {
            shape[2] as u32
        }
        _ => {
            return Err(DetectError::Model(
                "model input must be [1, 3, S, S] with a fixed S".into(),
            ));
        }
    };
    Ok((session, size))
}

#[cfg(not(feature = "coreml"))]
fn create_session(path: &Path) -> Result<(Session, u32), DetectError> {
    let (session, size, _labels) =
        reco_detect::create_ort_session(path, vec!["person".into()]).map_err(err)?;
    Ok((session, size))
}

impl RfDetrSeg {
    /// Load a model. `person_class` is the index of "person" in the logits
    /// (1 for RF-DETR COCO checkpoints, which use COCO category ids).
    pub fn load(path: &Path, person_class: usize) -> Result<Self, DetectError> {
        let (session, size) = create_session(path)?;
        let names = Self::resolve_outputs(&session)?;
        log::info!(
            "RF-DETR loaded: input {size}x{size}, outputs {:?}, person class {person_class}",
            names
        );
        Ok(Self {
            session,
            size,
            person_class,
            names,
        })
    }

    fn resolve_outputs(session: &Session) -> Result<Names, DetectError> {
        let mut boxes = None;
        let mut logits = None;
        let mut masks = None;
        for o in session.outputs() {
            let name = o.name().to_string();
            let rank = match o.dtype() {
                ort::value::ValueType::Tensor { shape, .. } => {
                    Some((shape.len(), shape.last().copied()))
                }
                _ => None,
            };
            match (name.as_str(), rank) {
                ("dets" | "boxes" | "pred_boxes", _) => boxes = Some(name),
                ("labels" | "logits" | "pred_logits", _) => logits = Some(name),
                ("masks" | "pred_masks", _) => masks = Some(name),
                (_, Some((4, _))) if masks.is_none() => masks = Some(name),
                (_, Some((3, Some(4)))) if boxes.is_none() => boxes = Some(name),
                (_, Some((3, _))) if logits.is_none() => logits = Some(name),
                _ => {}
            }
        }
        Ok(Names {
            boxes: boxes.ok_or_else(|| DetectError::Output("no box output".into()))?,
            logits: logits.ok_or_else(|| DetectError::Output("no logits output".into()))?,
            masks,
        })
    }

    fn preprocess(&self, tile: &RgbImage) -> Vec<f32> {
        let n = (self.size * self.size) as usize;
        let mut out = vec![0.0f32; 3 * n];
        for (i, px) in tile.data.as_chunks::<3>().0.iter().enumerate() {
            for c in 0..3 {
                out[c * n + i] = (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
        out
    }
}

impl PersonSegmenter for RfDetrSeg {
    fn input_size(&self) -> u32 {
        self.size
    }

    fn segment(&mut self, tile: &RgbImage) -> Result<Vec<PersonInstance>, DetectError> {
        if tile.width != self.size || tile.height != self.size {
            return Err(DetectError::Model(format!(
                "tile {}x{} != model input {}",
                tile.width, tile.height, self.size
            )));
        }
        let s = self.size as usize;
        let input = Tensor::from_array(([1usize, 3, s, s], self.preprocess(tile))).map_err(err)?;
        let names = self.names.clone();
        let outputs = self.session.run(ort::inputs![input]).map_err(err)?;

        let (bshape, boxes) = outputs[names.boxes.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(err)?;
        let (lshape, logits) = outputs[names.logits.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(err)?;
        let q = bshape[1] as usize;
        let c = lshape[2] as usize;
        if self.person_class >= c || lshape[1] as usize != q {
            return Err(DetectError::Output(format!(
                "logits shape {lshape:?} incompatible with boxes {bshape:?} / person class {}",
                self.person_class
            )));
        }
        let masks = match &names.masks {
            Some(n) => {
                let (mshape, m) = outputs[n.as_str()]
                    .try_extract_tensor::<f32>()
                    .map_err(err)?;
                Some((mshape[2] as usize, mshape[3] as usize, m))
            }
            None => None,
        };

        let sz = self.size as f32;
        let mut out = Vec::new();
        for i in 0..q {
            let score = sigmoid(logits[i * c + self.person_class]);
            if score < PREFILTER {
                continue;
            }
            let (cx, cy, w, h) = (
                boxes[i * 4],
                boxes[i * 4 + 1],
                boxes[i * 4 + 2],
                boxes[i * 4 + 3],
            );
            let bbox = BBox::new(
                (cx - w / 2.0) * sz,
                (cy - h / 2.0) * sz,
                (cx + w / 2.0) * sz,
                (cy + h / 2.0) * sz,
            );
            if bbox.area() < 1.0 {
                continue;
            }
            // With a mask, the box is snapped outwards to the mask grid so
            // the grid covers it exactly (never smaller than the raw box).
            let (bbox, mask) = match masks
                .as_ref()
                .and_then(|(mh, mw, m)| crop_mask(m, i, *mh, *mw, &bbox, sz))
            {
                Some((b, m)) => (b, Some(m)),
                None => (bbox, None),
            };
            out.push(PersonInstance {
                bbox,
                score,
                mask,
                truncated: false,
                passes: vec![],
            });
        }
        Ok(out)
    }
}

/// Cut the instance's mask logits (covering the whole input) down to the
/// mask cells overlapping its box, thresholded at logit 0. Returns the
/// cell-aligned box the mask covers.
fn crop_mask(
    m: &[f32],
    q: usize,
    mh: usize,
    mw: usize,
    b: &BBox,
    input: f32,
) -> Option<(BBox, InstanceMask)> {
    let sx = mw as f32 / input;
    let sy = mh as f32 / input;
    let x0 = ((b.x0 * sx).floor().max(0.0) as usize).min(mw.saturating_sub(1));
    let y0 = ((b.y0 * sy).floor().max(0.0) as usize).min(mh.saturating_sub(1));
    let x1 = ((b.x1 * sx).ceil() as usize).clamp(x0 + 1, mw);
    let y1 = ((b.y1 * sy).ceil() as usize).clamp(y0 + 1, mh);
    let (w, h) = (x1 - x0, y1 - y0);
    let base = q * mh * mw;
    let mut bits = Vec::with_capacity(w * h);
    for y in y0..y1 {
        for x in x0..x1 {
            bits.push(m[base + y * mw + x] > 0.0);
        }
    }
    if !bits.iter().any(|&b| b) {
        return None;
    }
    let snapped = BBox::new(
        x0 as f32 / sx,
        y0 as f32 / sy,
        x1 as f32 / sx,
        y1 as f32 / sy,
    );
    Some((
        snapped,
        InstanceMask {
            w: w as u32,
            h: h as u32,
            bits,
        },
    ))
}

/// `<model>.json` next to the ONNX file, written by the scripts in
/// `scripts/`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelManifest {
    pub id: String,
    pub sha256: String,
    pub license: String,
    /// "person-seg" (default), "person-det" (boxes only) or "text-db".
    #[serde(default = "default_kind")]
    pub kind: String,
    #[serde(default = "default_person_class")]
    pub person_class: usize,
}

fn default_kind() -> String {
    "person-seg".into()
}

fn default_person_class() -> usize {
    1
}

/// Read the manifest, check the model file's sha256 and that its kind is one
/// of `kinds`.
pub fn verify_manifest(
    path: &Path,
    kinds: &[&str],
) -> Result<(ModelManifest, anon_model::ModelRef), DetectError> {
    use sha2::{Digest, Sha256};
    let mut mp = path.as_os_str().to_owned();
    mp.push(".json");
    let mp = std::path::PathBuf::from(mp);
    let bytes = std::fs::read(&mp)
        .map_err(|e| DetectError::Model(format!("model manifest {} missing: {e}", mp.display())))?;
    let manifest: ModelManifest = serde_json::from_slice(&bytes)
        .map_err(|e| DetectError::Model(format!("invalid model manifest: {e}")))?;
    if !kinds.contains(&manifest.kind.as_str()) {
        return Err(DetectError::Model(format!(
            "{} is a '{}' model, expected '{}'",
            path.display(),
            manifest.kind,
            kinds.join("' or '")
        )));
    }
    let model_bytes = std::fs::read(path)
        .map_err(|e| DetectError::Model(format!("reading {}: {e}", path.display())))?;
    let sha = hex::encode(Sha256::digest(&model_bytes));
    if sha != manifest.sha256 {
        return Err(DetectError::Model(format!(
            "model hash {sha} does not match manifest {}; refusing unknown model",
            manifest.sha256
        )));
    }
    let r = anon_model::ModelRef {
        id: manifest.id.clone(),
        sha256: sha,
        license: manifest.license.clone(),
    };
    Ok((manifest, r))
}

/// The `kind` in a model's manifest, without checking its hash (for UI
/// hints; loading still goes through [`load_verified`]).
pub fn manifest_kind(path: &Path) -> Option<String> {
    let mut mp = path.as_os_str().to_owned();
    mp.push(".json");
    let bytes = std::fs::read(std::path::PathBuf::from(mp)).ok()?;
    serde_json::from_slice::<ModelManifest>(&bytes)
        .ok()
        .map(|m| m.kind)
}

/// True for detection-only person models: heads are then estimated from the
/// person box (top fifth, full width), which is much coarser than from a mask.
pub fn boxes_only(path: &Path) -> bool {
    manifest_kind(path).as_deref() == Some("person-det")
}

/// Load a person model (segmentation, or detection-only with box-based head
/// derivation) only if its sha256 matches its manifest.
pub fn load_verified(path: &Path) -> Result<(RfDetrSeg, anon_model::ModelRef), DetectError> {
    let (manifest, r) = verify_manifest(path, &["person-seg", "person-det"])?;
    Ok((RfDetrSeg::load(path, manifest.person_class)?, r))
}
