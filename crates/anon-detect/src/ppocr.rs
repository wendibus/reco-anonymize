//! PaddleOCR DB text detector (PP-OCR det, Apache-2.0) on ONNX Runtime.
//!
//! Input `[1, 3, H, W]`, BGR, `(x/255 - mean) / std` with ImageNet
//! mean/std (PaddleOCR's `NormalizeImage` on its BGR-decoded images);
//! output probability map `[1, 1, H, W]`.

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;

use crate::DetectError;
use crate::image::RgbImage;
use crate::rfdetr::verify_manifest;
use crate::text::{DbTextDetector, TextMapModel};

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct PpOcrDet {
    session: Session,
}

fn err(e: impl std::fmt::Display) -> DetectError {
    DetectError::Model(e.to_string())
}

impl PpOcrDet {
    pub fn load(path: &Path) -> Result<Self, DetectError> {
        let (session, _, _) =
            reco_detect::create_ort_session(path, vec!["text".into()]).map_err(err)?;
        Ok(Self { session })
    }
}

impl TextMapModel for PpOcrDet {
    fn prob_map(&mut self, img: &RgbImage) -> Result<Vec<f32>, DetectError> {
        let (w, h) = (img.width as usize, img.height as usize);
        let n = w * h;
        let mut x = vec![0.0f32; 3 * n];
        for (i, px) in img.data.chunks_exact(3).enumerate() {
            // BGR channel order
            for (c, &v) in [px[2], px[1], px[0]].iter().enumerate() {
                x[c * n + i] = (f32::from(v) / 255.0 - MEAN[c]) / STD[c];
            }
        }
        let input = Tensor::from_array(([1usize, 3, h, w], x)).map_err(err)?;
        let out = self.session.run(ort::inputs![input]).map_err(err)?;
        let (shape, data) = out[0].try_extract_tensor::<f32>().map_err(err)?;
        if shape.iter().product::<i64>() as usize != n {
            return Err(DetectError::Output(format!(
                "text map shape {shape:?} for {w}x{h} input"
            )));
        }
        Ok(data.to_vec())
    }
}

/// Load a PP-OCR det model whose sha256 matches its manifest
/// (`kind: "text-db"`).
pub fn load_verified(
    path: &Path,
) -> Result<(DbTextDetector<PpOcrDet>, anon_model::ModelRef), DetectError> {
    let (_, r) = verify_manifest(path, "text-db")?;
    Ok((
        DbTextDetector {
            model: PpOcrDet::load(path)?,
        },
        r,
    ))
}
