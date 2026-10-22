//! ORT text backend against a synthetic model with PP-OCR det's contract.

#![cfg(feature = "ort")]

use anon_detect::RgbImage;
use anon_detect::ppocr::PpOcrDet;
use anon_detect::text::{DbTextDetector, TextDetector};
use anon_model::BBox;
use std::path::Path;

#[test]
fn finds_bright_number_via_onnx() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_ppocr.onnx");
    let mut d = DbTextDetector {
        model: PpOcrDet::load(&path).expect("load"),
    };
    let mut img = RgbImage::from_raw(320, 240, vec![30; 320 * 240 * 3]);
    for y in 100..130 {
        for x in 60..90 {
            let i = ((y * 320 + x) * 3) as usize;
            img.data[i..i + 3].copy_from_slice(&[255, 255, 255]);
        }
    }
    let out = d
        .detect(&img, &[BBox::new(40.0, 40.0, 110.0, 220.0)])
        .unwrap();
    assert_eq!(out.len(), 1, "{out:?}");
    let b = out[0].bbox;
    assert!(
        b.x0 <= 60.0 && b.y0 <= 100.0 && b.x1 >= 90.0 && b.y1 >= 130.0,
        "{b:?}"
    );
}
