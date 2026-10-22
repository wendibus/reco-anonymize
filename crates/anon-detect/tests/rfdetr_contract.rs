//! Exercises the ORT backend against a synthetic model with RF-DETR-Seg's
//! I/O contract (see scripts/make_fake_rfdetr.py). Requires an ONNX Runtime
//! library: linked by default, or via ORT_DYLIB_PATH with `load-dynamic`.

#![cfg(feature = "ort")]

use anon_detect::rfdetr::RfDetrSeg;
use anon_detect::{PersonSegmenter, RgbImage};
use std::path::Path;

#[test]
fn decodes_person_query_with_mask() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_rfdetr.onnx");
    let mut m = RfDetrSeg::load(&path, 1).expect("load");
    assert_eq!(m.input_size(), 64);
    let out = m.segment(&RgbImage::new(64, 64)).expect("run");
    // query 1 is another class, query 2 is below the prefilter
    assert_eq!(out.len(), 1, "{out:?}");
    let p = &out[0];
    assert!((p.score - 0.982).abs() < 0.01);
    // box cx .5 w .25 -> 24..40, cy .5 h .5 -> 16..48, snapped to 4px cells
    assert_eq!(
        (p.bbox.x0, p.bbox.y0, p.bbox.x1, p.bbox.y1),
        (24.0, 16.0, 40.0, 48.0)
    );
    let mask = p.mask.as_ref().expect("mask");
    assert_eq!((mask.w, mask.h), (4, 8));
    assert!(mask.bits.iter().all(|&b| b));
}
