# Third-party components in release packages

| Component | License | Notes |
|---|---|---|
| [FFmpeg](https://ffmpeg.org) and codec libraries (x264, x265, …) | GPL-3.0-or-later (GPL build) | Shared libraries bundled unmodified. Source: https://ffmpeg.org/download.html; Windows builds from https://github.com/BtbN/FFmpeg-Builds |
| [ONNX Runtime](https://onnxruntime.ai) | MIT | Linked via the `ort` crate |
| [Slint](https://slint.dev) | GPL-3.0-only (via AGPL compatibility) / Slint licenses | UI toolkit of the review app |
| [reco video-stitcher](https://github.com/reco-project/video-stitcher) (`reco-detect`) | AGPL-3.0-only | ORT session factory |
| Rust crates | MIT / Apache-2.0 and compatible | See `Cargo.lock` |

Models are **not** bundled. `scripts/export_rfdetr.py` (RF-DETR, Apache-2.0) and
`scripts/fetch_ppocr.py` (PaddleOCR det, Apache-2.0) create them locally.

reco-anonymize itself is AGPL-3.0-only. The complete corresponding source is
https://github.com/wendibus/reco-anonymize at the release tag.
