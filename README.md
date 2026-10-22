# reco-anonymize

Opaque head anonymization for reco panorama videos (and any other footage), 100 % local.
Every detected human head is covered with a solid fill; resolution, timestamps (VFR included),
colour description, rotation and all audio streams stay as they were. Container metadata (GPS,
title, device) is stripped.

Built to match the [reco](https://github.com/reco-project/video-stitcher) architecture: Rust
workspace, `ffmpeg-next` 8, ONNX Runtime sessions created by `reco-detect` (CoreML on macOS),
AGPL-3.0-only.

> Automatic detection is never 100 % reliable. The tool enforces a human review step before
> any final export, and fails closed everywhere else (see `docs/SPEC.md` §8, §13).

## Workflow

```sh
reco-anon analyze game.mp4 --model models/rfdetr-seg-medium.onnx   # -> game.mp4.anon/
reco-anon status  game.mp4                                          # flagged frames
reco-anon zone add game.mp4 --label stands --points "0,0 5760,0 5760,380 0,420"
reco-anon render  game.mp4 --preview                                # game.mp4.anon/preview.mp4
reco-anon approve game.mp4 --i-reviewed-the-preview
reco-anon render  game.mp4 -o game-anon.mp4                         # gated, verified
reco-anon audit   game.mp4 -o game-anon.mp4 --model models/rfdetr-seg-medium.onnx
reco-anon status  game.mp4                                          # export CLEAN?
```

- Long analyses checkpoint every 250 frames; after a crash continue with `analyze … --resume`.
- `audit` re-runs detection on the rendered file and checks from the output pixels that every
  head it finds is covered. Hits must be fixed or acknowledged with a reason (`audit-ack`)
  before `status` reports the export as clean. A new render voids the previous audit.

- `render` refuses without a valid approval. Any change to tracks or zones voids it.
- Frames that were not analysed, errored, or whose timestamp differs at render time are painted
  fully opaque.
- The output is verified by decoding it (frame count and timing) before it gets its final name.
- `approve` refuses a clip in which nobody was detected (likely wrong model or class index)
  unless `--no-people-expected` is given.

## Model

RF-DETR-Seg (Apache-2.0 code and COCO weights). Export once on the Mac:

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install "rfdetr>=1.3" onnx onnxsim
python scripts/export_rfdetr.py --size medium --out models/rfdetr-seg-medium.onnx
```

This writes `models/rfdetr-seg-medium.onnx.json` (sha256, licence, person class). `analyze`
refuses models whose hash does not match their manifest. Only the Apache sizes (nano … large)
are allowed; XL/2XL use a different licence.

## Build

```sh
brew install ffmpeg pkg-config
cargo build --release -p anon-cli      # binary: target/release/reco-anon (CoreML EP on macOS)
cargo test --workspace
```

On machines where the ONNX Runtime binaries cannot be downloaded at build time, build with
`--features anon-cli/load-dynamic` and point `ORT_DYLIB_PATH` at a `libonnxruntime` 1.24.

## Layout

| crate | role |
|---|---|
| `anon-model` | project files, tracks, zones, frame table, approval hash, mask resolver |
| `anon-detect` | tiling, flip + full-frame passes, fusion, head derivation, RF-DETR backend |
| `anon-track` | bidirectional Kalman/ByteTrack tracker, gap bridging, extrapolation, motion padding |
| `anon-render` | conservative span rasterizer, opaque fill in planar YUV (8/10-bit, 4:2:0/4:2:2/4:4:4) |
| `anon-io` | probe, fingerprint, decode, VFR-safe re-encode, audio stream copy, metadata strip |
| `anon-pipeline` | analyze (resumable), flags, review gate, render, verification, audit |
| `anon-cli` | `reco-anon` |

Milestones and design: [`docs/SPEC.md`](docs/SPEC.md).
