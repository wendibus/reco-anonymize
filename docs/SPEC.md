# reco-anonymize — Specification (Step 2: crates, APIs, data model)

Status: v0.6, M1–M4 implemented (see §12–16) · License: AGPL-3.0-only (links reco crates) · Target: macOS M4 Pro first, Linux/Windows via reco's backends

## 1. Goals and non-goals

**Goals**
- Mask **every human head** in a video with an **opaque fill**: irreversible, and safer than pixelation.
- 100 % local, quality over speed, no resolution/length limits.
- Input: any video file (reco panorama, reco autocam output, other footage).
- Output: identical resolution, frame rate/timestamps (VFR-safe), colour metadata, untouched audio (stream copy), container metadata stripped.
- **Optional:** mask jersey numbers and the scoreboard (names).
- Mandatory human review before export.

**Non-goals (v1)**
- Audio anonymization (voices, called-out names). Flagged in the UI as a residual risk.
- Live/real-time use. The push API exists for later reco integration, but v1 is offline.

## 2. Workspace layout

Mirrors reco conventions: strict dependency direction, zero clippy warnings, `cargo deny`, fuzz targets.

```
reco-anonymize/
  crates/
    anon-model     Pure data types + (de)serialization + schema versioning. No I/O, no GPU.
    anon-detect    Detector trait, RF-DETR-Seg backend (via reco-detect/ORT), tiling, head derivation,
                   optional text detector for jersey numbers.
    anon-track     Offline bidirectional tracker, gap interpolation, temporal dilation.
    anon-render    Opaque span fill (ellipses + polygons) in planar YUV (see §12).
    anon-io        FFmpeg probe, decode, VFR-safe re-encode, audio stream copy.
    anon-pipeline  Orchestration: analyze / render / audit, checkpoints, push API.
    anon-cli       `reco-anon analyze | render | audit | info`
    anon-gui       Slint review app (timeline, flags, manual edits, approval).
  models/          Model manifests (sha256, license, source URL). Weights not committed.
  fuzz/            Fuzz targets for project-file parsing.
```

Dependencies (via git, pinned rev): `reco-core` (wgpu), `reco-io` (FFmpeg decode/encode), `reco-detect` (ORT CoreML/CUDA/CPU).
Direction: `cli`/`gui` → `pipeline` → `detect`/`track`/`render` → `model`; `render` → `reco-core`; `pipeline` → `reco-io`.

## 3. Pipeline (three phases)

```
ANALYZE  decode ─► Pass A: person seg, tiled, 2 scales ─┐
                   Pass B: same, horizontally flipped ──┼─► fuse ─► head regions ─► track fwd+bwd
                   Pass C: text det (optional) ─────────┘                         ─► interpolate/dilate
                                                                                  ─► flags ─► project
REVIEW   GUI: scrub, coverage heatmap, flagged frames, add/edit/delete, zones ─► approve
RENDER   decode ─► GPU opaque mask (tracks + zones) ─► encode ─► mux audio copy ─► strip metadata
AUDIT    re-decode output ─► run detector again ─► report any unmasked head ─► pass/fail
```

## 4. Detection

**Model:** RF-DETR-Seg (Apache-2.0 code and COCO weights), exported to ONNX and run through reco-detect's ORT session (CoreML EP).
Each model is recorded in `models/*.toml` with its sha256 and licence. Analysis refuses to run with an unknown hash.

**Tiling (SAHI-style)**
- Tile = model input size, 25 % overlap.
- Scale 1.0 for small/distant people, plus one downscaled full-frame pass for large, close people.
- Fusion: class-agnostic NMS at IoU 0.5 over all tiles, scales and the flip pass, followed by mask union per fused instance.

**Head derivation** (person mask → head region)
1. Take the mask's top rows down to 22 % of the instance height.
2. Find the narrowest row (neck) within 12–30 % of the height. If found, cut there.
3. Fit a box to the remaining mask pixels, then expand it 1.5× into an ellipse.

Phase 2 (optional): fine-tune RF-DETR with a `head` class on own footage, pre-annotated by this tool. It then runs as an extra pass, fused by union.

**Thresholds favour recall:** person score ≥ 0.15. Detections between 0.15 and 0.35 are kept *and* flagged.

**Option: jersey numbers.** A text detector runs only inside person boxes and masks the text polygons.
Candidate: PaddleOCR DB text detector (license to be verified before adoption).

**Option: scoreboard.** Handled as a static zone, since the scoreboard doesn't move.

## 5. Tracking

- **Forward pass:** ByteTrack-style association, with high-score matching first and low-score recovery second. Uses a constant-velocity Kalman filter with IoU cost on head ellipses.
- **Backward pass:** same algorithm on the reversed sequence. Forward and backward tracks are merged when their temporal IoU is ≥ 0.5.
- **Gap fill:** linear interpolation (position and size) across gaps of ≤ 30 frames (configurable).
- **Temporal dilation:** every track is extended ±3 frames, with its velocity extrapolated.
- **Motion padding:** the ellipse grows with speed, `r += k · |v|`, to cover motion blur.

## 6. Data model (anon-model)

Project = a directory next to the source: `clip.mp4.anon/`

```
project.json      metadata, settings, zones, review state
detections.jsonl  raw per-pass detections (debug / re-fusion without re-inference)
tracks.json       auto + manual tracks (keyframed)
frames.bin        per-frame status (analyzed / error / flagged), fixed-size records
checkpoint.json   resume point for long analyses
```

```rust
pub const SCHEMA_VERSION: u32 = 1;

pub struct Project {
    pub schema_version: u32,
    pub source: SourceInfo,
    pub settings: AnalysisSettings,
    pub zones: Vec<Zone>,
    pub review: ReviewState,
}

pub struct SourceInfo {
    pub path: PathBuf,
    pub fingerprint: Fingerprint,     // size + sha256 of first/last 8 MiB + duration
    pub width: u32, pub height: u32,
    pub time_base: Rational, pub frame_count: u64,
    pub color: ColorInfo,             // primaries, transfer, matrix, range
    pub rotation_deg: i32,
}

pub struct AnalysisSettings {
    pub models: Vec<ModelRef>,        // id + sha256
    pub person_threshold: f32,        // 0.15
    pub flag_band: (f32, f32),        // (0.15, 0.35)
    pub tile: TileSettings,
    pub head_expand: f32,             // 1.5
    pub gap_fill_frames: u32,         // 30
    pub dilation_frames: u32,         // 3
    pub mask_jersey_numbers: bool,    // option, default false
}

pub struct Frame { pub index: u64, pub pts: i64 }   // index for UI, pts is authoritative (VFR)

pub struct Zone {                     // static, user-drawn
    pub id: ZoneId, pub label: String,
    pub polygon: Vec<[f32; 2]>,       // source pixel coords
    pub range: Option<(u64, u64)>,    // None = whole clip
}

pub struct Track {
    pub id: TrackId,
    pub origin: Origin,               // Auto { passes } | Manual
    pub keyframes: Vec<Keyframe>,     // sorted by frame
    pub suppressed: Option<Suppression>, // false positive removed in review (who/when/why)
}

pub struct Keyframe { pub frame: u64, pub shape: Shape, pub interpolated: bool }

pub enum Shape {
    Ellipse { cx: f32, cy: f32, rx: f32, ry: f32 },
    Polygon(Vec<[f32; 2]>),           // text regions
}

pub struct ReviewState {
    pub approved: bool,
    pub approved_at: Option<DateTime>,
    pub approved_hash: Option<[u8; 32]>, // hash of tracks + zones at approval time
    pub open_flags: u32,
}
```

**Edit invariants**
- Re-running analysis never touches `Manual` tracks or suppressions.
- Any change to tracks or zones invalidates the approval, because the hash no longer matches.

## 7. Key APIs

```rust
// anon-detect
pub trait RegionDetector: Send {
    fn id(&self) -> &ModelRef;
    fn detect(&mut self, frame: &FrameView, ctx: &PassCtx) -> Result<Vec<RawDetection>>;
}

// anon-track
pub fn track_offline(dets: &DetectionStore, cfg: &TrackCfg) -> Vec<Track>;

// anon-render
pub struct MaskRenderer { /* wgpu pipeline, reuses reco-core device */ }
impl MaskRenderer {
    pub fn apply(&mut self, tex: &wgpu::Texture, shapes: &[Shape], fill: [f32; 4]) -> Result<()>;
}

// anon-pipeline
pub fn analyze(src: &Path, settings: AnalysisSettings, progress: impl Progress) -> Result<Project>;
pub fn render(project: &Project, out: &Path, enc: EncodeSettings) -> Result<RenderReport>;
pub fn audit(project: &Project, rendered: &Path) -> Result<AuditReport>;

// Push API for later reco integration (shaped like reco's StitchCore::submit_frame_*)
pub struct Anonymizer { /* online mode: detection + forward tracking only, conservative dilation */ }
impl Anonymizer {
    pub fn submit_frame_bgra(&mut self, frame: &FrameView) -> Result<MaskedFrame>;
}
```

## 8. Safety rules (fail closed)

1. **Render gate:** `render` refuses to run unless `review.approved` is true, `approved_hash` matches, and the source fingerprint matches.
2. **No silent gaps:** a frame without an `analyzed` record is rendered **fully opaque** and reported. It is never passed through unmasked.
3. **Decode/inference errors** mark the frame `error`, and it must be resolved in review before approval.
4. **Flags that need review:**
   - low-score band detections
   - tracks that start or end away from the frame border (likely missed elsewhere)
   - sudden drops in person count
   - interpolated spans > 10 frames
   - any `error` frame
5. **Audit after render:** re-decode the output and run the detector at its lowest threshold. Any person whose head region isn't ≥ 95 % covered gets listed. Export is "clean" only if the audit passes or every hit is acknowledged.
6. **Metadata:** strip global and stream tags (GPS, device, creation time). Keep only codec/colour/rotation essentials.

## 9. Encoding

- **Default:** HEVC via VideoToolbox at high quality, 10-bit if the source is 10-bit. Fallback: libx264 CRF 16, preset slow.
- **Timing:** copy pts/dts and time base from the source. No frame-rate conversion.
- **Colour:** copy primaries, transfer, matrix and range. Copy rotation side data.
- **Audio:** stream copy of all audio streams. Subtitles and data streams are dropped.
- **Verification:** compare output frame count and pts against the source. Abort on mismatch.

## 10. Scale (no limits)

- Frames are streamed and never accumulated. Detections go to JSONL, and per-frame status uses fixed-size records.
- Tile count grows with resolution; batch size is adapted to memory.
- Checkpoints are written every N frames, so `analyze --resume` continues after a crash.

## 11. Milestones

| M | Scope | Done when |
|---|-------|-----------|
| M1 | model, detect (tiled RF-DETR-Seg, head derivation), render, CLI analyze/render | Opaque heads on a test clip, A/V sync intact |
| M2 | Bidirectional tracking, gap fill, dilation, fail-closed rules, audit | Audit passes on 3 reference clips |
| M3 | Slint review GUI: timeline, flags, manual tracks, approval gate | Full workflow without CLI |
| M4 | Static zones UI (stands, scoreboard), jersey-number option | Option toggles verified |
| M5 | Fine-tuned head model from own footage (pre-annotated by tool) | Recall gain measured on reference clips |

## 12. Checked against reco source (commit cd3bf43)

| Question | Finding | Decision |
|---|---|---|
| reco-io decoder/encoder outside `StitchJob`? | `VideoDecoder::open` works standalone. `VideoEncoder` writes CFR timestamps from a frame counter, copies only the first audio stream, sets no colour tags. | Own I/O crate `anon-io` on `ffmpeg-next` 8 (same as reco). Required for VFR, all audio streams, colour. |
| reco-detect backend for non-YOLO output? | `UnifiedDetector` and postprocess are YOLO-shaped; `create_ort_session` is generic (EP selection, input size). | Link `reco-detect` for the session only; RF-DETR decoding lives in `anon-detect`. |
| wgpu 28 device sharing | Possible, but not needed: painting is a few spans per head. | `anon-render` paints on the CPU in the YUV domain, bit-exact, no readback. wgpu stays an option for the push API. |
| PaddleOCR license | Open | Still open for M4. |

## 13. M1 status (implemented)

Delivered: `anon-model`, `anon-detect` (tiling, flip + full-frame passes, fusion, head derivation, RF-DETR-Seg ORT backend), `anon-render`, `anon-io`, `anon-pipeline`, `reco-anon` CLI (`info | analyze | status | zone | approve | render [--preview]`). 33 tests.

Changes vs. v0.2, all in the fail-closed direction:
- **Reference-frame leak fixed by design:** decoded frames are made writable (private copy) before painting. Painting into decoder buffers leaks masks into later predicted frames. Covered by the e2e test (body must stay visible).
- **Verification decodes the output** instead of counting packets. Packet counting missed a dropped final frame (MP4 trims the last sample to its packet duration). Packet durations are now carried explicitly.
- **Output written as `*.partial.*`**, renamed only after verification, deleted on failure.
- **Approval refused when no person was detected in the whole clip** (wrong model or class index), unless explicitly overridden.
- **Model manifest required:** `<model>.onnx.json` with sha256, licence and person class; unknown hashes are refused.
- **Rotation:** detection runs on the display-oriented image, masks are mapped back; rotation metadata is preserved.
- **pts mismatch between analysis and render** paints the frame fully opaque and fails verification.
- M1 used a lightweight linker; replaced by the M2 tracker (§14).


## 14. M2 status (implemented)

**`anon-track`: bidirectional tracker**
- Per-axis constant-velocity Kalman filter on (cx, cy, w, h), noise scaled by head height.
- ByteTrack-style two-stage association: high-score detections (≥ 0.5) against tracked and lost tracks, then low-score detections against tracks seen in the previous frame.
- Deviation from ByteTrack, for recall: unmatched low-score detections also start tracks. Every detection ends up masked.
- Forward and backward passes are run separately. A backward track that bridges several forward fragments merges them (union-find).
- A merge is refused if it would put two detections of the same frame into one track, i.e. two people.
- Keyframes: the ellipse grows with speed (`r += |v|`, against motion blur). Velocity-extrapolated keyframes are added `dilation_frames` before the first and after the last observation.
- Gaps up to `gap_fill_frames` are bridged by interpolation. Bridged spans longer than 10 frames are flagged.

**Resumable analysis**
- Checkpoint every 250 frames: `checkpoint.json` (source fingerprint, settings hash, frames done), partial `frames.bin`, flushed `detections.jsonl`.
- `--resume` re-decodes the clip and skips inference for checkpointed frames, after checking that their pts match.
- Detections before the checkpoint come from the log, so tracks span the interruption.

**Audit (§8.5)**
- `audit` decodes the rendered file and runs the detector at threshold 0.10.
- A head counts as covered if ≥ 95 % of its box, or of the band above the person (masked heads may fall out of the silhouette), is fill-black in the output pixels.
- The check is independent of tracks and resolver, so it also catches tracking, resolving and painting bugs.
- Hits are written to `audit-report.json` and can be acknowledged with a mandatory reason.
- `status` reports the export as clean only when the audited file is unchanged since the audit and no hit is open. A new render deletes the previous audit.

**Tests:** 7 tracker tests, including forward fragmentation healed by the backward pass. E2E tests cover a clean audit passing, a deliberately suppressed track producing 50 audit hits, and a simulated crash at frame 270 resuming from 250 with only 50 frames re-inferred.

## 15. M3 status (implemented)

**`anon-review`: headless review session**

All UI behaviour lives here and is tested without a window:
- Frame-exact random access (`anon-io::FrameSeeker`: decodes forward for short steps, otherwise seeks to the previous keyframe and backs off further if the seek lands late).
- Two views: *Outline* shows the original picture with masks (auto yellow, manual green, suppressed red dashed, interpolated dashed, zones tinted); *Export* shows exactly what the render will paint, using the same resolver.
- Hit-testing picks the smallest mask under the cursor.
- Suppress requires a reason; restore undoes it.
- Manual heads are drawn as boxes. While a manual track is selected, a box on another frame adds a keyframe to it, and the frames in between are interpolated.
- Automatic tracks can only be suppressed, never deleted.
- Each edit is saved immediately and voids the approval.

**Stricter gate in the app**
- A flagged frame counts as reviewed once it has been displayed. This is stored in `project.json` (`review.reviewed_frames`) and reset by re-analysis.
- Approve stays disabled until every flagged frame has been reviewed and no frame is in error. The no-people check still applies.

**`anon-gui` (`reco-anon-review`, Slint 1.15 + fluent-dark, same stack as reco-gui)**
- Frame view with drag-to-add and click-to-select.
- Timeline strip: error red, unreviewed flag orange, reviewed green, people grey.
- Side panel: flags, review progress, masks in the frame, selection actions, approval readiness, export state.
- Keyboard: ←/→ (Shift ×10), `n`/`p` for flagged frames, `o` for the view, `Esc` to deselect.
- Jobs on worker threads: Analyze, Preview (opens the system player), Render (save dialog, then an automatic audit), Audit.
- The model path is remembered in `~/.reco-anonymize/last-model`.


## 16. M4 status (implemented)

**Zones in the review app**
- Click the corners of a polygon on the picture, give it a label and optionally an inclusive frame range ("Only frames").
- Zones can be removed again. Adding or removing a zone voids the approval.
- In the export view zones are painted exactly as in the render.

**Jersey numbers (opt-in)**
- `TextDetector` trait in `anon-detect`. The `DbTextDetector` runs a DB text-probability model on every person crop:
  - the crop is scaled to a long side of 320 and padded to multiples of 32;
  - the map is thresholded at 0.3 and split into connected components, components of fewer than 6 cells are dropped;
  - each box is expanded by area × 1.5 / perimeter (DB unclip) and clipped to the person box.
- ORT backend `ppocr::PpOcrDet`: PaddleOCR DB, BGR input, ImageNet normalisation. The PP-OCR det weights are Apache-2.0 per the model card; `scripts/fetch_ppocr.py` downloads them and writes the manifest.
- Model manifests now carry a `kind` (`person-seg` or `text-db`). Loading a model under the wrong kind is refused.
- Text detections go through the same bidirectional tracker as heads (`RegionKind::Text`, ellipse ×1.2), with gap bridging and extrapolation.
- Enabling the option without a text model is refused. The text model is recorded in `settings.models`, so it is part of the resume settings hash.
- Not covered: the audit checks heads only. Text coverage relies on the review.

**Tests:** connected components and unclip, number on a shirt mapped to source coordinates, an ORT contract test with a synthetic DB model, and an end-to-end test (number on a moving player masked, body visible, enabling without a model refused). Session test for zone add, frame range, export view and removal.
