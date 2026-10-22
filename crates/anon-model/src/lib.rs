//! Data model for reco-anonymize projects.
//!
//! Pure types, (de)serialization and the approval hash. No I/O beyond
//! reading/writing the project directory, no GPU, no FFmpeg.
//!
//! A project lives in a directory next to the source video
//! (`clip.mp4.anon/`) and holds:
//!
//! | file              | content                                             |
//! |-------------------|-----------------------------------------------------|
//! | `project.json`    | source info, settings, zones, review state          |
//! | `tracks.json`     | auto + manual mask tracks (keyframed)               |
//! | `frames.bin`      | per-frame status, fixed 16-byte records             |
//! | `detections.jsonl`| raw detections per pass (debug / re-fusion)         |

pub mod frames;
pub mod geometry;
pub mod hash;
pub mod project;
pub mod resolve;
pub mod store;
pub mod track;

pub use frames::{FrameRecord, FrameStatus, FrameTable};
pub use geometry::{BBox, Shape};
pub use project::*;
pub use resolve::MaskResolver;
pub use store::{ProjectDir, StoreError};
pub use track::*;

/// Schema version written to `project.json`. Bump on breaking changes.
pub const SCHEMA_VERSION: u32 = 1;
