//! `project.json`: source description, analysis settings, zones, review.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::geometry::Shape;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub schema_version: u32,
    pub source: SourceInfo,
    pub settings: AnalysisSettings,
    #[serde(default)]
    pub zones: Vec<Zone>,
    #[serde(default)]
    pub review: ReviewState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rational {
    pub num: i32,
    pub den: i32,
}

/// Cheap but collision-resistant identity of the source file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fingerprint {
    pub size: u64,
    /// sha256 over the first and last 8 MiB (hex).
    pub head_tail_sha256: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColorInfo {
    /// FFmpeg enum values (AVColorPrimaries, AVColorTransferCharacteristic,
    /// AVColorSpace, AVColorRange) stored raw so nothing is lost.
    pub primaries: i32,
    pub transfer: i32,
    pub matrix: i32,
    pub range: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    pub path: PathBuf,
    pub fingerprint: Fingerprint,
    pub width: u32,
    pub height: u32,
    pub pixel_format: String,
    pub time_base: Rational,
    /// Number of decoded video frames (exact, counted during analysis).
    pub frame_count: u64,
    pub color: ColorInfo,
    pub rotation_deg: i32,
    pub audio_streams: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef {
    pub id: String,
    pub sha256: String,
    pub license: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileSettings {
    /// Tile edge in source pixels at scale 1.0. `0` = model input size.
    pub tile: u32,
    /// Fractional overlap between neighbouring tiles.
    pub overlap: f32,
    /// Also run a whole-frame pass, downscaled to the model input.
    pub full_frame_pass: bool,
    /// Also run every tile horizontally flipped.
    pub flip_pass: bool,
}

impl Default for TileSettings {
    fn default() -> Self {
        Self {
            tile: 0,
            overlap: 0.25,
            full_frame_pass: true,
            flip_pass: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisSettings {
    pub models: Vec<ModelRef>,
    /// Minimum person score kept.
    pub person_threshold: f32,
    /// Detections with score in `[lo, hi)` are kept and flagged.
    pub flag_band: (f32, f32),
    pub tile: TileSettings,
    /// Head ellipse expansion factor.
    pub head_expand: f32,
    /// Max gap (frames) bridged by interpolation (M2 tracker).
    pub gap_fill_frames: u32,
    /// Every mask is extended this many frames before and after.
    pub dilation_frames: u32,
    pub mask_jersey_numbers: bool,
}

impl Default for AnalysisSettings {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            person_threshold: 0.15,
            flag_band: (0.15, 0.35),
            tile: TileSettings::default(),
            head_expand: 1.5,
            gap_fill_frames: 30,
            dilation_frames: 3,
            mask_jersey_numbers: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Zone {
    pub id: u32,
    pub label: String,
    /// Polygon in source pixel coordinates.
    pub polygon: Vec<[f32; 2]>,
    /// Inclusive frame range, `None` = whole clip.
    pub range: Option<(u64, u64)>,
}

impl Zone {
    pub fn shape(&self) -> Shape {
        Shape::Polygon {
            points: self.polygon.clone(),
        }
    }

    pub fn active_at(&self, frame: u64) -> bool {
        self.range.is_none_or(|(a, b)| frame >= a && frame <= b)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReviewState {
    pub approved: bool,
    /// Unix seconds.
    pub approved_at: Option<u64>,
    /// Hex sha256 of tracks + zones + frame table at approval time.
    pub approved_hash: Option<String>,
    pub note: Option<String>,
}
