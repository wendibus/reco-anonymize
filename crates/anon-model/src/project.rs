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
    /// Perspective tiling: from this fraction of the (displayed) frame height
    /// down, people are near the camera and large, so that band is tiled with
    /// the coarser `coarse_tile` (downscaled into the model input). `None` =
    /// fine tiles everywhere.
    #[serde(default)]
    pub coarse_below: Option<f32>,
    /// Tile edge in source pixels for the coarse band. `0` = twice the fine tile.
    #[serde(default)]
    pub coarse_tile: u32,
}

impl Default for TileSettings {
    fn default() -> Self {
        Self {
            tile: 0,
            overlap: 0.25,
            full_frame_pass: true,
            flip_pass: true,
            coarse_below: None,
            coarse_tile: 0,
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
    /// Run the detector on every n-th frame only; the tracker interpolates
    /// the frames in between. `1` = every frame.
    #[serde(default = "every_frame")]
    pub detect_every: u32,
    /// Time between detected frames; when set, `detect_every` is derived from
    /// it with the clip's frame rate (see [`AnalysisSettings::resolve_timing`]).
    #[serde(default)]
    pub detect_interval_ms: Option<u32>,
    /// Mask extension before and after every track; when set,
    /// `dilation_frames` is derived from it with the clip's frame rate.
    #[serde(default)]
    pub dilation_ms: Option<u32>,
}

fn every_frame() -> u32 {
    1
}

impl AnalysisSettings {
    /// Derive `detect_every` and `dilation_frames` from the millisecond
    /// options for a clip at `fps`. The dilation is rounded up (a mask never
    /// lasts shorter than asked), the detection step to the nearest frame.
    pub fn resolve_timing(&mut self, fps: f64) -> Result<(), String> {
        if self.detect_interval_ms.is_none() && self.dilation_ms.is_none() {
            return Ok(());
        }
        if !(fps.is_finite() && fps > 0.0) {
            return Err("unknown frame rate: cannot convert milliseconds to frames".into());
        }
        let frames = |ms: u32| f64::from(ms) * fps / 1000.0;
        if let Some(ms) = self.dilation_ms {
            self.dilation_frames = (frames(ms) - 1e-6).ceil().max(1.0) as u32;
        }
        if let Some(ms) = self.detect_interval_ms {
            self.detect_every = frames(ms).round().max(1.0) as u32;
        }
        Ok(())
    }
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
            detect_every: 1,
            detect_interval_ms: None,
            // 3 frames at 30 fps, and the same time at any other frame rate
            dilation_ms: Some(100),
        }
    }
}

#[cfg(test)]
mod timing_tests {
    use super::*;

    fn resolved(interval: Option<u32>, dilation: Option<u32>, fps: f64) -> (u32, u32) {
        let mut s = AnalysisSettings {
            detect_interval_ms: interval,
            dilation_ms: dilation,
            ..AnalysisSettings::default()
        };
        s.resolve_timing(fps).unwrap();
        (s.detect_every, s.dilation_frames)
    }

    #[test]
    fn same_time_at_any_frame_rate() {
        // (detect_every, dilation_frames)
        assert_eq!(resolved(Some(67), Some(100), 30.0), (2, 3));
        assert_eq!(resolved(Some(67), Some(100), 30000.0 / 1001.0), (2, 3));
        assert_eq!(resolved(Some(67), Some(100), 60.0), (4, 6));
        assert_eq!(resolved(Some(67), Some(100), 25.0), (2, 3));
        assert_eq!(resolved(Some(100), Some(100), 120.0), (12, 12));
    }

    #[test]
    fn frame_fields_kept_without_times() {
        let mut s = AnalysisSettings {
            detect_every: 2,
            dilation_frames: 5,
            dilation_ms: None,
            ..AnalysisSettings::default()
        };
        s.resolve_timing(f64::NAN).unwrap();
        assert_eq!((s.detect_every, s.dilation_frames), (2, 5));
    }

    #[test]
    fn never_below_one_frame_and_needs_a_frame_rate() {
        assert_eq!(resolved(Some(1), Some(1), 30.0), (1, 1));
        let mut s = AnalysisSettings::default();
        assert!(s.resolve_timing(0.0).is_err());
    }

    #[test]
    fn old_projects_keep_their_frame_counts() {
        let mut v = serde_json::to_value(AnalysisSettings::default()).unwrap();
        let o = v.as_object_mut().unwrap();
        o.remove("detect_interval_ms");
        o.remove("dilation_ms");
        let s: AnalysisSettings = serde_json::from_value(v).unwrap();
        assert_eq!((s.detect_interval_ms, s.dilation_ms), (None, None));
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
    /// Flagged frames a reviewer has looked at in the review app.
    #[serde(default)]
    pub reviewed_frames: Vec<u64>,
}
