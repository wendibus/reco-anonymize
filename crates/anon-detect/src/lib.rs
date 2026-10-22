//! Person detection for anonymization.
//!
//! The model contract is deliberately small: a [`PersonSegmenter`] takes a
//! square RGB tile at its input size and returns person instances in tile
//! coordinates. Everything that makes recall high — tiling, scales, flip
//! pass, fusion, head derivation — lives here, model-independent.

pub mod analyzer;
pub mod fuse;
pub mod head;
pub mod image;
#[cfg(feature = "ort")]
pub mod rfdetr;
pub mod tiling;

use anon_model::BBox;

pub use analyzer::{Analyzer, FrameDetections, HeadRegion};
pub use image::RgbImage;

#[derive(Debug, thiserror::Error)]
pub enum DetectError {
    #[error("model error: {0}")]
    Model(String),
    #[error("unexpected model output: {0}")]
    Output(String),
}

/// Binary mask covering `bbox`, `w × h` cells, row-major. Cell `(i, j)`
/// covers the source rectangle of size `bbox.width()/w × bbox.height()/h`.
#[derive(Debug, Clone, PartialEq)]
pub struct InstanceMask {
    pub w: u32,
    pub h: u32,
    pub bits: Vec<bool>,
}

impl InstanceMask {
    pub fn get(&self, x: u32, y: u32) -> bool {
        self.bits[(y * self.w + x) as usize]
    }
}

/// One detected person.
#[derive(Debug, Clone, PartialEq)]
pub struct PersonInstance {
    pub bbox: BBox,
    pub score: f32,
    pub mask: Option<InstanceMask>,
    /// Box touches an internal tile edge (person likely cut off).
    pub truncated: bool,
    /// Pass names that produced or merged into this instance.
    pub passes: Vec<&'static str>,
}

/// A model that finds people in a square RGB tile of `input_size()`.
pub trait PersonSegmenter: Send {
    fn input_size(&self) -> u32;
    /// Instances in tile pixel coordinates, any score (thresholding happens
    /// downstream).
    fn segment(&mut self, tile: &RgbImage) -> Result<Vec<PersonInstance>, DetectError>;
}
