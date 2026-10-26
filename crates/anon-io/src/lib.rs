//! FFmpeg I/O for reco-anonymize.
//!
//! Why not `reco_io::ffmpeg::VideoEncoder`: it writes constant-frame-rate
//! timestamps from a frame counter, copies only the first audio stream and
//! does not carry colour metadata. Anonymized exports must keep the source
//! timing (VFR), every audio stream and the colour description, so this
//! crate drives `ffmpeg-next` 9 directly (FFmpeg 6.1 up to 8.x).

pub mod probe;
pub mod reader;
pub mod transcode;

pub use ffmpeg_next;
pub use probe::{VideoProbe, fingerprint, probe};
pub use reader::{FrameReader, FrameSeeker, RgbConverter, THUMB_W, Thumbnailer};
pub use transcode::{
    Codec, EncodeSettings, FrameJob, TranscodeReport, decoded_pts_secs, transcode,
};

#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("ffmpeg: {0}")]
    Ffmpeg(#[from] ffmpeg_next::Error),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}

/// Initialise FFmpeg once and keep its own logging quiet.
pub fn init() -> Result<(), IoError> {
    ffmpeg_next::init()?;
    ffmpeg_next::util::log::set_level(ffmpeg_next::util::log::Level::Error);
    Ok(())
}
