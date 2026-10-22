//! Source inspection and fingerprinting.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anon_model::{ColorInfo, Fingerprint, Rational};
use ff::ffi;
use ffmpeg_next as ff;
use sha2::{Digest, Sha256};

use crate::IoError;

const FP_CHUNK: u64 = 8 * 1024 * 1024;

/// size + sha256(first 8 MiB ‖ last 8 MiB).
pub fn fingerprint(path: &Path) -> Result<Fingerprint, IoError> {
    let mut f = File::open(path)?;
    let size = f.metadata()?.len();
    let mut h = Sha256::new();
    let mut buf = vec![0u8; FP_CHUNK.min(size) as usize];
    f.read_exact(&mut buf)?;
    h.update(&buf);
    if size > FP_CHUNK {
        let tail = FP_CHUNK.min(size - FP_CHUNK);
        f.seek(SeekFrom::Start(size - tail))?;
        let mut buf = vec![0u8; tail as usize];
        f.read_exact(&mut buf)?;
        h.update(&buf);
    }
    Ok(Fingerprint {
        size,
        head_tail_sha256: hex::encode(h.finalize()),
    })
}

#[derive(Debug, Clone)]
pub struct VideoProbe {
    pub video_index: usize,
    pub width: u32,
    pub height: u32,
    pub pixel_format: String,
    pub time_base: Rational,
    pub avg_frame_rate: Rational,
    pub color: ColorInfo,
    /// Clockwise rotation a player applies on display (0/90/180/270).
    pub rotation_deg: i32,
    pub audio_streams: u32,
    pub duration_secs: Option<f64>,
}

/// Display rotation from the stream's display matrix, as clockwise degrees
/// normalised to 0/90/180/270.
pub(crate) fn rotation_of(par: *const ffi::AVCodecParameters) -> i32 {
    // SAFETY: par comes from a live stream; side data pointers are owned by it.
    unsafe {
        let sd = ffi::av_packet_side_data_get(
            (*par).coded_side_data,
            (*par).nb_coded_side_data,
            ffi::AVPacketSideDataType::AV_PKT_DATA_DISPLAYMATRIX,
        );
        if sd.is_null() || (*sd).size < 36 {
            return 0;
        }
        let ccw = ffi::av_display_rotation_get((*sd).data as *const i32);
        if ccw.is_nan() {
            return 0;
        }
        let cw = (-ccw).rem_euclid(360.0);
        ((cw / 90.0).round() as i32 * 90).rem_euclid(360)
    }
}

pub(crate) fn rational(r: ff::Rational) -> Rational {
    Rational {
        num: r.numerator(),
        den: r.denominator(),
    }
}

pub fn probe(path: &Path) -> Result<VideoProbe, IoError> {
    let ictx = ff::format::input(path)?;
    let st = ictx
        .streams()
        .best(ff::media::Type::Video)
        .ok_or_else(|| IoError::Other(format!("no video stream in {}", path.display())))?;
    let par = st.parameters();
    // SAFETY: read-only access to plain fields of a valid AVCodecParameters.
    let fmt = ff::codec::context::Context::from_parameters(par.clone())?
        .decoder()
        .video()?
        .format();
    let (w, h, color, rotation) = unsafe {
        let p = par.as_ptr();
        (
            (*p).width as u32,
            (*p).height as u32,
            ColorInfo {
                primaries: (*p).color_primaries as i32,
                transfer: (*p).color_trc as i32,
                matrix: (*p).color_space as i32,
                range: (*p).color_range as i32,
            },
            rotation_of(p),
        )
    };
    let audio_streams = ictx
        .streams()
        .filter(|s| s.parameters().medium() == ff::media::Type::Audio)
        .count() as u32;
    let duration_secs =
        (ictx.duration() > 0).then(|| ictx.duration() as f64 / f64::from(ffi::AV_TIME_BASE));
    Ok(VideoProbe {
        video_index: st.index(),
        width: w,
        height: h,
        pixel_format: fmt
            .descriptor()
            .map_or_else(|| format!("{fmt:?}"), |d| d.name().to_string()),
        time_base: rational(st.time_base()),
        avg_frame_rate: rational(st.avg_frame_rate()),
        color,
        rotation_deg: rotation,
        audio_streams,
        duration_secs,
    })
}
