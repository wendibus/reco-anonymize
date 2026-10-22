//! Sequential video decoding (analysis side) and RGB conversion.

use std::path::Path;

use anon_detect::RgbImage;
use ff::{Packet, decoder, format, frame, software::scaling};
use ffmpeg_next as ff;

use crate::IoError;

/// Open a software video decoder for the best video stream.
pub(crate) fn open_decoder(
    ictx: &format::context::Input,
) -> Result<(usize, decoder::Video, ff::Rational), IoError> {
    let st = ictx
        .streams()
        .best(ff::media::Type::Video)
        .ok_or_else(|| IoError::Other("no video stream".into()))?;
    let mut ctx = ff::codec::context::Context::from_parameters(st.parameters())?;
    ctx.set_threading(ff::threading::Config {
        kind: ff::threading::Type::Frame,
        count: 0,
    });
    let dec = ctx.decoder().video()?;
    Ok((st.index(), dec, st.time_base()))
}

/// Timestamp policy shared by analysis and render, so frame `i` gets the
/// same pts on both sides: best-effort timestamp, forced strictly
/// increasing.
#[derive(Debug, Default)]
pub(crate) struct PtsTracker {
    last: Option<i64>,
}

impl PtsTracker {
    pub fn next(&mut self, f: &frame::Video) -> i64 {
        let raw = f.timestamp().or(f.pts());
        let pts = match (raw, self.last) {
            (Some(p), Some(l)) if p > l => p,
            (Some(_), Some(l)) | (None, Some(l)) => {
                log::warn!("non-increasing or missing pts after {l}, using {}", l + 1);
                l + 1
            }
            (Some(p), None) => p,
            (None, None) => 0,
        };
        self.last = Some(pts);
        pts
    }
}

/// Decoded frames in presentation order with their (normalised) pts.
pub struct FrameReader {
    ictx: format::context::Input,
    dec: decoder::Video,
    stream: usize,
    pub time_base: ff::Rational,
    eof: bool,
    pts: PtsTracker,
}

impl FrameReader {
    pub fn open(path: &Path) -> Result<Self, IoError> {
        let ictx = format::input(path)?;
        let (stream, dec, time_base) = open_decoder(&ictx)?;
        Ok(Self {
            ictx,
            dec,
            stream,
            time_base,
            eof: false,
            pts: PtsTracker::default(),
        })
    }

    /// Next frame and its pts, `None` at end of stream.
    pub fn next_frame(&mut self) -> Result<Option<(frame::Video, i64)>, IoError> {
        let mut f = frame::Video::empty();
        loop {
            match self.dec.receive_frame(&mut f) {
                Ok(()) => {
                    let pts = self.pts.next(&f);
                    return Ok(Some((f, pts)));
                }
                Err(ff::Error::Eof) => return Ok(None),
                Err(ff::Error::Other { errno }) if errno == ff::util::error::EAGAIN => {}
                Err(e) => return Err(e.into()),
            }
            if self.eof {
                return Ok(None);
            }
            let mut pkt = Packet::empty();
            match pkt.read(&mut self.ictx) {
                Ok(()) => {
                    if pkt.stream() == self.stream {
                        self.dec.send_packet(&pkt)?;
                    }
                }
                Err(ff::Error::Eof) => {
                    self.eof = true;
                    self.dec.send_eof()?;
                }
                Err(e) => return Err(e.into()),
            }
        }
    }
}

/// Converts any decoded frame to packed RGB24 for the detector.
#[derive(Default)]
pub struct RgbConverter {
    ctx: Option<(scaling::Context, ff::format::Pixel, u32, u32)>,
}

impl RgbConverter {
    pub fn convert(&mut self, f: &frame::Video) -> Result<RgbImage, IoError> {
        let key = (f.format(), f.width(), f.height());
        let stale = !matches!(&self.ctx, Some((_, fm, w, h)) if (*fm, *w, *h) == key);
        if stale {
            let s = scaling::Context::get(
                key.0,
                key.1,
                key.2,
                ff::format::Pixel::RGB24,
                key.1,
                key.2,
                scaling::Flags::BILINEAR,
            )?;
            self.ctx = Some((s, key.0, key.1, key.2));
        }
        let (s, ..) = self.ctx.as_mut().expect("scaler");
        let mut rgb = frame::Video::empty();
        s.run(f, &mut rgb)?;
        let (w, h) = (rgb.width(), rgb.height());
        let stride = rgb.stride(0);
        let src = rgb.data(0);
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h as usize {
            data.extend_from_slice(&src[y * stride..y * stride + (w * 3) as usize]);
        }
        Ok(RgbImage::from_raw(w, h, data))
    }
}

/// Random access to frames by pts, for the review app. Steps forward by
/// decoding when the target is close, otherwise seeks to the preceding
/// keyframe and decodes up to the target.
pub struct FrameSeeker {
    ictx: format::context::Input,
    dec: decoder::Video,
    stream: usize,
    pub time_base: ff::Rational,
    last_pts: Option<i64>,
    eof: bool,
}

/// Targets this far (in seconds) ahead of the last decoded frame are
/// reached by decoding instead of seeking.
const STEP_WINDOW_SECS: f64 = 2.0;

impl FrameSeeker {
    pub fn open(path: &Path) -> Result<Self, IoError> {
        let ictx = format::input(path)?;
        let (stream, dec, time_base) = open_decoder(&ictx)?;
        Ok(Self {
            ictx,
            dec,
            stream,
            time_base,
            last_pts: None,
            eof: false,
        })
    }

    fn decode_next(&mut self) -> Result<Option<(frame::Video, i64)>, IoError> {
        let mut f = frame::Video::empty();
        loop {
            match self.dec.receive_frame(&mut f) {
                Ok(()) => {
                    let pts = f.timestamp().or(f.pts()).unwrap_or(i64::MIN);
                    self.last_pts = Some(pts);
                    return Ok(Some((f, pts)));
                }
                Err(ff::Error::Eof) => return Ok(None),
                Err(ff::Error::Other { errno }) if errno == ff::util::error::EAGAIN => {}
                Err(e) => return Err(e.into()),
            }
            if self.eof {
                return Ok(None);
            }
            let mut pkt = Packet::empty();
            match pkt.read(&mut self.ictx) {
                Ok(()) => {
                    if pkt.stream() == self.stream {
                        self.dec.send_packet(&pkt)?;
                    }
                }
                Err(ff::Error::Eof) => {
                    self.eof = true;
                    self.dec.send_eof()?;
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn seek_before(&mut self, pts: i64, back_secs: f64) -> Result<(), IoError> {
        let tb = f64::from(self.time_base);
        let us = ((pts as f64 * tb - back_secs).max(0.0) * 1e6) as i64;
        // ffmpeg-next passes the range end as avformat_seek_file max_ts
        self.ictx.seek(us, ..us)?;
        self.dec.flush();
        self.eof = false;
        self.last_pts = None;
        Ok(())
    }

    /// The frame with exactly this pts, or `None` if it cannot be found.
    pub fn frame_at(&mut self, pts: i64) -> Result<Option<frame::Video>, IoError> {
        let tb = f64::from(self.time_base);
        let close = self
            .last_pts
            .is_some_and(|l| l < pts && (pts - l) as f64 * tb <= STEP_WINDOW_SECS);
        let mut back = 0.0;
        for attempt in 0..5 {
            if attempt > 0 || !close {
                self.seek_before(pts, back)?;
            }
            let mut first = true;
            while let Some((f, p)) = self.decode_next()? {
                if p == pts {
                    return Ok(Some(f));
                }
                if p > pts {
                    if first {
                        break; // seek landed after the target: go further back
                    }
                    log::warn!("no frame with pts {pts} (next is {p})");
                    return Ok(None);
                }
                first = false;
            }
            back = if back == 0.0 { 1.0 } else { back * 4.0 };
        }
        Ok(None)
    }
}
