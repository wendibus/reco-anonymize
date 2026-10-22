//! Re-encode the video with opaque masks painted in, keeping timing,
//! colour description, rotation and every audio stream (stream copy).
//! Container and stream metadata are dropped (only audio language kept).

use std::collections::HashMap;
use std::path::Path;

use anon_render::{Coverage, PlaneMut, YuvLayout, paint_yuv};
use ff::ffi;
use ff::format::Pixel;
use ff::{Dictionary, Packet, codec, encoder, format, frame, software::scaling};
use ffmpeg_next as ff;

use crate::IoError;
use crate::reader::{PtsTracker, open_decoder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Hevc,
    H264,
}

#[derive(Debug, Clone)]
pub struct EncodeSettings {
    pub codec: Codec,
    /// Force an FFmpeg encoder by name (e.g. "libx264").
    pub encoder: Option<String>,
    /// 0..100, higher is better. 86 ≈ x264/x265 CRF 16.
    pub quality: u8,
}

impl Default for EncodeSettings {
    fn default() -> Self {
        Self {
            codec: Codec::Hevc,
            encoder: None,
            quality: 86,
        }
    }
}

/// What the paint callback is asked about.
#[derive(Debug, Clone, Copy)]
pub struct FrameJob {
    pub index: u64,
    pub pts: i64,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone)]
pub struct TranscodeReport {
    pub frames: u64,
    pub fully_opaque_frames: u64,
    pub encoder: String,
    pub pixel_format: String,
    pub audio_streams_copied: u32,
    /// Source pts of every frame, in order.
    pub pts: Vec<i64>,
}

/// Planar YUV formats painted in place. Anything else is converted first.
fn layout_of(p: Pixel) -> Option<YuvLayout> {
    let (w, h, bps, depth, full) = match p {
        Pixel::YUV420P => (1, 1, 1, 8, false),
        Pixel::YUVJ420P => (1, 1, 1, 8, true),
        Pixel::YUV422P => (1, 0, 1, 8, false),
        Pixel::YUVJ422P => (1, 0, 1, 8, true),
        Pixel::YUV444P => (0, 0, 1, 8, false),
        Pixel::YUVJ444P => (0, 0, 1, 8, true),
        Pixel::YUV420P10LE => (1, 1, 2, 10, false),
        Pixel::YUV422P10LE => (1, 0, 2, 10, false),
        Pixel::YUV444P10LE => (0, 0, 2, 10, false),
        _ => return None,
    };
    Some(YuvLayout {
        log2_chroma_w: w,
        log2_chroma_h: h,
        bytes_per_sample: bps,
        bit_depth: depth,
        full_range: full,
    })
}

fn bit_depth(p: Pixel) -> u32 {
    // SAFETY: av_pix_fmt_desc_get returns a static descriptor or null.
    unsafe {
        let d = ffi::av_pix_fmt_desc_get(p.into());
        if d.is_null() {
            8
        } else {
            (*d).comp[0].depth as u32
        }
    }
}

fn paint_format_for(src: Pixel) -> Pixel {
    if layout_of(src).is_some() {
        src
    } else if bit_depth(src) > 8 {
        Pixel::YUV420P10LE
    } else {
        Pixel::YUV420P
    }
}

fn encoder_candidates(s: &EncodeSettings) -> Vec<&str> {
    if let Some(e) = &s.encoder {
        return vec![e.as_str()];
    }
    match s.codec {
        // quality first; hardware encoders only when no software encoder
        // is built in (e.g. LGPL FFmpeg builds)
        Codec::Hevc => vec![
            "hevc_videotoolbox",
            "libx265",
            "libx264",
            "hevc_nvenc",
            "hevc_qsv",
            "hevc_amf",
            "hevc_mf",
        ],
        Codec::H264 => vec![
            "h264_videotoolbox",
            "libx264",
            "h264_nvenc",
            "h264_qsv",
            "h264_amf",
            "h264_mf",
        ],
    }
}

fn supported_formats(c: &ff::Codec) -> Option<Vec<Pixel>> {
    c.video()
        .ok()
        .and_then(|v| v.formats().map(|f| f.collect()))
}

/// Pick encoder + its input format. Prefers the paint format; otherwise a
/// format of the same bit depth the encoder accepts.
fn pick_encoder(s: &EncodeSettings, paint: Pixel) -> Result<(ff::Codec, Pixel), IoError> {
    let depth = bit_depth(paint);
    let fallbacks: &[Pixel] = if depth > 8 {
        &[Pixel::YUV420P10LE, Pixel::P010LE]
    } else {
        &[Pixel::YUV420P, Pixel::NV12]
    };
    let mut tried = Vec::new();
    for name in encoder_candidates(s) {
        let Some(c) = encoder::find_by_name(name) else {
            tried.push(format!("{name} (not built)"));
            continue;
        };
        let fmts = supported_formats(&c).unwrap_or_default();
        if fmts.is_empty() || fmts.contains(&paint) {
            return Ok((c, paint));
        }
        if let Some(f) = fallbacks.iter().find(|f| fmts.contains(f)) {
            return Ok((c, *f));
        }
        tried.push(format!("{name} (no {depth}-bit format)"));
    }
    Err(IoError::Other(format!(
        "no usable encoder: {}",
        tried.join(", ")
    )))
}

/// Quality knobs per encoder family.
fn encoder_options(
    name: &str,
    quality: u8,
    enc: &mut encoder::video::Video,
) -> Dictionary<'static> {
    let q = f32::from(quality.min(100));
    let mut d = Dictionary::new();
    if name.ends_with("_videotoolbox") {
        // Same mapping reco uses for VideoToolbox (40..75).
        let gq = (40.0 + q * 0.35).round() as i32;
        enc.set_global_quality(gq * ffi::FF_QP2LAMBDA);
        // SAFETY: plain field write on an unopened codec context.
        unsafe { (*enc.as_mut_ptr()).flags |= ffi::AV_CODEC_FLAG_QSCALE as i32 };
        d.set("allow_sw", "1");
    } else if name == "libx264" || name == "libx265" {
        let crf = (40.0 - q / 100.0 * 28.0).clamp(0.0, 51.0);
        d.set("crf", &format!("{crf:.1}"));
        d.set("preset", "slow");
        if name == "libx265" {
            d.set("x265-params", "log-level=error");
        }
    } else if name.ends_with("_nvenc") {
        let cq = (40.0 - q / 100.0 * 28.0).clamp(0.0, 51.0);
        d.set("preset", "p7");
        d.set("rc", "vbr");
        d.set("cq", &format!("{cq:.0}"));
    } else if name.ends_with("_qsv") {
        let gq = (40.0 - q / 100.0 * 28.0).clamp(1.0, 51.0);
        d.set("global_quality", &format!("{gq:.0}"));
    } else if name.ends_with("_amf") {
        d.set("quality", "quality");
    }
    d
}

struct Converter {
    ctx: Option<(scaling::Context, Pixel, Pixel, u32, u32)>,
}

impl Converter {
    fn run(&mut self, src: &frame::Video, dst_fmt: Pixel) -> Result<frame::Video, IoError> {
        let key = (src.format(), dst_fmt, src.width(), src.height());
        if !matches!(&self.ctx, Some((_, a, b, w, h)) if (*a, *b, *w, *h) == key) {
            let s = scaling::Context::get(
                key.0,
                key.2,
                key.3,
                key.1,
                key.2,
                key.3,
                scaling::Flags::BICUBIC,
            )?;
            self.ctx = Some((s, key.0, key.1, key.2, key.3));
        }
        let mut out = frame::Video::empty();
        self.ctx.as_mut().expect("scaler").0.run(src, &mut out)?;
        out.set_color_space(src.color_space());
        out.set_color_range(src.color_range());
        out.set_color_primaries(src.color_primaries());
        out.set_color_transfer_characteristic(src.color_transfer_characteristic());
        Ok(out)
    }
}

fn paint(f: &mut frame::Video, cov: &Coverage) -> Result<(), IoError> {
    let mut layout = layout_of(f.format())
        .ok_or_else(|| IoError::Other(format!("cannot paint {:?}", f.format())))?;
    layout.full_range |= f.color_range() == ff::util::color::Range::JPEG;
    // Decoded frames share their buffers with the decoder's reference
    // pictures. Writing into them would leak the mask into later frames
    // (predicted from the painted picture), so take a private copy first.
    // SAFETY: valid frame; av_frame_make_writable copies only if shared.
    let rc = unsafe { ffi::av_frame_make_writable(f.as_mut_ptr()) };
    if rc < 0 {
        return Err(IoError::Ffmpeg(ff::Error::from(rc)));
    }
    let strides = [f.stride(0), f.stride(1), f.stride(2)];
    // SAFETY: the three planes of a planar frame are disjoint allocations;
    // we only need simultaneous &mut access to them.
    let (p0, p1, p2) = unsafe {
        let ptr = f.as_mut_ptr();
        let h = (*ptr).height as usize;
        let ch = h.div_ceil(1 << layout.log2_chroma_h);
        (
            std::slice::from_raw_parts_mut((*ptr).data[0], strides[0] * h),
            std::slice::from_raw_parts_mut((*ptr).data[1], strides[1] * ch),
            std::slice::from_raw_parts_mut((*ptr).data[2], strides[2] * ch),
        )
    };
    let mut planes = [
        PlaneMut {
            data: p0,
            stride: strides[0],
        },
        PlaneMut {
            data: p1,
            stride: strides[1],
        },
        PlaneMut {
            data: p2,
            stride: strides[2],
        },
    ];
    paint_yuv(&mut planes, layout, layout.black(), cov);
    Ok(())
}

fn copy_display_matrix(src: *const ffi::AVCodecParameters, dst: *mut ffi::AVCodecParameters) {
    // SAFETY: both parameter sets are alive; new side data is owned by dst.
    unsafe {
        let sd = ffi::av_packet_side_data_get(
            (*src).coded_side_data,
            (*src).nb_coded_side_data,
            ffi::AVPacketSideDataType::AV_PKT_DATA_DISPLAYMATRIX,
        );
        if sd.is_null() {
            return;
        }
        let nsd = ffi::av_packet_side_data_new(
            &mut (*dst).coded_side_data,
            &mut (*dst).nb_coded_side_data,
            ffi::AVPacketSideDataType::AV_PKT_DATA_DISPLAYMATRIX,
            (*sd).size,
            0,
        );
        if !nsd.is_null() {
            std::ptr::copy_nonoverlapping((*sd).data, (*nsd).data, (*sd).size);
        }
    }
}

const fn mktag(a: u8, b: u8, c: u8, d: u8) -> u32 {
    (a as u32) | ((b as u32) << 8) | ((c as u32) << 16) | ((d as u32) << 24)
}

/// Decode `src`, ask `plan` what to paint on every frame, encode to `dst`.
pub fn transcode(
    src: &Path,
    dst: &Path,
    settings: &EncodeSettings,
    mut plan: impl FnMut(FrameJob) -> Coverage,
    mut progress: impl FnMut(u64),
) -> Result<TranscodeReport, IoError> {
    let mut ictx = format::input(src)?;
    let (vin, mut dec, in_tb) = open_decoder(&ictx)?;
    let paint_fmt = paint_format_for(dec.format());
    let (codec, enc_fmt) = pick_encoder(settings, paint_fmt)?;
    let enc_name = codec.name().to_string();

    let mut octx = format::output(dst)?;
    let global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);
    let mp4_like = matches!(octx.format().name(), n if n.contains("mp4") || n.contains("mov"));

    // --- video encoder ---------------------------------------------------
    let in_st = ictx.stream(vin).expect("video stream");
    let in_par = in_st.parameters();
    // SAFETY: plain enum field reads of valid parameters.
    let src_color = unsafe {
        let p = in_par.as_ptr();
        (
            (*p).color_primaries,
            (*p).color_trc,
            (*p).color_space,
            (*p).color_range,
        )
    };
    let avg_fps = in_st.avg_frame_rate();
    let mut enc = codec::context::Context::new_with_codec(codec)
        .encoder()
        .video()?;
    enc.set_width(dec.width());
    enc.set_height(dec.height());
    enc.set_format(enc_fmt);
    enc.set_time_base(in_tb);
    if avg_fps.numerator() > 0 && avg_fps.denominator() > 0 {
        enc.set_frame_rate(Some(avg_fps));
    }
    enc.set_aspect_ratio(dec.aspect_ratio());
    // SAFETY: plain enum field copies from source parameters.
    unsafe {
        let e = enc.as_mut_ptr();
        let p = in_par.as_ptr();
        (*e).color_primaries = (*p).color_primaries;
        (*e).color_trc = (*p).color_trc;
        (*e).colorspace = (*p).color_space;
        (*e).color_range = (*p).color_range;
        (*e).chroma_sample_location = (*p).chroma_location;
        if global_header {
            (*e).flags |= ffi::AV_CODEC_FLAG_GLOBAL_HEADER as i32;
        }
    }
    let opts = encoder_options(&enc_name, settings.quality, &mut enc);
    let mut enc = enc.open_with(opts)?;
    let vout = {
        let mut ost = octx.add_stream(codec)?;
        ost.set_parameters(&enc);
        ost.set_time_base(in_tb);
        // SAFETY: field writes on the freshly created output stream.
        unsafe {
            let par = (*ost.as_mut_ptr()).codecpar;
            (*par).codec_tag = if mp4_like && codec.id() == codec::Id::HEVC {
                mktag(b'h', b'v', b'c', b'1')
            } else {
                0
            };
            copy_display_matrix(in_par.as_ptr(), par);
            // colour description on the stream itself (muxers read it here)
            let ip = in_par.as_ptr();
            (*par).color_primaries = (*ip).color_primaries;
            (*par).color_trc = (*ip).color_trc;
            (*par).color_space = (*ip).color_space;
            (*par).color_range = (*ip).color_range;
            (*par).chroma_location = (*ip).chroma_location;
        }
        ost.index()
    };

    // --- audio passthrough ----------------------------------------------
    let mut audio_map: HashMap<usize, (usize, ff::Rational)> = HashMap::new();
    for ist in ictx.streams() {
        if ist.parameters().medium() != ff::media::Type::Audio {
            continue;
        }
        let mut ost = octx.add_stream(encoder::find(codec::Id::None))?;
        ost.set_parameters(ist.parameters());
        // SAFETY: let the muxer choose the tag for its container.
        unsafe { (*(*ost.as_mut_ptr()).codecpar).codec_tag = 0 };
        let mut md = Dictionary::new();
        if let Some(lang) = ist.metadata().get("language") {
            md.set("language", lang);
        }
        ost.set_metadata(md);
        audio_map.insert(ist.index(), (ost.index(), ist.time_base()));
    }
    let audio_streams_copied = audio_map.len() as u32;

    let mut mux_opts = Dictionary::new();
    if mp4_like {
        mux_opts.set("movflags", "+faststart");
    }
    octx.write_header_with(mux_opts)?;
    let vout_tb = octx.stream(vout).expect("vout").time_base();
    let audio_out_tb: HashMap<usize, ff::Rational> = audio_map
        .values()
        .map(|(o, _)| (*o, octx.stream(*o).expect("aout").time_base()))
        .collect();

    // --- main loop -------------------------------------------------------
    let mut conv_in = Converter { ctx: None };
    let mut conv_out = Converter { ctx: None };
    let mut pts_tracker = PtsTracker::default();
    let mut report = TranscodeReport {
        frames: 0,
        fully_opaque_frames: 0,
        encoder: enc_name.clone(),
        pixel_format: format!("{enc_fmt:?}"),
        audio_streams_copied,
        pts: Vec::new(),
    };

    // Fallback frame duration in the input time base (1 tick if unknown).
    let default_dur: i64 = if avg_fps.numerator() > 0 && in_tb.numerator() > 0 {
        let d = (i64::from(in_tb.denominator()) * i64::from(avg_fps.denominator()))
            / (i64::from(in_tb.numerator()) * i64::from(avg_fps.numerator()));
        d.max(1)
    } else {
        1
    };
    // pts -> duration of every submitted frame. Encoders do not always
    // carry durations to packets, and MP4 trims the last sample to its
    // packet duration, so a zero duration would drop the final frame.
    let mut durations: HashMap<i64, i64> = HashMap::new();

    let drain_encoder = |enc: &mut encoder::Video,
                         octx: &mut format::context::Output,
                         durations: &mut HashMap<i64, i64>|
     -> Result<(), IoError> {
        let mut pkt = Packet::empty();
        while enc.receive_packet(&mut pkt).is_ok() {
            if let Some(d) = pkt.pts().and_then(|p| durations.remove(&p))
                && pkt.duration() <= 0
            {
                pkt.set_duration(d);
            }
            pkt.set_stream(vout);
            pkt.rescale_ts(in_tb, vout_tb);
            pkt.set_position(-1);
            pkt.write_interleaved(octx)?;
        }
        Ok(())
    };

    let mut handle_frame = |f: frame::Video,
                            enc: &mut encoder::Video,
                            report: &mut TranscodeReport,
                            durations: &mut HashMap<i64, i64>|
     -> Result<(), IoError> {
        let pts = pts_tracker.next(&f);
        // SAFETY: read of a plain field.
        let dur = unsafe { (*f.as_ptr()).duration };
        durations.insert(pts, if dur > 0 { dur } else { default_dur });
        let job = FrameJob {
            index: report.frames,
            pts,
            width: f.width(),
            height: f.height(),
        };
        let cov = plan(job);
        if cov.is_full() {
            report.fully_opaque_frames += 1;
        }
        let mut work = if f.format() == paint_fmt {
            f
        } else {
            conv_in.run(&f, paint_fmt)?
        };
        paint(&mut work, &cov)?;
        let mut out = if work.format() == enc_fmt {
            work
        } else {
            conv_out.run(&work, enc_fmt)?
        };
        out.set_pts(Some(pts));
        out.set_kind(ff::picture::Type::None);
        // newer encoders take the colour description from the frame:
        // fill it from the stream when the decoder left it unspecified
        // SAFETY: plain enum field reads/writes on a valid frame.
        unsafe {
            let f = out.as_mut_ptr();
            if (*f).color_primaries == ffi::AVColorPrimaries::AVCOL_PRI_UNSPECIFIED {
                (*f).color_primaries = src_color.0;
            }
            if (*f).color_trc == ffi::AVColorTransferCharacteristic::AVCOL_TRC_UNSPECIFIED {
                (*f).color_trc = src_color.1;
            }
            if (*f).colorspace == ffi::AVColorSpace::AVCOL_SPC_UNSPECIFIED {
                (*f).colorspace = src_color.2;
            }
            if (*f).color_range == ffi::AVColorRange::AVCOL_RANGE_UNSPECIFIED {
                (*f).color_range = src_color.3;
            }
        }
        enc.send_frame(&out)?;
        report.pts.push(pts);
        report.frames += 1;
        progress(report.frames);
        Ok(())
    };

    let mut pkt = Packet::empty();
    loop {
        match pkt.read(&mut ictx) {
            Ok(()) => {}
            Err(ff::Error::Eof) => break,
            Err(e) => return Err(e.into()),
        }
        let si = pkt.stream();
        if si == vin {
            dec.send_packet(&pkt)?;
            let mut f = frame::Video::empty();
            while dec.receive_frame(&mut f).is_ok() {
                handle_frame(
                    std::mem::replace(&mut f, frame::Video::empty()),
                    &mut enc,
                    &mut report,
                    &mut durations,
                )?;
                drain_encoder(&mut enc, &mut octx, &mut durations)?;
            }
        } else if let Some((oi, itb)) = audio_map.get(&si) {
            pkt.rescale_ts(*itb, audio_out_tb[oi]);
            pkt.set_stream(*oi);
            pkt.set_position(-1);
            pkt.write_interleaved(&mut octx)?;
        }
    }
    dec.send_eof()?;
    let mut f = frame::Video::empty();
    while dec.receive_frame(&mut f).is_ok() {
        handle_frame(
            std::mem::replace(&mut f, frame::Video::empty()),
            &mut enc,
            &mut report,
            &mut durations,
        )?;
        drain_encoder(&mut enc, &mut octx, &mut durations)?;
    }
    enc.send_eof()?;
    drain_encoder(&mut enc, &mut octx, &mut durations)?;
    octx.write_trailer()?;
    Ok(report)
}

/// Presentation times (seconds) of every frame a player would show,
/// obtained by actually decoding the file (honours edit lists and
/// discarded packets, unlike counting packets).
pub fn decoded_pts_secs(path: &Path) -> Result<Vec<f64>, IoError> {
    let mut r = crate::FrameReader::open(path)?;
    let tb = f64::from(r.time_base);
    let mut v = Vec::new();
    while let Some((_, pts)) = r.next_frame()? {
        v.push(pts as f64 * tb);
    }
    Ok(v)
}
