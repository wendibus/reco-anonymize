//! Analysis pass: decode every frame, detect, track, flag, persist.
//!
//! Long clips are checkpointed every [`CHECKPOINT_EVERY`] frames
//! (`checkpoint.json` + partial `frames.bin` + `detections.jsonl`). With
//! `resume`, already analysed frames are decoded again (cheap) but not
//! re-inferred (expensive); their detections come from the log.

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anon_detect::PersonSegmenter;
use anon_detect::analyzer::{Analyzer, AnalyzerConfig};
use anon_detect::image::unrotate_point;
use anon_detect::text::TextDetector;
use anon_io::{FrameReader, RgbConverter};
use anon_model::frames::flag;
use anon_model::{
    AnalysisSettings, BBox, Fingerprint, FrameRecord, FrameStatus, FrameTable, ModelRef, Project,
    ProjectDir, RegionKind, ReviewState, SCHEMA_VERSION, SourceInfo, TrackSet,
};
use anon_track::{Det, FrameDets, TrackerConfig, pass_bits, track_offline};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::PipelineError;

/// Frames between checkpoints.
pub const CHECKPOINT_EVERY: u64 = 250;

pub struct AnalyzeOptions {
    pub settings: AnalysisSettings,
    pub model_ref: ModelRef,
    /// Discard an existing project for a different source.
    pub fresh: bool,
    /// Continue an interrupted analysis from its last checkpoint.
    pub resume: bool,
    /// Text detector for jersey numbers; required when
    /// `settings.mask_jersey_numbers` is set.
    pub text: Option<(Box<dyn TextDetector>, ModelRef)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalyzeSummary {
    pub frames: u64,
    pub resumed_from: u64,
    pub errors: u64,
    pub flagged: u64,
    pub tracks: usize,
    pub text_tracks: usize,
    pub max_persons: u16,
}

#[derive(Serialize, Deserialize)]
struct DetLine {
    frame: u64,
    pts: i64,
    person: [f32; 4],
    head: [f32; 4],
    score: f32,
    arms_up: bool,
    passes: Vec<String>,
    /// "head" or "text".
    #[serde(default = "head_kind")]
    kind: String,
}

fn head_kind() -> String {
    "head".into()
}

#[derive(Serialize, Deserialize, PartialEq)]
struct Checkpoint {
    fingerprint: Fingerprint,
    settings_sha256: String,
    frames_done: u64,
}

fn unrotate_box(b: &BBox, deg: i32, w: f32, h: f32) -> BBox {
    let (x0, y0) = unrotate_point(deg, w, h, b.x0, b.y0);
    let (x1, y1) = unrotate_point(deg, w, h, b.x1, b.y1);
    BBox::new(x0, y0, x1, y1)
}

fn near_border(b: &BBox, w: f32, h: f32) -> bool {
    let mx = w * 0.05;
    let my = h * 0.05;
    b.x0 <= mx || b.y0 <= my || b.x1 >= w - mx || b.y1 >= h - my
}

fn settings_hash(s: &AnalysisSettings) -> String {
    hex::encode(Sha256::digest(serde_json::to_vec(s).expect("serializable")))
}

fn det_of(line: &DetLine) -> Det {
    let names: Vec<&str> = line.passes.iter().map(String::as_str).collect();
    Det {
        head: BBox::new(line.head[0], line.head[1], line.head[2], line.head[3]),
        score: line.score,
        passes: pass_bits(&names),
    }
}

struct Resume {
    frames: FrameTable,
    dets: Vec<FrameDets>,
    text: Vec<FrameDets>,
}

/// Load a matching checkpoint and trim the logs to it.
fn load_resume(
    dir: &ProjectDir,
    fp: &Fingerprint,
    sh: &str,
) -> Result<Option<Resume>, PipelineError> {
    let Ok(bytes) = std::fs::read(dir.path("checkpoint.json")) else {
        return Ok(None);
    };
    let Ok(cp) = serde_json::from_slice::<Checkpoint>(&bytes) else {
        return Ok(None);
    };
    if &cp.fingerprint != fp || cp.settings_sha256 != sh {
        log::warn!("checkpoint does not match source or settings; starting over");
        return Ok(None);
    }
    let mut frames = dir.load_frames()?;
    if (frames.records.len() as u64) < cp.frames_done {
        return Ok(None);
    }
    frames.records.truncate(cp.frames_done as usize);
    let mut dets = vec![FrameDets::default(); cp.frames_done as usize];
    let mut text = vec![FrameDets::default(); cp.frames_done as usize];
    let log_path = dir.path("detections.jsonl");
    let mut kept = Vec::new();
    for line in BufReader::new(std::fs::File::open(&log_path)?).lines() {
        let line = line?;
        // a torn last line after a crash is simply dropped
        let Ok(d) = serde_json::from_str::<DetLine>(&line) else {
            continue;
        };
        if d.frame < cp.frames_done {
            let target = if d.kind == "text" {
                &mut text
            } else {
                &mut dets
            };
            target[d.frame as usize].dets.push(det_of(&d));
            kept.push(line);
        }
    }
    let mut w = BufWriter::new(std::fs::File::create(&log_path)?);
    for l in kept {
        writeln!(w, "{l}")?;
    }
    w.flush()?;
    Ok(Some(Resume { frames, dets, text }))
}

/// Analyse `src` into the project directory `dir`.
pub fn analyze(
    src: &Path,
    dir: &ProjectDir,
    model: Box<dyn PersonSegmenter>,
    opts: AnalyzeOptions,
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<AnalyzeSummary, PipelineError> {
    anon_io::init()?;
    let probe = anon_io::probe(src)?;
    let fingerprint = anon_io::fingerprint(src)?;
    dir.create()?;

    // Keep zones and manual tracks of an existing project for this source.
    let (zones, mut tracks) = match dir.load_project() {
        Ok(p) if p.source.fingerprint == fingerprint => (p.zones, dir.load_tracks()?),
        Ok(_) if !opts.fresh => {
            return Err(PipelineError::Refused(format!(
                "{} belongs to a different source file (use --fresh to discard it)",
                dir.root.display()
            )));
        }
        _ => (Vec::new(), TrackSet::default()),
    };

    let mut settings = opts.settings;
    let fps = (probe.avg_frame_rate.den > 0)
        .then(|| f64::from(probe.avg_frame_rate.num) / f64::from(probe.avg_frame_rate.den));
    settings
        .resolve_timing(fps.unwrap_or(f64::NAN))
        .map_err(PipelineError::Refused)?;
    settings.models = vec![opts.model_ref];
    let mut text_detector = match (settings.mask_jersey_numbers, opts.text) {
        (true, Some((d, r))) => {
            settings.models.push(r);
            Some(d)
        }
        (true, None) => {
            return Err(PipelineError::Refused(
                "jersey-number masking needs a text model".into(),
            ));
        }
        (false, _) => None,
    };
    let sh = settings_hash(&settings);
    let resume = if opts.resume {
        load_resume(dir, &fingerprint, &sh)?
    } else {
        None
    };
    let resumed_from = resume.as_ref().map_or(0, |r| r.frames.records.len() as u64);
    let (mut frames, mut all_dets, mut all_text) = match resume {
        Some(r) => (r.frames, r.dets, r.text),
        None => (FrameTable::default(), Vec::new(), Vec::new()),
    };
    let mut det_log = if resumed_from > 0 {
        BufWriter::new(
            OpenOptions::new()
                .append(true)
                .open(dir.path("detections.jsonl"))?,
        )
    } else {
        dir.detections_writer()?
    };

    let cfg = AnalyzerConfig {
        tile: settings.tile.clone(),
        person_threshold: settings.person_threshold,
        flag_below: settings.flag_band.1,
        head_expand: settings.head_expand,
    };
    let mut analyzer = Analyzer::new(model, cfg);
    let step = u64::from(settings.detect_every);
    if step == 0 || step > u64::from(settings.dilation_frames) + 1 {
        return Err(PipelineError::Refused(format!(
            "detecting every {step} frames needs 1..={} (dilation + 1), otherwise a track can end unmasked; \
             use a shorter detection interval or a longer dilation",
            settings.dilation_frames + 1
        )));
    }
    let rot = probe.rotation_deg;
    let (w, h) = (probe.width as f32, probe.height as f32);
    let (dw, dh) = if rot % 180 == 0 { (w, h) } else { (h, w) };
    let expected = probe
        .duration_secs
        .zip(fps)
        .map(|(d, fps)| (d * fps).round() as u64);

    let mut reader = FrameReader::open(src)?;
    let mut rgb = RgbConverter::default();
    let mut prev_persons: u16 = frames.records.last().map_or(0, |r| r.persons);
    let mut max_persons: u16 = frames.records.iter().map(|r| r.persons).max().unwrap_or(0);

    let mut index: u64 = 0;
    loop {
        let next = match reader.next_frame() {
            Ok(n) => n,
            Err(e) => {
                log::error!("decode error after frame {index}: {e}");
                // Recorded so approval stays blocked; the rest renders opaque.
                let mut rec = FrameRecord::missing(frames.records.last().map_or(0, |r| r.pts + 1));
                rec.status = FrameStatus::Error;
                rec.flags |= flag::ERROR;
                frames.records.push(rec);
                all_dets.push(FrameDets::default());
                all_text.push(FrameDets::default());
                index += 1;
                break;
            }
        };
        let Some((frame, pts)) = next else { break };

        if index < resumed_from {
            // Already analysed before the interruption: the decoder must
            // reproduce the same timeline, otherwise the logs are unusable.
            if frames.records[index as usize].pts != pts {
                return Err(PipelineError::Refused(format!(
                    "resume: frame {index} has pts {pts}, checkpoint has {}; run without --resume",
                    frames.records[index as usize].pts
                )));
            }
            index += 1;
            progress(index, expected);
            continue;
        }

        let mut rec = FrameRecord::missing(pts);
        let mut fdets = FrameDets::default();
        let mut tdets = FrameDets::default();
        let result = index.is_multiple_of(step).then(|| {
            rgb.convert(&frame)
                .map_err(|e| e.to_string())
                .and_then(|img| {
                    let shown = img.rotate_cw(rot);
                    let det = analyzer.analyze(&shown).map_err(|e| e.to_string())?;
                    let text = match text_detector.as_mut() {
                        Some(t) => {
                            let persons: Vec<BBox> = det.heads.iter().map(|h| h.person).collect();
                            t.detect(&shown, &persons).map_err(|e| e.to_string())?
                        }
                        None => Vec::new(),
                    };
                    Ok((det, text))
                })
        });
        match result {
            Some(Ok((det, text))) => {
                for tb in &text {
                    let b = unrotate_box(&tb.bbox, rot, dw, dh);
                    let line = DetLine {
                        frame: index,
                        pts,
                        person: [b.x0, b.y0, b.x1, b.y1],
                        head: [b.x0, b.y0, b.x1, b.y1],
                        score: tb.score,
                        arms_up: false,
                        passes: vec!["text".into()],
                        kind: "text".into(),
                    };
                    serde_json::to_writer(&mut det_log, &line).map_err(std::io::Error::other)?;
                    det_log.write_all(b"\n")?;
                    tdets.dets.push(det_of(&line));
                }
                let persons = det.heads.len().min(u16::MAX as usize) as u16;
                rec.status = FrameStatus::Analyzed;
                rec.persons = persons;
                if det.heads.iter().any(|h| h.low_score) {
                    rec.flags |= flag::LOW_SCORE;
                }
                if prev_persons >= 3
                    && prev_persons - persons.min(prev_persons) >= 2
                    && persons * 10 < prev_persons * 7
                {
                    rec.flags |= flag::COUNT_DROP;
                }
                prev_persons = persons;
                max_persons = max_persons.max(persons);
                for hd in &det.heads {
                    let line = DetLine {
                        frame: index,
                        pts,
                        person: {
                            let p = unrotate_box(&hd.person, rot, dw, dh);
                            [p.x0, p.y0, p.x1, p.y1]
                        },
                        head: {
                            let b = unrotate_box(&hd.head, rot, dw, dh);
                            [b.x0, b.y0, b.x1, b.y1]
                        },
                        score: hd.score,
                        arms_up: hd.arms_up,
                        passes: hd.passes.iter().map(|s| (*s).to_string()).collect(),
                        kind: head_kind(),
                    };
                    serde_json::to_writer(&mut det_log, &line).map_err(std::io::Error::other)?;
                    det_log.write_all(b"\n")?;
                    fdets.dets.push(det_of(&line));
                }
            }
            Some(Err(e)) => {
                log::error!("frame {index}: {e}");
                rec.status = FrameStatus::Error;
                rec.flags |= flag::ERROR;
            }
            None => {
                // between detected frames: no detections, the tracker bridges
                // the gap (step <= dilation + 1 covers a track ending here)
                rec.status = FrameStatus::Analyzed;
                rec.persons = prev_persons;
            }
        }
        frames.records.push(rec);
        all_dets.push(fdets);
        all_text.push(tdets);
        index += 1;
        progress(index, expected);

        if index.is_multiple_of(CHECKPOINT_EVERY) {
            det_log.flush()?;
            dir.save_frames(&frames)?;
            let cp = Checkpoint {
                fingerprint: fingerprint.clone(),
                settings_sha256: sh.clone(),
                frames_done: index,
            };
            std::fs::write(
                dir.path("checkpoint.json"),
                serde_json::to_vec(&cp).expect("serializable"),
            )?;
        }
    }
    det_log.flush()?;
    if index < resumed_from {
        return Err(PipelineError::Refused(format!(
            "resume: source now ends at frame {index}, checkpoint has {resumed_from}"
        )));
    }

    let tcfg = TrackerConfig {
        max_lost: settings.gap_fill_frames,
        extend_frames: settings.dilation_frames,
        head_expand: settings.head_expand,
        ..TrackerConfig::default()
    };
    let mut auto = track_offline(&all_dets, &tcfg);
    let text_tracks = if settings.mask_jersey_numbers {
        track_offline(
            &all_text,
            &TrackerConfig {
                kind: RegionKind::Text,
                head_expand: 1.2,
                ..tcfg.clone()
            },
        )
    } else {
        Vec::new()
    };

    // Review flags that need the whole track picture.
    let last = index.saturating_sub(1);
    for t in &auto {
        let observed: Vec<_> = t.keyframes.iter().filter(|k| !k.interpolated).collect();
        let (Some(first), Some(end)) = (observed.first(), observed.last()) else {
            continue;
        };
        for (k, at_clip_edge) in [(first, first.frame == 0), (end, end.frame == last)] {
            if !at_clip_edge && !near_border(&k.shape.bounds(), w, h) {
                frames.records[k.frame as usize].flags |= flag::TRACK_EDGE;
            }
        }
        for pair in observed.windows(2) {
            if pair[1].frame - pair[0].frame > 10 {
                frames.records[pair[0].frame as usize].flags |= flag::LONG_INTERP;
            }
        }
    }
    let auto_count = auto.len();
    let text_count = text_tracks.len();
    auto.extend(text_tracks);
    tracks.replace_auto(auto);

    let project = Project {
        schema_version: SCHEMA_VERSION,
        source: SourceInfo {
            path: src.canonicalize().unwrap_or_else(|_| src.to_path_buf()),
            fingerprint,
            width: probe.width,
            height: probe.height,
            pixel_format: probe.pixel_format,
            time_base: probe.time_base,
            frame_count: index,
            color: probe.color,
            rotation_deg: probe.rotation_deg,
            audio_streams: probe.audio_streams,
        },
        settings,
        zones,
        // Any re-analysis voids a previous approval.
        review: ReviewState::default(),
    };
    dir.save_frames(&frames)?;
    dir.save_tracks(&tracks)?;
    dir.save_project(&project)?;
    let _ = std::fs::remove_file(dir.path("checkpoint.json"));

    Ok(AnalyzeSummary {
        frames: index,
        resumed_from,
        errors: frames.count_status(FrameStatus::Error) as u64,
        flagged: frames.flagged().count() as u64,
        tracks: auto_count,
        text_tracks: text_count,
        max_persons,
    })
}
