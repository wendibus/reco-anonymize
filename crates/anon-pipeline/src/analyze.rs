//! Analysis pass: decode every frame, detect, link, flag, persist.

use std::io::Write;
use std::path::Path;

use anon_detect::PersonSegmenter;
use anon_detect::analyzer::{Analyzer, AnalyzerConfig};
use anon_detect::image::unrotate_point;
use anon_io::{FrameReader, RgbConverter};
use anon_model::frames::flag;
use anon_model::{
    AnalysisSettings, BBox, FrameRecord, FrameStatus, FrameTable, ModelRef, Project, ProjectDir,
    ReviewState, SCHEMA_VERSION, Shape, SourceInfo, TrackSet,
};
use serde::Serialize;

use crate::PipelineError;
use crate::link::{HeadObs, Linker};

pub struct AnalyzeOptions {
    pub settings: AnalysisSettings,
    pub model_ref: ModelRef,
    /// Discard an existing project for a different source.
    pub fresh: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalyzeSummary {
    pub frames: u64,
    pub errors: u64,
    pub flagged: u64,
    pub tracks: usize,
    pub max_persons: u16,
}

#[derive(Serialize)]
struct DetLine<'a> {
    frame: u64,
    pts: i64,
    person: [f32; 4],
    head: [f32; 4],
    score: f32,
    arms_up: bool,
    passes: &'a [&'static str],
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
    settings.models = vec![opts.model_ref];
    let cfg = AnalyzerConfig {
        tile: settings.tile.clone(),
        person_threshold: settings.person_threshold,
        flag_below: settings.flag_band.1,
        head_expand: settings.head_expand,
    };
    let mut analyzer = Analyzer::new(model, cfg);
    let rot = probe.rotation_deg;
    let (w, h) = (probe.width as f32, probe.height as f32);
    let (dw, dh) = if rot % 180 == 0 { (w, h) } else { (h, w) };
    let expected = probe
        .duration_secs
        .zip(
            (probe.avg_frame_rate.den > 0)
                .then(|| f64::from(probe.avg_frame_rate.num) / f64::from(probe.avg_frame_rate.den)),
        )
        .map(|(d, fps)| (d * fps).round() as u64);

    let mut reader = FrameReader::open(src)?;
    let mut rgb = RgbConverter::default();
    let mut det_log = dir.detections_writer()?;
    let mut frames = FrameTable::default();
    let mut linker = Linker::new(settings.gap_fill_frames);
    let mut prev_persons: u16 = 0;
    let mut max_persons: u16 = 0;

    let mut index: u64 = 0;
    loop {
        let next = match reader.next_frame() {
            Ok(n) => n,
            Err(e) => {
                // A broken stream ends analysis; everything after it stays
                // un-analysed and is rendered opaque.
                log::error!("decode error after frame {index}: {e}");
                // Record the failure so approval is blocked until resolved.
                let mut rec = FrameRecord::missing(frames.records.last().map_or(0, |r| r.pts + 1));
                rec.status = FrameStatus::Error;
                rec.flags |= flag::ERROR;
                frames.records.push(rec);
                index += 1;
                break;
            }
        };
        let Some((frame, pts)) = next else { break };
        let mut rec = FrameRecord::missing(pts);
        let result = rgb
            .convert(&frame)
            .map_err(|e| e.to_string())
            .and_then(|img| {
                analyzer
                    .analyze(&img.rotate_cw(rot))
                    .map_err(|e| e.to_string())
            });
        match result {
            Ok(det) => {
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
                let mut obs = Vec::with_capacity(det.heads.len());
                for hd in &det.heads {
                    let head = unrotate_box(&hd.head, rot, dw, dh);
                    let person = unrotate_box(&hd.person, rot, dw, dh);
                    let line = DetLine {
                        frame: index,
                        pts,
                        person: [person.x0, person.y0, person.x1, person.y1],
                        head: [head.x0, head.y0, head.x1, head.y1],
                        score: hd.score,
                        arms_up: hd.arms_up,
                        passes: &hd.passes,
                    };
                    serde_json::to_writer(&mut det_log, &line).map_err(std::io::Error::other)?;
                    det_log.write_all(b"\n")?;
                    obs.push(HeadObs {
                        frame: index,
                        head,
                        shape: Shape::ellipse_around(&head, settings.head_expand),
                        score: hd.score,
                        passes: hd.passes.iter().map(|s| (*s).to_string()).collect(),
                    });
                }
                linker.push_frame(index, obs);
            }
            Err(e) => {
                log::error!("frame {index}: {e}");
                rec.status = FrameStatus::Error;
                rec.flags |= flag::ERROR;
            }
        }
        frames.records.push(rec);
        index += 1;
        progress(index, expected);
    }
    det_log.flush()?;

    // Review flags that need the whole track picture.
    let auto = linker.finish();
    let last = index.saturating_sub(1);
    for t in &auto {
        let (Some(first), Some(end)) = (t.keyframes.first(), t.keyframes.last()) else {
            continue;
        };
        for (k, at_clip_edge) in [(first, first.frame == 0), (end, end.frame == last)] {
            if !at_clip_edge && !near_border(&k.shape.bounds(), w, h) {
                frames.records[k.frame as usize].flags |= flag::TRACK_EDGE;
            }
        }
        for pair in t.keyframes.windows(2) {
            if pair[1].frame - pair[0].frame > 10 {
                frames.records[pair[0].frame as usize].flags |= flag::LONG_INTERP;
            }
        }
    }
    let auto_count = auto.len();
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

    Ok(AnalyzeSummary {
        frames: index,
        errors: frames.count_status(FrameStatus::Error) as u64,
        flagged: frames.flagged().count() as u64,
        tracks: auto_count,
        max_persons,
    })
}
