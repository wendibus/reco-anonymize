//! Review gate and rendering.
//!
//! Fail-closed rules enforced here:
//! 1. Render requires an approval whose hash still matches tracks, zones
//!    and the frame table, and a source whose fingerprint still matches.
//! 2. Approval is refused while any frame is in error.
//! 3. Any frame without an `Analyzed` record, or whose pts differs from the
//!    analysed one, is painted fully opaque.
//! 4. The output is written under a temporary name, verified (frame count,
//!    timing), and only then moved into place. On failure it is deleted.

use std::path::{Path, PathBuf};

use anon_io::{EncodeSettings, decoded_pts_secs, transcode};
use anon_model::frames::flag;
use anon_model::hash::approval_hash;
use anon_model::resolve::FramePlan;
use anon_model::{FrameStatus, MaskResolver, ProjectDir};
use anon_render::Coverage;
use serde::Serialize;

use crate::{PipelineError, now_unix};

#[derive(Debug, Clone, Serialize)]
pub struct Status {
    pub source: PathBuf,
    pub source_matches: bool,
    pub frames: usize,
    pub analyzed: usize,
    pub errors: usize,
    pub flagged_frames: usize,
    /// First flagged frames with reasons (for CLI review).
    pub flagged: Vec<(u64, Vec<&'static str>)>,
    pub auto_tracks: usize,
    pub manual_tracks: usize,
    pub suppressed_tracks: usize,
    pub zones: usize,
    pub approved: bool,
    pub approval_valid: bool,
    /// Rendered output audited with no open hits, file unchanged since.
    pub export_clean: bool,
    pub export_note: String,
}

pub fn status(dir: &ProjectDir, max_flags: usize) -> Result<Status, PipelineError> {
    let p = dir.load_project()?;
    let tracks = dir.load_tracks()?;
    let frames = dir.load_frames()?;
    let source_matches =
        anon_io::fingerprint(&p.source.path).is_ok_and(|f| f == p.source.fingerprint);
    let hash = approval_hash(&p.source.fingerprint, &tracks, &p.zones, &frames);
    let (export_clean, export_note) = crate::audit::export_clean(dir)?;
    Ok(Status {
        source: p.source.path.clone(),
        source_matches,
        frames: frames.records.len(),
        analyzed: frames.count_status(FrameStatus::Analyzed),
        errors: frames.count_status(FrameStatus::Error),
        flagged_frames: frames.flagged().count(),
        flagged: frames
            .flagged()
            .take(max_flags)
            .map(|(i, r)| (i, flag::names(r.flags)))
            .collect(),
        auto_tracks: tracks.tracks.iter().filter(|t| !t.is_manual()).count(),
        manual_tracks: tracks.tracks.iter().filter(|t| t.is_manual()).count(),
        suppressed_tracks: tracks
            .tracks
            .iter()
            .filter(|t| t.suppressed.is_some())
            .count(),
        zones: p.zones.len(),
        approved: p.review.approved,
        approval_valid: p.review.approved
            && p.review.approved_hash.as_deref() == Some(hash.as_str()),
        export_clean,
        export_note,
    })
}

/// Record that a human reviewed the current masks.
///
/// A clip in which not a single person was detected is refused unless
/// `allow_no_people` is set: that result usually means a wrong model or
/// class index, and approving it would publish unmasked footage.
pub fn approve(
    dir: &ProjectDir,
    note: Option<String>,
    allow_no_people: bool,
) -> Result<String, PipelineError> {
    let mut p = dir.load_project()?;
    let tracks = dir.load_tracks()?;
    let frames = dir.load_frames()?;
    if frames.records.is_empty() {
        return Err(PipelineError::Refused(
            "project has not been analysed".into(),
        ));
    }
    let errors = frames.count_status(FrameStatus::Error);
    if errors > 0 {
        return Err(PipelineError::Refused(format!(
            "{errors} frame(s) in error state; resolve them first"
        )));
    }
    if !allow_no_people && frames.records.iter().all(|r| r.persons == 0) {
        return Err(PipelineError::Refused(
            "no person was detected in any frame. Check the model (person class index) and the \
             preview; if the clip really shows nobody, approve with the no-people override"
                .into(),
        ));
    }
    let fp = anon_io::fingerprint(&p.source.path)?;
    if fp != p.source.fingerprint {
        return Err(PipelineError::Refused(
            "source file changed since analysis".into(),
        ));
    }
    let hash = approval_hash(&p.source.fingerprint, &tracks, &p.zones, &frames);
    p.review.approved = true;
    p.review.approved_at = Some(now_unix());
    p.review.approved_hash = Some(hash.clone());
    p.review.note = note;
    dir.save_project(&p)?;
    Ok(hash)
}

pub struct RenderOptions {
    pub encode: EncodeSettings,
    /// Render without approval. Output goes to `<project>/preview.<ext>`
    /// only, never to a user-chosen path.
    pub preview: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RenderSummary {
    pub output: PathBuf,
    pub frames: u64,
    pub fully_opaque_frames: u64,
    pub pts_mismatches: u64,
    pub encoder: String,
    pub pixel_format: String,
    pub audio_streams: u32,
    pub preview: bool,
}

fn partial_path(out: &Path) -> PathBuf {
    let stem = out.file_stem().unwrap_or_default().to_string_lossy();
    let ext = out
        .extension()
        .map(|e| e.to_string_lossy().into_owned())
        .unwrap_or_else(|| "mp4".into());
    out.with_file_name(format!("{stem}.partial.{ext}"))
}

pub fn render(
    dir: &ProjectDir,
    out: &Path,
    opts: &RenderOptions,
) -> Result<RenderSummary, PipelineError> {
    anon_io::init()?;
    let p = dir.load_project()?;
    let tracks = dir.load_tracks()?;
    let frames = dir.load_frames()?;
    let src = p.source.path.clone();

    let fp = anon_io::fingerprint(&src)?;
    if fp != p.source.fingerprint {
        return Err(PipelineError::Refused(
            "source file changed since analysis".into(),
        ));
    }
    let out = if opts.preview {
        let ext = out
            .extension()
            .map(|e| e.to_string_lossy().into_owned())
            .unwrap_or_else(|| "mp4".into());
        dir.path(&format!("preview.{ext}"))
    } else {
        if !p.review.approved {
            return Err(PipelineError::Refused(
                "not approved; review the project and run `approve` first".into(),
            ));
        }
        let hash = approval_hash(&p.source.fingerprint, &tracks, &p.zones, &frames);
        if p.review.approved_hash.as_deref() != Some(hash.as_str()) {
            return Err(PipelineError::Refused(
                "masks changed after approval; review and approve again".into(),
            ));
        }
        out.to_path_buf()
    };
    if out.canonicalize().ok() == src.canonicalize().ok() && out.exists() {
        return Err(PipelineError::Refused(
            "output would overwrite the source".into(),
        ));
    }

    let resolver = MaskResolver::new(
        &tracks.tracks,
        &p.zones,
        &frames,
        p.settings.dilation_frames,
    );
    let mut mismatches = 0u64;
    let tmp = partial_path(&out);
    let result = transcode(
        &src,
        &tmp,
        &opts.encode,
        |job| {
            let full = || Coverage::full(job.width, job.height);
            match frames.get(job.index) {
                Some(r) if r.pts == job.pts => {}
                _ => {
                    mismatches += 1;
                    return full();
                }
            }
            match resolver.plan(job.index) {
                FramePlan::FullOpaque => full(),
                FramePlan::Shapes(shapes) => {
                    let mut c = Coverage::new(job.width, job.height);
                    for s in &shapes {
                        c.add_shape(s);
                    }
                    c
                }
            }
        },
        |_| {},
    );
    let report = match result {
        Ok(r) => r,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(e.into());
        }
    };

    // Verify before the file gets its final name.
    let fail = |msg: String| -> PipelineError {
        let _ = std::fs::remove_file(&tmp);
        PipelineError::Verify(msg)
    };
    if report.frames as usize != frames.records.len() && !opts.preview {
        return Err(fail(format!(
            "rendered {} frames, analysis has {}",
            report.frames,
            frames.records.len()
        )));
    }
    if mismatches > 0 && !opts.preview {
        return Err(fail(format!(
            "{mismatches} frame(s) did not match the analysed timestamps"
        )));
    }
    let out_pts = decoded_pts_secs(&tmp)?;
    let tb = f64::from(p.source.time_base.num) / f64::from(p.source.time_base.den);
    let mut src_pts: Vec<f64> = report.pts.iter().map(|&x| x as f64 * tb).collect();
    src_pts.sort_by(f64::total_cmp);
    if out_pts.len() != src_pts.len() {
        return Err(fail(format!(
            "output has {} frames, expected {}",
            out_pts.len(),
            src_pts.len()
        )));
    }
    let tol = 1e-3_f64.max(tb);
    if let Some((i, (a, b))) = src_pts
        .iter()
        .zip(&out_pts)
        .enumerate()
        .find(|(_, (a, b))| ((*a - src_pts[0]) - (*b - out_pts[0])).abs() > tol)
    {
        return Err(fail(format!(
            "timing differs at frame {i}: {a:.6}s vs {b:.6}s"
        )));
    }
    std::fs::rename(&tmp, &out)?;
    if !opts.preview {
        // a new export needs a new audit
        let _ = std::fs::remove_file(dir.path("audit-report.json"));
    }

    let summary = RenderSummary {
        output: out,
        frames: report.frames,
        fully_opaque_frames: report.fully_opaque_frames,
        pts_mismatches: mismatches,
        encoder: report.encoder,
        pixel_format: report.pixel_format,
        audio_streams: report.audio_streams_copied,
        preview: opts.preview,
    };
    let name = if opts.preview {
        "preview-report.json"
    } else {
        "render-report.json"
    };
    std::fs::write(
        dir.path(name),
        serde_json::to_vec_pretty(&summary).expect("serializable"),
    )?;
    Ok(summary)
}
