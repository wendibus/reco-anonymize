//! Headless review session behind the review app.
//!
//! Everything the UI can do lives here and is testable without a window:
//! frame access by index, overlay rendering (outline / exact opaque
//! preview), hit-testing, suppressing and restoring tracks, drawing manual
//! tracks, tracking which flagged frames were looked at, and the approval
//! gate (which additionally requires every flagged frame to be reviewed).

pub mod overlay;

use std::collections::BTreeSet;
use std::path::Path;

use anon_io::{FrameSeeker, RgbConverter};
use anon_model::frames::flag;
use anon_model::hash::approval_hash;
use anon_model::resolve::FramePlan;
use anon_model::{
    BBox, FrameStatus, FrameTable, Keyframe, MaskResolver, Origin, Project, ProjectDir, RegionKind,
    Shape, Suppression, Track, TrackSet,
};
use anon_render::Coverage;

pub use overlay::Rgba;

#[derive(Debug, thiserror::Error)]
pub enum ReviewError {
    #[error(transparent)]
    Store(#[from] anon_model::StoreError),
    #[error(transparent)]
    Io(#[from] anon_io::IoError),
    #[error(transparent)]
    Pipeline(#[from] anon_pipeline::PipelineError),
    #[error("{0}")]
    Refused(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    /// Original picture with mask outlines.
    Outline,
    /// Exactly what the export will show (opaque fill).
    Opaque,
}

/// What a drawn region belongs to.
#[derive(Debug, Clone, PartialEq)]
pub enum RegionRef {
    Track {
        id: u64,
        manual: bool,
        suppressed: bool,
        interpolated: bool,
        score: f32,
    },
    Zone {
        id: u32,
        label: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    pub what: RegionRef,
    pub shape: Shape,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Readiness {
    pub errors: usize,
    pub flagged: usize,
    pub unreviewed: usize,
    pub nobody_detected: bool,
    pub approval_valid: bool,
}

impl Readiness {
    pub fn can_approve(&self) -> bool {
        self.errors == 0 && self.unreviewed == 0
    }
}

pub const AUTO: [u8; 4] = [255, 210, 0, 255];
pub const MANUAL: [u8; 4] = [40, 220, 90, 255];
pub const SUPPRESSED: [u8; 4] = [255, 60, 60, 255];
pub const SELECTED: [u8; 4] = [0, 200, 255, 255];
pub const ZONE: [u8; 4] = [220, 60, 220, 90];

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

pub struct ReviewSession {
    pub dir: ProjectDir,
    pub project: Project,
    pub tracks: TrackSet,
    pub frames: FrameTable,
    pub current: u64,
    pub selected: Option<u64>,
    pub mode: ViewMode,
    pub dirty: bool,
    reviewed: BTreeSet<u64>,
    /// Polygon being drawn (source coordinates).
    pub zone_draft: Vec<[f32; 2]>,
    seeker: FrameSeeker,
    rgb: RgbConverter,
    cache: Option<(u64, anon_detect::RgbImage)>,
}

impl ReviewSession {
    /// Open a project directory or the source video next to it.
    pub fn open(path: &Path) -> Result<Self, ReviewError> {
        anon_io::init()?;
        let dir = if path.join("project.json").exists() {
            ProjectDir::open(path)
        } else {
            ProjectDir::for_source(path)
        };
        let project = dir.load_project()?;
        let tracks = dir.load_tracks()?;
        let frames = dir.load_frames()?;
        let seeker = FrameSeeker::open(&project.source.path)?;
        let reviewed = project.review.reviewed_frames.iter().copied().collect();
        Ok(Self {
            dir,
            project,
            tracks,
            frames,
            current: 0,
            selected: None,
            mode: ViewMode::Outline,
            dirty: false,
            zone_draft: Vec::new(),
            reviewed,
            seeker,
            rgb: RgbConverter::default(),
            cache: None,
        })
    }

    pub fn frame_count(&self) -> u64 {
        self.frames.records.len() as u64
    }

    pub fn size(&self) -> (u32, u32) {
        (self.project.source.width, self.project.source.height)
    }

    pub fn flagged(&self) -> Vec<u64> {
        self.frames.flagged().map(|(i, _)| i).collect()
    }

    pub fn flags_at(&self, frame: u64) -> Vec<&'static str> {
        self.frames
            .get(frame)
            .map(|r| flag::names(r.flags))
            .unwrap_or_default()
    }

    pub fn is_reviewed(&self, frame: u64) -> bool {
        self.reviewed.contains(&frame)
    }

    /// Next (`forward`) or previous flagged frame, optionally only those not
    /// reviewed yet.
    pub fn next_flag(&self, forward: bool, unreviewed_only: bool) -> Option<u64> {
        let pick = |f: &u64| !unreviewed_only || !self.reviewed.contains(f);
        let flagged = self.flagged();
        let cands: Vec<u64> = flagged.into_iter().filter(pick).collect();
        // wraps around at either end
        if forward {
            cands
                .iter()
                .copied()
                .find(|&f| f > self.current)
                .or_else(|| cands.first().copied().filter(|&f| f != self.current))
        } else {
            cands
                .iter()
                .copied()
                .rev()
                .find(|&f| f < self.current)
                .or_else(|| cands.last().copied().filter(|&f| f != self.current))
        }
    }

    /// Jump to a frame; showing a flagged frame marks it reviewed.
    pub fn goto(&mut self, frame: u64) {
        self.current = frame.min(self.frame_count().saturating_sub(1));
        let flagged = self.frames.get(self.current).is_some_and(|r| r.flags != 0);
        if flagged && self.reviewed.insert(self.current) {
            self.dirty = true;
        }
    }

    /// Every region at `frame`, including suppressed tracks (for display).
    pub fn regions_at(&self, frame: u64) -> Vec<Region> {
        let d = u64::from(self.project.settings.dilation_frames);
        let mut out: Vec<Region> = self
            .project
            .zones
            .iter()
            .filter(|z| z.active_at(frame))
            .map(|z| Region {
                what: RegionRef::Zone {
                    id: z.id,
                    label: z.label.clone(),
                },
                shape: z.shape(),
            })
            .collect();
        for t in &self.tracks.tracks {
            if let Some(shape) = MaskResolver::track_shape_at(t, frame, d) {
                let kf = t.keyframes.iter().find(|k| k.frame == frame);
                out.push(Region {
                    what: RegionRef::Track {
                        id: t.id,
                        manual: t.is_manual(),
                        suppressed: t.suppressed.is_some(),
                        interpolated: kf.is_none_or(|k| k.interpolated),
                        score: kf.map_or(0.0, |k| k.score),
                    },
                    shape,
                });
            }
        }
        out
    }

    /// Topmost (smallest) track region containing the point.
    pub fn hit_test(&self, x: f32, y: f32) -> Option<u64> {
        self.regions_at(self.current)
            .into_iter()
            .filter(|r| matches!(r.what, RegionRef::Track { .. }) && r.shape.contains(x, y))
            .min_by(|a, b| a.shape.bounds().area().total_cmp(&b.shape.bounds().area()))
            .and_then(|r| match r.what {
                RegionRef::Track { id, .. } => Some(id),
                RegionRef::Zone { .. } => None,
            })
    }

    fn decode(&mut self, frame: u64) -> Result<Option<anon_detect::RgbImage>, ReviewError> {
        if let Some((f, img)) = &self.cache
            && *f == frame
        {
            return Ok(Some(img.clone()));
        }
        let Some(rec) = self.frames.get(frame).copied() else {
            return Ok(None);
        };
        let Some(f) = self.seeker.frame_at(rec.pts)? else {
            return Ok(None);
        };
        let img = self.rgb.convert(&f)?;
        self.cache = Some((frame, img.clone()));
        Ok(Some(img))
    }

    /// The current frame with overlays for the chosen view mode.
    pub fn render_current(&mut self) -> Result<Rgba, ReviewError> {
        let frame = self.current;
        let (w, h) = self.size();
        let Some(img) = self.decode(frame)? else {
            // undecodable: show as the export would (opaque) with a marker
            let mut r = Rgba::filled(w, h, [0, 0, 0, 255]);
            r.rect(2.0, 2.0, w as f32 - 3.0, h as f32 - 3.0, 4, SUPPRESSED);
            return Ok(r);
        };
        let mut out = Rgba::from_rgb(&img);
        let regions = self.regions_at(frame);
        let thick = (w.max(h) / 640).clamp(2, 6);
        match self.mode {
            ViewMode::Opaque => {
                let resolver = MaskResolver::new(
                    &self.tracks.tracks,
                    &self.project.zones,
                    &self.frames,
                    self.project.settings.dilation_frames,
                );
                let cov = match resolver.plan(frame) {
                    FramePlan::FullOpaque => Coverage::full(w, h),
                    FramePlan::Shapes(shapes) => {
                        let mut c = Coverage::new(w, h);
                        for s in &shapes {
                            c.add_shape(s);
                        }
                        c
                    }
                };
                out.fill(&cov, [0, 0, 0, 255]);
            }
            ViewMode::Outline => {
                for r in &regions {
                    match &r.what {
                        RegionRef::Zone { .. } => out.fill_shape(&r.shape, ZONE),
                        RegionRef::Track {
                            id,
                            manual,
                            suppressed,
                            interpolated,
                            ..
                        } => {
                            let c = if self.selected == Some(*id) {
                                SELECTED
                            } else if *suppressed {
                                SUPPRESSED
                            } else if *manual {
                                MANUAL
                            } else {
                                AUTO
                            };
                            out.outline(&r.shape, thick, c, *interpolated || *suppressed);
                        }
                    }
                }
                if self
                    .frames
                    .get(frame)
                    .is_none_or(|r| r.status != FrameStatus::Analyzed)
                {
                    out.rect(2.0, 2.0, w as f32 - 3.0, h as f32 - 3.0, 6, SUPPRESSED);
                }
            }
        }
        if !self.zone_draft.is_empty() {
            let c = [255, 80, 255, 255];
            for p in &self.zone_draft {
                out.rect(p[0] - 4.0, p[1] - 4.0, p[0] + 4.0, p[1] + 4.0, thick, c);
            }
            if self.zone_draft.len() >= 2 {
                let mut pts = self.zone_draft.clone();
                if pts.len() == 2 {
                    pts.push(pts[1]);
                }
                out.outline(
                    &Shape::Polygon { points: pts },
                    thick,
                    c,
                    self.zone_draft.len() < 3,
                );
            }
        }
        Ok(out)
    }

    fn track_mut(&mut self, id: u64) -> Result<&mut Track, ReviewError> {
        self.tracks
            .tracks
            .iter_mut()
            .find(|t| t.id == id)
            .ok_or_else(|| ReviewError::Refused(format!("no track {id}")))
    }

    fn edited(&mut self) {
        self.dirty = true;
        self.project.review.approved = false;
    }

    /// Remove a track from the export as a false positive.
    pub fn suppress(&mut self, id: u64, reason: &str) -> Result<(), ReviewError> {
        if reason.trim().is_empty() {
            return Err(ReviewError::Refused("a reason is required".into()));
        }
        self.track_mut(id)?.suppressed = Some(Suppression {
            reason: reason.to_string(),
            at: now(),
        });
        self.edited();
        Ok(())
    }

    pub fn restore(&mut self, id: u64) -> Result<(), ReviewError> {
        self.track_mut(id)?.suppressed = None;
        self.edited();
        Ok(())
    }

    /// Add a head the detector missed. A box drawn while a manual track is
    /// selected becomes a keyframe of that track (interpolated in between);
    /// otherwise a new manual track starts. Returns the track id.
    pub fn add_manual_box(&mut self, b: BBox) -> Result<u64, ReviewError> {
        if b.width() < 2.0 || b.height() < 2.0 {
            return Err(ReviewError::Refused("box too small".into()));
        }
        let frame = self.current;
        let kf = Keyframe {
            frame,
            shape: Shape::ellipse_around(&b, 1.0),
            score: 1.0,
            interpolated: false,
        };
        let target = self.selected.filter(|id| {
            self.tracks
                .tracks
                .iter()
                .any(|t| t.id == *id && t.is_manual())
        });
        let id = match target {
            Some(id) => {
                let t = self.track_mut(id)?;
                t.keyframes.retain(|k| k.frame != frame);
                t.keyframes.push(kf);
                t.normalize();
                id
            }
            None => {
                let id = self.tracks.next_id();
                self.tracks.tracks.push(Track {
                    id,
                    kind: RegionKind::Head,
                    origin: Origin::Manual,
                    keyframes: vec![kf],
                    suppressed: None,
                });
                id
            }
        };
        self.selected = Some(id);
        self.edited();
        Ok(id)
    }

    /// Delete the current frame's keyframe of a manual track (the whole
    /// track when it was the last one).
    pub fn delete_manual_keyframe(&mut self, id: u64) -> Result<(), ReviewError> {
        let frame = self.current;
        let t = self.track_mut(id)?;
        if !t.is_manual() {
            return Err(ReviewError::Refused(
                "automatic tracks are suppressed, not deleted".into(),
            ));
        }
        t.keyframes.retain(|k| k.frame != frame);
        if t.keyframes.is_empty() {
            self.tracks.tracks.retain(|t| t.id != id);
            if self.selected == Some(id) {
                self.selected = None;
            }
        }
        self.edited();
        Ok(())
    }

    // ---- zones -----------------------------------------------------------

    pub fn zone_add_point(&mut self, x: f32, y: f32) {
        let (w, h) = self.size();
        self.zone_draft
            .push([x.clamp(0.0, w as f32), y.clamp(0.0, h as f32)]);
    }

    pub fn zone_undo_point(&mut self) {
        self.zone_draft.pop();
    }

    pub fn zone_cancel(&mut self) {
        self.zone_draft.clear();
    }

    /// Turn the draft into a static zone. `range` is inclusive, `None` =
    /// whole clip.
    pub fn zone_finish(
        &mut self,
        label: &str,
        range: Option<(u64, u64)>,
    ) -> Result<u32, ReviewError> {
        if self.zone_draft.len() < 3 {
            return Err(ReviewError::Refused(
                "a zone needs at least 3 points".into(),
            ));
        }
        if let Some((a, b)) = range
            && a > b
        {
            return Err(ReviewError::Refused("zone range: start after end".into()));
        }
        let id = self.project.zones.iter().map(|z| z.id).max().unwrap_or(0) + 1;
        let label = if label.trim().is_empty() {
            format!("zone {id}")
        } else {
            label.trim().to_string()
        };
        self.project.zones.push(anon_model::Zone {
            id,
            label,
            polygon: std::mem::take(&mut self.zone_draft),
            range,
        });
        self.edited();
        Ok(id)
    }

    pub fn zone_remove(&mut self, id: u32) -> Result<(), ReviewError> {
        let n = self.project.zones.len();
        self.project.zones.retain(|z| z.id != id);
        if self.project.zones.len() == n {
            return Err(ReviewError::Refused(format!("no zone {id}")));
        }
        self.edited();
        Ok(())
    }

    pub fn save(&mut self) -> Result<(), ReviewError> {
        self.project.review.reviewed_frames = self.reviewed.iter().copied().collect();
        self.dir.save_tracks(&self.tracks)?;
        self.dir.save_project(&self.project)?;
        self.dirty = false;
        Ok(())
    }

    pub fn readiness(&self) -> Readiness {
        let flagged = self.flagged();
        let hash = approval_hash(
            &self.project.source.fingerprint,
            &self.tracks,
            &self.project.zones,
            &self.frames,
        );
        Readiness {
            errors: self.frames.count_status(FrameStatus::Error),
            flagged: flagged.len(),
            unreviewed: flagged
                .iter()
                .filter(|f| !self.reviewed.contains(f))
                .count(),
            nobody_detected: self.frames.records.iter().all(|r| r.persons == 0),
            approval_valid: self.project.review.approved
                && self.project.review.approved_hash.as_deref() == Some(hash.as_str()),
        }
    }

    /// Save, check that every flagged frame was looked at, then approve.
    pub fn approve(
        &mut self,
        note: Option<String>,
        allow_no_people: bool,
    ) -> Result<String, ReviewError> {
        let r = self.readiness();
        if r.unreviewed > 0 {
            return Err(ReviewError::Refused(format!(
                "{} flagged frame(s) not reviewed yet",
                r.unreviewed
            )));
        }
        self.save()?;
        let hash = anon_pipeline::approve(&self.dir, note, allow_no_people)?;
        self.project = self.dir.load_project()?;
        Ok(hash)
    }

    /// Reload from disk (after a background analysis or render).
    pub fn reload(&mut self) -> Result<(), ReviewError> {
        self.project = self.dir.load_project()?;
        self.tracks = self.dir.load_tracks()?;
        self.frames = self.dir.load_frames()?;
        self.reviewed = self
            .project
            .review
            .reviewed_frames
            .iter()
            .copied()
            .collect();
        self.cache = None;
        self.seeker = FrameSeeker::open(&self.project.source.path)?;
        self.dirty = false;
        Ok(())
    }
}
