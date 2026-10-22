//! Turn tracks + zones into the exact set of shapes to paint per frame.

use crate::frames::{FrameStatus, FrameTable};
use crate::geometry::Shape;
use crate::project::Zone;
use crate::track::Track;

const BUCKET: u64 = 256;

/// What the renderer must do with a frame.
#[derive(Debug, Clone, PartialEq)]
pub enum FramePlan {
    /// Paint these shapes (may be empty).
    Shapes(Vec<Shape>),
    /// Frame was not analysed or errored: paint the whole frame.
    FullOpaque,
}

pub struct MaskResolver<'a> {
    tracks: Vec<&'a Track>,
    zones: &'a [Zone],
    frames: &'a FrameTable,
    dilation: u64,
    /// bucket index -> indices into `tracks` overlapping that bucket.
    buckets: Vec<Vec<u32>>,
}

impl<'a> MaskResolver<'a> {
    pub fn new(
        tracks: &'a [Track],
        zones: &'a [Zone],
        frames: &'a FrameTable,
        dilation: u32,
    ) -> Self {
        let dilation = u64::from(dilation);
        let tracks: Vec<&Track> = tracks
            .iter()
            .filter(|t| t.suppressed.is_none() && !t.keyframes.is_empty())
            .collect();
        let n_frames = frames.records.len() as u64;
        let n_buckets = n_frames.div_ceil(BUCKET).max(1) as usize;
        let mut buckets = vec![Vec::new(); n_buckets];
        for (i, t) in tracks.iter().enumerate() {
            let (a, b) = Self::range(t, dilation);
            let (ba, bb) = ((a / BUCKET) as usize, (b / BUCKET) as usize);
            for bucket in buckets
                .iter_mut()
                .take(bb.min(n_buckets - 1) + 1)
                .skip(ba.min(n_buckets - 1))
            {
                bucket.push(i as u32);
            }
        }
        Self {
            tracks,
            zones,
            frames,
            dilation,
            buckets,
        }
    }

    fn range(t: &Track, d: u64) -> (u64, u64) {
        let a = t.first_frame().unwrap_or(0).saturating_sub(d);
        let b = t.last_frame().unwrap_or(0).saturating_add(d);
        (a, b)
    }

    /// Shape of `t` at `frame`, honouring dilation and interpolation.
    pub fn track_shape_at(t: &Track, frame: u64, dilation: u64) -> Option<Shape> {
        let (a, b) = Self::range(t, dilation);
        if frame < a || frame > b {
            return None;
        }
        let kf = &t.keyframes;
        let idx = kf.partition_point(|k| k.frame <= frame);
        if idx == 0 {
            return Some(kf[0].shape.clone());
        }
        let prev = &kf[idx - 1];
        match kf.get(idx) {
            Some(next) if prev.frame != frame => {
                let span = (next.frame - prev.frame) as f32;
                let t = (frame - prev.frame) as f32 / span;
                Some(Shape::lerp(&prev.shape, &next.shape, t))
            }
            _ => Some(prev.shape.clone()),
        }
    }

    pub fn plan(&self, frame: u64) -> FramePlan {
        match self.frames.get(frame) {
            Some(r) if r.status == FrameStatus::Analyzed => {}
            _ => return FramePlan::FullOpaque,
        }
        let mut shapes: Vec<Shape> = self
            .zones
            .iter()
            .filter(|z| z.active_at(frame))
            .map(Zone::shape)
            .collect();
        if let Some(bucket) = self.buckets.get((frame / BUCKET) as usize) {
            for &i in bucket {
                if let Some(s) = Self::track_shape_at(self.tracks[i as usize], frame, self.dilation)
                {
                    shapes.push(s);
                }
            }
        }
        FramePlan::Shapes(shapes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::FrameRecord;
    use crate::track::{Keyframe, Origin, RegionKind};

    fn ell(cx: f32) -> Shape {
        Shape::Ellipse {
            cx,
            cy: 0.0,
            rx: 1.0,
            ry: 1.0,
        }
    }

    fn table(n: usize) -> FrameTable {
        FrameTable {
            records: (0..n)
                .map(|i| FrameRecord {
                    pts: i as i64,
                    status: FrameStatus::Analyzed,
                    flags: 0,
                    persons: 0,
                })
                .collect(),
        }
    }

    fn track(frames: &[(u64, f32)]) -> Track {
        Track {
            id: 1,
            kind: RegionKind::Head,
            origin: Origin::Manual,
            keyframes: frames
                .iter()
                .map(|&(f, cx)| Keyframe {
                    frame: f,
                    shape: ell(cx),
                    score: 1.0,
                    interpolated: false,
                })
                .collect(),
            suppressed: None,
        }
    }

    #[test]
    fn interpolates_and_dilates() {
        let tracks = vec![track(&[(300, 0.0), (310, 10.0)])];
        let t = table(600);
        let r = MaskResolver::new(&tracks, &[], &t, 3);
        assert_eq!(r.plan(296), FramePlan::Shapes(vec![]));
        assert_eq!(r.plan(297), FramePlan::Shapes(vec![ell(0.0)]));
        assert_eq!(r.plan(305), FramePlan::Shapes(vec![ell(5.0)]));
        assert_eq!(r.plan(313), FramePlan::Shapes(vec![ell(10.0)]));
        assert_eq!(r.plan(314), FramePlan::Shapes(vec![]));
    }

    #[test]
    fn missing_or_error_frames_fail_closed() {
        let mut t = table(3);
        t.records[1].status = FrameStatus::Error;
        t.records[2].status = FrameStatus::Missing;
        let r = MaskResolver::new(&[], &[], &t, 0);
        assert_eq!(r.plan(1), FramePlan::FullOpaque);
        assert_eq!(r.plan(2), FramePlan::FullOpaque);
        assert_eq!(r.plan(99), FramePlan::FullOpaque);
        assert_eq!(r.plan(0), FramePlan::Shapes(vec![]));
    }

    #[test]
    fn suppressed_tracks_are_skipped_zones_painted() {
        let mut tr = track(&[(0, 1.0)]);
        tr.suppressed = Some(crate::track::Suppression {
            reason: "fp".into(),
            at: 0,
        });
        let tracks = vec![tr];
        let zones = vec![Zone {
            id: 1,
            label: "stands".into(),
            polygon: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            range: Some((0, 0)),
        }];
        let t = table(2);
        let r = MaskResolver::new(&tracks, &zones, &t, 0);
        assert!(matches!(r.plan(0), FramePlan::Shapes(v) if v.len() == 1));
        assert_eq!(r.plan(1), FramePlan::Shapes(vec![]));
    }
}
