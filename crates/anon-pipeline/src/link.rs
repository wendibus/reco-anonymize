//! Lightweight offline linking of per-frame heads into tracks.
//!
//! M1 scope: greedy IoU / centre-distance association with gap tolerance.
//! A gap of up to `max_gap` frames is bridged by interpolation between the
//! two keyframes (coverage only grows). The full bidirectional
//! ByteTrack-style tracker replaces this in M2.

use anon_model::{BBox, Keyframe, Origin, RegionKind, Shape, Track};

#[derive(Debug, Clone)]
pub struct HeadObs {
    pub frame: u64,
    pub head: BBox,
    pub shape: Shape,
    pub score: f32,
    pub passes: Vec<String>,
}

struct Open {
    track: Track,
    last: BBox,
    last_frame: u64,
}

fn affinity(a: &BBox, b: &BBox) -> Option<f32> {
    let iou = a.iou(b);
    if iou >= 0.2 {
        return Some(1.0 + iou);
    }
    let (ax, ay) = a.center();
    let (bx, by) = b.center();
    let scale = a
        .width()
        .max(a.height())
        .max(b.width().max(b.height()))
        .max(1.0);
    let d = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt() / scale;
    (d <= 1.0).then_some(1.0 - d)
}

#[derive(Default)]
pub struct Linker {
    open: Vec<Open>,
    done: Vec<Track>,
    max_gap: u64,
}

impl Linker {
    pub fn new(max_gap: u32) -> Self {
        Self {
            max_gap: u64::from(max_gap),
            ..Self::default()
        }
    }

    /// Add all heads of one frame (frames must arrive in order).
    pub fn push_frame(&mut self, frame: u64, mut obs: Vec<HeadObs>) {
        // close tracks that exceeded the gap
        let max_gap = self.max_gap;
        let (keep, close): (Vec<Open>, Vec<Open>) = self
            .open
            .drain(..)
            .partition(|o| frame.saturating_sub(o.last_frame) <= max_gap + 1);
        self.open = keep;
        self.done.extend(close.into_iter().map(|o| o.track));

        // greedy: best affinities first
        let mut pairs = Vec::new();
        for (ti, t) in self.open.iter().enumerate() {
            for (oi, o) in obs.iter().enumerate() {
                if let Some(a) = affinity(&t.last, &o.head) {
                    pairs.push((a, ti, oi));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut used_t = vec![false; self.open.len()];
        let mut used_o = vec![false; obs.len()];
        for (_, ti, oi) in pairs {
            if used_t[ti] || used_o[oi] {
                continue;
            }
            used_t[ti] = true;
            used_o[oi] = true;
            let o = &obs[oi];
            let t = &mut self.open[ti];
            if let Origin::Auto { passes } = &mut t.track.origin {
                for p in &o.passes {
                    if !passes.contains(p) {
                        passes.push(p.clone());
                    }
                }
            }
            t.track.keyframes.push(Keyframe {
                frame,
                shape: o.shape.clone(),
                score: o.score,
                interpolated: false,
            });
            t.last = o.head;
            t.last_frame = frame;
        }
        for (oi, o) in obs.drain(..).enumerate() {
            if used_o[oi] {
                continue;
            }
            self.open.push(Open {
                last: o.head,
                last_frame: frame,
                track: Track {
                    id: 0,
                    kind: RegionKind::Head,
                    origin: Origin::Auto { passes: o.passes },
                    keyframes: vec![Keyframe {
                        frame,
                        shape: o.shape,
                        score: o.score,
                        interpolated: false,
                    }],
                    suppressed: None,
                },
            });
        }
    }

    pub fn finish(mut self) -> Vec<Track> {
        self.done.extend(self.open.into_iter().map(|o| o.track));
        self.done
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(frame: u64, x: f32) -> HeadObs {
        let b = BBox::new(x, 0.0, x + 10.0, 10.0);
        HeadObs {
            frame,
            head: b,
            shape: Shape::ellipse_around(&b, 1.0),
            score: 0.9,
            passes: vec!["tile".into()],
        }
    }

    #[test]
    fn links_moving_head_and_bridges_gap() {
        let mut l = Linker::new(5);
        l.push_frame(0, vec![obs(0, 0.0), obs(0, 100.0)]);
        l.push_frame(1, vec![obs(1, 3.0), obs(1, 103.0)]);
        // frames 2..4 missed, reappears within gap
        l.push_frame(5, vec![obs(5, 12.0)]);
        // far beyond gap for the second head
        l.push_frame(20, vec![obs(20, 104.0)]);
        let mut tracks = l.finish();
        tracks.sort_by_key(|t| (t.keyframes.len(), t.keyframes[0].frame));
        let frames: Vec<Vec<u64>> = tracks
            .iter()
            .map(|t| t.keyframes.iter().map(|k| k.frame).collect())
            .collect();
        assert_eq!(frames, vec![vec![20], vec![0, 1], vec![0, 1, 5]]);
    }
}
