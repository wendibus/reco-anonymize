//! Offline, bidirectional head tracker.
//!
//! 1. **Forward pass** and **backward pass** over the whole clip, each a
//!    ByteTrack-style tracker: Kalman prediction, high-score detections
//!    matched first (tracked + lost tracks), low-score detections second
//!    (tracked only). Unlike ByteTrack, *unmatched low-score detections
//!    still start tracks*: every detection must end up masked.
//! 2. **Merge:** a backward track that bridges several forward fragments
//!    joins them (union-find), unless that would put two detections of the
//!    same frame into one track (two people).
//! 3. **Keyframes:** one per detection, ellipse grown with speed (motion
//!    blur), plus velocity-extrapolated extension keyframes before the
//!    first and after the last observation. Gaps between keyframes are
//!    interpolated by the resolver.

pub mod kalman;

use anon_model::{BBox, Keyframe, Origin, RegionKind, Shape, Track};

use kalman::BoxFilter;

/// One head detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Det {
    pub head: BBox,
    pub score: f32,
    /// Bit set of analysis passes (see `pass_names`).
    pub passes: u8,
}

/// All detections of one frame. Frames must be consecutive from 0.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FrameDets {
    pub dets: Vec<Det>,
}

#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// Detections at or above this score are matched first.
    pub high_score: f32,
    /// Frames a track survives without a match (gap that can be bridged).
    pub max_lost: u32,
    /// Extrapolated extension before the first / after the last detection.
    pub extend_frames: u32,
    /// Ellipse growth per pixel/frame of speed.
    pub motion_pad: f32,
    /// The motion growth is capped at this multiple of the ellipse radius. A
    /// speed far above what a head can do comes from a wrong link (another
    /// person, a camera pan or cut) and would otherwise stretch the mask
    /// across half the frame.
    pub max_motion_pad: f32,
    pub head_expand: f32,
    /// Kind written to the produced tracks.
    pub kind: RegionKind,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            high_score: 0.5,
            max_lost: 30,
            extend_frames: 3,
            motion_pad: 1.0,
            max_motion_pad: 1.0,
            head_expand: 1.5,
            kind: RegionKind::Head,
        }
    }
}

/// Names for `Det::passes` bits, in bit order.
pub const PASS_NAMES: [&str; 4] = ["tile", "tile_flip", "full", "full_flip"];

pub fn pass_bits(names: &[&str]) -> u8 {
    names
        .iter()
        .filter_map(|n| PASS_NAMES.iter().position(|p| p == n))
        .fold(0, |m, i| m | (1 << i))
}

pub fn pass_names(bits: u8) -> Vec<String> {
    PASS_NAMES
        .iter()
        .enumerate()
        .filter(|(i, _)| bits & (1 << i) != 0)
        .map(|(_, n)| (*n).to_string())
        .collect()
}

/// Higher is better; `None` = cannot be the same head.
fn affinity(pred: &BBox, det: &BBox) -> Option<f32> {
    let iou = pred.iou(det);
    if iou >= 0.2 {
        return Some(1.0 + iou);
    }
    let (ax, ay) = pred.center();
    let (bx, by) = det.center();
    let scale = pred
        .width()
        .max(pred.height())
        .max(det.width().max(det.height()))
        .max(1.0);
    // size must be roughly compatible
    let ratio = (pred.area().max(1.0) / det.area().max(1.0))
        .max(det.area().max(1.0) / pred.area().max(1.0));
    if ratio > 4.0 {
        return None;
    }
    let d = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt() / scale;
    (d <= 1.0).then_some(1.0 - d)
}

struct Live {
    id: u32,
    filter: BoxFilter,
    lost: u32,
}

/// Greedy best-affinity matching of `tracks` (indices into `live`) to
/// `dets` (indices into the frame's detections).
fn greedy(
    live: &[Live],
    tracks: &[usize],
    dets: &[usize],
    frame: &FrameDets,
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for &t in tracks {
        let pred = live[t].filter.bbox();
        for &d in dets {
            if let Some(a) = affinity(&pred, &frame.dets[d].head) {
                pairs.push((a, t, d));
            }
        }
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut used_t = Vec::new();
    let mut used_d = Vec::new();
    let mut out = Vec::new();
    for (_, t, d) in pairs {
        if used_t.contains(&t) || used_d.contains(&d) {
            continue;
        }
        used_t.push(t);
        used_d.push(d);
        out.push((t, d));
    }
    out
}

/// One directional pass. Returns, per frame, the track id of each
/// detection (`order` gives the frame visiting order).
fn run_pass(
    frames: &[FrameDets],
    order: impl Iterator<Item = usize>,
    cfg: &TrackerConfig,
) -> Vec<Vec<u32>> {
    let mut assign: Vec<Vec<u32>> = frames
        .iter()
        .map(|f| vec![u32::MAX; f.dets.len()])
        .collect();
    let mut live: Vec<Live> = Vec::new();
    let mut next_id = 0u32;
    for fi in order {
        let frame = &frames[fi];
        for l in &mut live {
            l.filter.predict();
        }
        let (high, low): (Vec<usize>, Vec<usize>) =
            (0..frame.dets.len()).partition(|&d| frame.dets[d].score >= cfg.high_score);

        // stage 1: high-score dets vs all live tracks
        let all: Vec<usize> = (0..live.len()).collect();
        let m1 = greedy(&live, &all, &high, frame);
        let mut matched_t: Vec<usize> = m1.iter().map(|m| m.0).collect();
        let mut matched_d: Vec<usize> = m1.iter().map(|m| m.1).collect();
        // stage 2: low-score dets vs tracks updated last frame
        let recent: Vec<usize> = (0..live.len())
            .filter(|t| !matched_t.contains(t) && live[*t].lost == 0)
            .collect();
        let m2 = greedy(&live, &recent, &low, frame);
        matched_t.extend(m2.iter().map(|m| m.0));
        matched_d.extend(m2.iter().map(|m| m.1));

        for (t, d) in m1.into_iter().chain(m2) {
            live[t].filter.update(&frame.dets[d].head);
            live[t].lost = 0;
            assign[fi][d] = live[t].id;
        }
        for (t, l) in live.iter_mut().enumerate() {
            if !matched_t.contains(&t) {
                l.lost += 1;
            }
        }
        live.retain(|l| l.lost <= cfg.max_lost);
        // every unmatched detection starts a track (recall first)
        for (d, slot) in assign[fi].iter_mut().enumerate() {
            if matched_d.contains(&d) {
                continue;
            }
            live.push(Live {
                id: next_id,
                filter: BoxFilter::new(&frame.dets[d].head),
                lost: 0,
            });
            *slot = next_id;
            next_id += 1;
        }
    }
    assign
}

struct Groups {
    parent: Vec<usize>,
    frames: Vec<Vec<u64>>, // sorted frames per root
}

impl Groups {
    fn find(&mut self, x: usize) -> usize {
        let mut r = x;
        while self.parent[r] != r {
            r = self.parent[r];
        }
        let mut c = x;
        while self.parent[c] != r {
            let n = self.parent[c];
            self.parent[c] = r;
            c = n;
        }
        r
    }

    /// Union unless both groups observe the same frame.
    fn union(&mut self, a: usize, b: usize) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return true;
        }
        let (fa, fb) = (&self.frames[ra], &self.frames[rb]);
        let (mut i, mut j) = (0, 0);
        while i < fa.len() && j < fb.len() {
            match fa[i].cmp(&fb[j]) {
                std::cmp::Ordering::Equal => return false,
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        let mut merged = std::mem::take(&mut self.frames[rb]);
        merged.extend_from_slice(&self.frames[ra]);
        merged.sort_unstable();
        self.frames[ra] = merged;
        self.parent[rb] = ra;
        true
    }
}

/// Track all heads of a clip. `n_frames` bounds the extrapolated ends.
pub fn track_offline(frames: &[FrameDets], cfg: &TrackerConfig) -> Vec<Track> {
    if frames.is_empty() {
        return Vec::new();
    }
    let fwd = run_pass(frames, 0..frames.len(), cfg);
    let bwd = run_pass(frames, (0..frames.len()).rev(), cfg);

    let n_fwd = fwd
        .iter()
        .flatten()
        .filter(|&&id| id != u32::MAX)
        .map(|&id| id as usize + 1)
        .max()
        .unwrap_or(0);
    let mut groups = Groups {
        parent: (0..n_fwd).collect(),
        frames: vec![Vec::new(); n_fwd],
    };
    let mut fwd_size = vec![0usize; n_fwd];
    for (fi, ids) in fwd.iter().enumerate() {
        for &id in ids {
            groups.frames[id as usize].push(fi as u64);
            fwd_size[id as usize] += 1;
        }
    }
    // bwd track -> (fwd track -> shared count)
    let mut shared: std::collections::BTreeMap<u32, std::collections::BTreeMap<u32, usize>> =
        Default::default();
    for (fi, ids) in bwd.iter().enumerate() {
        for (d, &b) in ids.iter().enumerate() {
            *shared.entry(b).or_default().entry(fwd[fi][d]).or_default() += 1;
        }
    }
    for parts in shared.values() {
        // fwd fragments that lie mostly inside this backward track
        let members: Vec<u32> = parts
            .iter()
            .filter(|(f, c)| **c * 2 >= fwd_size[**f as usize])
            .map(|(f, _)| *f)
            .collect();
        for w in members.windows(2) {
            groups.union(w[0] as usize, w[1] as usize);
        }
    }

    // collect detections per final group
    let mut by_group: std::collections::BTreeMap<usize, Vec<(u64, Det)>> = Default::default();
    for (fi, ids) in fwd.iter().enumerate() {
        for (d, &id) in ids.iter().enumerate() {
            let g = groups.find(id as usize);
            by_group
                .entry(g)
                .or_default()
                .push((fi as u64, frames[fi].dets[d]));
        }
    }
    let last_frame = frames.len() as u64 - 1;
    by_group
        .into_values()
        .map(|mut obs| {
            obs.sort_by_key(|o| o.0);
            build_track(&obs, cfg, last_frame)
        })
        .collect()
}

/// [`track_offline`] per shot: `cuts` are the first frames of new shots
/// (scene cuts). No track links, interpolates or extrapolates across a cut,
/// so a head is never dragged from one camera view into the next.
pub fn track_shots(frames: &[FrameDets], cuts: &[u64], cfg: &TrackerConfig) -> Vec<Track> {
    let n = frames.len() as u64;
    let mut bounds: Vec<u64> = cuts.iter().copied().filter(|&c| c > 0 && c < n).collect();
    bounds.sort_unstable();
    bounds.dedup();
    bounds.insert(0, 0);
    bounds.push(n);
    let mut out = Vec::new();
    for w in bounds.windows(2) {
        let (a, b) = (w[0], w[1]);
        for mut t in track_offline(&frames[a as usize..b as usize], cfg) {
            for k in &mut t.keyframes {
                k.frame += a;
            }
            out.push(t);
        }
    }
    out
}

fn speed_at(obs: &[(u64, Det)], i: usize) -> (f32, f32) {
    let a = i.saturating_sub(1);
    let b = (i + 1).min(obs.len() - 1);
    if a == b {
        return (0.0, 0.0);
    }
    let (ax, ay) = obs[a].1.head.center();
    let (bx, by) = obs[b].1.head.center();
    let dt = (obs[b].0 - obs[a].0).max(1) as f32;
    ((bx - ax) / dt, (by - ay) / dt)
}

fn padded(b: &BBox, v: (f32, f32), cfg: &TrackerConfig) -> Shape {
    match Shape::ellipse_around(b, cfg.head_expand) {
        Shape::Ellipse { cx, cy, rx, ry } => Shape::Ellipse {
            cx,
            cy,
            rx: rx + (cfg.motion_pad * v.0.abs()).min(cfg.max_motion_pad * rx),
            ry: ry + (cfg.motion_pad * v.1.abs()).min(cfg.max_motion_pad * ry),
        },
        s => s,
    }
}

fn build_track(obs: &[(u64, Det)], cfg: &TrackerConfig, last_frame: u64) -> Track {
    let mut passes = 0u8;
    let mut keyframes = Vec::with_capacity(obs.len() + 2);
    for (i, (f, d)) in obs.iter().enumerate() {
        passes |= d.passes;
        keyframes.push(Keyframe {
            frame: *f,
            shape: padded(&d.head, speed_at(obs, i), cfg),
            score: d.score,
            interpolated: false,
        });
    }
    // velocity-extrapolated extensions
    let ext = u64::from(cfg.extend_frames);
    if ext > 0 {
        let (f0, d0) = obs[0];
        if f0 > 0 {
            let v = speed_at(obs, 0);
            let k = f0.min(ext);
            let b = d0.head.translate(-v.0 * k as f32, -v.1 * k as f32);
            keyframes.insert(
                0,
                Keyframe {
                    frame: f0 - k,
                    shape: padded(&b, v, cfg),
                    score: d0.score,
                    interpolated: true,
                },
            );
        }
        let (fl, dl) = obs[obs.len() - 1];
        if fl < last_frame {
            let v = speed_at(obs, obs.len() - 1);
            let k = (last_frame - fl).min(ext);
            let b = dl.head.translate(v.0 * k as f32, v.1 * k as f32);
            keyframes.push(Keyframe {
                frame: fl + k,
                shape: padded(&b, v, cfg),
                score: dl.score,
                interpolated: true,
            });
        }
    }
    Track {
        id: 0,
        kind: cfg.kind,
        origin: Origin::Auto {
            passes: pass_names(passes),
        },
        keyframes,
        suppressed: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, score: f32) -> Det {
        Det {
            head: BBox::new(x, y, x + 10.0, y + 12.0),
            score,
            passes: 1,
        }
    }

    fn observed(t: &Track) -> Vec<u64> {
        t.keyframes
            .iter()
            .filter(|k| !k.interpolated)
            .map(|k| k.frame)
            .collect()
    }

    #[test]
    fn two_crossing_heads_stay_separate() {
        // A moves right, B moves left, they pass each other vertically offset
        let frames: Vec<FrameDets> = (0..40)
            .map(|i| FrameDets {
                dets: vec![
                    det(i as f32 * 4.0, 0.0, 0.9),
                    det(160.0 - i as f32 * 4.0, 30.0, 0.9),
                ],
            })
            .collect();
        let tracks = track_offline(&frames, &TrackerConfig::default());
        assert_eq!(tracks.len(), 2);
        for t in &tracks {
            assert_eq!(observed(t).len(), 40);
        }
    }

    #[test]
    fn occlusion_gap_is_bridged_and_low_scores_kept() {
        let mut frames: Vec<FrameDets> = (0..30)
            .map(|i| FrameDets {
                dets: vec![det(i as f32 * 2.0, 0.0, 0.9)],
            })
            .collect();
        // missed for 8 frames, then seen with a low score
        for f in frames.iter_mut().take(18).skip(10) {
            f.dets.clear();
        }
        frames[18].dets[0].score = 0.2;
        let tracks = track_offline(&frames, &TrackerConfig::default());
        assert_eq!(tracks.len(), 1, "{tracks:?}");
        let obs = observed(&tracks[0]);
        assert_eq!(obs.len(), 22);
        assert!(obs.contains(&9) && obs.contains(&18));
    }

    #[test]
    fn isolated_low_score_detection_still_tracked() {
        let frames = vec![
            FrameDets::default(),
            FrameDets {
                dets: vec![det(50.0, 50.0, 0.16)],
            },
            FrameDets::default(),
        ];
        let tracks = track_offline(&frames, &TrackerConfig::default());
        assert_eq!(tracks.len(), 1);
        // extended one frame each way (clip bounds)
        let frames: Vec<u64> = tracks[0].keyframes.iter().map(|k| k.frame).collect();
        assert_eq!(frames, vec![0, 1, 2]);
    }

    #[test]
    fn motion_pad_and_extrapolation() {
        let frames: Vec<FrameDets> = (0..20)
            .map(|i| FrameDets {
                dets: if (5..15).contains(&i) {
                    vec![det(i as f32 * 5.0, 0.0, 0.9)]
                } else {
                    vec![]
                },
            })
            .collect();
        let cfg = TrackerConfig::default();
        let t = &track_offline(&frames, &cfg)[0];
        let first = &t.keyframes[0];
        assert_eq!(first.frame, 2);
        assert!(first.interpolated);
        let Shape::Ellipse { cx, rx, .. } = first.shape else {
            panic!()
        };
        // centre extrapolated 3 frames back at 5 px/frame: 25 + 5 - 15 = 15
        assert!((cx - 15.0).abs() < 0.01, "cx {cx}");
        let still = Shape::ellipse_around(&BBox::new(0.0, 0.0, 10.0, 12.0), cfg.head_expand);
        let Shape::Ellipse { rx: rx0, .. } = still else {
            panic!()
        };
        assert!((rx - (rx0 + 5.0)).abs() < 0.01);
    }

    #[test]
    fn same_frame_detections_never_merged() {
        // two heads close together, both seen every frame
        let frames: Vec<FrameDets> = (0..10)
            .map(|_| FrameDets {
                dets: vec![det(0.0, 0.0, 0.9), det(6.0, 0.0, 0.9)],
            })
            .collect();
        let tracks = track_offline(&frames, &TrackerConfig::default());
        assert_eq!(tracks.len(), 2);
        for t in &tracks {
            let f = observed(t);
            let mut d = f.clone();
            d.dedup();
            assert_eq!(f, d);
        }
    }

    #[test]
    fn motion_padding_is_capped() {
        // 8 px/frame with a motion pad of 25 px per px/frame: an implausible
        // 200 px of growth, as a wrong link would produce
        let frames: Vec<FrameDets> = (0..6u32)
            .map(|i| FrameDets {
                dets: vec![det(i as f32 * 8.0, 0.0, 0.9)],
            })
            .collect();
        let slow = track_offline(&frames, &TrackerConfig::default());
        let base = TrackerConfig {
            motion_pad: 25.0,
            ..TrackerConfig::default()
        };
        let fast = track_offline(&frames, &base);
        let rx = |t: &Track| match t.keyframes[2].shape {
            Shape::Ellipse { rx, .. } => rx,
            _ => panic!("ellipse expected"),
        };
        let r0 = match Shape::ellipse_around(&BBox::new(0.0, 0.0, 10.0, 12.0), 1.5) {
            Shape::Ellipse { rx, .. } => rx,
            _ => unreachable!(),
        };
        assert!(rx(&slow[0]) < 2.0 * r0);
        assert!(
            (rx(&fast[0]) - 2.0 * r0).abs() < 1e-3,
            "{} vs {}",
            rx(&fast[0]),
            r0
        );
    }

    #[test]
    fn tracks_end_at_a_cut() {
        // the same position before and after a cut at frame 10: without the
        // cut it is one track, with it two, neither reaching across
        let frames: Vec<FrameDets> = (0..20u32)
            .map(|_| FrameDets {
                dets: vec![det(50.0, 50.0, 0.9)],
            })
            .collect();
        let cfg = TrackerConfig::default();
        assert_eq!(track_shots(&frames, &[], &cfg).len(), 1);
        let mut t = track_shots(&frames, &[10], &cfg);
        assert_eq!(t.len(), 2);
        t.sort_by_key(|t| t.keyframes[0].frame);
        assert_eq!(t[0].keyframes.last().unwrap().frame, 9);
        assert_eq!(t[1].keyframes[0].frame, 10);
        assert_eq!(t[1].keyframes.last().unwrap().frame, 19);
    }

    #[test]
    fn backward_pass_bridges_what_forward_misses() {
        // stationary at x=0 (frames 0..10), unseen 10..15, then at x=15 walking
        // right at 3 px/frame. Forward prediction stays at 0 and misses the
        // re-appearance; backward prediction walks left into the old spot.
        let frames: Vec<FrameDets> = (0..30u32)
            .map(|i| FrameDets {
                dets: match i {
                    0..=9 => vec![det(0.0, 0.0, 0.9)],
                    10..=14 => vec![],
                    _ => vec![det(15.0 + (i - 15) as f32 * 3.0, 0.0, 0.9)],
                },
            })
            .collect();
        let cfg = TrackerConfig::default();
        let fwd = run_pass(&frames, 0..frames.len(), &cfg);
        assert_ne!(fwd[9][0], fwd[15][0], "forward alone fragments");
        let tracks = track_offline(&frames, &cfg);
        assert_eq!(tracks.len(), 1, "merged by the backward pass");
        assert_eq!(observed(&tracks[0]).len(), 25);
    }
}
