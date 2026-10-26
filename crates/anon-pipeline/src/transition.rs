//! Scene transitions (hard cuts, crossfades, wipes) from tiny greyscale
//! thumbnails, so tracks are not linked from one shot into the next.
//!
//! The change between consecutive thumbnails is the mean absolute difference
//! at the best small offset (so a camera pan counts as little change). A
//! frame whose change is far above the recent level opens a transition
//! window; it stays open while such frames keep coming. Fast swings of the
//! camera open short windows too, which is harmless: the analysis then
//! detects on every frame there instead of relying on the tracker.
//!
//! Tuned on a basketball highlight reel (crossfades and a wipe graphic over
//! 5-15 frames, change 5-15 against a recent median below 1) and a fixed
//! soccer panorama (change never above 0.7, no window).

use std::collections::VecDeque;

/// A window opens when the change exceeds this multiple of (recent median + 1)...
const RATIO: f32 = 3.0;
/// ...and this absolute level (grey values, 0..255).
const MIN_CHANGE: f32 = 4.0;
/// Frames of recent change the median is taken over.
const HISTORY: usize = 15;
/// A window closes after this many calm frames in a row.
const CLOSE_AFTER: u32 = 3;
/// Horizontal / vertical offsets tried when comparing thumbnails.
const SHIFT_X: i32 = 3;
const SHIFT_Y: i32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    /// Ordinary frame.
    None,
    /// First frame of a transition window: a new shot starts here.
    Start,
    /// Inside an open window.
    Inside,
}

#[derive(Debug, Default)]
pub struct TransitionDetector {
    prev: Option<(u32, u32, Vec<u8>)>,
    history: VecDeque<f32>,
    open: bool,
    calm: u32,
}

/// Mean absolute difference of two equally sized thumbnails at the best
/// offset within `SHIFT_X` x `SHIFT_Y`.
pub fn change(w: u32, h: u32, a: &[u8], b: &[u8]) -> f32 {
    let (w, h) = (w as i32, h as i32);
    let mut best = f32::MAX;
    for dy in -SHIFT_Y..=SHIFT_Y {
        for dx in -SHIFT_X..=SHIFT_X {
            let (ow, oh) = (w - dx.abs(), h - dy.abs());
            if ow <= 0 || oh <= 0 {
                continue;
            }
            let mut sum = 0u64;
            for y in 0..oh {
                let ya = (y + dy.max(0)) * w;
                let yb = (y + (-dy).max(0)) * w;
                for x in 0..ow {
                    let pa = a[(ya + x + dx.max(0)) as usize];
                    let pb = b[(yb + x + (-dx).max(0)) as usize];
                    sum += u64::from(pa.abs_diff(pb));
                }
            }
            best = best.min(sum as f32 / (ow * oh) as f32);
        }
    }
    best
}

impl TransitionDetector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed the next frame's thumbnail.
    pub fn push(&mut self, w: u32, h: u32, thumb: Vec<u8>) -> Transition {
        let c = match &self.prev {
            Some((pw, ph, p)) if (*pw, *ph) == (w, h) => change(w, h, p, &thumb),
            // first frame, or the frame size changed: no comparison possible
            Some(_) => f32::MAX,
            None => 0.0,
        };
        self.prev = Some((w, h, thumb));
        let median = {
            let mut v: Vec<f32> = self.history.iter().copied().collect();
            v.sort_by(f32::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(0.0)
        };
        if self.history.len() == HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(c.min(255.0));
        let jump = c > MIN_CHANGE && c > RATIO * (median + 1.0);
        if jump {
            self.calm = 0;
            if !self.open {
                self.open = true;
                return Transition::Start;
            }
            return Transition::Inside;
        }
        if self.open {
            self.calm += 1;
            if self.calm >= CLOSE_AFTER {
                self.open = false;
                return Transition::None;
            }
            return Transition::Inside;
        }
        Transition::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 64;
    const H: u32 = 36;

    /// A textured scene; `seed` picks the scene, `shift` pans it.
    fn scene(seed: u32, shift: i32) -> Vec<u8> {
        (0..H)
            .flat_map(|y| {
                (0..W).map(move |x| {
                    let gx = (x as i32 + shift) as u32;
                    let v =
                        (gx.wrapping_mul(37) ^ y.wrapping_mul(91) ^ seed.wrapping_mul(2654435761))
                            % 200;
                    (v + 20) as u8
                })
            })
            .collect()
    }

    fn blend(a: &[u8], b: &[u8], t: f32) -> Vec<u8> {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x as f32 * (1.0 - t) + y as f32 * t).round() as u8)
            .collect()
    }

    fn run(frames: Vec<Vec<u8>>) -> Vec<Transition> {
        let mut d = TransitionDetector::new();
        frames.into_iter().map(|f| d.push(W, H, f)).collect()
    }

    #[test]
    fn steady_pan_is_not_a_transition() {
        let out = run((0..60).map(|i| scene(1, i * 2)).collect());
        assert!(out.iter().all(|&t| t == Transition::None), "{out:?}");
    }

    #[test]
    fn hard_cut_starts_a_new_shot() {
        let mut f: Vec<Vec<u8>> = (0..30).map(|_| scene(1, 0)).collect();
        f.extend((0..30).map(|_| scene(2, 0)));
        let out = run(f);
        let starts: Vec<usize> = (0..out.len())
            .filter(|&i| out[i] == Transition::Start)
            .collect();
        assert_eq!(starts, vec![30]);
        // the window closes again shortly after
        assert!(out[40..].iter().all(|&t| t == Transition::None));
    }

    #[test]
    fn crossfade_is_one_window() {
        let (a, b) = (scene(1, 0), scene(2, 0));
        let mut f: Vec<Vec<u8>> = (0..30).map(|_| a.clone()).collect();
        f.extend((1..=10).map(|i| blend(&a, &b, i as f32 / 10.0)));
        f.extend((0..30).map(|_| b.clone()));
        let out = run(f);
        let starts: Vec<usize> = (0..out.len())
            .filter(|&i| out[i] == Transition::Start)
            .collect();
        assert_eq!(starts, vec![30], "{out:?}");
        assert!((31..40).all(|i| out[i] == Transition::Inside), "{out:?}");
        assert!(out[50..].iter().all(|&t| t == Transition::None));
    }

    #[test]
    fn change_ignores_small_offsets() {
        let a = scene(1, 0);
        assert_eq!(change(W, H, &a, &scene(1, 3)), 0.0);
        assert!(change(W, H, &a, &scene(2, 0)) > 20.0);
    }
}
