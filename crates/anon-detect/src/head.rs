//! Derive a head box from a person instance.
//!
//! Works from any viewpoint (front, side, back) because it only uses the
//! silhouette. Errors are biased towards masking more: raised arms extend
//! the head band instead of risking a miss.

use anon_model::BBox;

use crate::{InstanceMask, PersonInstance};

/// Fraction of person height used when no neck is found.
pub const DEFAULT_CUT: f32 = 0.22;
/// Neck search window (fraction of person height from the top).
pub const NECK_WINDOW: (f32, f32) = (0.12, 0.30);
/// Head band when raised arms are suspected.
pub const ARMS_UP_CUT: f32 = 0.40;
/// Head width is at least this fraction of the head band height.
const MIN_ASPECT: f32 = 0.6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerivedHead {
    pub bbox: BBox,
    pub arms_up: bool,
}

fn widen(b: BBox) -> BBox {
    let min_w = b.height() * MIN_ASPECT;
    if b.width() >= min_w {
        return b;
    }
    let (cx, _) = b.center();
    BBox::new(cx - min_w / 2.0, b.y0, cx + min_w / 2.0, b.y1)
}

/// A head band wider than this multiple of its height may hold several heads.
pub const SPLIT_ASPECT: f32 = 1.6;
/// A split part must be at least this fraction of the head height wide to
/// count as a head. A narrower part reaching up to head level (a raised
/// hand) cancels the split; a narrower one lower down (a shoulder) is ignored.
const MIN_HEAD_WIDTH: f32 = 0.6;

/// The head(s) of one person instance. A mask that holds several people (a
/// row on a bench, a group segmented as one) yields one head per peak of its
/// upper contour instead of one flat band across all of them.
pub fn derive_heads(p: &PersonInstance) -> Vec<DerivedHead> {
    let one = derive_head(p);
    match &p.mask {
        Some(m) if one.bbox.width() > SPLIT_ASPECT * one.bbox.height() => {
            split_heads(&p.bbox, m, &one).unwrap_or_else(|| vec![one])
        }
        _ => vec![one],
    }
}

/// Columns whose top edge lies at most half a head height below the highest
/// point within half a head height around them belong to a head; each run of
/// such columns is one head. Shoulders next to a head are lower than that and
/// drop out. Every column within half a head height of the mask's top stays
/// inside some head (its neighbourhood cannot be higher than the top), so the
/// heads cover what the band covered there. `None` (keep the band) unless it
/// yields at least two head-wide parts and no raised hand.
fn split_heads(b: &BBox, m: &InstanceMask, one: &DerivedHead) -> Option<Vec<DerivedHead>> {
    let cw = b.width() / m.w as f32;
    let ch = b.height() / m.h as f32;
    // head height in cells: the plain top-of-person share, not the taller
    // raised-arms band (several heads side by side also look like that)
    let rows = (0..m.h).filter(|&y| (0..m.w).any(|x| m.get(x, y))).count() as f32;
    let band = ((rows * DEFAULT_CUT).round() as i64).max(2);
    let top: Vec<Option<i64>> = (0..m.w)
        .map(|x| (0..m.h).find(|&y| m.get(x, y)).map(i64::from))
        .collect();
    let reach = (band / 2).max(1);
    let w = i64::from(m.w);
    let head_col: Vec<bool> = (0..w)
        .map(|x| {
            let Some(t) = top[x as usize] else {
                return false;
            };
            let near = ((x - reach).max(0)..=(x + reach).min(w - 1))
                .filter_map(|i| top[i as usize])
                .min()
                .unwrap_or(t);
            t <= near + band / 2
        })
        .collect();
    let mut runs = Vec::new();
    let mut x = 0;
    while x < w {
        if !head_col[x as usize] {
            x += 1;
            continue;
        }
        let l = x;
        while x < w && head_col[x as usize] {
            x += 1;
        }
        let t0 = (l..x).filter_map(|i| top[i as usize]).min()?;
        runs.push((l, x, t0));
    }
    let wide = |&(l, r, _): &(i64, i64, i64)| (r - l) as f32 >= MIN_HEAD_WIDTH * band as f32;
    let head_level = runs.iter().filter(|r| wide(r)).map(|r| r.2).max()?;
    if runs.iter().any(|r| !wide(r) && r.2 <= head_level) {
        return None;
    }
    let mut heads = Vec::new();
    for &(l, r, t0) in runs.iter().filter(|r| wide(r)) {
        heads.push(DerivedHead {
            bbox: widen(BBox::new(
                b.x0 + l as f32 * cw,
                b.y0 + t0 as f32 * ch,
                b.x0 + r as f32 * cw,
                b.y0 + (t0 + band) as f32 * ch,
            )),
            arms_up: one.arms_up,
        });
    }
    (heads.len() >= 2).then_some(heads)
}

pub fn derive_head(p: &PersonInstance) -> DerivedHead {
    match &p.mask {
        Some(m) => from_mask(&p.bbox, m).unwrap_or_else(|| from_box(&p.bbox)),
        None => from_box(&p.bbox),
    }
}

fn from_box(b: &BBox) -> DerivedHead {
    DerivedHead {
        bbox: widen(BBox::new(b.x0, b.y0, b.x1, b.y0 + b.height() * DEFAULT_CUT)),
        arms_up: false,
    }
}

struct RowStats {
    count: u32,
    runs: u32,
    min_x: u32,
    max_x: u32,
    sum_x: u64,
}

fn row_stats(m: &InstanceMask, y: u32) -> RowStats {
    let mut s = RowStats {
        count: 0,
        runs: 0,
        min_x: u32::MAX,
        max_x: 0,
        sum_x: 0,
    };
    let mut prev = false;
    for x in 0..m.w {
        let on = m.get(x, y);
        if on {
            s.count += 1;
            s.min_x = s.min_x.min(x);
            s.max_x = s.max_x.max(x);
            s.sum_x += u64::from(x);
            if !prev {
                s.runs += 1;
            }
        }
        prev = on;
    }
    s
}

fn centroid(rows: &[RowStats]) -> Option<f32> {
    let (sum, n) = rows.iter().fold((0u64, 0u64), |(s, n), r| {
        (s + r.sum_x, n + u64::from(r.count))
    });
    (n > 0).then(|| sum as f32 / n as f32)
}

fn from_mask(b: &BBox, m: &InstanceMask) -> Option<DerivedHead> {
    let rows: Vec<RowStats> = (0..m.h).map(|y| row_stats(m, y)).collect();
    let top = rows.iter().position(|r| r.count > 0)? as u32;
    let bottom = rows.iter().rposition(|r| r.count > 0)? as u32;
    let mh = (bottom - top + 1) as f32;
    let at = |f: f32| (top + (f * mh) as u32).min(bottom);

    // Raised arms: several separate blobs near the top, or the top of the
    // silhouette far off the torso's centre line.
    let top_band = &rows[top as usize..=at(0.15) as usize];
    let torso = &rows[at(0.30) as usize..=at(0.60) as usize];
    let torso_width = torso
        .iter()
        .filter(|r| r.count > 0)
        .map(|r| (r.max_x - r.min_x + 1) as f32)
        .fold(0.0f32, f32::max)
        .max(1.0);
    let off_centre = match (centroid(top_band), centroid(torso)) {
        (Some(a), Some(b)) => (a - b).abs() > 0.25 * torso_width,
        _ => false,
    };
    let arms_up = top_band.iter().any(|r| r.runs >= 2) || off_centre;

    let cut = if arms_up {
        at(ARMS_UP_CUT)
    } else {
        // Neck = narrowest (3-row smoothed) row in the window, accepted only
        // if clearly narrower than the widest row above it.
        let (lo, hi) = (at(NECK_WINDOW.0), at(NECK_WINDOW.1));
        let width = |y: u32| -> f32 {
            let r = &rows[y as usize];
            if r.count == 0 {
                0.0
            } else {
                (r.max_x - r.min_x + 1) as f32
            }
        };
        let smooth = |y: u32| -> f32 {
            let a = y.saturating_sub(1).max(top);
            let c = (y + 1).min(bottom);
            (width(a) + width(y) + width(c)) / 3.0
        };
        let neck = (lo..=hi).min_by(|&a, &b| smooth(a).total_cmp(&smooth(b)));
        let head_max = (top..lo.max(top + 1)).map(width).fold(0.0f32, f32::max);
        match neck {
            Some(n) if smooth(n) > 0.0 && smooth(n) < 0.8 * head_max => n,
            _ => at(DEFAULT_CUT),
        }
    };

    let band = &rows[top as usize..=cut as usize];
    let min_x = band.iter().filter(|r| r.count > 0).map(|r| r.min_x).min()?;
    let max_x = band.iter().filter(|r| r.count > 0).map(|r| r.max_x).max()?;
    let cw = b.width() / m.w as f32;
    let ch = b.height() / m.h as f32;
    let head = BBox::new(
        b.x0 + min_x as f32 * cw,
        b.y0 + top as f32 * ch,
        b.x0 + (max_x + 1) as f32 * cw,
        b.y0 + (cut + 1) as f32 * ch,
    );
    Some(DerivedHead {
        bbox: widen(head),
        arms_up,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Draw a stick figure into a mask: list of (y_from, y_to, x_from, x_to).
    fn figure(w: u32, h: u32, parts: &[(u32, u32, u32, u32)]) -> InstanceMask {
        let mut bits = vec![false; (w * h) as usize];
        for &(y0, y1, x0, x1) in parts {
            for y in y0..y1 {
                for x in x0..x1 {
                    bits[(y * w + x) as usize] = true;
                }
            }
        }
        InstanceMask { w, h, bits }
    }

    fn person(m: InstanceMask) -> PersonInstance {
        PersonInstance {
            bbox: BBox::new(100.0, 200.0, 100.0 + m.w as f32, 200.0 + m.h as f32),
            score: 0.9,
            mask: Some(m),
            truncated: false,
            passes: vec![],
        }
    }

    #[test]
    fn row_of_seated_people_gives_one_head_each() {
        // three heads (6 wide, 6 tall) on one continuous band of shoulders
        let m = figure(
            40,
            30,
            &[(0, 6, 3, 9), (2, 8, 17, 23), (1, 7, 31, 37), (8, 30, 0, 40)],
        );
        let p = person(m);
        assert!(derive_head(&p).bbox.width() > 30.0, "one band across all");
        let heads = derive_heads(&p);
        assert_eq!(heads.len(), 3, "{heads:?}");
        for (h, x) in heads.iter().zip([6.0, 20.0, 34.0]) {
            let (cx, _) = h.bbox.center();
            assert!((cx - (100.0 + x)).abs() < 2.5, "{h:?}");
            assert!(h.bbox.width() < 12.0, "{h:?}");
        }
    }

    #[test]
    fn single_person_is_not_split() {
        let m = figure(
            20,
            100,
            &[
                (0, 12, 6, 14),
                (12, 15, 8, 12),
                (15, 60, 0, 20),
                (60, 100, 4, 16),
            ],
        );
        let p = person(m);
        assert_eq!(derive_heads(&p), vec![derive_head(&p)]);
    }

    #[test]
    fn raised_hand_next_to_heads_keeps_the_band() {
        // two heads and a narrow raised hand between them: no split, so the
        // band keeps covering everything
        let m = figure(
            40,
            30,
            &[(2, 8, 3, 9), (2, 8, 29, 35), (0, 8, 18, 20), (8, 30, 0, 40)],
        );
        let p = person(m);
        assert_eq!(derive_heads(&p), vec![derive_head(&p)]);
    }

    #[test]
    fn finds_neck() {
        // head 0..12 (width 8), neck 12..15 (width 4), torso 15..60 (width 20)
        let m = figure(
            20,
            100,
            &[
                (0, 12, 6, 14),
                (12, 15, 8, 12),
                (15, 60, 0, 20),
                (60, 100, 4, 16),
            ],
        );
        let h = derive_head(&person(m));
        assert!(!h.arms_up);
        assert!(
            h.bbox.y1 <= 200.0 + 16.0 && h.bbox.y1 >= 200.0 + 12.0,
            "{:?}",
            h.bbox
        );
        assert!(h.bbox.x0 >= 100.0 + 4.0 && h.bbox.x1 <= 100.0 + 16.0);
    }

    #[test]
    fn raised_arms_extend_band() {
        // two hands above the head
        let m = figure(
            30,
            100,
            &[
                (0, 30, 2, 6),
                (0, 30, 24, 28),
                (10, 22, 11, 19),
                (22, 25, 13, 17),
                (25, 70, 5, 25),
                (70, 100, 8, 22),
            ],
        );
        let h = derive_head(&person(m));
        assert!(h.arms_up);
        // band must reach below the real head (rows 10..22)
        assert!(h.bbox.y1 >= 200.0 + 22.0);
    }

    #[test]
    fn no_mask_uses_top_of_box() {
        let p = PersonInstance {
            bbox: BBox::new(0.0, 0.0, 40.0, 100.0),
            score: 0.5,
            mask: None,
            truncated: false,
            passes: vec![],
        };
        let h = derive_head(&p);
        assert_eq!(h.bbox, BBox::new(0.0, 0.0, 40.0, 22.0));
    }
}
