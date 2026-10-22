//! Fusion of detections from all tiles, scales and the flip pass.
//!
//! Policy is recall-first: instances are only merged when they clearly
//! describe the same person. Two overlapping people stay separate, so each
//! still gets its own head.

use anon_model::BBox;

use crate::{InstanceMask, PersonInstance};

/// IoU above which two detections are the same person.
pub const SAME_IOU: f32 = 0.5;
/// Intersection-over-smaller above which a cut-off fragment belongs to a
/// larger detection.
pub const FRAGMENT_IOS: f32 = 0.8;
const MAX_MASK_CELLS: f32 = 512.0;

fn density(b: &BBox, m: &Option<InstanceMask>) -> (f32, f32) {
    match m {
        Some(m) if b.width() > 0.0 && b.height() > 0.0 => {
            (m.w as f32 / b.width(), m.h as f32 / b.height())
        }
        _ => (0.0, 0.0),
    }
}

/// Sample a mask (or a full box when it has none) at a source point.
fn covered(b: &BBox, m: &Option<InstanceMask>, x: f32, y: f32) -> bool {
    if x < b.x0 || x >= b.x1 || y < b.y0 || y >= b.y1 {
        return false;
    }
    match m {
        None => true,
        Some(m) => {
            let cx = (((x - b.x0) / b.width()) * m.w as f32) as u32;
            let cy = (((y - b.y0) / b.height()) * m.h as f32) as u32;
            m.get(cx.min(m.w - 1), cy.min(m.h - 1))
        }
    }
}

/// Union of two instances' boxes and masks. A missing mask counts as a
/// fully covered box (conservative).
pub fn merge_masks(
    a: &BBox,
    am: &Option<InstanceMask>,
    b: &BBox,
    bm: &Option<InstanceMask>,
) -> (BBox, Option<InstanceMask>) {
    let u = a.union_box(b);
    if am.is_none() && bm.is_none() {
        return (u, None);
    }
    let (ax, ay) = density(a, am);
    let (bx, by) = density(b, bm);
    let w = (u.width() * ax.max(bx)).ceil().clamp(1.0, MAX_MASK_CELLS) as u32;
    let h = (u.height() * ay.max(by)).ceil().clamp(1.0, MAX_MASK_CELLS) as u32;
    let mut bits = vec![false; (w * h) as usize];
    for j in 0..h {
        let y = u.y0 + (j as f32 + 0.5) * u.height() / h as f32;
        for i in 0..w {
            let x = u.x0 + (i as f32 + 0.5) * u.width() / w as f32;
            bits[(j * w + i) as usize] = covered(a, am, x, y) || covered(b, bm, x, y);
        }
    }
    (u, Some(InstanceMask { w, h, bits }))
}

fn absorb(into: &mut PersonInstance, other: &PersonInstance) {
    let (bbox, mask) = merge_masks(&into.bbox, &into.mask, &other.bbox, &other.mask);
    into.bbox = bbox;
    into.mask = mask;
    into.score = into.score.max(other.score);
    into.truncated = into.truncated && other.truncated;
    for p in &other.passes {
        if !into.passes.contains(p) {
            into.passes.push(p);
        }
    }
}

/// Fuse raw detections (already in source coordinates). Repeats until no
/// further merge happens, because clusters grow while absorbing fragments.
pub fn fuse(dets: Vec<PersonInstance>) -> Vec<PersonInstance> {
    let mut cur = dets;
    loop {
        let n = cur.len();
        cur = fuse_once(cur);
        if cur.len() == n {
            return cur;
        }
    }
}

fn fuse_once(mut dets: Vec<PersonInstance>) -> Vec<PersonInstance> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut out: Vec<PersonInstance> = Vec::new();
    'next: for d in dets {
        for c in out.iter_mut() {
            let same = c.bbox.iou(&d.bbox) >= SAME_IOU;
            let fragment = (d.truncated || c.truncated) && c.bbox.ios(&d.bbox) >= FRAGMENT_IOS;
            if same || fragment {
                absorb(c, &d);
                continue 'next;
            }
        }
        out.push(d);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(b: BBox, s: f32, truncated: bool) -> PersonInstance {
        PersonInstance {
            bbox: b,
            score: s,
            mask: None,
            truncated,
            passes: vec!["t"],
        }
    }

    #[test]
    fn duplicates_merge_neighbours_stay() {
        let a = p(BBox::new(0.0, 0.0, 10.0, 30.0), 0.9, false);
        let a2 = p(BBox::new(1.0, 0.0, 11.0, 30.0), 0.5, false);
        // overlapping second person, IoU < 0.5
        let b = p(BBox::new(6.0, 0.0, 16.0, 30.0), 0.8, false);
        let out = fuse(vec![a, a2, b]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].score, 0.9);
        assert_eq!(out[0].bbox, BBox::new(0.0, 0.0, 11.0, 30.0));
    }

    #[test]
    fn truncated_fragment_absorbed_untruncated_child_kept() {
        let full = p(BBox::new(0.0, 0.0, 10.0, 30.0), 0.9, false);
        let frag = p(BBox::new(0.0, 0.0, 10.0, 12.0), 0.4, true);
        // a small person standing in front: inside but not truncated
        let child = p(BBox::new(2.0, 15.0, 6.0, 29.0), 0.7, false);
        let out = fuse(vec![full, frag, child]);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn mask_union_is_conservative() {
        let a = BBox::new(0.0, 0.0, 4.0, 4.0);
        let am = Some(InstanceMask {
            w: 2,
            h: 2,
            bits: vec![true, false, false, false],
        });
        let b = BBox::new(4.0, 0.0, 8.0, 4.0);
        let (u, m) = merge_masks(&a, &am, &b, &None);
        let m = m.unwrap();
        assert_eq!(u, BBox::new(0.0, 0.0, 8.0, 4.0));
        assert_eq!((m.w, m.h), (4, 2));
        assert_eq!(
            m.bits,
            vec![true, false, true, true, false, false, true, true]
        );
    }
}
