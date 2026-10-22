//! Opaque masking in the YUV domain.
//!
//! Painting happens directly on the decoded planar YUV frame, before the
//! encoder, so no colour conversion touches unmasked pixels. Coverage is
//! conservative: a pixel is painted if the shape touches any part of it,
//! and a chroma sample is painted if any luma pixel it covers is painted
//! (no colour bleed from under the mask).
//!
//! Why not a wgpu compute pass: the work is a few spans per head; a CPU
//! span fill costs microseconds per frame, needs no upload/readback and is
//! bit-exact on every platform. GPU stays an option for the push API.

use anon_model::{BBox, Shape};

/// Row spans `[x0, x1)` that must be painted, luma resolution.
#[derive(Debug, Clone)]
pub struct Coverage {
    pub width: u32,
    pub height: u32,
    rows: Vec<Vec<(u32, u32)>>,
    full: bool,
}

impl Coverage {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            rows: vec![Vec::new(); height as usize],
            full: false,
        }
    }

    /// Everything is painted (fail-closed frames).
    pub fn full(width: u32, height: u32) -> Self {
        let mut c = Self::new(width, height);
        c.full = true;
        c
    }

    pub fn is_full(&self) -> bool {
        self.full
    }

    pub fn is_empty(&self) -> bool {
        !self.full && self.rows.iter().all(Vec::is_empty)
    }

    fn push_span(&mut self, y: u32, x0: f32, x1: f32) {
        if y >= self.height {
            return;
        }
        let a = x0.floor().max(0.0) as u32;
        let b = (x1.ceil().max(0.0) as u32).min(self.width);
        if a < b {
            self.rows[y as usize].push((a, b));
        }
    }

    pub fn add_shape(&mut self, s: &Shape) {
        if self.full {
            return;
        }
        let bounds = s.bounds();
        if bounds.x1 < 0.0
            || bounds.y1 < 0.0
            || bounds.x0 >= self.width as f32
            || bounds.y0 >= self.height as f32
        {
            return;
        }
        let y_start = bounds.y0.floor().max(0.0) as u32;
        let y_end = (bounds.y1.ceil().max(0.0) as u32).min(self.height);
        match s {
            Shape::Ellipse { cx, cy, rx, ry } => {
                if *rx <= 0.0 || *ry <= 0.0 {
                    return;
                }
                for y in y_start..y_end {
                    // Widest point of the ellipse within this pixel row.
                    let (top, bot) = (y as f32, y as f32 + 1.0);
                    let yy = cy.clamp(top, bot);
                    let t = (yy - cy) / ry;
                    let k = 1.0 - t * t;
                    if k < 0.0 {
                        continue;
                    }
                    let hw = rx * k.sqrt();
                    self.push_span(y, cx - hw, cx + hw);
                }
            }
            Shape::Polygon { points } => {
                let n = points.len();
                if n < 3 {
                    return;
                }
                let mut xs = Vec::new();
                for y in y_start..y_end {
                    // Sample at three heights in the row so thin slivers and
                    // pixel-row-crossing edges are not missed.
                    let mut spans: Vec<(f32, f32)> = Vec::new();
                    for sy in [y as f32 + 0.01, y as f32 + 0.5, y as f32 + 0.99] {
                        xs.clear();
                        let mut j = n - 1;
                        for i in 0..n {
                            let (xi, yi) = (points[i][0], points[i][1]);
                            let (xj, yj) = (points[j][0], points[j][1]);
                            if (yi > sy) != (yj > sy) {
                                xs.push(xi + (sy - yi) * (xj - xi) / (yj - yi));
                            }
                            j = i;
                        }
                        xs.sort_by(f32::total_cmp);
                        for pair in xs.as_chunks::<2>().0.iter() {
                            spans.push((pair[0], pair[1]));
                        }
                    }
                    for (a, b) in spans {
                        self.push_span(y, a, b);
                    }
                }
            }
        }
    }

    /// Sorted, merged spans of row `y`.
    pub fn row(&self, y: u32) -> Vec<(u32, u32)> {
        if self.full {
            return vec![(0, self.width)];
        }
        merge(self.rows.get(y as usize).cloned().unwrap_or_default())
    }

    pub fn contains(&self, x: u32, y: u32) -> bool {
        self.row(y).iter().any(|&(a, b)| x >= a && x < b)
    }

    /// Fraction of pixels of `b` that are painted.
    pub fn covered_fraction(&self, b: &BBox) -> f32 {
        let x0 = b.x0.floor().max(0.0) as u32;
        let x1 = (b.x1.ceil().max(0.0) as u32).min(self.width);
        let y0 = b.y0.floor().max(0.0) as u32;
        let y1 = (b.y1.ceil().max(0.0) as u32).min(self.height);
        if x0 >= x1 || y0 >= y1 {
            return 1.0;
        }
        let mut covered = 0u64;
        for y in y0..y1 {
            for (a, c) in self.row(y) {
                let lo = a.max(x0);
                let hi = c.min(x1);
                if lo < hi {
                    covered += u64::from(hi - lo);
                }
            }
        }
        covered as f32 / (u64::from(x1 - x0) * u64::from(y1 - y0)) as f32
    }
}

fn merge(mut spans: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    spans.sort_unstable();
    let mut out: Vec<(u32, u32)> = Vec::with_capacity(spans.len());
    for (a, b) in spans {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// One plane of a planar frame.
pub struct PlaneMut<'a> {
    pub data: &'a mut [u8],
    /// Bytes per row.
    pub stride: usize,
}

/// Planar YUV layout description.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YuvLayout {
    /// log2 of horizontal / vertical chroma subsampling (1,1 = 4:2:0).
    pub log2_chroma_w: u32,
    pub log2_chroma_h: u32,
    /// 1 for 8-bit, 2 for 9..16-bit little-endian samples.
    pub bytes_per_sample: usize,
    pub bit_depth: u32,
    pub full_range: bool,
}

impl YuvLayout {
    /// Opaque black for this layout: Y, U, V sample values.
    pub fn black(&self) -> [u16; 3] {
        let shift = self.bit_depth.saturating_sub(8);
        let y = if self.full_range { 0 } else { 16u16 << shift };
        let c = 128u16 << shift;
        [y, c, c]
    }
}

fn fill_span(plane: &mut PlaneMut<'_>, y: u32, x0: u32, x1: u32, bps: usize, v: u16) {
    let start = y as usize * plane.stride + x0 as usize * bps;
    let end = y as usize * plane.stride + x1 as usize * bps;
    let row = &mut plane.data[start..end];
    if bps == 1 {
        row.fill(v as u8);
    } else {
        let b = v.to_le_bytes();
        for px in row.as_chunks_mut::<2>().0 {
            *px = b;
        }
    }
}

/// Paint `cov` with `value` (Y, U, V) into a planar YUV frame.
pub fn paint_yuv(
    planes: &mut [PlaneMut<'_>; 3],
    layout: YuvLayout,
    value: [u16; 3],
    cov: &Coverage,
) {
    if cov.is_empty() {
        return;
    }
    let bps = layout.bytes_per_sample;
    for y in 0..cov.height {
        for (a, b) in cov.row(y) {
            fill_span(&mut planes[0], y, a, b, bps, value[0]);
        }
    }
    let sx = layout.log2_chroma_w;
    let sy = layout.log2_chroma_h;
    let cw = cov.width.div_ceil(1 << sx);
    let ch = cov.height.div_ceil(1 << sy);
    for cy in 0..ch {
        let mut spans = Vec::new();
        for ly in (cy << sy)..((cy + 1) << sy).min(cov.height) {
            for (a, b) in cov.row(ly) {
                spans.push((a >> sx, (b.div_ceil(1 << sx)).min(cw)));
            }
        }
        for (a, b) in merge(spans) {
            fill_span(&mut planes[1], cy, a, b, bps, value[1]);
            fill_span(&mut planes[2], cy, a, b, bps, value[2]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ellipse_coverage_is_conservative() {
        let mut c = Coverage::new(100, 100);
        let e = Shape::Ellipse {
            cx: 50.0,
            cy: 50.0,
            rx: 10.3,
            ry: 5.2,
        };
        c.add_shape(&e);
        // every pixel whose any corner is inside must be painted
        for y in 0..100u32 {
            for x in 0..100u32 {
                let touched = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)]
                    .iter()
                    .any(|(dx, dy)| e.contains(x as f32 + dx, y as f32 + dy));
                if touched {
                    assert!(c.contains(x, y), "missed {x},{y}");
                }
            }
        }
    }

    #[test]
    fn polygon_spans() {
        let mut c = Coverage::new(20, 20);
        c.add_shape(&Shape::Polygon {
            points: vec![[2.0, 2.0], [10.0, 2.0], [10.0, 6.0], [2.0, 6.0]],
        });
        assert!(c.contains(2, 2) && c.contains(9, 5));
        assert!(!c.contains(10, 3) && !c.contains(5, 6) && !c.contains(1, 3));
        assert!((c.covered_fraction(&BBox::new(2.0, 2.0, 10.0, 6.0)) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn paints_luma_and_full_chroma_blocks_420() {
        let (w, h) = (8u32, 4u32);
        let mut y = vec![200u8; 8 * 4];
        let mut u = vec![50u8; 4 * 2];
        let mut v = vec![60u8; 4 * 2];
        let mut cov = Coverage::new(w, h);
        // single pixel at (3,1): chroma block (1,0) must be painted
        cov.add_shape(&Shape::Polygon {
            points: vec![[3.2, 1.2], [3.8, 1.2], [3.8, 1.8], [3.2, 1.8]],
        });
        let layout = YuvLayout {
            log2_chroma_w: 1,
            log2_chroma_h: 1,
            bytes_per_sample: 1,
            bit_depth: 8,
            full_range: false,
        };
        let mut planes = [
            PlaneMut {
                data: &mut y,
                stride: 8,
            },
            PlaneMut {
                data: &mut u,
                stride: 4,
            },
            PlaneMut {
                data: &mut v,
                stride: 4,
            },
        ];
        paint_yuv(&mut planes, layout, layout.black(), &cov);
        assert_eq!(y[8 + 3], 16);
        assert_eq!(y.iter().filter(|&&p| p == 16).count(), 1);
        assert_eq!(u[1], 128);
        assert_eq!(v[1], 128);
        assert_eq!(u.iter().filter(|&&p| p == 128).count(), 1);
    }

    #[test]
    fn ten_bit_full_range_black() {
        let l = YuvLayout {
            log2_chroma_w: 1,
            log2_chroma_h: 1,
            bytes_per_sample: 2,
            bit_depth: 10,
            full_range: true,
        };
        assert_eq!(l.black(), [0, 512, 512]);
        let mut y = vec![0xffu8; 4 * 2 * 2];
        let mut u = vec![0u8; 2 * 2];
        let mut v = vec![0u8; 2 * 2];
        let mut planes = [
            PlaneMut {
                data: &mut y,
                stride: 8,
            },
            PlaneMut {
                data: &mut u,
                stride: 4,
            },
            PlaneMut {
                data: &mut v,
                stride: 4,
            },
        ];
        paint_yuv(&mut planes, l, l.black(), &Coverage::full(4, 2));
        assert!(y.iter().all(|&b| b == 0));
        assert_eq!(u, vec![0x00, 0x02, 0x00, 0x02]);
    }
}
