//! Overlapping tile grid (SAHI-style) so small, distant people are seen at
//! native resolution.

/// A tile in source pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl TileRect {
    /// Which of this tile's edges lie inside the image (not on its border).
    /// Detections touching an internal edge may be cut off.
    pub fn internal_edges(&self, img_w: u32, img_h: u32) -> [bool; 4] {
        [
            self.x > 0,              // left
            self.y > 0,              // top
            self.x + self.w < img_w, // right
            self.y + self.h < img_h, // bottom
        ]
    }
}

fn axis(len: u32, tile: u32, overlap: f32) -> Vec<u32> {
    if len <= tile {
        return vec![0];
    }
    let stride = ((tile as f32) * (1.0 - overlap.clamp(0.0, 0.9))).max(1.0) as u32;
    let mut v = Vec::new();
    let mut p = 0;
    loop {
        if p + tile >= len {
            v.push(len - tile);
            break;
        }
        v.push(p);
        p += stride;
    }
    v.dedup();
    v
}

/// Tiles of `tile × tile` covering `width × height` with at least `overlap`
/// fractional overlap. The last row/column is snapped to the image edge so
/// every pixel is covered and no tile leaves the image.
pub fn plan_tiles(width: u32, height: u32, tile: u32, overlap: f32) -> Vec<TileRect> {
    let xs = axis(width, tile, overlap);
    let ys = axis(height, tile, overlap);
    let mut out = Vec::with_capacity(xs.len() * ys.len());
    for &y in &ys {
        for &x in &xs {
            out.push(TileRect {
                x,
                y,
                w: tile.min(width),
                h: tile.min(height),
            });
        }
    }
    out
}

/// A tile together with the tile edge it was planned for (its scale).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaledTile {
    pub rect: TileRect,
    pub tile: u32,
}

/// [`plan_tiles`] over the horizontal band `y0..y1`, in image coordinates.
fn plan_band(width: u32, y0: u32, y1: u32, tile: u32, overlap: f32) -> Vec<ScaledTile> {
    plan_tiles(width, y1 - y0, tile, overlap)
        .into_iter()
        .map(|r| ScaledTile {
            rect: TileRect { y: r.y + y0, ..r },
            tile,
        })
        .collect()
}

/// Perspective tiling: `fine` tiles above row `split`, `coarse` tiles from
/// there down. The bands overlap (the fine band reaches half a fine tile
/// below `split`, the coarse band a quarter coarse tile above it) so a person
/// standing on the border is seen whole by at least one of them. Without a
/// usable `split` this is [`plan_tiles`] with `fine`.
pub fn plan_perspective(
    width: u32,
    height: u32,
    fine: u32,
    coarse: u32,
    split: Option<u32>,
    overlap: f32,
) -> Vec<ScaledTile> {
    match split {
        Some(s) if s > 0 && s < height && coarse > fine => {
            let fine_end = (s + fine / 2).min(height);
            let coarse_start = s.saturating_sub(coarse / 4);
            let mut v = plan_band(width, 0, fine_end, fine, overlap);
            v.extend(plan_band(width, coarse_start, height, coarse, overlap));
            v
        }
        _ => plan_band(width, 0, height, fine, overlap),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perspective_covers_every_pixel_with_fewer_tiles() {
        let (w, h) = (4096, 1152);
        let full = plan_tiles(w, h, 432, 0.25);
        let tiles = plan_perspective(w, h, 432, 864, Some(518), 0.25);
        assert!(
            tiles.len() * 3 < full.len() * 2,
            "{} vs {}",
            tiles.len(),
            full.len()
        );
        let mut covered = vec![false; (w * h) as usize];
        for t in &tiles {
            let r = t.rect;
            assert!(r.x + r.w <= w && r.y + r.h <= h, "{r:?}");
            for y in r.y..r.y + r.h {
                for x in r.x..r.x + r.w {
                    covered[(y * w + x) as usize] = true;
                }
            }
        }
        assert!(covered.iter().all(|&c| c));
        // rows above the split are only ever seen by fine tiles
        assert!(
            tiles
                .iter()
                .filter(|t| t.rect.y < 518 - 216)
                .all(|t| t.tile == 432)
        );
    }

    #[test]
    fn perspective_without_split_is_plain_tiling() {
        let plain: Vec<_> = plan_tiles(1920, 1080, 432, 0.25);
        for split in [None, Some(0), Some(1080), Some(5000)] {
            let p = plan_perspective(1920, 1080, 432, 864, split, 0.25);
            assert_eq!(p.iter().map(|t| t.rect).collect::<Vec<_>>(), plain);
        }
    }

    #[test]
    fn covers_every_pixel() {
        for (w, h, t) in [
            (5760, 1080, 432),
            (1920, 1080, 640),
            (300, 200, 432),
            (1000, 433, 432),
        ] {
            let tiles = plan_tiles(w, h, t, 0.25);
            let mut covered = vec![false; (w * h) as usize];
            for r in &tiles {
                assert!(r.x + r.w <= w && r.y + r.h <= h);
                for y in r.y..r.y + r.h {
                    for x in r.x..r.x + r.w {
                        covered[(y * w + x) as usize] = true;
                    }
                }
            }
            assert!(covered.iter().all(|&c| c), "{w}x{h}");
        }
    }

    #[test]
    fn overlap_at_least_requested() {
        let xs = axis(5760, 432, 0.25);
        for p in xs.windows(2) {
            assert!(p[1] - p[0] <= 324);
        }
    }
}
