//! Minimal RGB8 image ops needed for tiling (crop, pad, flip, resize).

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    /// Row-major RGB, no padding, `width * height * 3` bytes.
    pub data: Vec<u8>,
}

impl RgbImage {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0; (width * height * 3) as usize],
        }
    }

    pub fn from_raw(width: u32, height: u32, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), (width * height * 3) as usize, "RGB buffer size");
        Self {
            width,
            height,
            data,
        }
    }

    #[inline]
    pub fn px(&self, x: u32, y: u32) -> [u8; 3] {
        let i = ((y * self.width + x) * 3) as usize;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }

    /// Crop `w × h` at `(x, y)` into a `size × size` canvas (top-left
    /// aligned, rest filled with mid-grey, as the model's padding).
    pub fn crop_padded(&self, x: u32, y: u32, w: u32, h: u32, size: u32) -> RgbImage {
        let mut out = RgbImage {
            width: size,
            height: size,
            data: vec![114; (size * size * 3) as usize],
        };
        let w = w.min(size).min(self.width.saturating_sub(x));
        let h = h.min(size).min(self.height.saturating_sub(y));
        for row in 0..h {
            let src = (((y + row) * self.width + x) * 3) as usize;
            let dst = ((row * size) * 3) as usize;
            out.data[dst..dst + (w * 3) as usize]
                .copy_from_slice(&self.data[src..src + (w * 3) as usize]);
        }
        out
    }

    pub fn flip_horizontal(&self) -> RgbImage {
        let mut out = RgbImage::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let s = ((y * self.width + x) * 3) as usize;
                let d = ((y * self.width + (self.width - 1 - x)) * 3) as usize;
                out.data[d..d + 3].copy_from_slice(&self.data[s..s + 3]);
            }
        }
        out
    }

    /// Bilinear resize to `w × h`.
    pub fn resize(&self, w: u32, h: u32) -> RgbImage {
        let mut out = RgbImage::new(w, h);
        let sx = self.width as f32 / w as f32;
        let sy = self.height as f32 / h as f32;
        for y in 0..h {
            let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (self.height - 1) as f32);
            let y0 = fy.floor() as u32;
            let y1 = (y0 + 1).min(self.height - 1);
            let ty = fy - y0 as f32;
            for x in 0..w {
                let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (self.width - 1) as f32);
                let x0 = fx.floor() as u32;
                let x1 = (x0 + 1).min(self.width - 1);
                let tx = fx - x0 as f32;
                let (a, b, c, d) = (
                    self.px(x0, y0),
                    self.px(x1, y0),
                    self.px(x0, y1),
                    self.px(x1, y1),
                );
                let i = ((y * w + x) * 3) as usize;
                for ch in 0..3 {
                    let top = a[ch] as f32 * (1.0 - tx) + b[ch] as f32 * tx;
                    let bot = c[ch] as f32 * (1.0 - tx) + d[ch] as f32 * tx;
                    out.data[i + ch] = (top * (1.0 - ty) + bot * ty).round() as u8;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_pads_and_flip_mirrors() {
        let mut img = RgbImage::new(3, 2);
        img.data[0] = 9; // (0,0).r
        let c = img.crop_padded(0, 0, 3, 2, 4);
        assert_eq!(c.px(0, 0)[0], 9);
        assert_eq!(c.px(3, 3), [114, 114, 114]);
        let f = img.flip_horizontal();
        assert_eq!(f.px(2, 0)[0], 9);
    }

    #[test]
    fn resize_keeps_flat_colour() {
        let img = RgbImage::from_raw(4, 4, vec![50; 48]);
        let r = img.resize(7, 3);
        assert!(r.data.iter().all(|&v| v == 50));
    }
}

impl RgbImage {
    /// Rotate clockwise by 0/90/180/270 degrees (display orientation).
    pub fn rotate_cw(&self, deg: i32) -> RgbImage {
        let (w, h) = (self.width, self.height);
        match deg.rem_euclid(360) {
            90 | 270 => {
                let mut out = RgbImage::new(h, w);
                for y in 0..h {
                    for x in 0..w {
                        let (nx, ny) = if deg.rem_euclid(360) == 90 {
                            (h - 1 - y, x)
                        } else {
                            (y, w - 1 - x)
                        };
                        let s = ((y * w + x) * 3) as usize;
                        let d = ((ny * h + nx) * 3) as usize;
                        out.data[d..d + 3].copy_from_slice(&self.data[s..s + 3]);
                    }
                }
                out
            }
            180 => {
                let mut out = RgbImage::new(w, h);
                for y in 0..h {
                    for x in 0..w {
                        let s = ((y * w + x) * 3) as usize;
                        let d = (((h - 1 - y) * w + (w - 1 - x)) * 3) as usize;
                        out.data[d..d + 3].copy_from_slice(&self.data[s..s + 3]);
                    }
                }
                out
            }
            _ => self.clone(),
        }
    }
}

/// Map a point from a `rotate_cw(deg)` image back to the stored image of
/// size `w × h` (continuous coordinates).
pub fn unrotate_point(deg: i32, w: f32, h: f32, x: f32, y: f32) -> (f32, f32) {
    match deg.rem_euclid(360) {
        90 => (y, h - x),
        180 => (w - x, h - y),
        270 => (w - y, x),
        _ => (x, y),
    }
}

#[cfg(test)]
mod rot_tests {
    use super::*;

    #[test]
    fn rotation_roundtrip_of_points() {
        let mut img = RgbImage::new(4, 2);
        // mark stored pixel (3, 0)
        img.data[9] = 255;
        for deg in [0, 90, 180, 270] {
            let r = img.rotate_cw(deg);
            let (mut fx, mut fy) = (0, 0);
            for y in 0..r.height {
                for x in 0..r.width {
                    if r.px(x, y)[0] == 255 {
                        (fx, fy) = (x, y);
                    }
                }
            }
            // centre of the marked display pixel maps back to stored centre
            let (sx, sy) = unrotate_point(deg, 4.0, 2.0, fx as f32 + 0.5, fy as f32 + 0.5);
            assert_eq!((sx, sy), (3.5, 0.5), "deg {deg}");
        }
    }
}
