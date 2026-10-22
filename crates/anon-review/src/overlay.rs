//! RGBA frame buffer and overlay drawing for the review view.

use anon_detect::RgbImage;
use anon_model::Shape;
use anon_render::Coverage;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rgba {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Rgba {
    pub fn from_rgb(img: &RgbImage) -> Self {
        let mut data = Vec::with_capacity((img.width * img.height * 4) as usize);
        for px in img.data.chunks_exact(3) {
            data.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
        Self {
            width: img.width,
            height: img.height,
            data,
        }
    }

    pub fn filled(width: u32, height: u32, c: [u8; 4]) -> Self {
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..width * height {
            data.extend_from_slice(&c);
        }
        Self {
            width,
            height,
            data,
        }
    }

    #[inline]
    fn blend(&mut self, x: u32, y: u32, c: [u8; 4]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let i = ((y * self.width + x) * 4) as usize;
        let a = u16::from(c[3]);
        for (dst, &src) in self.data[i..i + 3].iter_mut().zip(&c[..3]) {
            *dst = ((u16::from(src) * a + u16::from(*dst) * (255 - a)) / 255) as u8;
        }
    }

    pub fn px(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        [
            self.data[i],
            self.data[i + 1],
            self.data[i + 2],
            self.data[i + 3],
        ]
    }

    /// Paint coverage spans with `c` (alpha-blended).
    pub fn fill(&mut self, cov: &Coverage, c: [u8; 4]) {
        for y in 0..cov.height.min(self.height) {
            for (a, b) in cov.row(y) {
                for x in a..b.min(self.width) {
                    self.blend(x, y, c);
                }
            }
        }
    }

    pub fn fill_shape(&mut self, s: &Shape, c: [u8; 4]) {
        let mut cov = Coverage::new(self.width, self.height);
        cov.add_shape(s);
        self.fill(&cov, c);
    }

    fn stamp(&mut self, x: f32, y: f32, t: u32, c: [u8; 4]) {
        let r = t as i64 / 2;
        let (cx, cy) = (x.round() as i64, y.round() as i64);
        for yy in cy - r..=cy + r {
            for xx in cx - r..=cx + r {
                if xx >= 0 && yy >= 0 {
                    self.blend(xx as u32, yy as u32, c);
                }
            }
        }
    }

    fn line(&mut self, a: [f32; 2], b: [f32; 2], t: u32, c: [u8; 4], dashed: bool) {
        let len = ((b[0] - a[0]).hypot(b[1] - a[1])).max(1.0);
        let steps = len.ceil() as u32;
        for i in 0..=steps {
            if dashed && (i / 6) % 2 == 1 {
                continue;
            }
            let f = i as f32 / steps as f32;
            self.stamp(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, t, c);
        }
    }

    /// Outline a shape with thickness `t` pixels.
    pub fn outline(&mut self, s: &Shape, t: u32, c: [u8; 4], dashed: bool) {
        match s {
            Shape::Ellipse { cx, cy, rx, ry } => {
                let n = ((rx + ry) * std::f32::consts::PI)
                    .ceil()
                    .clamp(16.0, 4096.0) as usize;
                let pts: Vec<[f32; 2]> = (0..=n)
                    .map(|i| {
                        let a = i as f32 / n as f32 * std::f32::consts::TAU;
                        [cx + rx * a.cos(), cy + ry * a.sin()]
                    })
                    .collect();
                for (i, w) in pts.windows(2).enumerate() {
                    if dashed && (i / 4) % 2 == 1 {
                        continue;
                    }
                    self.line(w[0], w[1], t, c, false);
                }
            }
            Shape::Polygon { points } => {
                for i in 0..points.len() {
                    self.line(points[i], points[(i + 1) % points.len()], t, c, dashed);
                }
            }
        }
    }

    pub fn rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, t: u32, c: [u8; 4]) {
        self.outline(
            &Shape::Polygon {
                points: vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            },
            t,
            c,
            false,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outline_hits_ellipse_edge_not_centre() {
        let mut img = Rgba::filled(100, 100, [0, 0, 0, 255]);
        let e = Shape::Ellipse {
            cx: 50.0,
            cy: 50.0,
            rx: 20.0,
            ry: 10.0,
        };
        img.outline(&e, 2, [255, 0, 0, 255], false);
        assert_eq!(img.px(70, 50)[0], 255);
        assert_eq!(img.px(50, 50)[0], 0);
        img.fill_shape(&e, [0, 255, 0, 128]);
        assert!(img.px(50, 50)[1] > 100);
    }
}
