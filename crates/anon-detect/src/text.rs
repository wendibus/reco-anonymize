//! Text regions (jersey numbers, names on shirts) inside person boxes.
//!
//! The model contract matches PaddleOCR's DB text detector (PP-OCR det):
//! input `[1, 3, H, W]` (H, W multiples of 32), output a probability map
//! `[1, 1, H, W]`. Post-processing (threshold → connected components →
//! box → "unclip" expansion) is model-independent and lives here.

use anon_model::BBox;

use crate::DetectError;
use crate::image::RgbImage;

/// Probability above which a pixel counts as text.
pub const BIN_THRESH: f32 = 0.3;
/// Components smaller than this many map pixels are ignored.
pub const MIN_AREA: usize = 6;
/// DB "unclip" ratio: boxes grow by area * ratio / perimeter on each side.
pub const UNCLIP: f32 = 1.5;
/// Longest side a person crop is scaled to before detection.
pub const CROP_SIDE: u32 = 320;

/// A model producing a per-pixel text probability map.
pub trait TextMapModel: Send {
    /// Probability map, `w × h` row-major, for an RGB image whose sides are
    /// multiples of 32.
    fn prob_map(&mut self, img: &RgbImage) -> Result<Vec<f32>, DetectError>;
}

/// Finds text inside person boxes of a frame.
pub trait TextDetector: Send {
    /// Text boxes (source coordinates) inside the given person boxes.
    fn detect(&mut self, frame: &RgbImage, persons: &[BBox]) -> Result<Vec<TextBox>, DetectError>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextBox {
    pub bbox: BBox,
    pub score: f32,
}

/// Connected components (4-neighbourhood) of `map > thresh`, as boxes in
/// map pixels with their mean probability.
pub fn components(map: &[f32], w: usize, h: usize, thresh: f32) -> Vec<(BBox, f32, usize)> {
    let mut seen = vec![false; map.len()];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for start in 0..map.len() {
        if seen[start] || map[start] <= thresh {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        let (mut n, mut sum) = (0usize, 0.0f32);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            n += 1;
            sum += map[i];
            let mut push = |j: usize| {
                if !seen[j] && map[j] > thresh {
                    seen[j] = true;
                    stack.push(j);
                }
            };
            if x > 0 {
                push(i - 1);
            }
            if x + 1 < w {
                push(i + 1);
            }
            if y > 0 {
                push(i - w);
            }
            if y + 1 < h {
                push(i + w);
            }
        }
        out.push((
            BBox::new(x0 as f32, y0 as f32, (x1 + 1) as f32, (y1 + 1) as f32),
            sum / n as f32,
            n,
        ));
    }
    out
}

/// DB-style expansion of a shrunk text kernel back to the full text area.
pub fn unclip(b: &BBox, ratio: f32) -> BBox {
    let area = b.area();
    let perim = 2.0 * (b.width() + b.height());
    let d = if perim > 0.0 {
        area * ratio / perim
    } else {
        0.0
    };
    BBox::new(b.x0 - d, b.y0 - d, b.x1 + d, b.y1 + d)
}

/// Runs a [`TextMapModel`] on each person crop.
pub struct DbTextDetector<M: TextMapModel> {
    pub model: M,
}

impl<M: TextMapModel> TextDetector for DbTextDetector<M> {
    fn detect(&mut self, frame: &RgbImage, persons: &[BBox]) -> Result<Vec<TextBox>, DetectError> {
        let mut out = Vec::new();
        for p in persons {
            let x0 = p.x0.floor().max(0.0) as u32;
            let y0 = p.y0.floor().max(0.0) as u32;
            let x1 = (p.x1.ceil().max(0.0) as u32).min(frame.width);
            let y1 = (p.y1.ceil().max(0.0) as u32).min(frame.height);
            if x1 <= x0 + 4 || y1 <= y0 + 4 {
                continue;
            }
            let (cw, ch) = (x1 - x0, y1 - y0);
            let k = (CROP_SIDE as f32 / cw.max(ch) as f32).min(4.0);
            let (sw, sh) = (
                ((cw as f32 * k).round() as u32).max(1),
                ((ch as f32 * k).round() as u32).max(1),
            );
            let (pw, ph) = (sw.div_ceil(32) * 32, sh.div_ceil(32) * 32);
            let input = frame.crop(x0, y0, cw, ch).resize(sw, sh).pad_to(pw, ph, 0);
            let map = self.model.prob_map(&input)?;
            if map.len() != (input.width * input.height) as usize {
                return Err(DetectError::Output(format!(
                    "text map has {} values for a {}x{} input",
                    map.len(),
                    input.width,
                    input.height
                )));
            }
            for (b, score, n) in components(
                &map,
                input.width as usize,
                input.height as usize,
                BIN_THRESH,
            ) {
                if n < MIN_AREA || b.x0 >= sw as f32 || b.y0 >= sh as f32 {
                    continue; // noise or padding
                }
                let u = unclip(&b, UNCLIP);
                out.push(TextBox {
                    bbox: BBox::new(
                        (x0 as f32 + u.x0 / k).max(p.x0),
                        (y0 as f32 + u.y0 / k).max(p.y0),
                        (x0 as f32 + u.x1 / k).min(p.x1),
                        (y0 as f32 + u.y1 / k).min(p.y1),
                    ),
                    score,
                });
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Text = bright pixels.
    struct Bright;
    impl TextMapModel for Bright {
        fn prob_map(&mut self, img: &RgbImage) -> Result<Vec<f32>, DetectError> {
            Ok(img
                .data
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| {
                    if p.iter().all(|&c| c > 200) {
                        0.9
                    } else {
                        0.05
                    }
                })
                .collect())
        }
    }

    #[test]
    fn components_and_unclip() {
        let (w, h) = (10, 6);
        let mut m = vec![0.0; w * h];
        for y in 1..3 {
            for x in 2..6 {
                m[y * w + x] = 0.8;
            }
        }
        m[5 * w + 9] = 0.9;
        let c = components(&m, w, h, 0.3);
        assert_eq!(c.len(), 2);
        let (b, s, n) = c[0];
        assert_eq!((b, n), (BBox::new(2.0, 1.0, 6.0, 3.0), 8));
        assert!((s - 0.8).abs() < 1e-6);
        let u = unclip(&b, 1.5);
        // area 8, perimeter 12 -> grows by 1 on each side
        assert_eq!(u, BBox::new(1.0, 0.0, 7.0, 4.0));
    }

    #[test]
    fn finds_number_on_shirt_in_source_coordinates() {
        // 400x300 dark frame, person box 100..160 x 50..230, white number at 115..145 x 110..150
        let mut img = RgbImage::from_raw(400, 300, vec![30; 400 * 300 * 3]);
        for y in 110..150 {
            for x in 115..145 {
                let i = ((y * 400 + x) * 3) as usize;
                img.data[i..i + 3].copy_from_slice(&[255, 255, 255]);
            }
        }
        // a bright spot outside the person must be ignored
        for y in 10..20 {
            for x in 300..320 {
                let i = ((y * 400 + x) * 3) as usize;
                img.data[i..i + 3].copy_from_slice(&[255, 255, 255]);
            }
        }
        let mut d = DbTextDetector { model: Bright };
        let out = d
            .detect(&img, &[BBox::new(100.0, 50.0, 160.0, 230.0)])
            .unwrap();
        assert_eq!(out.len(), 1, "{out:?}");
        let b = out[0].bbox;
        assert!(
            b.x0 <= 115.0 && b.y0 <= 110.0 && b.x1 >= 145.0 && b.y1 >= 150.0,
            "{b:?}"
        );
        assert!(
            b.x0 >= 100.0 && b.x1 <= 160.0,
            "clipped to the person: {b:?}"
        );
    }
}
