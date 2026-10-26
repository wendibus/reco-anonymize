//! Multi-pass analysis of one frame: tiles (native resolution), optional
//! whole-frame pass, optional flipped copies, fusion, head derivation.

use anon_model::{BBox, Shape, TileSettings};

use crate::fuse::fuse;
use crate::fuse::resample_mask;
use crate::head::derive_heads;
use crate::image::RgbImage;
use crate::tiling::{TileRect, plan_perspective};
use crate::{DetectError, InstanceMask, PersonInstance, PersonSegmenter};

/// Source pixels within which a box counts as touching a tile edge.
const EDGE_EPS: f32 = 2.0;

#[derive(Debug, Clone)]
pub struct AnalyzerConfig {
    pub tile: TileSettings,
    pub person_threshold: f32,
    /// Scores below this (but above `person_threshold`) are flagged.
    pub flag_below: f32,
    pub head_expand: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HeadRegion {
    pub head: BBox,
    pub person: BBox,
    pub shape: Shape,
    pub score: f32,
    pub low_score: bool,
    pub arms_up: bool,
    pub passes: Vec<&'static str>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FrameDetections {
    pub heads: Vec<HeadRegion>,
    /// Number of raw (pre-fusion) detections above threshold, for diagnostics.
    pub raw_count: usize,
}

pub struct Analyzer {
    model: Box<dyn PersonSegmenter>,
    cfg: AnalyzerConfig,
}

fn flip_instance(p: &mut PersonInstance, s: f32) {
    p.bbox = BBox::new(s - p.bbox.x1, p.bbox.y0, s - p.bbox.x0, p.bbox.y1);
    if let Some(m) = &mut p.mask {
        let mut bits = vec![false; m.bits.len()];
        for y in 0..m.h {
            for x in 0..m.w {
                bits[(y * m.w + (m.w - 1 - x)) as usize] = m.get(x, y);
            }
        }
        *m = InstanceMask {
            w: m.w,
            h: m.h,
            bits,
        };
    }
}

impl Analyzer {
    pub fn new(model: Box<dyn PersonSegmenter>, cfg: AnalyzerConfig) -> Self {
        Self { model, cfg }
    }

    pub fn config(&self) -> &AnalyzerConfig {
        &self.cfg
    }

    /// Run the model on the region `r` of `img`, scaled by `k` into the
    /// model's square input, once plain and optionally once flipped.
    /// Results are mapped back to source coordinates.
    fn run_region(
        &mut self,
        img: &RgbImage,
        r: TileRect,
        k: f32,
        name: &'static str,
        edges: [bool; 4],
        out: &mut Vec<PersonInstance>,
    ) -> Result<(), DetectError> {
        let s = self.model.input_size();
        let tile = if (k - 1.0).abs() < 1e-6 {
            img.crop_padded(r.x, r.y, r.w, r.h, s)
        } else {
            let region = img.crop_padded(r.x, r.y, r.w, r.h, r.w.max(r.h));
            let side = ((r.w.max(r.h) as f32) * k).round().max(1.0) as u32;
            region.resize(side, side).crop_padded(0, 0, side, side, s)
        };
        // Content extent inside the tile (excludes padding).
        let cw = (r.w as f32 * k).min(s as f32);
        let ch = (r.h as f32 * k).min(s as f32);

        let mut variants: Vec<(RgbImage, bool, &'static str)> = vec![(tile, false, name)];
        if self.cfg.tile.flip_pass {
            let f = variants[0].0.flip_horizontal();
            variants.push((
                f,
                true,
                if name == "tile" {
                    "tile_flip"
                } else {
                    "full_flip"
                },
            ));
        }
        for (t, flipped, pass) in variants {
            for mut p in self.model.segment(&t)? {
                if p.score < self.cfg.person_threshold {
                    continue;
                }
                if flipped {
                    flip_instance(&mut p, s as f32);
                }
                // Clip to real content, drop anything living in padding.
                let b = BBox::new(
                    p.bbox.x0.max(0.0),
                    p.bbox.y0.max(0.0),
                    p.bbox.x1.min(cw),
                    p.bbox.y1.min(ch),
                );
                if b.area() <= 0.0 {
                    continue;
                }
                if b != p.bbox {
                    // keep the part of the mask inside the clipped box
                    p.mask = p.mask.take().and_then(|m| resample_mask(&p.bbox, &m, &b));
                }
                let src = BBox::new(
                    r.x as f32 + b.x0 / k,
                    r.y as f32 + b.y0 / k,
                    r.x as f32 + b.x1 / k,
                    r.y as f32 + b.y1 / k,
                );
                let truncated = (edges[0] && b.x0 <= EDGE_EPS * k)
                    || (edges[1] && b.y0 <= EDGE_EPS * k)
                    || (edges[2] && b.x1 >= cw - EDGE_EPS * k)
                    || (edges[3] && b.y1 >= ch - EDGE_EPS * k);
                out.push(PersonInstance {
                    bbox: src,
                    score: p.score,
                    mask: p.mask,
                    truncated,
                    passes: vec![pass],
                });
            }
        }
        Ok(())
    }

    pub fn analyze(&mut self, img: &RgbImage) -> Result<FrameDetections, DetectError> {
        let (persons, raw_count) = self.persons(img)?;
        let heads = persons
            .iter()
            .flat_map(|p| {
                derive_heads(p).into_iter().map(|h| HeadRegion {
                    shape: Shape::ellipse_around(&h.bbox, self.cfg.head_expand),
                    head: h.bbox,
                    person: p.bbox,
                    score: p.score,
                    low_score: p.score < self.cfg.flag_below,
                    arms_up: h.arms_up,
                    passes: p.passes.clone(),
                })
            })
            .collect();
        Ok(FrameDetections { heads, raw_count })
    }

    /// All passes over `img`, fused into person instances (source
    /// coordinates), plus the number of raw detections before fusion.
    pub fn persons(&mut self, img: &RgbImage) -> Result<(Vec<PersonInstance>, usize), DetectError> {
        let s = self.model.input_size();
        let tile = if self.cfg.tile.tile == 0 {
            s
        } else {
            self.cfg.tile.tile
        };
        let coarse = if self.cfg.tile.coarse_tile == 0 {
            tile * 2
        } else {
            self.cfg.tile.coarse_tile
        };
        let split = self
            .cfg
            .tile
            .coarse_below
            .map(|f| (f.clamp(0.0, 1.0) * img.height as f32).round() as u32);
        let mut raw = Vec::new();

        for t in plan_perspective(
            img.width,
            img.height,
            tile,
            coarse,
            split,
            self.cfg.tile.overlap,
        ) {
            let edges = t.rect.internal_edges(img.width, img.height);
            let k = s as f32 / t.tile as f32;
            self.run_region(img, t.rect, k, "tile", edges, &mut raw)?;
        }
        let needs_full = img.width > tile || img.height > tile;
        if self.cfg.tile.full_frame_pass && needs_full {
            let r = TileRect {
                x: 0,
                y: 0,
                w: img.width,
                h: img.height,
            };
            let kf = s as f32 / img.width.max(img.height) as f32;
            self.run_region(img, r, kf, "full", [false; 4], &mut raw)?;
        }

        let raw_count = raw.len();
        Ok((fuse(raw), raw_count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fake model: reports a person wherever a pure-red pixel block exists,
    /// box = bounding box of red pixels in the tile.
    struct RedFinder(u32);

    impl PersonSegmenter for RedFinder {
        fn input_size(&self) -> u32 {
            self.0
        }
        fn segment(&mut self, t: &RgbImage) -> Result<Vec<PersonInstance>, DetectError> {
            let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
            for y in 0..t.height {
                for x in 0..t.width {
                    if t.px(x, y) == [255, 0, 0] {
                        x0 = x0.min(x);
                        y0 = y0.min(y);
                        x1 = x1.max(x + 1);
                        y1 = y1.max(y + 1);
                    }
                }
            }
            if x0 == u32::MAX {
                return Ok(vec![]);
            }
            Ok(vec![PersonInstance {
                bbox: BBox::new(x0 as f32, y0 as f32, x1 as f32, y1 as f32),
                score: 0.9,
                mask: None,
                truncated: false,
                passes: vec![],
            }])
        }
    }

    fn cfg() -> AnalyzerConfig {
        AnalyzerConfig {
            tile: TileSettings::default(),
            person_threshold: 0.15,
            flag_below: 0.35,
            head_expand: 1.5,
        }
    }

    #[test]
    fn finds_small_person_and_maps_coordinates_back() {
        let mut img = RgbImage::new(400, 120);
        for y in 50..90 {
            for x in 300..310 {
                let i = ((y * 400 + x) * 3) as usize;
                img.data[i] = 255;
            }
        }
        let mut a = Analyzer::new(Box::new(RedFinder(64)), cfg());
        let d = a.analyze(&img).unwrap();
        assert_eq!(d.heads.len(), 1, "{:?}", d.heads);
        let p = d.heads[0].person;
        // union with the coarse full-frame pass may only grow the box
        assert!(
            (p.x0 - 300.0).abs() < 1.5 && p.x1 >= 310.0 && p.x1 < 316.0,
            "{p:?}"
        );
        assert!(
            (p.y0 - 50.0).abs() < 1.5 && (p.y1 - 90.0).abs() < 1.5,
            "{p:?}"
        );
        // passes from plain + flipped + full frame merged
        assert!(d.heads[0].passes.contains(&"tile"));
        assert!(d.heads[0].passes.contains(&"tile_flip"));
        // head sits at top of person
        assert!(d.heads[0].head.y1 <= 50.0 + 40.0 * 0.25);
    }

    fn paint_red(img: &mut RgbImage, xs: std::ops::Range<u32>, ys: std::ops::Range<u32>) {
        for y in ys {
            for x in xs.clone() {
                let i = ((y * img.width + x) * 3) as usize;
                img.data[i] = 255;
            }
        }
    }

    #[test]
    fn perspective_tiling_finds_far_and_near_people() {
        let mut img = RgbImage::new(800, 240);
        paint_red(&mut img, 100..106, 30..42); // small, far away
        paint_red(&mut img, 500..540, 140..220); // large, near the camera
        let mut c = cfg();
        c.tile.flip_pass = false;
        c.tile.full_frame_pass = false;
        c.tile.coarse_below = Some(0.5);
        let mut a = Analyzer::new(Box::new(RedFinder(64)), c);
        let d = a.analyze(&img).unwrap();
        assert_eq!(d.heads.len(), 2, "{:?}", d.heads);
        let mut persons: Vec<BBox> = d.heads.iter().map(|h| h.person).collect();
        persons.sort_by(|a, b| a.x0.total_cmp(&b.x0));
        let (far, near) = (persons[0], persons[1]);
        assert!(
            (far.x0 - 100.0).abs() < 1.5 && (far.y0 - 30.0).abs() < 1.5,
            "{far:?}"
        );
        // coarse tiles are downscaled 2x: boxes are accurate to ~2 source px
        assert!(
            (near.x0 - 500.0).abs() < 3.0 && (near.y1 - 220.0).abs() < 3.0,
            "{near:?}"
        );
    }
}
