//! Geometric primitives in source-pixel coordinates.

use serde::{Deserialize, Serialize};

/// Axis-aligned box, `x0 <= x1`, `y0 <= y1`, source pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl BBox {
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self {
            x0: x0.min(x1),
            y0: y0.min(y1),
            x1: x0.max(x1),
            y1: y0.max(y1),
        }
    }

    pub fn width(&self) -> f32 {
        (self.x1 - self.x0).max(0.0)
    }

    pub fn height(&self) -> f32 {
        (self.y1 - self.y0).max(0.0)
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    pub fn center(&self) -> (f32, f32) {
        ((self.x0 + self.x1) * 0.5, (self.y0 + self.y1) * 0.5)
    }

    pub fn intersection(&self, o: &BBox) -> f32 {
        let w = (self.x1.min(o.x1) - self.x0.max(o.x0)).max(0.0);
        let h = (self.y1.min(o.y1) - self.y0.max(o.y0)).max(0.0);
        w * h
    }

    pub fn iou(&self, o: &BBox) -> f32 {
        let inter = self.intersection(o);
        let union = self.area() + o.area() - inter;
        if union <= 0.0 { 0.0 } else { inter / union }
    }

    /// Intersection over the smaller box. Used for fusing tile duplicates,
    /// where a person cut at a tile border yields a box contained in the
    /// full one.
    pub fn ios(&self, o: &BBox) -> f32 {
        let inter = self.intersection(o);
        let min = self.area().min(o.area());
        if min <= 0.0 { 0.0 } else { inter / min }
    }

    pub fn union_box(&self, o: &BBox) -> BBox {
        BBox::new(
            self.x0.min(o.x0),
            self.y0.min(o.y0),
            self.x1.max(o.x1),
            self.y1.max(o.y1),
        )
    }

    pub fn translate(&self, dx: f32, dy: f32) -> BBox {
        BBox::new(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)
    }

    pub fn scale(&self, s: f32) -> BBox {
        BBox::new(self.x0 * s, self.y0 * s, self.x1 * s, self.y1 * s)
    }
}

/// A region to make opaque.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Shape {
    /// Axis-aligned ellipse (heads).
    Ellipse { cx: f32, cy: f32, rx: f32, ry: f32 },
    /// Closed polygon (text regions, static zones).
    Polygon { points: Vec<[f32; 2]> },
}

impl Shape {
    /// Ellipse inscribed in `b`, scaled by `expand` around the centre.
    pub fn ellipse_around(b: &BBox, expand: f32) -> Shape {
        let (cx, cy) = b.center();
        // An ellipse inscribed in a box misses the corners; multiply by
        // sqrt(2) so the ellipse *circumscribes* the box before expansion.
        let k = std::f32::consts::SQRT_2 * 0.5 * expand;
        Shape::Ellipse {
            cx,
            cy,
            rx: b.width() * k,
            ry: b.height() * k,
        }
    }

    pub fn bounds(&self) -> BBox {
        match self {
            Shape::Ellipse { cx, cy, rx, ry } => BBox::new(cx - rx, cy - ry, cx + rx, cy + ry),
            Shape::Polygon { points } => {
                let mut b = BBox::new(f32::MAX, f32::MAX, f32::MIN, f32::MIN);
                for p in points {
                    b.x0 = b.x0.min(p[0]);
                    b.y0 = b.y0.min(p[1]);
                    b.x1 = b.x1.max(p[0]);
                    b.y1 = b.y1.max(p[1]);
                }
                b
            }
        }
    }

    /// Point-in-shape test (pixel centres). Polygons use even-odd rule.
    pub fn contains(&self, x: f32, y: f32) -> bool {
        match self {
            Shape::Ellipse { cx, cy, rx, ry } => {
                if *rx <= 0.0 || *ry <= 0.0 {
                    return false;
                }
                let dx = (x - cx) / rx;
                let dy = (y - cy) / ry;
                dx * dx + dy * dy <= 1.0
            }
            Shape::Polygon { points } => {
                let n = points.len();
                if n < 3 {
                    return false;
                }
                let mut inside = false;
                let mut j = n - 1;
                for i in 0..n {
                    let (xi, yi) = (points[i][0], points[i][1]);
                    let (xj, yj) = (points[j][0], points[j][1]);
                    if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
                        inside = !inside;
                    }
                    j = i;
                }
                inside
            }
        }
    }

    /// Linear interpolation between two shapes of the same kind. Polygons
    /// with differing vertex counts return `a` (held).
    pub fn lerp(a: &Shape, b: &Shape, t: f32) -> Shape {
        match (a, b) {
            (
                Shape::Ellipse {
                    cx: ax,
                    cy: ay,
                    rx: arx,
                    ry: ary,
                },
                Shape::Ellipse {
                    cx: bx,
                    cy: by,
                    rx: brx,
                    ry: bry,
                },
            ) => Shape::Ellipse {
                cx: ax + (bx - ax) * t,
                cy: ay + (by - ay) * t,
                // Never shrink below either endpoint while interpolating:
                // fail towards covering more.
                rx: arx.max(*brx),
                ry: ary.max(*bry),
            },
            (Shape::Polygon { points: pa }, Shape::Polygon { points: pb })
                if pa.len() == pb.len() =>
            {
                Shape::Polygon {
                    points: pa
                        .iter()
                        .zip(pb)
                        .map(|(p, q)| [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t])
                        .collect(),
                }
            }
            _ => a.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_and_ios() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(5.0, 0.0, 15.0, 10.0);
        assert!((a.iou(&b) - 50.0 / 150.0).abs() < 1e-6);
        let inner = BBox::new(2.0, 2.0, 4.0, 4.0);
        assert!((a.ios(&inner) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ellipse_covers_box_corners() {
        let b = BBox::new(10.0, 20.0, 30.0, 60.0);
        let e = Shape::ellipse_around(&b, 1.0);
        for (x, y) in [(10.0, 20.0), (30.0, 20.0), (10.0, 60.0), (30.0, 60.0)] {
            // corners lie on the boundary; nudge inwards by epsilon
            let (cx, cy) = b.center();
            assert!(e.contains(x + (cx - x) * 1e-3, y + (cy - y) * 1e-3));
        }
    }

    #[test]
    fn polygon_contains() {
        let p = Shape::Polygon {
            points: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        };
        assert!(p.contains(5.0, 5.0));
        assert!(!p.contains(11.0, 5.0));
    }

    #[test]
    fn lerp_never_shrinks() {
        let a = Shape::Ellipse {
            cx: 0.0,
            cy: 0.0,
            rx: 10.0,
            ry: 5.0,
        };
        let b = Shape::Ellipse {
            cx: 10.0,
            cy: 0.0,
            rx: 4.0,
            ry: 8.0,
        };
        let Shape::Ellipse { cx, rx, ry, .. } = Shape::lerp(&a, &b, 0.5) else {
            panic!()
        };
        assert_eq!(cx, 5.0);
        assert_eq!((rx, ry), (10.0, 8.0));
    }
}
