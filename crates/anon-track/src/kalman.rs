//! Constant-velocity Kalman filter for a head box (cx, cy, w, h), one
//! independent 2-state filter per axis. Noise scales with box height as in
//! SORT/ByteTrack (position σ = h/20, velocity σ = h/160).

use anon_model::BBox;

const STD_POS: f32 = 1.0 / 20.0;
const STD_VEL: f32 = 1.0 / 160.0;

#[derive(Debug, Clone, Copy)]
struct Axis {
    x: f32,
    v: f32,
    // covariance [[p00, p01], [p01, p11]]
    p00: f32,
    p01: f32,
    p11: f32,
}

impl Axis {
    fn new(x: f32, sp: f32, sv: f32) -> Self {
        Self {
            x,
            v: 0.0,
            p00: (2.0 * sp).powi(2),
            p01: 0.0,
            p11: (10.0 * sv).powi(2),
        }
    }

    fn predict(&mut self, sp: f32, sv: f32) {
        self.x += self.v;
        // P = F P Fᵀ + Q, F = [[1,1],[0,1]]
        let p00 = self.p00 + 2.0 * self.p01 + self.p11 + sp * sp;
        let p01 = self.p01 + self.p11;
        let p11 = self.p11 + sv * sv;
        self.p00 = p00;
        self.p01 = p01;
        self.p11 = p11;
    }

    fn update(&mut self, z: f32, sm: f32) {
        let s = self.p00 + sm * sm;
        let k0 = self.p00 / s;
        let k1 = self.p01 / s;
        let y = z - self.x;
        self.x += k0 * y;
        self.v += k1 * y;
        let p00 = (1.0 - k0) * self.p00;
        let p01 = (1.0 - k0) * self.p01;
        let p11 = self.p11 - k1 * self.p01;
        self.p00 = p00;
        self.p01 = p01;
        self.p11 = p11;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoxFilter {
    axes: [Axis; 4],
}

impl BoxFilter {
    pub fn new(b: &BBox) -> Self {
        let (cx, cy) = b.center();
        let h = b.height().max(1.0);
        let (sp, sv) = (STD_POS * h, STD_VEL * h);
        Self {
            axes: [
                Axis::new(cx, sp, sv),
                Axis::new(cy, sp, sv),
                Axis::new(b.width(), sp, sv),
                Axis::new(b.height(), sp, sv),
            ],
        }
    }

    fn h(&self) -> f32 {
        self.axes[3].x.max(1.0)
    }

    pub fn predict(&mut self) {
        let h = self.h();
        for a in &mut self.axes {
            a.predict(STD_POS * h, STD_VEL * h);
        }
        // size can not go negative
        self.axes[2].x = self.axes[2].x.max(1.0);
        self.axes[3].x = self.axes[3].x.max(1.0);
    }

    pub fn update(&mut self, b: &BBox) {
        let h = self.h();
        let (cx, cy) = b.center();
        for (a, z) in self.axes.iter_mut().zip([cx, cy, b.width(), b.height()]) {
            a.update(z, STD_POS * h);
        }
    }

    pub fn bbox(&self) -> BBox {
        let [cx, cy, w, h] = self.axes.map(|a| a.x);
        BBox::new(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
    }

    /// Centre velocity in pixels per frame.
    pub fn velocity(&self) -> (f32, f32) {
        (self.axes[0].v, self.axes[1].v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_constant_velocity() {
        let mut f = BoxFilter::new(&BBox::new(0.0, 0.0, 10.0, 10.0));
        for i in 1..30 {
            f.predict();
            let x = i as f32 * 3.0;
            f.update(&BBox::new(x, 0.0, x + 10.0, 10.0));
        }
        let (vx, vy) = f.velocity();
        assert!((vx - 3.0).abs() < 0.2, "vx {vx}");
        assert!(vy.abs() < 0.1);
        f.predict();
        let (cx, _) = f.bbox().center();
        assert!((cx - (30.0 * 3.0 + 5.0)).abs() < 1.0, "cx {cx}");
    }
}
