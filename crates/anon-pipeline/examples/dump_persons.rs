//! Debug view of the fused person instances of one frame: every mask in its
//! own colour, its box in white and the derived head box in black.
//!
//!     cargo run --release -p anon-pipeline --example dump_persons -- \
//!         video.mp4 model.onnx FRAME out.ppm [coarse_below]

use std::io::Write;
use std::path::Path;

use anon_detect::analyzer::{Analyzer, AnalyzerConfig};
use anon_detect::head::derive_heads;
use anon_io::{FrameReader, RgbConverter};
use anon_model::{BBox, TileSettings};

const COLOURS: [[u8; 3]; 8] = [
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
];

fn rect(img: &mut [u8], w: u32, h: u32, b: &BBox, c: [u8; 3]) {
    let (x0, y0) = (b.x0.max(0.0) as u32, b.y0.max(0.0) as u32);
    let (x1, y1) = ((b.x1 as u32).min(w - 1), (b.y1 as u32).min(h - 1));
    for x in x0..=x1 {
        for y in [y0, y1] {
            img[((y * w + x) * 3) as usize..][..3].copy_from_slice(&c);
        }
    }
    for y in y0..=y1 {
        for x in [x0, x1] {
            img[((y * w + x) * 3) as usize..][..3].copy_from_slice(&c);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let (video, model, frame, out) = (&a[1], &a[2], a[3].parse::<u64>()?, &a[4]);
    let coarse_below = a.get(5).map(|s| s.parse::<f32>()).transpose()?;
    anon_io::init()?;
    let mut reader = FrameReader::open(Path::new(video))?;
    let mut n = 0;
    let f = loop {
        let (f, _) = reader.next_frame()?.ok_or("frame out of range")?;
        if n == frame {
            break f;
        }
        n += 1;
    };
    let img = RgbConverter::default().convert(&f)?;
    let (m, _) = anon_detect::rfdetr::load_verified(Path::new(model))?;
    let mut an = Analyzer::new(
        Box::new(m),
        AnalyzerConfig {
            tile: TileSettings {
                coarse_below,
                ..TileSettings::default()
            },
            person_threshold: 0.15,
            flag_below: 0.35,
            head_expand: 1.5,
        },
    );
    let (persons, raw) = an.persons(&img)?;
    eprintln!("{raw} raw detections, {} persons", persons.len());
    let (w, h) = (img.width, img.height);
    let mut px = img.data.clone();
    for (i, p) in persons.iter().enumerate() {
        let c = COLOURS[i % COLOURS.len()];
        let heads: Vec<BBox> = derive_heads(p).iter().map(|h| h.bbox).collect();
        let head = heads[0];
        let masked = p
            .mask
            .as_ref()
            .map_or(0, |m| m.bits.iter().filter(|&&b| b).count());
        eprintln!(
            "#{i:<3} score {:.2} box {:>4.0},{:>4.0} {:>4.0}x{:<4.0} head {:>4.0}x{:<4.0} ({} heads) mask {:?} on {masked} passes {:?}",
            p.score,
            p.bbox.x0,
            p.bbox.y0,
            p.bbox.width(),
            p.bbox.height(),
            head.width(),
            head.height(),
            heads.len(),
            p.mask.as_ref().map(|m| (m.w, m.h)),
            p.passes
        );
        if let Some(m) = &p.mask {
            for my in 0..m.h {
                for mx in 0..m.w {
                    if !m.get(mx, my) {
                        continue;
                    }
                    let x0 = p.bbox.x0 + mx as f32 * p.bbox.width() / m.w as f32;
                    let x1 = p.bbox.x0 + (mx + 1) as f32 * p.bbox.width() / m.w as f32;
                    let y0 = p.bbox.y0 + my as f32 * p.bbox.height() / m.h as f32;
                    let y1 = p.bbox.y0 + (my + 1) as f32 * p.bbox.height() / m.h as f32;
                    for y in (y0.max(0.0) as u32)..(y1 as u32).min(h) {
                        for x in (x0.max(0.0) as u32)..(x1 as u32).min(w) {
                            let k = ((y * w + x) * 3) as usize;
                            for ch in 0..3 {
                                px[k + ch] = ((u16::from(px[k + ch]) + u16::from(c[ch])) / 2) as u8;
                            }
                        }
                    }
                }
            }
        }
        rect(&mut px, w, h, &p.bbox, [255, 255, 255]);
        for hb in &heads {
            rect(&mut px, w, h, hb, [0, 0, 0]);
        }
    }
    let mut o = std::fs::File::create(out)?;
    write!(o, "P6\n{w} {h}\n255\n")?;
    o.write_all(&px)?;
    Ok(())
}
