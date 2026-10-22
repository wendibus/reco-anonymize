//! Full workflow on a synthetic clip with a moving "person" (red box), using
//! a colour-threshold stand-in for the model: analyze → gate → approve →
//! render → verify. Skipped without ffmpeg/ffprobe.

use std::path::{Path, PathBuf};
use std::process::Command;

use anon_detect::{DetectError, PersonInstance, PersonSegmenter, RgbImage};
use anon_io::{Codec, EncodeSettings};
use anon_model::{AnalysisSettings, BBox, ModelRef, ProjectDir, Suppression, TileSettings, Zone};
use anon_pipeline::{
    AnalyzeOptions, PipelineError, RenderOptions, acknowledge, analyze, approve, audit, render,
    status,
};

struct RedFinder;

impl PersonSegmenter for RedFinder {
    fn input_size(&self) -> u32 {
        128
    }
    fn segment(&mut self, t: &RgbImage) -> Result<Vec<PersonInstance>, DetectError> {
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
        for y in 0..t.height {
            for x in 0..t.width {
                let [r, g, b] = t.px(x, y);
                if r > 170 && g < 90 && b < 90 {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x + 1);
                    y1 = y1.max(y + 1);
                }
            }
        }
        if x0 == u32::MAX || (x1 - x0) * (y1 - y0) < 20 {
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

fn have(bin: &str) -> bool {
    Command::new(bin)
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn rgb_at(path: &Path, x: u32, y: u32, frame: u32) -> [u8; 3] {
    let o = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-vf",
            &format!("select=eq(n\\,{frame}),format=rgb24,crop=1:1:{x}:{y}"),
            "-frames:v",
            "1",
        ])
        .args(["-f", "rawvideo", "-"])
        .output()
        .unwrap();
    [o.stdout[0], o.stdout[1], o.stdout[2]]
}

fn opts() -> AnalyzeOptions {
    AnalyzeOptions {
        settings: AnalysisSettings {
            tile: TileSettings {
                tile: 0,
                overlap: 0.25,
                full_frame_pass: true,
                flip_pass: true,
            },
            ..AnalysisSettings::default()
        },
        model_ref: ModelRef {
            id: "red-finder".into(),
            sha256: "0".into(),
            license: "test".into(),
        },
        fresh: false,
        resume: false,
        text: None,
    }
}

#[test]
fn full_workflow_masks_heads_and_enforces_gate() {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("skipped: ffmpeg/ffprobe not installed");
        return;
    }
    let dir: PathBuf = std::env::temp_dir().join(format!("anon-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("game.mp4");
    // grey court, a red 20x80 "player" walking right at 60 px/s
    let st = Command::new("ffmpeg")
        .args(["-v", "error", "-y"])
        .args([
            "-f",
            "lavfi",
            "-i",
            "color=c=0x606060:size=320x240:rate=25:duration=2",
        ])
        .args(["-f", "lavfi", "-i", "sine=f=440:duration=2"])
        .args([
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=20x80:rate=25:duration=2",
        ])
        .args([
            "-filter_complex",
            "[0:v][2:v]overlay=x='20+t*60':y=80[v]",
            "-map",
            "[v]",
            "-map",
            "1:a",
        ])
        .args([
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
        ])
        .arg(&src)
        .status()
        .unwrap();
    assert!(st.success());

    let proj = ProjectDir::for_source(&src);
    let s = analyze(&src, &proj, Box::new(RedFinder), opts(), |_, _| {}).unwrap();
    assert_eq!(s.frames, 50);
    assert_eq!(s.errors, 0);
    assert_eq!(s.max_persons, 1);
    assert_eq!(s.tracks, 1, "one linked track for one walking player");

    // gate: no approval, no render
    let out = dir.join("anon.mp4");
    let ro = RenderOptions {
        encode: EncodeSettings {
            codec: Codec::H264,
            encoder: Some("libx264".into()),
            quality: 80,
        },
        preview: false,
    };
    assert!(matches!(
        render(&proj, &out, &ro),
        Err(PipelineError::Refused(_))
    ));

    // preview is allowed and lands inside the project only
    let pv = render(
        &proj,
        &dir.join("ignored.mp4"),
        &RenderOptions {
            preview: true,
            ..ro_clone(&ro)
        },
    )
    .unwrap();
    assert!(pv.output.starts_with(&proj.root));
    assert!(!dir.join("ignored.mp4").exists());

    approve(&proj, Some("test".into()), false).unwrap();
    assert!(status(&proj, 5).unwrap().approval_valid);

    // edits after approval void it
    let mut p = proj.load_project().unwrap();
    p.zones.push(Zone {
        id: 1,
        label: "scoreboard".into(),
        polygon: vec![[250.0, 10.0], [310.0, 10.0], [310.0, 40.0], [250.0, 40.0]],
        range: None,
    });
    proj.save_project(&p).unwrap();
    assert!(!status(&proj, 5).unwrap().approval_valid);
    assert!(matches!(
        render(&proj, &out, &ro),
        Err(PipelineError::Refused(_))
    ));
    approve(&proj, None, false).unwrap();

    let r = render(&proj, &out, &ro).unwrap();
    assert_eq!(r.frames, 50);
    assert_eq!(r.fully_opaque_frames, 0);
    assert_eq!(r.audio_streams, 1);
    assert!(out.exists());

    for frame in [0u32, 24, 49] {
        let t = frame as f32 / 25.0;
        let x = (20.0 + t * 60.0) as u32 + 10;
        let head = rgb_at(&out, x, 84, frame);
        assert!(
            head.iter().all(|&c| c < 40),
            "frame {frame}: head not masked {head:?}"
        );
        let legs = rgb_at(&out, x, 150, frame);
        assert!(
            legs[0] > 170 && legs[1] < 90,
            "frame {frame}: body should stay visible {legs:?}"
        );
    }
    let board = rgb_at(&out, 280, 25, 10);
    assert!(board.iter().all(|&c| c < 40), "zone not masked {board:?}");

    // audit the clean export: no hits
    let rep = audit(&proj, &out, Box::new(RedFinder), "test", |_| {}).unwrap();
    assert_eq!(rep.frames, 50);
    assert!(rep.passed(), "unexpected hits: {:?}", rep.hits);
    assert!(status(&proj, 0).unwrap().export_clean);

    // simulate a missed head: suppress the track, approve, render, audit
    let mut ts = proj.load_tracks().unwrap();
    for t in &mut ts.tracks {
        t.suppressed = Some(Suppression {
            reason: "test".into(),
            at: 0,
        });
    }
    proj.save_tracks(&ts).unwrap();
    approve(&proj, None, false).unwrap();
    let leaky = dir.join("leaky.mp4");
    render(&proj, &leaky, &ro).unwrap();
    assert!(!status(&proj, 0).unwrap().export_clean, "not audited yet");
    let rep = audit(&proj, &leaky, Box::new(RedFinder), "test", |_| {}).unwrap();
    assert_eq!(rep.open_hits(), 50, "every frame leaks a head");
    assert!(rep.hits.iter().all(|h| h.coverage < 0.2));
    let st = status(&proj, 0).unwrap();
    assert!(!st.export_clean && st.export_note.contains("50 open"));
    assert!(acknowledge(&proj, None, " ").is_err());
    assert_eq!(acknowledge(&proj, Some(&[0, 1]), "test only").unwrap(), 2);
    assert!(!status(&proj, 0).unwrap().export_clean);

    // source changed after analysis -> refused
    std::fs::write(&src, b"tampered").unwrap();
    assert!(render(&proj, &dir.join("x.mp4"), &ro).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

fn ro_clone(r: &RenderOptions) -> RenderOptions {
    RenderOptions {
        encode: r.encode.clone(),
        preview: r.preview,
    }
}

#[test]
fn approval_refused_when_nobody_was_detected() {
    if !have("ffmpeg") {
        return;
    }
    let dir: PathBuf = std::env::temp_dir().join(format!("anon-empty-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("empty.mp4");
    let st = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:size=160x120:rate=10:duration=1",
        ])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(&src)
        .status()
        .unwrap();
    assert!(st.success());
    let proj = ProjectDir::for_source(&src);
    analyze(&src, &proj, Box::new(RedFinder), opts(), |_, _| {}).unwrap();
    assert!(matches!(
        approve(&proj, None, false),
        Err(PipelineError::Refused(_))
    ));
    approve(&proj, None, true).unwrap();
    std::fs::remove_dir_all(&dir).ok();
}

/// Wraps RedFinder: counts calls, panics after `panic_after` calls.
struct Counting {
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    panic_after: usize,
}

impl PersonSegmenter for Counting {
    fn input_size(&self) -> u32 {
        128
    }
    fn segment(&mut self, t: &RgbImage) -> Result<Vec<PersonInstance>, DetectError> {
        let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if n >= self.panic_after {
            panic!("simulated crash");
        }
        RedFinder.segment(t)
    }
}

#[test]
fn interrupted_analysis_resumes_from_checkpoint() {
    if !have("ffmpeg") {
        return;
    }
    let dir: PathBuf = std::env::temp_dir().join(format!("anon-resume-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("long.mp4");
    let st = Command::new("ffmpeg")
        .args(["-v", "error", "-y"])
        .args([
            "-f",
            "lavfi",
            "-i",
            "color=c=0x606060:size=320x240:rate=25:duration=12",
        ])
        .args([
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=20x80:rate=25:duration=12",
        ])
        .args([
            "-filter_complex",
            "[0:v][1:v]overlay=x='20+t*20':y=80",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&src)
        .status()
        .unwrap();
    assert!(st.success());
    let proj = ProjectDir::for_source(&src);

    let per_frame = 20; // 3x3 tiles (+flip) + full frame (+flip)
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let crash = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let m = Counting {
            calls: calls.clone(),
            panic_after: per_frame * 270,
        };
        analyze(&src, &proj, Box::new(m), opts(), |_, _| {}).unwrap();
    }));
    assert!(crash.is_err());
    assert!(proj.path("checkpoint.json").exists());

    let calls2 = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let m = Counting {
        calls: calls2.clone(),
        panic_after: usize::MAX,
    };
    let mut o = opts();
    o.resume = true;
    let s = analyze(&src, &proj, Box::new(m), o, |_, _| {}).unwrap();
    assert_eq!(s.frames, 300);
    assert_eq!(s.resumed_from, 250);
    assert_eq!(
        calls2.load(std::sync::atomic::Ordering::SeqCst),
        per_frame * 50
    );
    assert_eq!(
        s.tracks, 1,
        "detections before the crash are part of the track"
    );
    assert!(!proj.path("checkpoint.json").exists());
    let fr = proj.load_frames().unwrap();
    assert!(fr.records.iter().all(|r| r.persons == 1));
    std::fs::remove_dir_all(&dir).ok();
}

/// Text = bright pixels (stand-in for PP-OCR det).
struct Bright;

impl anon_detect::text::TextMapModel for Bright {
    fn prob_map(&mut self, img: &RgbImage) -> Result<Vec<f32>, DetectError> {
        Ok(img
            .data
            .chunks_exact(3)
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
fn jersey_numbers_are_masked_when_enabled() {
    if !have("ffmpeg") || !have("ffprobe") {
        return;
    }
    let dir: PathBuf = std::env::temp_dir().join(format!("anon-text-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("numbers.mp4");
    // red player with a white 10x14 "number" on the shirt, walking right
    let st = Command::new("ffmpeg")
        .args(["-v", "error", "-y"])
        .args([
            "-f",
            "lavfi",
            "-i",
            "color=c=0x606060:size=320x240:rate=25:duration=1",
        ])
        .args([
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=24x90:rate=25:duration=1",
        ])
        .args([
            "-f",
            "lavfi",
            "-i",
            "color=c=white:size=10x14:rate=25:duration=1",
        ])
        .args([
            "-filter_complex",
            "[0:v][1:v]overlay=x='40+t*60':y=70[a];[a][2:v]overlay=x='47+t*60':y=110",
        ])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(&src)
        .status()
        .unwrap();
    assert!(st.success());
    let proj = ProjectDir::for_source(&src);

    // enabled without a text model -> refused
    let mut o = opts();
    o.settings.mask_jersey_numbers = true;
    assert!(matches!(
        analyze(&src, &proj, Box::new(RedFinder), o, |_, _| {}),
        Err(PipelineError::Refused(_))
    ));

    let mut o = opts();
    o.settings.mask_jersey_numbers = true;
    o.text = Some((
        Box::new(anon_detect::text::DbTextDetector { model: Bright }),
        ModelRef {
            id: "bright".into(),
            sha256: "0".into(),
            license: "test".into(),
        },
    ));
    let s = analyze(&src, &proj, Box::new(RedFinder), o, |_, _| {}).unwrap();
    assert_eq!(s.tracks, 1);
    assert_eq!(s.text_tracks, 1, "one number track");
    assert_eq!(proj.load_project().unwrap().settings.models.len(), 2);
    approve(&proj, None, false).unwrap();
    let out = dir.join("out.mp4");
    let ro = RenderOptions {
        encode: EncodeSettings {
            codec: Codec::H264,
            encoder: Some("libx264".into()),
            quality: 80,
        },
        preview: false,
    };
    render(&proj, &out, &ro).unwrap();
    for frame in [0u32, 12, 24] {
        let t = frame as f32 / 25.0;
        let nx = (47.0 + t * 60.0) as u32 + 5;
        let number = rgb_at(&out, nx, 117, frame);
        assert!(
            number.iter().all(|&c| c < 40),
            "frame {frame}: number visible {number:?}"
        );
        let legs = rgb_at(&out, nx, 150, frame);
        assert!(
            legs[0] > 170 && legs[1] < 90,
            "frame {frame}: body hidden {legs:?}"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn exports_reviewed_head_labels_as_coco() {
    if !have("ffmpeg") {
        return;
    }
    let dir: PathBuf = std::env::temp_dir().join(format!("anon-export-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("clip.mp4");
    let st = Command::new("ffmpeg")
        .args(["-v", "error", "-y"])
        .args([
            "-f",
            "lavfi",
            "-i",
            "color=c=0x606060:size=320x240:rate=25:duration=2",
        ])
        .args([
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=20x80:rate=25:duration=2",
        ])
        .args(["-filter_complex", "[0:v][1:v]overlay=x='20+t*60':y=80"])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(&src)
        .status()
        .unwrap();
    assert!(st.success());
    let proj = ProjectDir::for_source(&src);
    analyze(&src, &proj, Box::new(RedFinder), opts(), |_, _| {}).unwrap();
    let out = dir.join("dataset");
    let eo = anon_pipeline::ExportOptions {
        every: 5,
        block: 10,
        ..Default::default()
    };
    // unapproved -> refused
    assert!(
        anon_pipeline::export_dataset(std::slice::from_ref(&proj), &out, &eo, |_, _| {}).is_err()
    );
    approve(&proj, None, false).unwrap();
    let s =
        anon_pipeline::export_dataset(std::slice::from_ref(&proj), &out, &eo, |_, _| {}).unwrap();
    assert_eq!(s.images.iter().sum::<usize>(), 10);
    assert_eq!(
        s.boxes.iter().sum::<usize>(),
        10,
        "one head per exported frame"
    );
    assert!(s.images[0] > 0);
    // the box of frame 0 sits on the player's head
    let mut found = false;
    for split in ["train", "valid", "test"] {
        let doc: serde_json::Value = serde_json::from_slice(
            &std::fs::read(out.join(split).join("_annotations.coco.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(doc["categories"][0]["name"], "head");
        for img in doc["images"].as_array().unwrap() {
            let f = img["file_name"].as_str().unwrap();
            assert!(out.join(split).join(f).exists());
            if f.ends_with("_0000000.jpg") {
                let id = &img["id"];
                let ann = doc["annotations"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|a| &a["image_id"] == id)
                    .unwrap();
                let b: Vec<f64> = ann["bbox"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                assert!(
                    (b[0] - 20.0).abs() < 3.0 && (b[1] - 80.0).abs() < 3.0,
                    "{b:?}"
                );
                assert!(b[3] < 30.0, "head, not the whole body: {b:?}");
                found = true;
            }
        }
    }
    assert!(found);
    assert!(out.join("README.txt").exists());
    std::fs::remove_dir_all(&dir).ok();
}
