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
