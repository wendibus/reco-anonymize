//! Review session on a synthetic clip: navigation, overlays, edits, gate.

use std::path::PathBuf;
use std::process::Command;

use anon_detect::{DetectError, PersonInstance, PersonSegmenter, RgbImage};
use anon_model::{AnalysisSettings, BBox, ModelRef, ProjectDir};
use anon_pipeline::AnalyzeOptions;
use anon_review::{RegionRef, ReviewSession, ViewMode};

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
            score: 0.3, // low score -> every frame flagged
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

#[test]
fn review_workflow() {
    if !have("ffmpeg") {
        return;
    }
    let dir: PathBuf = std::env::temp_dir().join(format!("anon-review-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("clip.mp4");
    let st = Command::new("ffmpeg")
        .args(["-v", "error", "-y"])
        .args([
            "-f",
            "lavfi",
            "-i",
            "color=c=0x606060:size=320x240:rate=10:duration=2",
        ])
        .args([
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=20x80:rate=10:duration=2",
        ])
        .args(["-filter_complex", "[0:v][1:v]overlay=x='20+t*60':y=80"])
        .args(["-c:v", "libx264", "-g", "5", "-pix_fmt", "yuv420p"])
        .arg(&src)
        .status()
        .unwrap();
    assert!(st.success());
    let proj = ProjectDir::for_source(&src);
    anon_pipeline::analyze(
        &src,
        &proj,
        Box::new(RedFinder),
        AnalyzeOptions {
            settings: AnalysisSettings::default(),
            model_ref: ModelRef {
                id: "t".into(),
                sha256: "0".into(),
                license: "t".into(),
            },
            fresh: false,
            resume: false,
            text: None,
        },
        |_, _| {},
    )
    .unwrap();

    let mut s = ReviewSession::open(&src).unwrap();
    assert_eq!(s.frame_count(), 20);
    let r = s.readiness();
    assert_eq!(r.flagged, 20, "low scores flag every frame");
    assert_eq!(r.unreviewed, 20);
    assert!(!r.can_approve());
    assert!(s.approve(None, false).is_err());

    // frame 10: player at x = 20 + 1.0*60 = 80..100, head at the top
    s.goto(10);
    let regions = s.regions_at(10);
    let auto_id = regions
        .iter()
        .find_map(|r| match r.what {
            RegionRef::Track {
                id, manual: false, ..
            } => Some(id),
            _ => None,
        })
        .expect("auto track");
    assert_eq!(s.hit_test(90.0, 85.0), Some(auto_id));
    assert_eq!(s.hit_test(300.0, 200.0), None);

    // outline view shows the picture; opaque view blacks out the head only
    let img = s.render_current().unwrap();
    assert!(img.px(90, 150)[0] > 170, "body visible");
    s.mode = ViewMode::Opaque;
    let img = s.render_current().unwrap();
    assert_eq!(&img.px(90, 84)[..3], &[0, 0, 0]);
    assert!(img.px(90, 150)[0] > 170);
    assert!(img.px(250, 200)[0] > 60, "background untouched");

    // random access in both directions returns the right picture
    s.goto(2);
    let a = s.render_current().unwrap();
    s.goto(18);
    s.render_current().unwrap();
    s.goto(2);
    assert_eq!(s.render_current().unwrap(), a);

    // suppress needs a reason, then the head is exposed in the opaque view
    assert!(s.suppress(auto_id, "").is_err());
    s.suppress(auto_id, "test").unwrap();
    s.goto(10);
    let img = s.render_current().unwrap();
    assert!(img.px(90, 84)[0] > 170, "suppressed track not painted");
    s.restore(auto_id).unwrap();

    // manual track: draw on 5 and 15, interpolated in between
    s.selected = None;
    s.goto(5);
    let m = s
        .add_manual_box(BBox::new(200.0, 20.0, 220.0, 40.0))
        .unwrap();
    s.goto(15);
    assert_eq!(
        s.add_manual_box(BBox::new(260.0, 20.0, 280.0, 40.0))
            .unwrap(),
        m
    );
    s.goto(10);
    assert_eq!(s.hit_test(240.0, 30.0), Some(m));
    s.delete_manual_keyframe(m).unwrap(); // no keyframe at 10: no-op
    s.goto(15);
    s.delete_manual_keyframe(m).unwrap();
    s.goto(5);
    s.delete_manual_keyframe(m).unwrap();
    assert!(!s.tracks.tracks.iter().any(|t| t.id == m), "track gone");
    assert!(s.delete_manual_keyframe(auto_id).is_err());

    // zones: draw, finish (range), paints in export view, remove
    s.goto(4);
    s.zone_add_point(250.0, 10.0);
    s.zone_add_point(310.0, 10.0);
    assert!(s.zone_finish("too few", None).is_err());
    s.zone_add_point(310.0, 40.0);
    s.zone_add_point(250.0, 40.0);
    assert!(s.zone_finish("board", Some((5, 3))).is_err());
    let z = s.zone_finish("scoreboard", Some((3, 6))).unwrap();
    assert!(s.zone_draft.is_empty());
    s.mode = ViewMode::Opaque;
    let img = s.render_current().unwrap();
    assert_eq!(&img.px(280, 25)[..3], &[0, 0, 0], "zone painted in range");
    s.goto(8);
    let img = s.render_current().unwrap();
    assert!(img.px(280, 25)[0] > 60, "outside the zone's frame range");
    assert!(
        s.project
            .zones
            .iter()
            .any(|zz| zz.id == z && zz.label == "scoreboard")
    );
    s.zone_remove(z).unwrap();
    assert!(s.zone_remove(z).is_err());
    s.mode = ViewMode::Outline;

    // visit every flagged frame, then approval works and persists
    while let Some(f) = s.next_flag(true, true) {
        s.goto(f);
    }
    s.goto(0);
    assert_eq!(s.readiness().unreviewed, 0);
    s.approve(Some("ok".into()), false).unwrap();
    assert!(s.readiness().approval_valid);
    let s2 = ReviewSession::open(&src).unwrap();
    assert!(s2.readiness().approval_valid);
    assert_eq!(s2.readiness().unreviewed, 0);
    std::fs::remove_dir_all(&dir).ok();
}
