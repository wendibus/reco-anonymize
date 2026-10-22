//! End-to-end transcode checks against a synthetic clip built with the
//! ffmpeg CLI: VFR timing, two audio streams, colour tags, rotation and
//! identifying metadata. Skipped when ffmpeg/ffprobe are not installed.

use std::path::{Path, PathBuf};
use std::process::Command;

use anon_io::{Codec, EncodeSettings, decoded_pts_secs, probe, transcode};
use anon_model::Shape;
use anon_render::Coverage;

fn have(bin: &str) -> bool {
    Command::new(bin)
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("anon-io-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn make_clip(path: &Path) {
    let st = Command::new("ffmpeg")
        .args(["-v", "error", "-y"])
        .args([
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30:duration=2",
        ])
        .args(["-f", "lavfi", "-i", "sine=f=440:duration=2"])
        .args(["-f", "lavfi", "-i", "sine=f=880:duration=2"])
        // drop 2 of every 5 frames -> irregular frame spacing (VFR)
        .args([
            "-vf",
            "select='lt(mod(n\\,5)\\,3)'",
            "-fps_mode",
            "passthrough",
        ])
        .args(["-map", "0:v", "-map", "1:a", "-map", "2:a"])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"])
        .args([
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
        ])
        .args([
            "-metadata",
            "title=Secret Game",
            "-metadata",
            "location=+52.2689+010.5268/",
        ])
        .args([
            "-metadata:s:a:0",
            "language=deu",
            "-metadata:s:a:1",
            "language=eng",
        ])
        .arg(path)
        .status()
        .unwrap();
    assert!(st.success());
}

fn ffprobe(path: &Path, args: &[&str]) -> String {
    let o = Command::new("ffprobe")
        .args(["-v", "error"])
        .args(args)
        .arg(path)
        .output()
        .unwrap();
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn gray_at(path: &Path, x: u32, y: u32, frame: u32) -> u8 {
    let o = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-vf",
            &format!("select=eq(n\\,{frame}),format=gray,crop=1:1:{x}:{y}"),
            "-frames:v",
            "1",
        ])
        .args(["-f", "rawvideo", "-"])
        .output()
        .unwrap();
    o.stdout[0]
}

#[test]
fn keeps_timing_audio_colour_and_strips_metadata() {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("skipped: ffmpeg/ffprobe not installed");
        return;
    }
    anon_io::init().unwrap();
    let dir = tmpdir("vfr");
    let src = dir.join("src.mp4");
    let dst = dir.join("out.mp4");
    make_clip(&src);

    let p = probe(&src).unwrap();
    assert_eq!(p.audio_streams, 2);

    let settings = EncodeSettings {
        codec: Codec::H264,
        encoder: Some("libx264".into()),
        quality: 86,
    };
    let report = transcode(
        &src,
        &dst,
        &settings,
        |job| {
            let mut c = Coverage::new(job.width, job.height);
            c.add_shape(&Shape::Polygon {
                points: vec![[0.0, 0.0], [160.0, 0.0], [160.0, 240.0], [0.0, 240.0]],
            });
            c
        },
        |_| {},
    )
    .unwrap();
    assert_eq!(report.audio_streams_copied, 2);

    // timing: identical packet timestamps relative to the first frame
    let a = decoded_pts_secs(&src).unwrap();
    let b = decoded_pts_secs(&dst).unwrap();
    assert_eq!(a.len(), b.len(), "frame count");
    assert_eq!(report.frames as usize, a.len());
    for (x, y) in a.iter().zip(&b) {
        assert!(((x - a[0]) - (y - b[0])).abs() < 1e-3, "{x} vs {y}");
    }
    // the clip really is VFR
    let gaps: std::collections::BTreeSet<i64> = a
        .windows(2)
        .map(|w| ((w[1] - w[0]) * 1000.0).round() as i64)
        .collect();
    assert!(gaps.len() > 1, "test clip should be VFR: {gaps:?}");

    // audio: two streams, languages kept, codec copied
    let audio = ffprobe(
        &dst,
        &[
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_name:stream_tags=language",
            "-of",
            "csv=p=0",
        ],
    );
    assert_eq!(audio.lines().count(), 2, "{audio}");
    assert!(
        audio.contains("aac") && audio.contains("deu") && audio.contains("eng"),
        "{audio}"
    );

    // colour description carried over
    let col = ffprobe(
        &dst,
        &[
            "-select_streams",
            "v",
            "-show_entries",
            "stream=color_primaries,color_transfer,color_space",
            "-of",
            "csv=p=0",
        ],
    );
    assert_eq!(col.trim(), "bt709,bt709,bt709");

    // identifying container metadata gone
    let tags = ffprobe(
        &dst,
        &["-show_entries", "format_tags", "-of", "default=nw=1"],
    );
    assert!(
        !tags.contains("Secret") && !tags.contains("52.26"),
        "{tags}"
    );

    // left half black, right half untouched (testsrc2 is colourful)
    for f in [0, 7, 20] {
        assert!(gray_at(&dst, 40, 120, f) <= 20, "frame {f} left");
        assert!(gray_at(&dst, 159, 10, f) <= 20, "frame {f} edge");
    }
    let right_src = gray_at(&src, 250, 120, 7);
    let right_dst = gray_at(&dst, 250, 120, 7);
    assert!((i32::from(right_src) - i32::from(right_dst)).abs() < 12);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn keeps_rotation_and_paints_unrotated_frame() {
    if !have("ffmpeg") || !have("ffprobe") {
        return;
    }
    anon_io::init().unwrap();
    let dir = tmpdir("rot");
    let plain = dir.join("plain.mp4");
    let src = dir.join("rot.mp4");
    let dst = dir.join("out.mp4");
    let st = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=25:duration=1",
        ])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(&plain)
        .status()
        .unwrap();
    assert!(st.success());
    let st = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-display_rotation", "90", "-i"])
        .arg(&plain)
        .args(["-c", "copy"])
        .arg(&src)
        .status()
        .unwrap();
    assert!(st.success());
    let p = probe(&src).unwrap();
    assert_eq!(
        p.rotation_deg, 270,
        "display_rotation 90 is counter-clockwise"
    );
    transcode(
        &src,
        &dst,
        &EncodeSettings {
            codec: Codec::H264,
            encoder: Some("libx264".into()),
            quality: 60,
        },
        |j| Coverage::new(j.width, j.height),
        |_| {},
    )
    .unwrap();
    assert_eq!(probe(&dst).unwrap().rotation_deg, 270);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn ten_bit_source_stays_ten_bit() {
    if !have("ffmpeg")
        || !have("ffprobe")
        || anon_io::ffmpeg_next::encoder::find_by_name("libx265").is_none()
    {
        return;
    }
    anon_io::init().unwrap();
    let dir = tmpdir("10bit");
    let src = dir.join("src.mkv");
    let dst = dir.join("out.mkv");
    let st = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=25:duration=1",
        ])
        .args([
            "-c:v",
            "libx265",
            "-pix_fmt",
            "yuv420p10le",
            "-x265-params",
            "log-level=error",
        ])
        .arg(&src)
        .status()
        .unwrap();
    assert!(st.success());
    let r = transcode(
        &src,
        &dst,
        &EncodeSettings {
            codec: Codec::Hevc,
            encoder: Some("libx265".into()),
            quality: 70,
        },
        |j| Coverage::full(j.width, j.height),
        |_| {},
    )
    .unwrap();
    assert_eq!(r.fully_opaque_frames, 25);
    let fmt = ffprobe(
        &dst,
        &[
            "-select_streams",
            "v",
            "-show_entries",
            "stream=pix_fmt",
            "-of",
            "csv=p=0",
        ],
    );
    assert_eq!(fmt.trim(), "yuv420p10le");
    assert!(gray_at(&dst, 100, 100, 12) <= 20);
    std::fs::remove_dir_all(dir).ok();
}
