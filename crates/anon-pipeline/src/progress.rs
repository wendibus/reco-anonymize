//! Frame rate and remaining time for the long frame loops (analyze, audit,
//! render), shared by the command line and the review app.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Rate is measured over this much recent wall time, so frames skipped on
/// `--resume` or a slow stretch stop skewing the estimate after a while.
const WINDOW: Duration = Duration::from_secs(60);
/// No estimate before the first sample is this old.
const MIN_SPAN: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    pub frames_per_sec: f64,
    /// `None` when the total frame count is unknown.
    pub remaining: Option<Duration>,
}

impl Estimate {
    /// "3.2 fps", or "4.1 s" per frame when slower than one frame a second.
    pub fn rate_text(&self, per_frame_unit: &str) -> String {
        if self.frames_per_sec >= 1.0 {
            format!("{:.1} fps", self.frames_per_sec)
        } else {
            format!("{:.1} {per_frame_unit}", 1.0 / self.frames_per_sec)
        }
    }
}

/// Sliding-window frame rate and remaining time.
#[derive(Debug, Default)]
pub struct Eta {
    samples: VecDeque<(Instant, u64)>,
}

impl Eta {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `done` frames are finished.
    pub fn update(&mut self, done: u64, total: Option<u64>) -> Option<Estimate> {
        self.update_at(Instant::now(), done, total)
    }

    fn update_at(&mut self, now: Instant, done: u64, total: Option<u64>) -> Option<Estimate> {
        if self.samples.back().is_some_and(|&(_, d)| done < d) {
            self.samples.clear();
        }
        self.samples.push_back((now, done));
        // keep one sample at least WINDOW old so the span never shrinks below it
        while self.samples.len() > 2 && now - self.samples[1].0 >= WINDOW {
            self.samples.pop_front();
        }
        let &(t0, d0) = self.samples.front()?;
        let span = now - t0;
        if span < MIN_SPAN || done <= d0 {
            return None;
        }
        let fps = (done - d0) as f64 / span.as_secs_f64();
        Some(Estimate {
            frames_per_sec: fps,
            remaining: total
                .filter(|&t| t >= done)
                .map(|t| Duration::from_secs_f64((t - done) as f64 / fps)),
        })
    }
}

/// "~2 h 05 min", "~14 min", "<1 min".
pub fn format_remaining(d: Duration) -> String {
    let min = (d.as_secs() + 30) / 60;
    match min {
        0 => "<1 min".into(),
        1..=59 => format!("~{min} min"),
        _ => format!("~{} h {:02} min", min / 60, min % 60),
    }
}

/// Calls at most every `interval`, plus always for the last frame, so a
/// slow loop still updates promptly and a fast one does not flood.
#[derive(Debug)]
pub struct Throttle {
    interval: Duration,
    last: Option<Instant>,
}

impl Throttle {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last: None,
        }
    }

    pub fn ready(&mut self, done: u64, total: Option<u64>) -> bool {
        let now = Instant::now();
        let due = self.last.is_none_or(|l| now - l >= self.interval) || total == Some(done);
        if due {
            self.last = Some(now);
        }
        due
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_from_recent_rate() {
        let t = Instant::now();
        let mut e = Eta::new();
        assert_eq!(e.update_at(t, 0, Some(1000)), None);
        // too early for an estimate
        assert_eq!(e.update_at(t + Duration::from_secs(1), 2, Some(1000)), None);
        let est = e
            .update_at(t + Duration::from_secs(10), 20, Some(1000))
            .unwrap();
        assert!((est.frames_per_sec - 2.0).abs() < 1e-9);
        assert_eq!(est.remaining, Some(Duration::from_secs(490)));
    }

    #[test]
    fn resume_burst_ages_out_of_the_window() {
        let t = Instant::now();
        let mut e = Eta::new();
        // 500 frames replayed from a checkpoint in one second
        e.update_at(t, 0, Some(2000));
        e.update_at(t + Duration::from_secs(1), 500, Some(2000));
        // then 1 frame per second for two minutes
        let mut last = None;
        for s in 2..=121 {
            last = e.update_at(t + Duration::from_secs(s), 500 + (s - 1), Some(2000));
        }
        let est = last.unwrap();
        assert!((est.frames_per_sec - 1.0).abs() < 0.05, "{est:?}");
    }

    #[test]
    fn new_job_resets() {
        let t = Instant::now();
        let mut e = Eta::new();
        e.update_at(t, 900, None);
        e.update_at(t + Duration::from_secs(5), 950, None);
        assert_eq!(e.update_at(t + Duration::from_secs(6), 10, None), None);
    }

    #[test]
    fn formats_remaining_time() {
        assert_eq!(format_remaining(Duration::from_secs(20)), "<1 min");
        assert_eq!(format_remaining(Duration::from_secs(14 * 60)), "~14 min");
        assert_eq!(
            format_remaining(Duration::from_secs(2 * 3600 + 5 * 60)),
            "~2 h 05 min"
        );
    }
}
