//! `tracks.json`: keyframed mask tracks.

use serde::{Deserialize, Serialize};

use crate::geometry::Shape;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionKind {
    Head,
    Text,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Origin {
    /// Produced by analysis; `passes` lists the passes that contributed.
    Auto { passes: Vec<String> },
    /// Drawn by a reviewer. Never touched by re-analysis.
    Manual,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Keyframe {
    pub frame: u64,
    pub shape: Shape,
    /// Best detection score behind this keyframe (1.0 for manual).
    pub score: f32,
    pub interpolated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Suppression {
    pub reason: String,
    /// Unix seconds.
    pub at: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Track {
    pub id: u64,
    pub kind: RegionKind,
    pub origin: Origin,
    /// Sorted by `frame`, unique frames.
    pub keyframes: Vec<Keyframe>,
    /// Set when a reviewer removed this track as a false positive.
    pub suppressed: Option<Suppression>,
}

impl Track {
    pub fn first_frame(&self) -> Option<u64> {
        self.keyframes.first().map(|k| k.frame)
    }

    pub fn last_frame(&self) -> Option<u64> {
        self.keyframes.last().map(|k| k.frame)
    }

    pub fn is_manual(&self) -> bool {
        matches!(self.origin, Origin::Manual)
    }

    /// Restore the sorted/unique keyframe invariant.
    pub fn normalize(&mut self) {
        self.keyframes.sort_by_key(|k| k.frame);
        self.keyframes.dedup_by_key(|k| k.frame);
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrackSet {
    pub tracks: Vec<Track>,
}

impl TrackSet {
    pub fn next_id(&self) -> u64 {
        self.tracks.iter().map(|t| t.id).max().map_or(1, |m| m + 1)
    }

    /// Replace all auto tracks, keeping manual tracks and suppressions of
    /// manual tracks untouched (edit invariant).
    pub fn replace_auto(&mut self, mut auto: Vec<Track>) {
        self.tracks.retain(Track::is_manual);
        for (id, t) in (self.next_id()..).zip(auto.iter_mut()) {
            t.id = id;
            t.normalize();
        }
        self.tracks.extend(auto);
    }
}
