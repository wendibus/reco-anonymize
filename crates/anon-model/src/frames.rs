//! `frames.bin`: one fixed-size 16-byte little-endian record per decoded
//! video frame, in decode-output (presentation) order.
//!
//! ```text
//! offset size field
//!      0    8 pts (stream time base)
//!      8    1 status
//!      9    1 reserved (0)
//!     10    2 flags (bit set)
//!     12    2 person count
//!     14    2 reserved (0)
//! ```

use serde::{Deserialize, Serialize};

pub const RECORD_SIZE: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum FrameStatus {
    /// Not analysed. Rendered fully opaque (fail closed).
    Missing = 0,
    Analyzed = 1,
    /// Decode or inference error. Rendered fully opaque, blocks approval.
    Error = 2,
}

impl FrameStatus {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Analyzed,
            2 => Self::Error,
            _ => Self::Missing,
        }
    }
}

/// Reasons a frame needs a human look.
pub mod flag {
    /// A detection in the low-score band.
    pub const LOW_SCORE: u16 = 1 << 0;
    /// Person count dropped sharply versus the previous frame.
    pub const COUNT_DROP: u16 = 1 << 1;
    /// A track starts or ends away from the frame border.
    pub const TRACK_EDGE: u16 = 1 << 2;
    /// Mask comes from a long interpolated span.
    pub const LONG_INTERP: u16 = 1 << 3;
    /// Frame errored (see status).
    pub const ERROR: u16 = 1 << 4;

    pub fn names(bits: u16) -> Vec<&'static str> {
        let mut v = Vec::new();
        for (b, n) in [
            (LOW_SCORE, "low_score"),
            (COUNT_DROP, "count_drop"),
            (TRACK_EDGE, "track_edge"),
            (LONG_INTERP, "long_interp"),
            (ERROR, "error"),
        ] {
            if bits & b != 0 {
                v.push(n);
            }
        }
        v
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRecord {
    pub pts: i64,
    pub status: FrameStatus,
    pub flags: u16,
    pub persons: u16,
}

impl FrameRecord {
    pub fn missing(pts: i64) -> Self {
        Self {
            pts,
            status: FrameStatus::Missing,
            flags: 0,
            persons: 0,
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.pts.to_le_bytes());
        out.push(self.status as u8);
        out.push(0);
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(&self.persons.to_le_bytes());
        out.extend_from_slice(&[0, 0]);
    }

    fn decode(b: &[u8]) -> Self {
        Self {
            pts: i64::from_le_bytes(b[0..8].try_into().expect("8 bytes")),
            status: FrameStatus::from_u8(b[8]),
            flags: u16::from_le_bytes([b[10], b[11]]),
            persons: u16::from_le_bytes([b[12], b[13]]),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrameTable {
    pub records: Vec<FrameRecord>,
}

impl FrameTable {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.records.len() * RECORD_SIZE);
        for r in &self.records {
            r.encode(&mut out);
        }
        out
    }

    /// Parse; a trailing partial record (crash mid-write) is dropped.
    pub fn from_bytes(b: &[u8]) -> Self {
        Self {
            records: b
                .as_chunks::<RECORD_SIZE>()
                .0
                .iter()
                .map(|c| FrameRecord::decode(c))
                .collect(),
        }
    }

    pub fn get(&self, index: u64) -> Option<&FrameRecord> {
        usize::try_from(index)
            .ok()
            .and_then(|i| self.records.get(i))
    }

    pub fn count_status(&self, s: FrameStatus) -> usize {
        self.records.iter().filter(|r| r.status == s).count()
    }

    pub fn flagged(&self) -> impl Iterator<Item = (u64, &FrameRecord)> {
        self.records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.flags != 0)
            .map(|(i, r)| (i as u64, r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_truncation() {
        let t = FrameTable {
            records: vec![
                FrameRecord {
                    pts: -3,
                    status: FrameStatus::Analyzed,
                    flags: flag::LOW_SCORE | flag::COUNT_DROP,
                    persons: 7,
                },
                FrameRecord::missing(1001),
            ],
        };
        let mut b = t.to_bytes();
        assert_eq!(b.len(), 32);
        assert_eq!(FrameTable::from_bytes(&b), t);
        b.extend_from_slice(&[1, 2, 3]);
        assert_eq!(FrameTable::from_bytes(&b), t);
    }
}
