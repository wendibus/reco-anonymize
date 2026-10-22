//! Approval hash: binds a review approval to the exact masks it approved.

use sha2::{Digest, Sha256};

use crate::frames::FrameTable;
use crate::project::{Fingerprint, Zone};
use crate::track::TrackSet;

/// sha256 over source fingerprint, tracks, zones and the frame table.
/// Any edit to what will be painted changes the hash and voids approval.
pub fn approval_hash(
    fp: &Fingerprint,
    tracks: &TrackSet,
    zones: &[Zone],
    frames: &FrameTable,
) -> String {
    let mut h = Sha256::new();
    h.update(b"reco-anonymize/approval/v1\0");
    h.update(serde_json::to_vec(fp).expect("serializable"));
    h.update(b"\0");
    h.update(serde_json::to_vec(tracks).expect("serializable"));
    h.update(b"\0");
    h.update(serde_json::to_vec(zones).expect("serializable"));
    h.update(b"\0");
    h.update(frames.to_bytes());
    hex::encode(h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::FrameRecord;

    #[test]
    fn changes_with_any_input() {
        let fp = Fingerprint {
            size: 1,
            head_tail_sha256: "aa".into(),
        };
        let tracks = TrackSet::default();
        let mut frames = FrameTable {
            records: vec![FrameRecord::missing(0)],
        };
        let a = approval_hash(&fp, &tracks, &[], &frames);
        assert_eq!(a, approval_hash(&fp, &tracks, &[], &frames));
        frames.records[0].flags = 1;
        assert_ne!(a, approval_hash(&fp, &tracks, &[], &frames));
    }
}
