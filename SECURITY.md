# Security and privacy

reco-anonymize exists to keep people in sports videos unidentifiable. A frame
that leaves the tool with a visible face, or with metadata pointing to a person
or place, is treated as a security bug.

## Reporting

Please report leaks or bypasses of the review gate privately via
[GitHub security advisories](https://github.com/wendibus/reco-anonymize/security/advisories/new)
instead of a public issue. Do **not** attach unanonymized footage; describe the
scene or use synthetic test clips.

## Guarantees the code enforces

- Nothing leaves the machine. The only network request is the optional update check against
  the GitHub releases API of this repository (can be switched off in the app); it sends no
  footage, paths or identifiers. Updates are installed only when the archive's SHA-256 matches
  the release's `SHA256SUMS.txt`.
- Exported training datasets contain un-anonymized frames and carry a README saying so.
- Frames that were not analysed, failed, or no longer match the analysis are
  rendered fully opaque.
- Exports require a human approval bound to a hash of the exact masks, and are
  verified (frame count, timing) before they get their final name.
- `audit` re-detects people on the finished file and checks coverage from the
  output pixels.

Automatic detection is never 100 % reliable. Always review before publishing.
