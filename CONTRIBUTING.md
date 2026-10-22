# Contributing

- Rust stable (MSRV 1.92, same as reco), FFmpeg 6.1+ development libraries.
- Before a PR: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace` (the integration tests need the `ffmpeg` and `ffprobe` binaries).
- Changes must keep the fail-closed rules in [`docs/SPEC.md`](docs/SPEC.md) §8 intact. Anything
  that could let a frame through unmasked needs a test that proves it cannot.
- Never commit real footage of people. Tests build synthetic clips with the ffmpeg CLI.
- By contributing you agree that your contribution is licensed under AGPL-3.0-only.
