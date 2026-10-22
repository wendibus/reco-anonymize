#!/usr/bin/env bash
# Package for macOS arm64: bundle Homebrew dylibs (FFmpeg and codecs) next to
# the binaries via dylibbundler, then ad-hoc sign everything (required on
# Apple silicon after rewriting load commands).
#
#   packaging/package-macos.sh <version> <target-dir> <out-dir>
set -euo pipefail
VERSION="$1"; TARGET="${2:-target/release}"; OUT="${3:-dist}"
NAME="reco-anonymize-${VERSION}"
STAGE="$(mktemp -d)/${NAME}"
mkdir -p "$STAGE/lib" "$OUT"
for bin in reco-anon reco-anon-review; do
  install -m 755 "$TARGET/$bin" "$STAGE/$bin"
done
dylibbundler -od -b -cd -x "$STAGE/reco-anon" -x "$STAGE/reco-anon-review" \
  -d "$STAGE/lib/" -p @executable_path/lib/
for f in "$STAGE"/lib/*.dylib "$STAGE/reco-anon" "$STAGE/reco-anon-review"; do
  codesign --force --sign - "$f"
done
cp README.md LICENSE SECURITY.md THIRD-PARTY-NOTICES.md "$STAGE/"
mkdir -p "$STAGE/scripts" && cp scripts/export_rfdetr.py scripts/fetch_ppocr.py "$STAGE/scripts/"
tar -C "$(dirname "$STAGE")" -czf "$OUT/${NAME}-macos-arm64.tar.gz" "$NAME"
echo "$OUT/${NAME}-macos-arm64.tar.gz"
