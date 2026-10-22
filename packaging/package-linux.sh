#!/usr/bin/env bash
# Package reco-anon + reco-anon-review for Linux x86_64 with their shared
# libraries (FFmpeg and codecs) bundled in lib/, found via RPATH $ORIGIN/lib.
# Graphics/windowing/system libraries stay with the host system.
#
#   packaging/package-linux.sh <version> <target-dir> <out-dir>
set -euo pipefail
VERSION="$1"; TARGET="${2:-target/release}"; OUT="${3:-dist}"
NAME="reco-anonymize-${VERSION}"
STAGE="$(mktemp -d)/${NAME}"
mkdir -p "$STAGE/lib" "$OUT"

for bin in reco-anon reco-anon-review; do
  install -m 755 "$TARGET/$bin" "$STAGE/$bin"
done

# never bundle: libc family, GPU/windowing stack, fonts, audio servers
SKIP='^(linux-vdso|ld-linux|libc\.|libm\.|libdl\.|librt\.|libpthread\.|libgcc_s|libstdc\+\+|libGL|libEGL|libGLX|libGLdispatch|libOpenGL|libX|libxcb|libxkb|libwayland|libdrm|libgbm|libfontconfig|libfreetype|libexpat|libz\.|libasound|libpulse|libdbus|libsystemd|libudev|libcap|libgpg|libgcrypt|liblzma|libzstd|liblz4|libbsd|libmd\.|libffi|libuuid|libblkid|libmount|libselinux|libpcre)'

collect() {
  ldd "$1" | awk '/=> \//{print $1" "$3}' | while read -r soname path; do
    if [[ "$soname" =~ $SKIP ]]; then continue; fi
    if [ ! -e "$STAGE/lib/$soname" ]; then
      cp -L "$path" "$STAGE/lib/$soname"
      chmod 644 "$STAGE/lib/$soname"
      collect "$path"
    fi
  done
}
for bin in reco-anon reco-anon-review; do collect "$STAGE/$bin"; done

# DT_RPATH (not RUNPATH) so transitive dependencies resolve from lib/ too
for bin in reco-anon reco-anon-review; do
  patchelf --force-rpath --set-rpath '$ORIGIN/lib' "$STAGE/$bin"
done
for so in "$STAGE"/lib/*.so*; do
  patchelf --force-rpath --set-rpath '$ORIGIN' "$so"
done

cp README.md LICENSE SECURITY.md THIRD-PARTY-NOTICES.md "$STAGE/"
mkdir -p "$STAGE/scripts" && cp scripts/export_rfdetr.py scripts/fetch_ppocr.py "$STAGE/scripts/"

# sanity: every bundled dependency must resolve inside the package
if ldd "$STAGE/reco-anon-review" "$STAGE/reco-anon" | grep -q "not found"; then
  ldd "$STAGE/reco-anon-review" "$STAGE/reco-anon" | grep "not found"
  echo "unresolved libraries" >&2; exit 1
fi
tar -C "$(dirname "$STAGE")" -czf "$OUT/${NAME}-linux-x86_64.tar.gz" "$NAME"
echo "$OUT/${NAME}-linux-x86_64.tar.gz ($(ls "$STAGE/lib" | wc -l) bundled libraries)"
