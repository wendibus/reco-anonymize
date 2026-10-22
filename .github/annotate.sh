#!/usr/bin/env bash
# Run a command; on failure turn its error lines into GitHub annotations
# (readable via the checks API even where raw logs are not accessible).
set -o pipefail
"$@" 2>&1 | tee .annotate.log
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  grep -E "^(error|warning)(\[|:)|^ +--> |panicked at|^test .* FAILED|^error: " .annotate.log \
    | head -60 | while IFS= read -r line; do echo "::error::${line//%/%25}"; done
fi
exit "$status"
