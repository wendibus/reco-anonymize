#!/usr/bin/env bash
# Run a command; on failure turn its error lines into GitHub annotations
# (readable via the checks API even where raw logs are not accessible).
# GitHub keeps 10 annotations per step, so each error is joined with its
# location line into one annotation.
set -o pipefail
"$@" 2>&1 | tee .annotate.log
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  perl -ne '
    s/\e\[[0-9;]*[A-Za-z]//g; chomp;
    if (/^(error|warning)(\[|:)|panicked at|^test .* FAILED|failed to run custom build/) {
      print "::error::$cur\n" if defined $cur; $cur = $_; next;
    }
    if (defined $cur && /^\s+--> (.*)/) { $cur .= " @ $1"; print "::error::$cur\n"; undef $cur; next; }
    END { print "::error::$cur\n" if defined $cur; }
  ' .annotate.log > .annotate.out
  { grep -v "^::error::warning" .annotate.out; grep "^::error::warning" .annotate.out | grep -v "build failed"; } | head -10
fi
exit "$status"
