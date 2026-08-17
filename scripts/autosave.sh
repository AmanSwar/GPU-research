#!/usr/bin/env bash
# Spot-instance insurance: snapshot whatever the research fleet has written and
# push it, every INTERVAL seconds. A reclaimed instance then costs at most one
# interval of work rather than the whole session.
#
# Snapshots are deliberately noisy and get squashed into curated commits later;
# losing the corpus is a worse outcome than an untidy history.
#
#   scripts/autosave.sh [interval_seconds] [max_iterations]

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

INTERVAL="${1:-180}"
MAX="${2:-400}"

for ((i = 0; i < MAX; i++)); do
  sleep "$INTERVAL"
  if [[ -n "$(git status --porcelain)" ]]; then
    n=$(git status --porcelain | wc -l)
    git add -A
    git commit -q -m "autosave: research fleet snapshot ($n paths)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" || continue
    git push -q origin main 2>/dev/null || echo "[autosave] push failed, will retry next interval"
    echo "[autosave] $(date -Is) committed $n paths"
  fi
done
