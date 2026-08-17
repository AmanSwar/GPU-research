#!/usr/bin/env bash
# Cheap half of the citation audit: does every URL in the corpus resolve?
#
# A dead or invented URL is the single most likely defect in research written at
# speed, and finding one costs an HTTP request rather than a model call. This
# catches fabricated links, typos and moved pages. It does NOT check that a live
# page says what the citing document claims it says -- that still needs a reader.
#
#   scripts/check_links.sh [concurrency]
#
# Writes results to scripts/link-check.tsv as: status <TAB> url <TAB> files

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

CONC="${1:-12}"
OUT="scripts/link-check.tsv"
URLS="$(mktemp)"
trap 'rm -f "$URLS"' EXIT

# Pull bare URLs out of markdown, strip trailing punctuation and closing
# delimiters that regularly get swept up in the match.
grep -rhoE 'https?://[^ )>"`'"'"']+' --include='*.md' . \
  | sed -E 's/[.,;:]+$//; s/\]+$//; s/\)+$//' \
  | sort -u > "$URLS"

total=$(wc -l < "$URLS")
echo "checking $total distinct URLs at concurrency $CONC..." >&2

probe() {
  url="$1"
  # HEAD first; many hosts (notably arxiv and some CDNs) reject HEAD, so fall
  # back to a ranged GET before believing a failure.
  code=$(curl -sS -o /dev/null -w '%{http_code}' -L --max-time 20 \
           -A 'Mozilla/5.0 (link-check; research corpus verification)' \
           -I "$url" 2>/dev/null)
  if [[ "$code" == "000" || "$code" == "40"* || "$code" == "50"* ]]; then
    code=$(curl -sS -o /dev/null -w '%{http_code}' -L --max-time 20 \
             -A 'Mozilla/5.0 (link-check; research corpus verification)' \
             -r 0-1024 "$url" 2>/dev/null)
  fi
  printf '%s\t%s\n' "$code" "$url"
}
export -f probe

xargs -a "$URLS" -P "$CONC" -I{} bash -c 'probe "$@"' _ {} > "$OUT.raw" 2>/dev/null

# Annotate each result with the files that cite it, so a bad link is actionable.
: > "$OUT"
while IFS=$'\t' read -r code url; do
  files=$(grep -rl -F "$url" --include='*.md' . | sed 's|^\./||' | paste -sd, -)
  printf '%s\t%s\t%s\n' "$code" "$url" "$files" >> "$OUT"
done < "$OUT.raw"
rm -f "$OUT.raw"

echo >&2
echo "=== summary ===" >&2
awk -F'\t' '{c[$1]++} END {for (k in c) printf "  %-6s %d\n", k, c[k]}' "$OUT" | sort -rn -k2 >&2
echo >&2
bad=$(awk -F'\t' '$1 ~ /^(000|4|5)/' "$OUT" | wc -l)
echo "$bad URLs need review; see $OUT" >&2
