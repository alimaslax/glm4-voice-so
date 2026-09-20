#!/usr/bin/env bash
# VM loop (runs inside the training image): follow the Mac's Scribe run.
#  1. scribe/ transcripts + window_status.jsonl  <- hf://buckets/lewenberg/so-duplex-transcripts/scribe
#  2. clean.flac of every episode that has a window in window_status.jsonl <- lewenberg/so-duplex-processed
#     (only clean.flac: the windows are offsets into it). Never deletes; stops downloading below MIN_FREE_GB.
set -uo pipefail
T=hf://buckets/lewenberg/so-duplex-transcripts
P=hf://buckets/lewenberg/so-duplex-processed
MIN_FREE_GB="${MIN_FREE_GB:-40}"
EVERY="${EVERY:-600}"
while true; do
  hf buckets sync "$T/scribe" "$SO_WORK/scribe" --exclude "*.request.json" --format quiet >/dev/null 2>&1 \
    && echo "$(date -u +%FT%TZ) transcripts synced" || echo "$(date -u +%FT%TZ) transcripts sync FAILED"
  python3 - "$SO_WORK/scribe/window_status.jsonl" > "$SO_DATA/.scribe_flac.list" <<'PY'
import json, sys
eps = sorted({(r["channel"], r["episode"]) for r in map(json.loads, open(sys.argv[1])) if r["channel"] != "omar"})
print("\n".join(f"{c}/{e}/clean.flac" for c, e in eps))
PY
  missing=$(while read -r f; do [ -s "$SO_DATA/processed/$f" ] || echo "$f"; done < "$SO_DATA/.scribe_flac.list")
  n_all=$(wc -l < "$SO_DATA/.scribe_flac.list"); n_miss=$(printf '%s' "$missing" | grep -c . || true)
  free=$(df -BG --output=avail "$SO_DATA" | tail -1 | tr -dc 0-9)
  echo "$(date -u +%FT%TZ) episodes $n_all, missing clean.flac $n_miss, free ${free}G"
  # 4 at a time, each retried with backoff: the Hub rate-limits bursts (the Mac's own bucket sync shares the account).
  printf '%s\n' "$missing" | grep . | xargs -P 4 -I{} sh -c '
      free=$(df -BG --output=avail "$0" | tail -1 | tr -dc 0-9); [ "$free" -gt "$2" ] || exit 0
      mkdir -p "$(dirname "$0/processed/{}")"
      for i in 1 2 3 4 5; do
        err=$(hf buckets cp "$1/{}" "$0/processed/{}.part" --format quiet 2>&1 >/dev/null) \
          && mv "$0/processed/{}.part" "$0/processed/{}" && exit 0
        rm -f "$0/processed/{}.part"; sleep $((i * 20))
      done
      echo "failed {}: $(printf "%s" "$err" | tail -1 | cut -c1-160)"' \
      "$SO_DATA" "$P" "$MIN_FREE_GB"
  echo "$(date -u +%FT%TZ) clean.flac on disk: $(find "$SO_DATA/processed" -name clean.flac | wc -l), free $(df -BG --output=avail "$SO_DATA" | tail -1 | tr -dc 0-9)G"
  [ "${ONCE:-0}" = 1 ] && break
  sleep "$EVERY"
done
