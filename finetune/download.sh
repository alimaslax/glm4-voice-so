#!/usr/bin/env bash
# Pull training data from the HF buckets into $SO_DATA (idempotent: sync skips what's already there).
#   transcripts : lewenberg/so-duplex-transcripts mai/ (-> transcripts/) and scribe/ (-> transcripts_scribe/)
#   omar        : only omar/ from lewenberg/so-duplex-processed
#   clean.flac  : only the <channel>/<episode>/clean.flac that have a transcript (the windows are just
#                 offsets into these files; no other processed files are downloaded)
set -euo pipefail
: "${HF_TOKEN:?HF_TOKEN not set (put it in .env)}"
: "${SO_DATA:?}"
T=hf://buckets/lewenberg/so-duplex-transcripts
P=hf://buckets/lewenberg/so-duplex-processed
mkdir -p "$SO_DATA/transcripts" "$SO_DATA/processed"

echo "== transcripts (mai/ = original MAI-Transcribe-2 labels; scribe/ = ElevenLabs Scribe v2 relabel, if present)"
hf buckets sync "$T/mai" "$SO_DATA/transcripts" --format quiet
hf buckets sync "$T/scribe" "$SO_DATA/transcripts_scribe" --format quiet 2>/dev/null || echo "   (no scribe/ yet)"
echo "== omar + clean.flac (in parallel)"
hf buckets sync "$P" "$SO_DATA/processed" --include "omar/*" --format quiet &
omar_pid=$!

# Direct parallel copies: a --filter-from sync would fnmatch 522 patterns against all ~300k bucket files first.
list="$SO_DATA/.clean_flac.list"
( cd "$SO_DATA/transcripts" && find . -path '*/diarized/transcripts.diarized.json' \
    | sed -E 's#^\./(.*)/diarized/transcripts\.diarized\.json$#\1/clean.flac#' | sort ) > "$list"
echo "episodes: $(wc -l < "$list")"
while read -r f; do [ -s "$SO_DATA/processed/$f" ] || echo "$f"; done < "$list" \
  | xargs -P 16 -I{} sh -c 'mkdir -p "$(dirname "$0/{}")" && hf buckets cp "$1/{}" "$0/{}.part" --format quiet >/dev/null && mv "$0/{}.part" "$0/{}"' \
      "$SO_DATA/processed" "$P"
wait "$omar_pid"

echo "== summary"
echo "transcripts: $(find "$SO_DATA/transcripts" -type f | wc -l) files"
echo "omar wavs:   $(find "$SO_DATA/processed/omar" -name '*.wav' | wc -l)"
echo "clean.flac:  $(find "$SO_DATA/processed" -name clean.flac | wc -l)"
du -sh "$SO_DATA"
