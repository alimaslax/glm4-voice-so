#!/usr/bin/env bash
# Pull training data from the HF buckets into $SO_DATA (idempotent: sync skips what's already there).
#   transcripts : all of lewenberg/so-duplex-transcripts
#   omar        : only omar/ from lewenberg/so-duplex-processed
#   clean.flac  : only the <channel>/<episode>/clean.flac that have a transcript (the windows are just
#                 offsets into these files; no other processed files are downloaded)
set -euo pipefail
: "${HF_TOKEN:?HF_TOKEN not set (put it in .env)}"
: "${SO_DATA:?}"
T=hf://buckets/lewenberg/so-duplex-transcripts
P=hf://buckets/lewenberg/so-duplex-processed
mkdir -p "$SO_DATA/transcripts" "$SO_DATA/processed"

echo "== transcripts"
hf buckets sync "$T" "$SO_DATA/transcripts" --format quiet
echo "== omar"
hf buckets sync "$P" "$SO_DATA/processed" --include "omar/*" --format quiet

echo "== clean.flac for transcribed episodes"
filter="$SO_DATA/.clean_flac.filter"
( cd "$SO_DATA/transcripts" && find . -path '*/diarized/transcripts.diarized.json' \
    | sed -E 's#^\./(.*)/diarized/transcripts\.diarized\.json$#+ \1/clean.flac#' | sort ) > "$filter"
echo "- *" >> "$filter"
echo "episodes: $(($(wc -l < "$filter") - 1))"
hf buckets sync "$P" "$SO_DATA/processed" --filter-from "$filter" --format quiet

echo "== summary"
echo "transcripts: $(find "$SO_DATA/transcripts" -type f | wc -l) files"
echo "omar wavs:   $(find "$SO_DATA/processed/omar" -name '*.wav' | wc -l)"
echo "clean.flac:  $(find "$SO_DATA/processed" -name clean.flac | wc -l)"
du -sh "$SO_DATA"
