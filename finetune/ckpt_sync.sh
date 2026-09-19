#!/usr/bin/env bash
# Back up $SO_WORK/runs (checkpoints incl. resume state, final/ models, tensorboard) to a private HF bucket.
# Loops every CKPT_EVERY_MIN minutes; ONCE=1 syncs a single time. Never deletes remote files, so checkpoints
# rotated out locally (save_total_limit) stay in the bucket. Restore: finetune/run.sh ckpt_pull
set -euo pipefail
: "${HF_TOKEN:?HF_TOKEN not set}"; : "${SO_WORK:?}"
BUCKET="${CKPT_BUCKET:-lewenberg/so-train-checkpoints}"
EVERY="${CKPT_EVERY_MIN:-10}"
hf buckets create "$BUCKET" --private --exist-ok --format quiet
while true; do
  if hf buckets sync "$SO_WORK/runs" "hf://buckets/$BUCKET/runs" --exclude '*.tmp' --format quiet; then
    echo "$(date -u +%FT%TZ) synced $SO_WORK/runs -> hf://buckets/$BUCKET/runs"
  else
    echo "$(date -u +%FT%TZ) sync failed (will retry)"
  fi
  [ "${ONCE:-0}" = 1 ] && break
  sleep $((EVERY * 60))
done
