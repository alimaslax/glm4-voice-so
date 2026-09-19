#!/usr/bin/env bash
# Run the whole pipeline end to end, skipping stages already completed (markers in $SO_WORK/.done/).
#   finetune/pipeline.sh                 # everything
#   finetune/pipeline.sh train_flow      # from that stage on
#   FORCE=1 finetune/pipeline.sh build_sft   # redo from build_sft even if marked done
# Training stages themselves resume from their last checkpoint when re-run.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SO_WORK="${SO_WORK:-/workspace/so-train}"
STAGES=(download selftest prepare_somali prepare_omar tokenize_somali tokenize_omar build_sft resynth train_flow train_lora)
START="${1:-${STAGES[0]}}"
mkdir -p "$SO_WORK/.done"
started=0
for s in "${STAGES[@]}"; do
  [ "$s" = "$START" ] && started=1
  [ "$started" = 1 ] || continue
  if [ -f "$SO_WORK/.done/$s" ] && [ "${FORCE:-0}" != 1 ]; then echo "== $s: done, skipping"; continue; fi
  echo "== $s: $(date -u)"
  "$HERE/run.sh" "$s"
  date -u > "$SO_WORK/.done/$s"
done
echo "== pipeline finished $(date -u)"
