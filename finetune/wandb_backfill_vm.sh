#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-glm4voice-train-blackwell:0.1}
ROOT=/workspace/so-train
UPLOADER=$ROOT/wandb_tensorboard_upload.py

upload() {
  local event=$1 project=$2 name=$3 run_id=$4 component=$5 round_name=$6 log_file=$7
  docker run --rm --network host \
    -v /workspace:/workspace \
    -v /root/.netrc:/root/.netrc:ro \
    -e PYTHONPATH=$ROOT/wandb-site \
    -e WANDB_SILENT=true \
    -e WANDB_DIR=$ROOT/wandb \
    "$IMAGE" python -u "$UPLOADER" "$event" \
      --entity lewenberg-student --project "$project" --name "$name" --run-id "$run_id" \
      --component "$component" --round "$round_name" --log-file "$log_file"
}

upload "$ROOT/runs/lora_somali/tb/events.out.tfevents.1789818186.tall-time-completes-fin-03.1.0" \
  glm4-voice-somali-lora glm4-somali-round1 lora-somali-r1-20260919 glm4-voice-lora round-1 \
  "$ROOT/logs/train_lora.20260919T114257Z.log"
upload "$ROOT/runs/lora_somali/tb/events.out.tfevents.1789823546.tall-time-completes-fin-03.1.0" \
  glm4-voice-somali-lora glm4-somali-round1-restart1 lora-somali-r1-restart1-20260919 glm4-voice-lora round-1 \
  "$ROOT/logs/train_lora.20260919T131218Z.log"
upload "$ROOT/runs/lora_somali/tb/events.out.tfevents.1789823629.tall-time-completes-fin-03.1.0" \
  glm4-voice-somali-lora glm4-somali-round1-restart2 lora-somali-r1-restart2-20260919 glm4-voice-lora round-1 \
  "$ROOT/logs/train_lora.20260919T131341Z.log"
upload "$ROOT/runs/lora_somali_scribe/tb/events.out.tfevents.1789837696.tall-time-completes-fin-03.1.0" \
  glm4-voice-somali-lora glm4-somali-scribe-round2 lora-somali-scribe-r2-20260919 glm4-voice-lora round-2 \
  "$ROOT/logs/train_lora.20260919T170808Z.log"

upload "$ROOT/runs/asr_mms/tb/events.out.tfevents.1789803320.tall-time-completes-fin-03.1.0" \
  glm4-voice-somali-asr mms-round1 asr-mms-r1-20260919 mms-asr round-1 \
  "$ROOT/logs/train_asr.20260919T073510Z.log"

upload "$ROOT/runs/flow_omar/tb/events.out.tfevents.1789807905.tall-time-completes-fin-03.1.0" \
  glm4-voice-somali-flow flow-omar-round1 flow-omar-r1-20260919 flow-matching round-1 \
  "$ROOT/logs/train_flow.20260919T085133Z.log"
