#!/usr/bin/env bash
# Run one pipeline stage inside the training container (on the GPU VM).
#   finetune/run.sh <stage> [extra args passed to the stage]
# Stages, in order:
#   download        pull transcripts + omar + transcribed clean.flac from the HF buckets
#   selftest        check the GLM prompt/token format against the demo's string tokenization
#   prepare_somali  transcripts -> cleaned clip/pair manifests            (CPU)
#   prepare_omar    Omar clips  -> single-speaker, ECAPA-verified manifest (GPU)
#   tokenize_somali Somali clips -> speech tokens                          (GPU)
#   tokenize_omar   Omar clips   -> speech tokens + 22.05 kHz mels         (GPU)
#   build_sft       Somali tokens -> ASR / TTS / dialogue SFT datasets     (CPU)
#   resynth         go/no-go: tokenizer+decoder round trip, Whisper CER, mel-config check
#   train_flow      Track B: flow decoder fine-tune on Omar
#   train_lora      Track A: bf16 LoRA on glm-4-voice-9b
#   shell           interactive shell in the container
# Paths (override via env): SO_DATA, SO_WORK, SO_MODELS. Secrets: .env at the repo root (HF_TOKEN).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="${1:?usage: finetune/run.sh <stage> [args]}"; shift || true
IMAGE="${IMAGE:-glm4voice-train-blackwell:0.1}"
export SO_DATA="${SO_DATA:-/workspace/so-data}"
export SO_WORK="${SO_WORK:-/workspace/so-train}"
export SO_MODELS="${SO_MODELS:-/workspace/glm-4-voice/models}"

if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"

case "$STAGE" in
  download)        CMD=(bash download.sh) ;;
  selftest)        CMD=(python -u selftest.py) ;;
  prepare_somali)  CMD=(python -u prepare_somali.py) ;;
  prepare_omar)    CMD=(python -u prepare_omar.py) ;;
  tokenize_somali) CMD=(python -u tokenize_audio.py --corpus somali) ;;
  tokenize_omar)   CMD=(python -u tokenize_audio.py --corpus omar) ;;
  build_sft)       CMD=(python -u build_sft.py) ;;
  resynth)         CMD=(python -u check_resynthesis.py) ;;
  train_flow)      CMD=(python -u train_flow.py) ;;
  train_lora)      CMD=(python -u train_lora.py) ;;
  shell)           CMD=(bash) ;;
  *) echo "unknown stage: $STAGE"; exit 2 ;;
esac

DOCKER=(docker run --rm --gpus all --ipc=host --network host --name "so-$STAGE"
  -e HF_TOKEN -e SO_DATA -e SO_WORK -e SO_MODELS -e HF_HOME=/workspace/cache -e PYTHONUNBUFFERED=1
  -v /workspace:/workspace -w "$REPO/finetune")
if [ "$STAGE" = shell ]; then exec "${DOCKER[@]}" -it "$IMAGE" bash; fi

mkdir -p "$SO_WORK/logs"
LOG="$SO_WORK/logs/$STAGE.$(date -u +%Y%m%dT%H%M%SZ).log"
echo "[$STAGE] image=$IMAGE commit=$(git -C "$REPO" rev-parse --short HEAD) log=$LOG"
"${DOCKER[@]}" "$IMAGE" "${CMD[@]}" "$@" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
