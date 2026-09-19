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
#   resynth         go/no-go: tokenizer+decoder round trip (stock decoder), MMS-som CER, A/B wavs
#   prepare_asr     Somali segments -> 16 kHz audio + normalized text for MMS
#   eval_asr        CER/WER of any model: --model <hf id or dir> --tag <name> [--splits test,val]
#   eval_asr_stock  CER/WER of stock facebook/mms-1b-all (som) on held-out episodes
#   train_asr       fine-tune MMS-1b-all Somali on all segments
#   eval_asr_ft     CER/WER of the fine-tuned MMS
#   publish_asr     private HF repo lewenberg/mms-1b-somali (base commit, then fine-tune)
#   train_flow      Track B: flow decoder fine-tune on Omar
#   resynth_flow    same A/B check with the Omar flow (and the fine-tuned MMS as judge)
#   publish_flow    private HF repo lewenberg/glm-4-voice-decoder-omar (base commit, then fine-tune)
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

# Judge for resynthesis checks: the fine-tuned MMS once it exists (container path == host path).
ASR_JUDGE=facebook/mms-1b-all
[ -d "$SO_WORK/runs/asr_mms/final" ] && ASR_JUDGE="$SO_WORK/runs/asr_mms/final"

case "$STAGE" in
  download)        CMD=(bash download.sh) ;;
  selftest)        CMD=(python -u selftest.py) ;;
  prepare_somali)  CMD=(python -u prepare_somali.py) ;;
  prepare_omar)    CMD=(python -u prepare_omar.py) ;;
  tokenize_somali) CMD=(python -u tokenize_audio.py --corpus somali) ;;
  tokenize_omar)   CMD=(python -u tokenize_audio.py --corpus omar) ;;
  build_sft)       CMD=(python -u build_sft.py) ;;
  resynth)         CMD=(python -u check_resynthesis.py --asr-model "$ASR_JUDGE") ;;
  prepare_asr)     CMD=(python -u prepare_asr.py) ;;
  eval_asr)        CMD=(python -u eval_asr.py) ;;                 # ad hoc: --model ... --tag ...
  eval_asr_stock)  CMD=(python -u eval_asr.py --model facebook/mms-1b-all --tag stock) ;;
  train_asr)       CMD=(python -u train_asr.py) ;;
  eval_asr_ft)     CMD=(python -u eval_asr.py --model "$SO_WORK/runs/asr_mms/final" --tag finetuned) ;;
  publish_asr)     CMD=(python -u publish_hf.py asr) ;;
  resynth_flow)    CMD=(python -u check_resynthesis.py --flow "$SO_WORK/runs/flow_omar/latest/flow.pt"
                        --tag flow_omar --asr-model "$ASR_JUDGE") ;;
  publish_flow)    CMD=(python -u publish_hf.py flow) ;;
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
