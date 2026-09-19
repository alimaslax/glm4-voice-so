#!/usr/bin/env bash
# One-time setup of a GPU VM for training (idempotent). Run ON the VM:
#   curl -fsSL https://raw.githubusercontent.com/alimaslax/glm4-voice-so/main/deploy/train/bootstrap.sh | bash
# or, if the repo is already there: /workspace/glm4-voice-so/deploy/train/bootstrap.sh
# Needs: Docker + nvidia-container-toolkit, the /workspace network volume (models + migration/).
# Then put HF_TOKEN in $REPO_DIR/.env (from your laptop: deploy/train/push_env.sh <vm-ip>).
set -euo pipefail
REPO_URL="${REPO_URL:-https://github.com/alimaslax/glm4-voice-so.git}"
REPO_DIR="${REPO_DIR:-/workspace/glm4-voice-so}"
BASE_IMAGE=glm4voice-runtime-blackwell:0.2
TRAIN_IMAGE="${IMAGE:-glm4voice-train-blackwell:0.1}"
MODELS="${SO_MODELS:-/workspace/glm-4-voice/models}"

if [ -d "$REPO_DIR/.git" ]; then git -C "$REPO_DIR" pull --ff-only; else git clone "$REPO_URL" "$REPO_DIR"; fi
git -C "$REPO_DIR" submodule update --init --recursive

for m in glm-4-voice-9b glm-4-voice-tokenizer glm-4-voice-decoder; do
  test -d "$MODELS/$m" || { echo "missing $MODELS/$m (see deploy/README.md)"; exit 1; }
done

docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || /workspace/migration/restore-images.sh
docker build -f "$REPO_DIR/deploy/docker/Dockerfile.train" -t "$TRAIN_IMAGE" "$REPO_DIR"

# The voice demo holds ~13 GB of VRAM; training wants the whole card.
if docker ps --format '{{.Names}}' | grep -qx glm4voice; then docker stop glm4voice; fi

[ -f "$REPO_DIR/.env" ] || echo "WARNING: $REPO_DIR/.env missing - run deploy/train/push_env.sh <vm-ip> from your laptop"
echo "ready: $REPO_DIR/finetune/pipeline.sh"
