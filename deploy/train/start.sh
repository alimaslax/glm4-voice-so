#!/usr/bin/env bash
# From your laptop: update the VM checkout and (re)start the pipeline in a detached tmux session,
# plus the checkpoint backup loop (tmux so-ckpt) if it isn't running.
#   deploy/train/start.sh <vm-ip> [first-stage]      then: ssh ... tmux attach -t so-train
set -euo pipefail
IP="${1:?usage: start.sh <vm-ip> [first-stage]}"
KEY="${KEY:-$HOME/.ssh/verda_cpu_runner_20260830}"
REPO_DIR="${REPO_DIR:-/workspace/glm4-voice-so}"
ssh -i "$KEY" "root@$IP" "set -e; git -C $REPO_DIR pull --ff-only; \
  mkdir -p /workspace/so-train; tmux kill-session -t so-train 2>/dev/null || true; \
  tmux new-session -d -s so-train '$REPO_DIR/finetune/pipeline.sh ${2:-} 2>&1 | tee -a /workspace/so-train/pipeline.log'; \
  tmux has-session -t so-ckpt 2>/dev/null || tmux new-session -d -s so-ckpt '$REPO_DIR/finetune/run.sh ckpt_sync'; \
  echo started; tmux ls"
