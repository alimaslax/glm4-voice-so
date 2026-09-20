#!/usr/bin/env bash
# From your laptop: copy ONLY the Hugging Face token (and the OpenRouter / ElevenLabs keys, if set) from the local .env\n# to the VM's repo .env.
#   deploy/train/push_env.sh <vm-ip> [ssh-key]
set -euo pipefail
IP="${1:?usage: push_env.sh <vm-ip> [ssh-key]}"
KEY="${2:-$HOME/.ssh/verda_cpu_runner_20260830}"
REPO_DIR="${REPO_DIR:-/workspace/glm4-voice-so}"
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
[ -n "$TOKEN" ] || { echo "no HF_TOKEN / HUGGINGFACE_TOKEN in .env"; exit 1; }
OR="${OPENROUTER_API_KEY:-${OPEN_ROUTER:-}}"
OR2="${OPENROUTER_API_KEY2:-${OPEN_ROUTER2:-}}"
{ printf 'HF_TOKEN=%s\n' "$TOKEN"; [ -z "$OR" ] || printf 'OPENROUTER_API_KEY=%s\n' "$OR"
  [ -z "$OR2" ] || printf 'OPENROUTER_API_KEY2=%s\n' "$OR2"
  [ -z "${ELEVENLABS_API_KEY:-}" ] || printf 'ELEVENLABS_API_KEY=%s\n' "$ELEVENLABS_API_KEY"
  [ -z "${ELEVENLABS_API_KEY2:-}" ] || printf 'ELEVENLABS_API_KEY2=%s\n' "$ELEVENLABS_API_KEY2"
  [ -z "${ELEVENLABS_API_KEY3:-}" ] || printf 'ELEVENLABS_API_KEY3=%s\n' "$ELEVENLABS_API_KEY3"; } | ssh -i "$KEY" "root@$IP" "umask 077; cat > $REPO_DIR/.env"
echo "wrote $REPO_DIR/.env on $IP"
