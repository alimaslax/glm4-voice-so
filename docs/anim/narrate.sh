#!/usr/bin/env bash
# Generate the narration wavs with OpenRouter (openai/gpt-audio-mini, streamed
# pcm16 @ 24 kHz). Needs OPEN_ROUTER (or OPENROUTER_API_KEY) in the repo .env.
#   docs/anim/narrate.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
set -a; . "$here/../../.env"; set +a
exec python3 "$here/narrate.py"
