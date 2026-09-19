#!/usr/bin/env bash
set -euo pipefail

test -d /workspace/glm-4-voice/repo
test -d /workspace/glm-4-voice/models/glm-4-voice-9b
test -d /workspace/glm-4-voice/models/glm-4-voice-tokenizer
test -d /workspace/glm-4-voice/models/glm-4-voice-decoder

# Blackwell-capable image (torch 2.9.1+cu128, bitsandbytes). DTYPE: bfloat16 | int4
IMAGE="${IMAGE:-glm4voice-runtime-blackwell:0.2}"
DTYPE="${DTYPE:-int4}"
UI_SCRIPT="${UI_SCRIPT:-orb_demo.py}"   # orb_demo.py (voice orb) | turn_demo.py (Gradio)
echo "Using runtime image: $IMAGE"

docker rm -f glm4voice 2>/dev/null || true
docker run -d \
  --name glm4voice \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  --network host \
  -e HF_HOME=/workspace/cache \
  -e UI_SCRIPT="$UI_SCRIPT" \
  -v /workspace/glm-4-voice:/workspace \
  -w /workspace/repo \
  "$IMAGE" \
  bash -lc "python -u model_server.py --host 127.0.0.1 --model-path /workspace/models/glm-4-voice-9b --port 10000 --dtype $DTYPE --device cuda:0 > /workspace/model-server.log 2>&1 & exec python -u \$UI_SCRIPT --host 127.0.0.1 --port 8890 --tokenizer-path /workspace/models/glm-4-voice-tokenizer --model-path /workspace/models/glm-4-voice-9b --flow-path /workspace/models/glm-4-voice-decoder --capture-dir /workspace/captures/glm4voice > /workspace/web-demo.log 2>&1"

echo "Waiting for Model Server (port 10000) and Web Demo (port 8890) to be ready..."
for i in $(seq 1 60); do
  model_ready=0
  demo_ready=0
  if curl -s http://127.0.0.1:10000/health >/dev/null 2>&1 || curl -s http://127.0.0.1:10000/docs >/dev/null 2>&1; then
    model_ready=1
  fi
  if curl -s http://127.0.0.1:8890/ >/dev/null 2>&1; then
    demo_ready=1
  fi
  if [ "$model_ready" -eq 1 ] && [ "$demo_ready" -eq 1 ]; then
    echo "GLM-4-Voice is up and healthy!"
    echo "  - Model Server: http://127.0.0.1:10000/ ($DTYPE)"
    echo "  - Web Demo:     http://127.0.0.1:8890/ (Real-time turn stream)"
    exit 0
  fi
  sleep 2
done

echo "Warning: Services took longer than 120s to start. Check /workspace/glm-4-voice/model-server.log and web-demo.log"
