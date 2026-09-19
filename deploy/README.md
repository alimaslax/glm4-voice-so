# GLM-4-Voice — fast voice-orb deployment (INT4, Blackwell-ready)

This is the exact setup that was running and measured: ~520–590 ms server-side to first audio,
~0.8–1.1 s felt from a laptop over an SSH tunnel (network RTT ~140–210 ms is part of that).

## What runs
- `model_server.py` — GLM-4-Voice-9B, **INT4 (bitsandbytes NF4)** by default, with a `/cancel` endpoint
  (used for barge-in and dropped speculative turns). Port 10000, localhost only.
- `orb_demo.py` + `static/orb.html` — the UI/server on port 8890: Silero-VAD orb page, live captions,
  mic audio streamed while you talk, reply audio streamed back as raw PCM over a WebSocket.
  Flow decoder is patched at startup (CFG passes batched, 6 Euler steps; `GLM_FLOW_STEPS=10` restores stock).
- `speech_tokenizer/utils.py` — reads audio with `soundfile` (torchaudio 2.9 needs torchcodec otherwise).

Everything else in this repo is the untouched upstream GLM-4-Voice code (`cosyvoice/`, `flow_inference.py`, …).

## How it works — what was changed and why

**1. INT4 quantization** (`model_server.py`)
- Upstream already had `--dtype int4` (bitsandbytes NF4, double-quant, bf16 compute). Only the 9B language model is quantized; the speech tokenizer and flow/vocoder decoder stay full precision.
- It crashed with `ValueError: .to is not supported for 4-bit or 8-bit bitsandbytes models` because the model was loaded with `device_map={"": 0}`. Fix: `device_map="auto"` when INT4 (line ~101), plus `_patched_to` (line ~29) which swallows that specific `.to()` error from accelerate/transformers.
- The image needs `bitsandbytes>=0.46` (0.50.2 installed) for Blackwell. Result: ~10 GB VRAM instead of ~21 GB, and ~15.8 vs 17.8 ms/token.

**2. Blackwell support** (`deploy/docker/`)
- The original image's torch 2.3 has no `sm_120` kernels. `Dockerfile.blackwell` installs torch/torchaudio 2.9.1+cu128; `Dockerfile.fix` upgrades `typing_extensions` (torch 2.9 needs `TypeIs`) and adds bitsandbytes.
- torchaudio 2.9 needs torchcodec to save/load audio, so `speech_tokenizer/utils.py` reads with `soundfile` instead.

**3. Lower latency**
- *Flow decoder* (`orb_demo.py: _speed_up_flow`): the stock decoder ran 10 Euler steps × 2 classifier-free-guidance passes = 20 tiny launch-bound GPU calls (~330 ms per chunk regardless of size). The two guidance passes now run as one batch-2 call (identical maths, 333 → 172 ms) and the solver uses 6 steps (→ ~108 ms; mel differs ~1% from 10-step). `GLM_FLOW_STEPS=10` restores stock.
- *First audio chunk* (`BLOCK_SCHEDULE`): first decode after 10 audio tokens (was 25), then blocks grow because the decoder re-reads its whole prompt each call.
- *Transport*: reply audio goes out as raw int16 PCM over a WebSocket (no HLS playlist), played gap-free with WebAudio.
- *Pause detection* (`static/orb.html`): Silero VAD runs in the browser, so end-of-speech is decided locally. `redemptionMs` (default 200 ms) is the "start thinking" silence.
- *Speculative start*: the utterance is sent to the model after that short silence, but the reply audio is held until 600 ms of silence (`commitTurn`, setting "Start speaking after silence"). If you resume first, the guess is cancelled (`/cancel`) and your two chunks are merged into one turn. So a false start costs nothing audible.
- *Streaming upload*: mic frames are uploaded while you talk (`stream_start` / `stream_end`), so only the last frames are left to send when you stop. This matters most over a WAN tunnel.
- *Barge-in*: talking over the reply cancels playback and the GPU generation (`CancelCriteria` + `/cancel` in `model_server.py`); the interrupted reply is kept in history.
- *Warm-up*: `orb_demo.py` runs a dummy tokenize/decode/LM pass at startup so the first real turn isn't cold.

**4. Conversation memory / context** (`orb_demo.py`)
- Model window is 8192 tokens (`config.seq_length`); audio costs ~12.5 tokens per second, so full-audio history only fits ~12–15 turns.
- History is stored per browser id (`Convo` / `CONVOS`, id passed as `/ws?sid=`), so it survives page reloads and reconnects. Before this, each websocket got a fresh empty history.
- Budget `MAX_CONTEXT_TOKENS = 7000` (leaves room for the 600-token reply + system prompt). `history()` drops the oldest turns only if still over budget.
- Compaction: turns older than the last `KEEP_FULL_TURNS = 8` are rendered as text only (`render(..., compact=True)`) — ~3× smaller, so roughly 40 turns fit. The browser's transcript of your speech (`user_text` message) replaces your audio tokens when available.
- Tests: with 17 turns, all four settings (no compaction / plain-text last-3 / marker last-3 / marker last-8) produced audio on every turn and recalled a code word from turn 1. An earlier 22-turn run had two silent replies with the plain-text format, so any silent reply is logged as `SILENT REPLY`.
- The UI shows `memory: N turns · X/7k tokens` (the server sends a `ctx` message).

## What's in this repo (relative to a clean upstream clone)
- **Changed:** `model_server.py` (INT4 device-map fix + `/cancel` endpoint), `speech_tokenizer/utils.py` (soundfile audio loading).
- **New:** `orb_demo.py`, `static/orb.html`, and everything under `deploy/`.
- **Submodule:** `third_party/Matcha-TTS` must be initialised (`git submodule update --init`); the server imports it.
- **Deliberately left out:** the Gradio fallback `turn_demo.py`, benchmark scratch scripts, `.bak` copies, model weights, the Docker image tarball.
- These changes are **not committed** in git; `git diff` shows the two patched files.

## Where things live
| What | Where |
|---|---|
| Model weights (18 GB + 1.4 GB + 0.5 GB) | GPU VM network volume: `/workspace/glm-4-voice/models/{glm-4-voice-9b,glm-4-voice-tokenizer,glm-4-voice-decoder}` |
| Runtime copy of this repo | `/workspace/glm-4-voice/repo` (mounted into the container as `/workspace/repo`) |
| Saved replies (WAV) | `/workspace/glm-4-voice/captures/glm4voice` |
| **Exact Docker image** `glm4voice-runtime-blackwell:0.2` (torch 2.9.1+cu128, bitsandbytes 0.50.2) | `/workspace/migration/docker-image-glm4voice-blackwell.tar.zst` (+ `.sha256`) |
| Launcher / restore scripts | `/workspace/migration/{start-glm.sh,restore-images.sh}` — same files as `deploy/` here |

Weights and the image tarball are **not** in git (≈30 GB); they stay on the network volume.
`deploy/image-pip-freeze.txt` is the full `pip freeze` of the image.

## Run it (fresh GPU VM with the network volume mounted at /workspace)
Needs Docker + nvidia-container-toolkit and a driver that supports CUDA 12.8 (Blackwell needs this; the old torch 2.3 image cannot run on it).

```bash
/workspace/migration/restore-images.sh     # once per new VM: verifies sha256, docker-loads the image (~10 GB)
/workspace/migration/start-glm.sh          # INT4 + orb UI on 127.0.0.1:8890
```
Then on your laptop:
```bash
deploy/tunnel.sh <vm-ip>                   # -> http://localhost:8000/
```
Options: `DTYPE=bfloat16 start-glm.sh` (~2x the VRAM, a bit slower per token on this GPU), `IMAGE=… start-glm.sh`.
Health: `curl 127.0.0.1:8890/`, `curl 127.0.0.1:10000/docs`, `nvidia-smi`.

## Push code changes from this repo to the VM
```bash
VM=root@<vm-ip>; KEY=~/.ssh/verda_cpu_runner_20260830
rsync -av -e "ssh -i $KEY" orb_demo.py model_server.py $VM:/workspace/glm-4-voice/repo/
rsync -av -e "ssh -i $KEY" static/ $VM:/workspace/glm-4-voice/repo/static/
rsync -av -e "ssh -i $KEY" speech_tokenizer/utils.py $VM:/workspace/glm-4-voice/repo/speech_tokenizer/
ssh -i $KEY $VM /workspace/migration/start-glm.sh      # restart (page-only changes need no restart)
```

## Docker files
- `deploy/docker/Dockerfile.blackwell` — `FROM glm4voice-runtime:0.1`, installs torch/torchaudio 2.9.1+cu128.
- `deploy/docker/Dockerfile.fix` — upgrades `typing_extensions`, adds `bitsandbytes>=0.46` → `glm4voice-runtime-blackwell:0.2`.
- `glm4voice-runtime:0.1` (the base) exists only inside `/workspace/migration/docker-images.tar.zst`; it is a repaired
  build of `zhipuai/glm-4-voice:0.1`. The saved `…blackwell.tar.zst` already contains all layers, so the two Dockerfiles are only
  needed if that tarball is missing (`restore-images.sh` falls back to them automatically).

## Behaviour notes
- Conversation memory is kept server-side per browser id (survives reload/reconnect). Model window is 8192 tokens;
  history budget is 7000. The last 8 turns stay as audio, older ones are stored as text. A silent (text-only) reply is
  logged as `SILENT REPLY` in `/workspace/glm-4-voice/web-demo.log`.
- Only one generation runs at a time; a new request cancels the previous one.
- `deploy/tools/ws_test.py` is a WebSocket latency/barge-in smoke test (run it inside the container).

## What's on the network drive (`/workspace/migration`)
| File | Purpose |
|---|---|
| `docker-image-glm4voice-blackwell.tar.zst` (+ `.sha256`) | The exact image this runs on, 9.7 GB, self-contained. **Use this one.** |
| `restore-images.sh` | Verifies the sha256 and `docker load`s the tarball; falls back to `restore-images.legacy.sh` + the two Dockerfiles if it's missing. |
| `start-glm.sh`, `tunnel.sh` | Launcher (INT4 + orb UI by default) and the laptop-side tunnel. |
| `docker-blackwell/` | The two Dockerfiles. |
| `image-pip-freeze.txt` | Every package version in the image. |
| `README.md` | This file (the previous one is kept as `README.old.md`). |
| `docker-images.tar.zst`, `docker-image-glm4voice-int4.tar.zst`, `start-glm.bf16-backup.sh` | Older artifacts (original runtime image, old INT4 image, old BF16 launcher). Kept, not needed for the current setup. |

## Not verified
- The image tarball was integrity-checked (full decompress, manifest tag `glm4voice-runtime-blackwell:0.2`, sha256) but **never test-restored onto a clean Docker host**.
- The Dockerfiles alone cannot rebuild the image from scratch: their base `glm4voice-runtime:0.1` only exists inside the older `docker-images.tar.zst`.
- Timings above were measured on one RTX PRO 6000 Blackwell VM, with typed turns and a recorded clip fed through the browser code; the real microphone and live captions were not exercised in an automated test.
