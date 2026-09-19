"""Low-latency ChatGPT-style voice chat for GLM-4-Voice.

Browser does VAD + live transcript; this server tokenizes the utterance, streams tokens from
model_server.py, decodes audio in small growing blocks and pushes raw PCM over a WebSocket.

Wire format
  client -> server  binary : uint32 LE turn-id | int16 LE PCM @16 kHz     (id 0 = streamed mic chunk, else a whole turn)
                    json   : {type:"stream_start", keep} | {type:"stream_abort"} | {type:"stream_end", id}
                             | {type:"text_turn", id, text} | {type:"cancel", id, keep:bool} | {type:"ping", t}
                             | {type:"reset"} | {type:"config", temperature, top_p, max_tokens}
  server -> client  binary : uint32 LE turn-id | int16 LE PCM @22050 Hz         (reply audio)
                    json   : {type:"ready"} | {type:"text", id, text} | {type:"first_audio", id, ms}
                             | {type:"done", id, text, ms, audio_ms, file, interrupted}
                             | {type:"error", id, message} | {type:"pong", t}
"""
import asyncio
import json
import os
import re
import struct
import sys
import threading
import time
import uuid
from argparse import ArgumentParser

import numpy as np
import requests
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from transformers import AutoTokenizer, WhisperFeatureExtractor

sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "./third_party/Matcha-TTS")

from flow_inference import AudioDecoder
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token

MODEL_URL = "http://127.0.0.1:10000"
SAMPLE_RATE_OUT = 22050
# Audio tokens per flow-decoder block. Small first block = low time-to-first-audio; later blocks grow
# because the decoder re-reads the whole prompt each call.
BLOCK_SCHEDULE = [10, 15, 25, 40, 60, 80, 100]
# Model window is 8192 tokens (config.seq_length); leave room for the 600-token reply + system prompt.
MAX_CONTEXT_TOKENS = 7000
# Turns older than this many are stored as text only (no audio tokens): ~3x smaller, so far more history fits.
# 17-turn tests with 3 and 8 full turns gave no silent replies; the last 8 stay full to keep the audio pattern strong.
KEEP_FULL_TURNS = 8

SYSTEM_PROMPT = (
    "User will provide you with an instruction. "
    "You are an English-speaking voice assistant. "
    "You must always reply strictly in fluent, natural English. "
    "Never speak, write, or respond in Chinese. Always answer in clear English. "
    "Keep answers short and conversational. "
    "Reply naturally and concisely in an interleaved manner, with 13 text tokens followed by 26 audio tokens."
)


def clean_text(tokenizer, ids):
    text = tokenizer.decode(ids, skip_special_tokens=False)
    text = re.sub(r"<\|[^>]+\|>", "", text)
    return re.sub(r"\s+", " ", text).strip()


parser = ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8890)
parser.add_argument("--flow-path", required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--tokenizer-path", required=True)
parser.add_argument("--capture-dir", default="/workspace/captures/glm4voice")
parser.add_argument("--static-dir", default="static")
args = parser.parse_args()

os.makedirs(args.capture_dir, exist_ok=True)
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
speech_encoder = WhisperVQEncoder.from_pretrained(args.tokenizer_path).eval().to(device)
feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tokenizer_path)
decoder = AudioDecoder(
    config_path=os.path.join(args.flow_path, "config.yaml"),
    flow_ckpt_path=os.path.join(args.flow_path, "flow.pt"),
    hift_ckpt_path=os.path.join(args.flow_path, "hift.pt"),
    device=device,
)


def _speed_up_flow(dec, steps):
    """Flow decoder latency: run the two classifier-free-guidance passes as one batch-2 call (identical
    maths, half the kernel launches) and use fewer Euler steps (the stock 10 are launch-bound: ~330 ms/call)."""
    import types
    cfm = dec.flow.decoder
    orig_forward = cfm.forward

    def solve_batched(self, x, t_span, mu, mask, spks, cond):
        t, dt = t_span[0], t_span[1] - t_span[0]
        mu2, spks2 = torch.cat([mu, torch.zeros_like(mu)]), torch.cat([spks, torch.zeros_like(spks)])
        cond2, mask2 = torch.cat([cond, torch.zeros_like(cond)]), torch.cat([mask, mask])
        for step in range(1, len(t_span)):
            cond_pred, uncond_pred = self.estimator(torch.cat([x, x]), mask2, mu2, t, spks2, cond2).chunk(2)
            x = x + dt * ((1.0 + self.inference_cfg_rate) * cond_pred - self.inference_cfg_rate * uncond_pred)
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x

    cfm.solve_euler = types.MethodType(solve_batched, cfm)
    cfm.forward = types.MethodType(lambda self, mu, mask, n_timesteps, **kw: orig_forward(mu, mask, steps, **kw), cfm)


_speed_up_flow(decoder, int(os.environ.get("GLM_FLOW_STEPS", "6")))
audio_offset = tokenizer.convert_tokens_to_ids("<|audio_0|>")
end_token_id = tokenizer.convert_tokens_to_ids("<|user|>")

gpu_lock = threading.Lock()  # tokenizer/decoder state is not re-entrant


class Convo:
    """Conversation state, keyed by a browser-supplied session id so it survives websocket reconnects."""

    def __init__(self):
        self.turns = []  # committed turn records (see Session._run_turn)
        self.cfg = {"temperature": 0.2, "top_p": 0.8, "max_tokens": 600,
                    "keep_full": KEEP_FULL_TURNS, "marker": 1}  # last two: history-compaction knobs


CONVOS = {}


def text_len(text):
    return len(tokenizer(text)["input_ids"]) + 6


class Session:
    """Per-connection wrapper around a Convo plus the single in-flight turn."""

    def __init__(self, ws, loop, convo):
        self.ws = ws
        self.loop = loop
        self.out = asyncio.Queue()
        self.turns = convo.turns  # shared, mutated in place
        self.cfg = convo.cfg
        self.job = None  # {id, cancel, keep, thread}
        self.buf = bytearray()  # mic audio streamed while the user is still talking
        self.mark = 0

    # ---- thread -> socket helpers
    def emit(self, item):
        self.loop.call_soon_threadsafe(self.out.put_nowait, item)

    def emit_json(self, **kw):
        self.emit(json.dumps(kw))

    # ---- conversation
    def render(self, turn, compact):
        """(prompt text, token count) of one past turn, full (interleaved audio) or compact (text only)."""
        if not compact:
            return turn["full"], turn["full_n"]
        if turn.get("user_text") and "text_n" not in turn:
            turn["text_n"] = text_len(turn["user_text"])
        user, user_n = (turn["user_text"], turn["text_n"]) if turn.get("user_text") else (turn["user"], turn["user_n"])
        tag = "streaming_transcription\n" if self.cfg.get("marker") else "\n"
        return f"<|user|>\n{user}<|assistant|>{tag}{turn['asst_text']}", user_n + turn["asst_n"]

    def history(self):
        """Render history, compacting old turns and dropping the oldest if still over budget."""
        while True:
            n = len(self.turns)
            parts = [self.render(t, i < n - int(self.cfg["keep_full"])) for i, t in enumerate(self.turns)]
            total = sum(c for _, c in parts) + 120  # system prompt + new turn framing
            if total <= MAX_CONTEXT_TOKENS or not self.turns:
                return "".join(p for p, _ in parts), total
            self.turns.pop(0)

    def send_ctx(self):
        _, total = self.history()
        self.emit_json(type="ctx", turns=len(self.turns), tokens=total, limit=MAX_CONTEXT_TOKENS)

    def build_prompt(self, user_input):
        hist, _ = self.history()
        base = f"<|system|>\n{SYSTEM_PROMPT}" + hist
        return base, base + f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

    def stop_job(self, keep=False):
        job = self.job
        if job and job["thread"].is_alive():
            job["keep"] = keep
            job["cancel"].set()
            try:
                requests.post(f"{MODEL_URL}/cancel", timeout=2)
            except Exception:
                pass
            job["thread"].join(timeout=5)

    def start_turn(self, turn_id, pcm=None, text=None):
        self.stop_job(keep=False)
        job = {"id": turn_id, "cancel": threading.Event(), "keep": False}
        t = threading.Thread(target=self._run_turn, args=(job, pcm, text), daemon=True)
        job["thread"] = t
        self.job = job
        t.start()

    # ---- the turn itself (worker thread)
    def _run_turn(self, job, pcm, text):
        cancel, tid = job["cancel"], job["id"]
        t0 = time.perf_counter()
        try:
            if pcm is not None:
                wav = torch.from_numpy(pcm.astype(np.float32) / 32768.0).unsqueeze(0)
                with gpu_lock:
                    speech_tokens = extract_speech_token(speech_encoder, feature_extractor, [(wav, 16000)])[0]
                if not speech_tokens:
                    self.emit_json(type="error", id=tid, message="no speech")
                    return
                user_input = "<|begin_of_audio|>" + "".join(f"<|audio_{t}|>" for t in speech_tokens) + "<|end_of_audio|>"
                n_user = len(speech_tokens) + 8
            else:
                user_input = text
                n_user = len(tokenizer(text)["input_ids"]) + 8
            base, prompt = self.build_prompt(user_input)
            t_tok = time.perf_counter()

            resp = requests.post(
                f"{MODEL_URL}/generate_stream",
                json={"prompt": prompt, "temperature": self.cfg["temperature"], "top_p": self.cfg["top_p"],
                      "max_new_tokens": int(self.cfg["max_tokens"])},
                stream=True, timeout=300,
            )
            resp.raise_for_status()

            complete_ids, text_ids, chunk = [], [], []
            speeches, mels = [], []
            prompt_tok = torch.zeros(1, 0, dtype=torch.int64, device=device)
            prompt_feat = torch.zeros(1, 0, 80, device=device)
            this_uuid = str(uuid.uuid4())
            block_idx, first_audio_ms, last_text_n = 0, None, 0
            header = struct.pack("<I", tid)

            def decode(finalize):
                nonlocal prompt_tok, prompt_feat, chunk, first_audio_ms
                tok = torch.tensor(chunk, device=device).unsqueeze(0)
                if mels:
                    prompt_feat = torch.cat(mels, dim=-1).transpose(1, 2)
                with gpu_lock, torch.inference_mode():
                    speech, mel = decoder.token2wav(tok, uuid=this_uuid, prompt_token=prompt_tok,
                                                    prompt_feat=prompt_feat, finalize=finalize)
                mels.append(mel)
                prompt_tok = torch.cat((prompt_tok, tok), dim=-1)
                chunk = []
                sp = speech.squeeze(0).float().clamp(-1, 1).cpu()
                if sp.numel():
                    speeches.append(sp)
                    self.emit(header + (sp.numpy() * 32767).astype("<i2").tobytes())
                    if first_audio_ms is None:
                        first_audio_ms = int((time.perf_counter() - t0) * 1000)
                        self.emit_json(type="first_audio", id=tid, ms=first_audio_ms,
                                       tokenize_ms=int((t_tok - t0) * 1000))

            for line in resp.iter_lines():
                if cancel.is_set():
                    break
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("error_code", 0):
                    raise RuntimeError(payload.get("text", "model server error"))
                token_id = int(payload["token_id"])
                if token_id == end_token_id:
                    break
                complete_ids.append(token_id)
                if token_id >= audio_offset:
                    chunk.append(token_id - audio_offset)
                    if len(chunk) >= BLOCK_SCHEDULE[block_idx]:
                        block_idx = min(block_idx + 1, len(BLOCK_SCHEDULE) - 1)
                        decode(False)
                else:
                    text_ids.append(token_id)
                    if len(text_ids) - last_text_n >= 2:
                        last_text_n = len(text_ids)
                        self.emit_json(type="text", id=tid, text=clean_text(tokenizer, text_ids))

            interrupted = cancel.is_set()
            resp.close()
            if not interrupted and chunk:
                decode(True)

            final_text = clean_text(tokenizer, text_ids)
            audio_ms = int(sum(s.numel() for s in speeches) / SAMPLE_RATE_OUT * 1000)
            wav_path = None
            if speeches:
                wav_path = os.path.join(args.capture_dir, f"reply-{int(time.time())}-{uuid.uuid4().hex[:8]}.wav")
                sf.write(wav_path, torch.cat(speeches).numpy(), SAMPLE_RATE_OUT, subtype="PCM_16")
            if not interrupted or job["keep"]:
                if complete_ids:
                    completion = tokenizer.decode(complete_ids, spaces_between_special_tokens=False)
                    self.turns.append({
                        "id": tid, "user": user_input, "user_n": n_user, "user_text": text or job.get("user_text"),
                        "full": f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n{completion}",
                        "full_n": n_user + len(complete_ids), "asst_text": final_text, "asst_n": len(text_ids) + 6,
                    })
            if not interrupted and text_ids and not speeches:
                print(f"SILENT REPLY turn {tid}: text but no audio tokens, history turns={len(self.turns)}", flush=True)
            self.send_ctx()
            self.emit_json(type="done", id=tid, text=final_text, ms=int((time.perf_counter() - t0) * 1000),
                           audio_ms=audio_ms, file=os.path.basename(wav_path) if wav_path else None,
                           interrupted=interrupted, first_audio_ms=first_audio_ms)
        except Exception as e:  # noqa: BLE001
            print("turn error:", repr(e), flush=True)
            self.emit_json(type="error", id=tid, message=str(e))


app = FastAPI()


@app.get("/")
async def index():
    return FileResponse(os.path.join(args.static_dir, "orb.html"), headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/captures/{name}")
async def capture(name: str):
    path = os.path.join(args.capture_dir, os.path.basename(name))
    return FileResponse(path, media_type="audio/wav")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    s = Session(ws, loop, CONVOS.setdefault(ws.query_params.get("sid", "default")[:64], Convo()))

    async def sender():
        while True:
            item = await s.out.get()
            if isinstance(item, bytes):
                await ws.send_bytes(item)
            else:
                await ws.send_text(item)

    send_task = asyncio.create_task(sender())
    await ws.send_text(json.dumps({"type": "ready"}))
    s.send_ctx()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                data = msg["bytes"]
                tid = struct.unpack("<I", data[:4])[0]
                if tid == 0:
                    s.buf.extend(data[4:])
                else:
                    pcm = np.frombuffer(data[4:], dtype="<i2")
                    await loop.run_in_executor(None, lambda: s.start_turn(tid, pcm=pcm))
            elif msg.get("text") is not None:
                m = json.loads(msg["text"])
                kind = m.get("type")
                if kind == "stream_start":
                    if not m.get("keep"):
                        s.buf.clear()
                    s.mark = len(s.buf)
                elif kind == "stream_abort":
                    del s.buf[s.mark:]
                elif kind == "stream_end":
                    pcm = np.frombuffer(bytes(s.buf), dtype="<i2")
                    await loop.run_in_executor(None, lambda: s.start_turn(m["id"], pcm=pcm))
                elif kind == "ping":
                    s.emit_json(type="pong", t=m.get("t"))
                elif kind == "text_turn":
                    await loop.run_in_executor(None, lambda: s.start_turn(m["id"], text=m["text"]))
                elif kind == "cancel":
                    await loop.run_in_executor(None, lambda: s.stop_job(keep=bool(m.get("keep"))))
                elif kind == "reset":
                    await loop.run_in_executor(None, s.stop_job)
                    s.turns.clear()
                    s.send_ctx()
                elif kind == "user_text":  # browser's transcript of a spoken turn: lets old turns be stored as text
                    for t in s.turns:
                        if t["id"] == m.get("id") and m.get("text"):
                            t["user_text"] = str(m["text"])[:2000]
                            t.pop("text_n", None)
                    if s.job and s.job["id"] == m.get("id") and m.get("text"):
                        s.job["user_text"] = str(m["text"])[:2000]
                elif kind == "config":
                    for k in ("temperature", "top_p", "max_tokens", "keep_full", "marker"):
                        if k in m:
                            s.cfg[k] = float(m[k])
    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()
        await loop.run_in_executor(None, s.stop_job)


def warmup():
    """Pay CUDA/kernel warm-up now instead of on the user's first turn."""
    print("warming up…", flush=True)
    t = time.perf_counter()
    with gpu_lock:
        extract_speech_token(speech_encoder, feature_extractor, [(torch.zeros(1, 16000 * 2), 16000)])
        tok = torch.randint(0, 4000, (1, 10), device=device)
        with torch.inference_mode():
            decoder.token2wav(tok, uuid="warm", finalize=False)
            decoder.token2wav(tok, uuid="warm", finalize=True)
    for _ in range(180):  # model_server may still be loading weights
        try:
            requests.get(f"{MODEL_URL}/health", timeout=1)
            break
        except Exception:
            time.sleep(1)
    try:
        r = requests.post(f"{MODEL_URL}/generate_stream", stream=True, timeout=120,
                          json={"prompt": f"<|system|>\n{SYSTEM_PROMPT}<|user|>\nHi<|assistant|>streaming_transcription\n",
                                "temperature": 0.2, "top_p": 0.8, "max_new_tokens": 8})
        for _ in r.iter_lines():
            pass
    except Exception as e:  # noqa: BLE001
        print("LM warm-up failed:", e, flush=True)
    print(f"warm-up done in {time.perf_counter() - t:.1f}s", flush=True)


if __name__ == "__main__":
    warmup()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
