"""Re-label the original ~42 s windows with ElevenLabs Scribe v2 (language forced to Somali, diarization on).

MAI-Transcribe-2 (the original labels) has no Somali and auto-detects ur/et/en/ar/...; Scribe v2 with
language_code=som returned Somali on clips where MAI produced gibberish (outputs/muse/README.md).

  python scribe_windows.py [--workers 10] [--limit N]

Input : window list + offsets from $SO_DATA/transcripts/*/*/diarized/transcripts.diarized.json (MAI run),
        audio cut in memory from $SO_DATA/processed/<channel>/<episode>/clean.flac (never written to disk)
Output: $SO_WORK/scribe/<channel>/<episode>/diarized/
          <window>.scribe.json    raw ElevenLabs response
          <window>.response.json  same schema as the MAI files (text, segments[speaker], words[speaker])
          transcripts.diarized.json  per-episode index, same schema -> prepare_somali.py can read either
        pushed to hf://buckets/lewenberg/so-duplex-transcripts/scribe/ by `run.sh scribe_push`.
Resumable; a clip that got a response is never re-sent. Stops cleanly if the account runs out of credits.
"""
import argparse
import io
import json
import os
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from common import PROCESSED_DIR, TRANSCRIPTS_DIR, WORK, log

L = log("scribe_windows")
OUT = WORK / "scribe"
MODEL = "scribe_v2"
LANGUAGE = "som"
ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"
SKIP_CHANNELS = {"omar"}          # Omar's transcripts already come from ElevenLabs
SEG_GAP = 0.7                     # new segment after a pause this long ...
SEG_MAX = 15.0                    # ... or at sentence end, and never longer than this
STOP = threading.Event()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def wav_bytes(audio, start, dur):
    import soundfile as sf
    info = sf.info(str(audio))
    x, sr = sf.read(str(audio), start=int(start * info.samplerate), stop=int((start + dur) * info.samplerate),
                    dtype="int16", always_2d=True)
    buf = io.BytesIO()
    sf.write(buf, x.mean(axis=1).astype("int16") if x.shape[1] > 1 else x[:, 0], sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def request_once(wav, key):
    fields = dict(model_id=MODEL, language_code=LANGUAGE, diarize="true", tag_audio_events="false",
                  timestamps_granularity="word")
    b = uuid.uuid4().hex
    body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items())
    body += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="window.wav"\r\n'
             f'Content-Type: audio/wav\r\n\r\n').encode() + wav + f"\r\n--{b}--\r\n".encode()
    req = Request(ENDPOINT, data=body, method="POST",
                  headers={"xi-api-key": key, "Content-Type": f"multipart/form-data; boundary={b}"})
    with urlopen(req, timeout=300) as r:
        return json.load(r)


def normalize(raw):
    """ElevenLabs words -> the MAI schema (words/segments with integer speakers) used by prepare_somali.py."""
    spk_ids, words = {}, []
    for w in raw.get("words") or []:
        if w.get("type") != "word" or not (w.get("text") or "").strip():
            continue
        s = w.get("speaker_id")
        spk = spk_ids.setdefault(s, len(spk_ids)) if s is not None else None
        words.append(dict(word=w["text"].strip(), start=float(w["start"]), end=float(w["end"]), speaker=spk))
    segs = []
    for w in words:
        cur = segs[-1] if segs else None
        if (cur and cur["speaker"] == w["speaker"] and w["start"] - cur["end"] <= SEG_GAP
                and not cur["text"].rstrip().endswith((".", "?", "!")) and w["end"] - cur["start"] <= SEG_MAX):
            cur["text"] += " " + w["word"]
            cur["end"] = w["end"]
        else:
            segs.append(dict(id=len(segs), start=w["start"], end=w["end"], text=w["word"], speaker=w["speaker"]))
    return dict(text=(raw.get("text") or "").strip(), language=raw.get("language_code"),
                language_probability=raw.get("language_probability"), segments=segs, words=words,
                _model=MODEL, _language_forced=LANGUAGE)


def episodes():
    for idx in sorted(TRANSCRIPTS_DIR.glob("*/*/diarized/transcripts.diarized.json")):
        ep = idx.parent.parent
        if ep.parent.name in SKIP_CHANNELS:
            continue
        key = f"{ep.parent.name}/{ep.name}"
        audio = PROCESSED_DIR / key / "clean.flac"
        if audio.exists():
            yield key, audio, json.loads(idx.read_text())


def write_index(key, meta):
    out = OUT / key / "diarized"
    wins = []
    for w in meta.get("windows", []):
        resp = out / f"{w['window_id']}.response.json"
        if not resp.exists():
            continue
        d = json.loads(resp.read_text())
        speakers = sorted({x["speaker"] for x in d["words"] if x["speaker"] is not None})
        wins.append(dict({k: w[k] for k in ("window_id", "offset_seconds", "source_start_seconds", "source_end_seconds",
                                            "duration_seconds", "source_start_sample", "source_end_sample",
                                            "num_samples", "sampling_rate") if k in w},
                         diarization_status="ok" if speakers else "no_speaker_labels", speakers=speakers,
                         language=d.get("language"), response_file=resp.name, raw_file=f"{w['window_id']}.scribe.json"))
    atomic(out / "transcripts.diarized.json", json.dumps(dict(
        recording=key, model=f"elevenlabs/{MODEL}", language=LANGUAGE, diarization=True, windows=wins),
        ensure_ascii=False, indent=2) + "\n")
    atomic(out / "transcript.diarized.txt", "\n\n".join(
        f"[{w['window_id']}]\n" + "\n".join(f"[{s['start']:.3f}-{s['end']:.3f}] {s['speaker']}: {s['text']}"
                                             for s in json.loads((out / w["response_file"]).read_text())["segments"])
        for w in wins) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--retry-failed", action="store_true")
    a = p.parse_args()
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise SystemExit("ELEVENLABS_API_KEY not set (put it in .env)")

    eps = list(episodes())
    todo, stats = [], Counter()
    for k, audio, meta in eps:
        for w in meta.get("windows", []):
            d = OUT / k / "diarized"
            resp, marker = d / f"{w['window_id']}.response.json", d / f"{w['window_id']}.request.json"
            if resp.exists():
                stats["already_done"] += 1
            elif marker.exists() and not (a.retry_failed and json.loads(marker.read_text()).get("status") != "dispatching"):
                stats["skipped_marker"] += 1
            else:
                todo.append((k, audio, w))
    if a.limit:
        todo = todo[:a.limit]
    hours = sum(float(w["duration_seconds"]) for _, _, w in todo) / 3600
    L.info("%d episodes; %s; sending %d windows (%.1f h), %d workers", len(eps), dict(stats), len(todo), hours, a.workers)

    done, lock, t0 = Counter(), threading.Lock(), time.time()

    def one(k, audio, w):
        if STOP.is_set():
            return "stopped", 0
        d = OUT / k / "diarized"
        resp, raw_p, marker = (d / f"{w['window_id']}.response.json", d / f"{w['window_id']}.scribe.json",
                               d / f"{w['window_id']}.request.json")
        wav = wav_bytes(audio, float(w["source_start_seconds"]), float(w["duration_seconds"]))
        for attempt in range(8):
            if STOP.is_set():
                return "stopped", 0
            atomic(marker, json.dumps(dict(window=w["window_id"], status="dispatching", at=now())) + "\n")
            try:
                raw = request_once(wav, key)
                atomic(raw_p, json.dumps(raw, ensure_ascii=False) + "\n")
                atomic(resp, json.dumps(normalize(raw), ensure_ascii=False, indent=1) + "\n")
                atomic(marker, json.dumps(dict(window=w["window_id"], status="saved", at=now())) + "\n")
                return "ok", float(w["duration_seconds"])
            except HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:500]
                if e.code == 429 or "concurrent" in body or "rate" in body.lower():
                    atomic(marker, json.dumps(dict(window=w["window_id"], status="rate_limited", at=now())) + "\n")
                    time.sleep(min(60, 3 * 2 ** attempt))     # rejected before processing -> safe to resend
                    continue
                atomic(marker, json.dumps(dict(window=w["window_id"], status=f"http_{e.code}", body=body, at=now())) + "\n")
                if "quota" in body or "credits" in body or e.code in (401, 402):
                    STOP.set()
                    L.error("stopping: %s", body[:300])
                return f"http_{e.code}", 0
            except (URLError, TimeoutError, OSError) as e:
                atomic(marker, json.dumps(dict(window=w["window_id"], status="network_error", error=str(e), at=now())) + "\n")
                return "network_error", 0
        return "rate_limited", 0

    by_ep = Counter(k for k, _, _ in todo)
    metas = {k: meta for k, _, meta in eps}
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, *t): t[0] for t in todo}
        for i, f in enumerate(as_completed(futs), 1):
            status, dur = f.result()
            k = futs[f]
            with lock:
                done[status] += 1
                done["seconds"] += dur
                by_ep[k] -= 1
                if by_ep[k] == 0:
                    write_index(k, metas[k])
                if i % 100 == 0 or i == len(todo):
                    el = time.time() - t0
                    L.info("%d/%d %s  %.1f h done  %.2f win/s  eta %.0f min", i, len(todo),
                           {s: n for s, n in done.items() if s != "seconds"}, done["seconds"] / 3600, i / el,
                           (len(todo) - i) / (i / el) / 60)
    for k in {k for k, _, _ in todo}:
        write_index(k, metas[k])
    L.info("finished: %s", dict(done))


if __name__ == "__main__":
    main()
