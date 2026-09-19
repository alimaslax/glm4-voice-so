"""Single-speaker Somali re-transcription: diarized speaker runs -> MAI-Transcribe-2 (language forced to Somali).

The original windows were sent with automatic language detection, which labelled Somali as ur/et/en/ar/...
Here every single-speaker stretch is re-sent on its own, with language=so and diarization on, so a clip
counts as single-speaker only if our diarization AND the new response both hear one voice.

  python single_speaker.py plan                     # speaker runs -> plan.jsonl + projected cost (no requests)
  python single_speaker.py transcribe [--limit 20]  # one request per clip, resumable, never re-sends a clip
  python single_speaker.py build                    # responses -> manifest.jsonl (+ Somali / single-voice checks)

Audio is cut in memory from $SO_DATA/processed/<channel>/<episode>/clean.flac and never written to disk.
Output ($SO_WORK/single_speaker/, text only): plan.jsonl, responses/<channel>/<episode>/<clip>.json,
manifest.jsonl, stats.json -- pushed to the bucket by `run.sh push_single_speaker`.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from common import PROCESSED_DIR, TRANSCRIPTS_DIR, WORK, log, read_jsonl, split_of, write_jsonl

L = log("single_speaker")
OUT = WORK / "single_speaker"
MODEL = "microsoft/mai-transcribe-2"
ENDPOINT = "https://openrouter.ai/api/v1/audio/transcriptions"
LANGUAGE = "so"
COST_PER_HOUR_USD = 0.10
EDGE = 0.15      # a word this close to a window edge may be cut off -> dropped
PAD = 0.2        # silence kept around a run, never crossing another speaker's word
SKIP_CHANNELS = {"omar"}   # already single-speaker (prepare_omar.py verifies him with ECAPA)

# Somali function words / very common forms (a text check: the API echoes language=so when we force it).
SOMALI_COMMON = set("""
waa iyo oo ku ka u la in ay uu aan ah ee ayaa waxaa waxay wuxuu waxaan si soo ma mid kale ha ahaa
laga loo lagu kaga kala sidaas sida hadda markii marka haddii laakiin ama aad baa bay buu yahay yihiin
ahayd tahay jira jiray jirta waxa isaga iyada iyaga annaga aniga adiga idinka dadka dal dalka ilaahay
waxbaa hal labo laba sax haa maya weeye waaye waayo sababtoo intaas kuwa kuwaas taas tan kan kaas
halkan halkaas maxaa sidee yaa goorma xaggee meeshaas wixii kii tii ugu aya kuma kama uma lama
""".split())
ENGLISH_COMMON = set("the and of to is that it you for was with this are have be not".split())
NON_LATIN = re.compile(r"[^\W\d_a-zA-ZÀ-ɏ]")      # any letter outside Latin (Arabic, Cyrillic, ...)
WORD = re.compile(r"[a-zA-Z'’À-ɏ]+")


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# ---------------------------------------------------------------- plan
def split_run(words, max_dur):
    """Split a same-speaker word run at its largest pauses until every piece is <= max_dur."""
    if words[-1]["end"] - words[0]["start"] <= max_dur or len(words) < 2:
        return [words]
    gaps = [(words[i + 1]["start"] - words[i]["end"], i) for i in range(len(words) - 1)]
    _, i = max(gaps)
    return split_run(words[:i + 1], max_dur) + split_run(words[i + 1:], max_dur)


def plan(a):
    rows, stats = [], Counter()
    for idx in sorted(TRANSCRIPTS_DIR.glob("*/*/diarized/transcripts.diarized.json")):
        ep_dir = idx.parent.parent
        ep_key = f"{ep_dir.parent.name}/{ep_dir.name}"
        if ep_dir.parent.name in SKIP_CHANNELS:
            stats["episode_skipped_channel"] += 1
            continue
        audio = PROCESSED_DIR / ep_key / "clean.flac"
        if not audio.exists():
            stats["episode_missing_audio"] += 1
            continue
        stats["episodes"] += 1
        for w in json.loads(idx.read_text()).get("windows", []):
            resp = idx.parent / w.get("response_file", f"{w['window_id']}.response.json")
            if not resp.exists():
                continue
            wdur, off = float(w["duration_seconds"]), float(w["source_start_seconds"])
            words = [x for x in (json.loads(resp.read_text()).get("words") or [])
                     if x.get("speaker") is not None and x.get("end", 0) > x.get("start", 0)]
            if not words:
                stats["window_no_speakers"] += 1
                continue
            stats["windows"] += 1
            runs = [[words[0]]]
            for x in words[1:]:
                (runs[-1].append(x) if x["speaker"] == runs[-1][-1]["speaker"] else runs.append([x]))
            k = 0
            for ri, run in enumerate(runs):
                prev_end = runs[ri - 1][-1]["end"] if ri else 0.0
                next_start = runs[ri + 1][0]["start"] if ri + 1 < len(runs) else wdur
                if run[0]["start"] < EDGE:
                    run = run[1:]            # may be cut by the window start
                if run and run[-1]["end"] > wdur - EDGE:
                    run = run[:-1]
                if not run:
                    continue
                pieces = split_run(run, a.max_dur)
                for pi, piece in enumerate(pieces):
                    lo = prev_end if pi == 0 else (pieces[pi - 1][-1]["end"] + piece[0]["start"]) / 2
                    hi = next_start if pi == len(pieces) - 1 else (piece[-1]["end"] + pieces[pi + 1][0]["start"]) / 2
                    s = max(lo, piece[0]["start"] - PAD, 0.0)
                    e = min(hi, piece[-1]["end"] + PAD, wdur)
                    if e - s < a.min_dur:
                        stats["too_short"] += 1
                        continue
                    rows.append(dict(
                        id=f"{ep_key}/{w['window_id']}/r{k:03d}", episode=ep_key, split=split_of(ep_key),
                        audio=f"{ep_key}/clean.flac", window_id=w["window_id"], speaker=piece[0]["speaker"],
                        start=round(off + s, 3), end=round(off + e, 3), dur=round(e - s, 3),
                        orig_text=" ".join(x["word"] for x in piece)))
                    k += 1
    write_jsonl(OUT / "plan.jsonl", rows)
    hours = sum(r["dur"] for r in rows) / 3600
    stats.update(clips=len(rows), hours=round(hours, 2), projected_cost_usd=round(hours * COST_PER_HOUR_USD, 2),
                 **{f"clips_{s}": sum(r["split"] == s for r in rows) for s in ("train", "val", "test")})
    atomic(OUT / "plan_stats.json", json.dumps(dict(stats), indent=2) + "\n")
    L.info("plan: %s", dict(stats))


# ---------------------------------------------------------------- transcribe
def wav_bytes(row):
    import soundfile as sf
    info = sf.info(str(PROCESSED_DIR / row["audio"]))
    x, sr = sf.read(str(PROCESSED_DIR / row["audio"]), start=int(row["start"] * info.samplerate),
                    stop=int(row["end"] * info.samplerate), dtype="int16", always_2d=True)
    buf = io.BytesIO()
    sf.write(buf, x.mean(axis=1).astype("int16") if x.shape[1] > 1 else x[:, 0], sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def request_once(wav, key):
    payload = {
        "model": MODEL,
        "input_audio": {"data": base64.b64encode(wav).decode("ascii"), "format": "wav"},
        "language": LANGUAGE,
        "response_format": "verbose_json",
        "timestamp_granularities": ["word"],
        "provider": {"options": {"azure": {
            "diarization": {"enabled": True},
            "enhancedMode": {"modelOptions": {"transcribeStyle": "verbatim"}},
        }}},
    }
    req = Request(ENDPOINT, data=json.dumps(payload).encode(), method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/alimaslax/glm4-voice-so", "X-Title": "Somali single-speaker ASR"})
    with urlopen(req, timeout=180) as r:
        out = json.load(r)
        out["_generation_id"] = r.headers.get("X-Generation-Id")
        return out


def paths(row):
    base = OUT / "responses" / row["episode"] / f"{row['window_id']}_{row['id'].rsplit('/', 1)[1]}"
    return base.with_suffix(".json"), base.with_suffix(".request.json")


def transcribe(a):
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set (put OPEN_ROUTER / OPENROUTER_API_KEY in .env)")
    rows = read_jsonl(OUT / "plan.jsonl")
    todo, stats = [], Counter()
    for r in rows:
        resp, marker = paths(r)
        if resp.exists():
            stats["already_done"] += 1
        elif marker.exists() and not (a.retry_failed and json.loads(marker.read_text()).get("status") != "dispatching"):
            stats["skipped_marker"] += 1     # dispatched before without a saved response: never re-send
        else:
            todo.append(r)
    if a.limit:
        todo = sorted(todo, key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest())[:a.limit]   # stable spread
    hours = sum(r["dur"] for r in todo) / 3600
    L.info("%s; sending %d clips, %.2f h, ~$%.2f, %d workers", dict(stats), len(todo), hours,
           hours * COST_PER_HOUR_USD, a.workers)
    lock, done = threading.Lock(), Counter()
    t0 = time.time()

    def one(r):
        resp, marker = paths(r)
        wav = wav_bytes(r)
        for attempt in range(6):
            atomic(marker, json.dumps(dict(id=r["id"], status="dispatching", at=now())) + "\n")
            try:
                out = request_once(wav, key)
                atomic(resp, json.dumps(out, ensure_ascii=False, indent=1) + "\n")
                atomic(marker, json.dumps(dict(id=r["id"], status="saved", at=now(),
                                               generation_id=out.get("_generation_id"))) + "\n")
                return "ok", r["dur"]
            except HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:500]
                if e.code == 429 or (e.code in (502, 503) and "rate" in body.lower()):
                    atomic(marker, json.dumps(dict(id=r["id"], status="rate_limited", at=now())) + "\n")
                    time.sleep(min(60, 5 * 2 ** attempt))    # rejected before processing -> safe to resend
                    continue
                atomic(marker, json.dumps(dict(id=r["id"], status=f"http_{e.code}", body=body, at=now())) + "\n")
                return f"http_{e.code}", 0
            except (URLError, TimeoutError, OSError) as e:
                atomic(marker, json.dumps(dict(id=r["id"], status="network_error", error=str(e), at=now())) + "\n")
                return "network_error", 0
        return "rate_limited", 0

    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, r) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            status, dur = f.result()
            with lock:
                done[status] += 1
                done["seconds"] += dur
                if i % 200 == 0 or i == len(todo):
                    el = time.time() - t0
                    L.info("%d/%d  %s  %.2f h sent  $%.2f  %.1f clips/s  eta %.0f min", i, len(todo),
                           {k: v for k, v in done.items() if k != "seconds"}, done["seconds"] / 3600,
                           done["seconds"] / 3600 * COST_PER_HOUR_USD, i / el, (len(todo) - i) / (i / el) / 60)
    L.info("transcribe finished: %s", dict(done))


# ---------------------------------------------------------------- build
def somali_check(text):
    toks = [t.lower().replace("’", "'") for t in WORD.findall(text)]
    letters = [c for c in text if c.isalpha()]
    non_latin = sum(bool(NON_LATIN.match(c)) for c in letters) / max(1, len(letters))
    so = sum(t in SOMALI_COMMON for t in toks) / max(1, len(toks))
    en = sum(t in ENGLISH_COMMON for t in toks) / max(1, len(toks))
    ok = bool(toks) and non_latin < 0.02 and en < 0.3 and (so >= 0.12 or len(toks) < 6)
    return ok, dict(tokens=len(toks), somali_ratio=round(so, 3), english_ratio=round(en, 3),
                    non_latin_ratio=round(non_latin, 3))


def build(a):
    rows, stats, out = read_jsonl(OUT / "plan.jsonl"), Counter(), []
    for r in rows:
        resp, _ = paths(r)
        if not resp.exists():
            stats["no_response"] += 1
            continue
        d = json.loads(resp.read_text())
        text = " ".join((d.get("text") or "").split())
        speakers = sorted({x.get("speaker") for k in ("segments", "words") for x in (d.get(k) or [])
                           if x.get("speaker") is not None}, key=str)
        is_so, so_info = somali_check(text)
        cps = len(text) / r["dur"]
        single = len(speakers) <= 1
        ok = bool(text) and is_so and single and 3 <= cps <= 30
        stats["responses"] += 1
        stats["ok"] += ok
        stats["not_somali"] += not is_so
        stats["multi_speaker"] += not single
        stats["bad_rate"] += not (3 <= cps <= 30)
        stats["hours_ok"] += r["dur"] / 3600 if ok else 0
        stats[f"ok_{r['split']}"] += ok
        out.append(dict(r, text=text, language=d.get("language"), response_speakers=len(speakers),
                        single_speaker=single, somali=is_so, somali_check=so_info, chars_per_sec=round(cps, 2),
                        ok=ok, generation_id=d.get("_generation_id"), usage=d.get("usage")))
    write_jsonl(OUT / "manifest.jsonl", out)
    stats = {k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()}
    stats.update(model=MODEL, language=LANGUAGE, built_at=now())
    atomic(OUT / "stats.json", json.dumps(stats, indent=2) + "\n")
    L.info("build: %s", stats)
    for r in [x for x in out if x["ok"]][:5] + [x for x in out if not x["somali"]][:3]:
        L.info("%s %s | %s", "OK " if r["ok"] else "BAD", r["id"], r["text"][:120])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["plan", "transcribe", "build"])
    p.add_argument("--min-dur", type=float, default=2.0)
    p.add_argument("--max-dur", type=float, default=20.0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="send only this many (pilot)")
    p.add_argument("--retry-failed", action="store_true", help="re-send clips whose request got an HTTP/network "
                   "error; clips left 'dispatching' (outcome unknown) are never re-sent")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dict(plan=plan, transcribe=transcribe, build=build)[a.step](a)


if __name__ == "__main__":
    main()
