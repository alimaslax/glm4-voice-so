"""Resend the windows Scribe returned EMPTY (with language forced to som) once more, language auto-detected.

  python scribe_retry_empty.py [--probe 20] [--min-good 0.5] [--workers 8]

Probe first: send --probe windows (the ones MAI had the most text for). Only if at least --min-good of them
come back non-empty AND Somali (Scribe language 'som' + the Somali word check) are the rest sent.
A result replaces the empty one only when it passes the same check; the empty raw reply is kept as
<window>.scribe.empty.json. Every attempt is logged in $SO_WORK/scribe_retry_empty.jsonl (never re-sent).
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from common import TRANSCRIPTS_DIR, WORK, log
from scribe_windows import OUT, atomic, episodes, normalize, request_once, wav_bytes, write_index
from single_speaker import somali_check

L = log("scribe_retry_empty")
LOG = WORK / "scribe_retry_empty.jsonl"


def mai_chars(key, wid):
    p = TRANSCRIPTS_DIR / key / "diarized" / f"{wid}.response.json"
    if not p.exists():
        return 0
    return sum(len(s.get("text") or "") for s in json.loads(p.read_text()).get("segments", []))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", type=int, default=20)
    p.add_argument("--min-good", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    keys = [k for k in (os.environ.get("ELEVENLABS_API_KEY2"), os.environ.get("ELEVENLABS_API_KEY")) if k]

    tried = {json.loads(l)["window"] for l in LOG.read_text().splitlines()} if LOG.exists() else set()
    todo, metas = [], {}
    for k, audio, meta in episodes():
        metas[k] = meta
        d = OUT / k / "diarized"
        for w in meta.get("windows", []):
            raw = d / f"{w['window_id']}.scribe.json"
            wid = f"{k}/{w['window_id']}"
            if raw.exists() and not (json.loads(raw.read_text()).get("text") or "").strip() and wid not in tried:
                todo.append((mai_chars(k, w["window_id"]), k, audio, w))
    todo.sort(key=lambda t: -t[0])                  # most likely to hold speech first
    L.info("%d empty windows to resend (%.1f h)", len(todo), sum(float(t[3]["duration_seconds"]) for t in todo) / 3600)

    def one(t):
        _, k, audio, w = t
        wav = wav_bytes(audio, float(w["source_start_seconds"]), float(w["duration_seconds"]))
        err = None
        for key in keys:
            try:
                raw = request_once(wav, key, language=None)
                break
            except Exception as e:                  # out of credits on this key -> next key
                err = str(e)[:200]
        else:
            return dict(window=f"{k}/{w['window_id']}", status="error", error=err)
        text = (raw.get("text") or "").strip()
        is_so, info = somali_check(text)
        good = bool(text) and raw.get("language_code") == "som" and is_so
        d = OUT / k / "diarized"
        if good:
            old = d / f"{w['window_id']}.scribe.json"
            atomic(d / f"{w['window_id']}.scribe.empty.json", old.read_text())
            atomic(old, json.dumps(raw, ensure_ascii=False) + "\n")
            atomic(d / f"{w['window_id']}.response.json",
                   json.dumps(normalize(raw, language=None), ensure_ascii=False, indent=1) + "\n")
        return dict(window=f"{k}/{w['window_id']}", status="somali" if good else ("empty" if not text else "not_somali"),
                    language=raw.get("language_code"), text=text[:200], check=info)

    def run(batch):
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(one, batch))
        with LOG.open("a") as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for k in {t[1] for t in batch}:
            write_index(k, metas[k])
        return res

    probe = run(todo[:a.probe])
    good = sum(r["status"] == "somali" for r in probe) / max(1, len(probe))
    for r in probe:
        L.info("probe %-11s %-4s %s | %s", r["status"], r.get("language"), r["window"], r.get("text", r.get("error", ""))[:100])
    L.info("probe: %.0f%% Somali (need %.0f%%)", 100 * good, 100 * a.min_good)
    if good < a.min_good:
        L.info("stopping: not sending the other %d windows", len(todo) - len(probe))
        return
    rest = run(todo[a.probe:])
    from collections import Counter
    L.info("rest: %s", dict(Counter(r["status"] for r in rest)))


if __name__ == "__main__":
    main()
