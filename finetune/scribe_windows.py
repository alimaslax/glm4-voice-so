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
If $SO_WORK/scribe/window_status.jsonl is present, every window recorded there is also treated as attempted.
This preserves empty and rejected results when resuming from a compact bucket status file.
"""
import argparse
import io
import json
import os
import threading
import time
import uuid
from collections import Counter, deque
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


def attempted_from_status():
    path = OUT / "window_status.jsonl"
    attempted = set()
    if not path.exists():
        return attempted
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                attempted.add((row["channel"], row["episode"], row.get("window", row.get("window_id"))))
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"invalid {path}:{line_no}: {e}") from e
    return attempted


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


def request_once(wav, key, language=LANGUAGE):
    fields = dict(model_id=MODEL, diarize="true", tag_audio_events="false", timestamps_granularity="word")
    if language:                                   # None -> let Scribe detect the language
        fields["language_code"] = language
    b = uuid.uuid4().hex
    body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items())
    body += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="window.wav"\r\n'
             f'Content-Type: audio/wav\r\n\r\n').encode() + wav + f"\r\n--{b}--\r\n".encode()
    req = Request(ENDPOINT, data=body, method="POST",
                  headers={"xi-api-key": key, "Content-Type": f"multipart/form-data; boundary={b}"})
    with urlopen(req, timeout=300) as r:
        return json.load(r)


def normalize(raw, language=LANGUAGE):
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
                _model=MODEL, _language_forced=language)


def episode_indexes_round_robin():
    """Yield episode indexes deterministically, interleaved across channel folders."""
    by_channel = {}
    for idx in sorted(TRANSCRIPTS_DIR.glob("*/*/diarized/transcripts.diarized.json")):
        channel = idx.parents[2].name
        if channel in SKIP_CHANNELS:
            continue
        by_channel.setdefault(channel, deque()).append(idx)
    channels = sorted(by_channel)
    while channels:
        remaining = []
        for channel in channels:
            indexes = by_channel[channel]
            yield indexes.popleft()
            if indexes:
                remaining.append(channel)
        channels = remaining


def episodes():
    for idx in episode_indexes_round_robin():
        ep = idx.parent.parent
        key = f"{ep.parent.name}/{ep.name}"
        audio = PROCESSED_DIR / key / "clean.flac"
        if audio.exists():
            yield key, audio, json.loads(idx.read_text())


WIN_KEYS = ("source_start_seconds", "source_end_seconds", "duration_seconds", "source_start_sample",
            "source_end_sample", "num_samples", "sampling_rate")
SS_GAP, SS_MIN, SS_MAX, SS_PAD = 1.0, 3.0, 30.0, 0.2   # single-speaker windows from the episode diarization


def single_windows(ep_dir, taken, sr):
    """New one-speaker windows from <episode>/diarization.json: consecutive turns of one speaker (gap <= SS_GAP,
    <= SS_MAX long) that no other speaker's turn overlaps, padded by SS_PAD without touching a neighbour, and not
    overlapping any existing (duplex) window of the episode, which is already transcribed or queued."""
    p = ep_dir / "diarization.json"
    if not p.exists():
        return []
    turns = sorted((t for t in json.loads(p.read_text()).get("turns", []) if t["end"] > t["start"]),
                   key=lambda t: t["start"])
    out, i = [], 0
    while i < len(turns):
        spk, s, e, j = turns[i]["speaker"], turns[i]["start"], turns[i]["end"], i + 1
        while (j < len(turns) and turns[j]["speaker"] == spk and turns[j]["start"] - e <= SS_GAP
               and turns[j]["end"] - s <= SS_MAX):
            e = max(e, turns[j]["end"]); j += 1
        i = j
        others = [t for t in turns if t["speaker"] != spk]
        if any(t["start"] < e and t["end"] > s for t in others):
            continue
        lo = max([t["end"] for t in others if t["end"] <= s] + [0.0])
        hi = min([t["start"] for t in others if t["start"] >= e] + [float("inf")])
        s, e = max(s - SS_PAD, lo, 0.0), min(e + SS_PAD, hi)
        if e - s < SS_MIN or any(a < e and b > s for a, b in taken):
            continue
        out.append(dict(window_id=f"single_{len(out) + 1:04d}", kind="single", offset_seconds=round(s, 3),
                        source_start_seconds=round(s, 3), source_end_seconds=round(e, 3),
                        duration_seconds=round(e - s, 3), source_start_sample=int(s * sr),
                        source_end_sample=int(e * sr), num_samples=int(e * sr) - int(s * sr), sampling_rate=sr,
                        speakers_diarization=[spk]))
    return out


def processed_episodes(single):
    """Episodes straight from so-duplex-processed: <channel>/<episode>/windows/windows.json (+ single windows)."""
    for wj in sorted(PROCESSED_DIR.glob("*/*/windows/windows.json")):
        ep = wj.parent.parent
        if ep.parent.name in SKIP_CHANNELS or not (ep / "clean.flac").exists():
            continue
        try:
            m = json.loads(wj.read_text())
        except json.JSONDecodeError:
            continue
        sr = int(m.get("source_sampling_rate", 24000))
        wins = [dict({k: w[k] for k in WIN_KEYS if k in w}, window_id=w["id"], kind="duplex",
                     offset_seconds=w.get("offset", w["source_start_seconds"])) for w in m.get("windows", [])]
        if single:
            wins += single_windows(ep, [(w["source_start_seconds"], w["source_end_seconds"]) for w in wins], sr)
        yield f"{ep.parent.name}/{ep.name}", ep / "clean.flac", dict(windows=wins)


def round_robin(items, key):
    """Interleave items by key (first of each key, then second of each, ...), keeping order within a key."""
    groups = {}
    for it in items:
        groups.setdefault(key(it), deque()).append(it)
    order, out = sorted(groups), []
    while order:
        order = [g for g in order if groups[g]]
        out += [groups[g].popleft() for g in order]
    return out


def status_row(k, w, raw):
    """One window_status.jsonl row: what came back for this window (same fields as the bucket file)."""
    from single_speaker import somali_check
    text = (raw.get("text") or "").strip()
    is_so, info = somali_check(text)
    words = [x for x in raw.get("words") or [] if x.get("type") == "word"]
    return dict(channel=k.split("/")[0], episode=k.split("/", 1)[1], window=w["window_id"],
                kind=w.get("kind", "duplex"), start=w["source_start_seconds"], duration=w["duration_seconds"],
                status="empty" if not text else ("somali" if is_so else "not_somali_check"),
                language=raw.get("language_code"), language_forced=LANGUAGE, chars=len(text), words=len(words),
                speakers=len({x.get("speaker_id") for x in words if x.get("speaker_id") is not None}),
                somali_check=info, text_preview=text[:120], at=now())


def write_index(key, meta):
    out = OUT / key / "diarized"
    wins = []
    for w in meta.get("windows", []):
        resp = out / f"{w['window_id']}.response.json"
        if not resp.exists():
            continue
        d = json.loads(resp.read_text())
        speakers = sorted({x["speaker"] for x in d["words"] if x["speaker"] is not None})
        wins.append(dict({k: w[k] for k in ("window_id", "kind", "offset_seconds", "source_start_seconds", "source_end_seconds",
                                            "duration_seconds", "source_start_sample", "source_end_sample",
                                            "num_samples", "sampling_rate") if k in w},
                         diarization_status="ok" if speakers else "no_speaker_labels", speakers=speakers,
                         language=d.get("language"), response_file=resp.name, raw_file=f"{w['window_id']}.scribe.json"))
    # Keep windows an earlier run indexed for this episode (e.g. the relabel of the MAI windows): rewriting the
    # index with only this run's windows would orphan their responses.
    idx = out / "transcripts.diarized.json"
    if idx.exists():
        have = {w["window_id"] for w in wins}
        wins = [w for w in json.loads(idx.read_text()).get("windows", [])
                if w["window_id"] not in have and (out / w.get("response_file", "")).is_file()] + wins
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
    p.add_argument("--source", choices=["transcripts", "processed"], default="transcripts",
                   help="transcripts = windows of the MAI index; processed = every windows/windows.json in "
                        "so-duplex-processed (episodes never transcribed)")
    p.add_argument("--single", action="store_true", help="with --source processed: also new single-speaker windows")
    p.add_argument("--plan-only", action="store_true", help="print what would be sent (per channel/kind/hours), send nothing")
    a = p.parse_args()
    # Keys are used in order: when one runs out of credits, switch to the next (ELEVENLABS_API_KEY2, ...).
    keys = [k for k in (
        os.environ.get("ELEVENLABS_API_KEY"),
        os.environ.get("ELEVENLABS_API_KEY2"),
        os.environ.get("ELEVENLABS_API_KEY3"),
    ) if k]
    if not keys:
        raise SystemExit("ELEVENLABS_API_KEY not set (put it in .env)")
    cur = [0]

    eps = list(processed_episodes(a.single) if a.source == "processed" else episodes())
    attempted = attempted_from_status()
    todo, stats = [], Counter()
    for k, audio, meta in eps:
        channel, episode = k.split("/", 1)
        for w in meta.get("windows", []):
            d = OUT / k / "diarized"
            resp, marker = d / f"{w['window_id']}.response.json", d / f"{w['window_id']}.request.json"
            if resp.exists():
                stats["already_done"] += 1
            elif (channel, episode, w["window_id"]) in attempted:
                stats["skipped_status"] += 1
            elif marker.exists() and not (a.retry_failed and json.loads(marker.read_text()).get("status") != "dispatching"):
                stats["skipped_marker"] += 1
            else:
                todo.append((k, audio, w))
    # Round-robin over channels only: channels take turns, but inside a channel one episode is finished before the
    # next (episodes already started first), duplex and single windows alternating. If credits run out, every
    # channel got an equal share, and the work sits in as few episodes as possible, so a training machine needs
    # few clean.flac downloads (~50 MB each) to use it.
    for _, _, meta in eps:
        n = Counter()
        for w in meta["windows"]:
            kind = w.get("kind", "duplex")
            w["_order"] = (n[kind], kind != "duplex"); n[kind] += 1
    per_channel = {}
    for t in sorted(todo, key=lambda t: t[2]["_order"]):
        per_channel.setdefault(t[0].split("/")[0], []).append(t)
    started = {(r["channel"], r["episode"]) for r in map(json.loads, (OUT / "window_status.jsonl").open())} \
        if (OUT / "window_status.jsonl").exists() else set()
    todo = round_robin([t for c in sorted(per_channel) for t in sorted(
        per_channel[c], key=lambda t: (tuple(t[0].split("/", 1)) not in started, t[0], t[2]["_order"]))],
        key=lambda t: t[0].split("/")[0])
    if a.limit:
        todo = todo[:a.limit]
    L.info("by channel: %s; by kind: %s", dict(Counter(t[0].split("/")[0] for t in todo)),
           dict(Counter(t[2].get("kind", "duplex") for t in todo)))
    hours = sum(float(w["duration_seconds"]) for _, _, w in todo) / 3600
    L.info("%d episodes; %s; sending %d windows (%.1f h), %d workers", len(eps), dict(stats), len(todo), hours, a.workers)
    if a.plan_only:
        hrs = Counter()
        for k, _, w in todo:
            hrs[(k.split("/")[0], w.get("kind", "duplex"))] += float(w["duration_seconds"]) / 3600
        for (c, kind), h in sorted(hrs.items()):
            L.info("plan %-14s %-6s %7.1f h", c, kind, h)
        L.info("first 12: %s", [f"{k.split('/')[0]}:{w['window_id']}" for k, _, w in todo[:12]])
        return

    done, lock, t0 = Counter(), threading.Lock(), time.time()

    def one(k, audio, w):
        if STOP.is_set():
            return "stopped", 0
        d = OUT / k / "diarized"
        resp, raw_p, marker = (d / f"{w['window_id']}.response.json", d / f"{w['window_id']}.scribe.json",
                               d / f"{w['window_id']}.request.json")
        try:
            wav = wav_bytes(audio, float(w["source_start_seconds"]), float(w["duration_seconds"]))
        except Exception as e:                        # e.g. clean.flac still downloading / truncated
            L.warning("unreadable audio %s %s: %s", k, w["window_id"], str(e)[:120])
            return "bad_audio", 0
        for attempt in range(8):
            if STOP.is_set():
                return "stopped", 0
            atomic(marker, json.dumps(dict(window=w["window_id"], status="dispatching", at=now())) + "\n")
            ki = cur[0]
            try:
                raw = request_once(wav, keys[ki])
                atomic(raw_p, json.dumps(raw, ensure_ascii=False) + "\n")
                atomic(resp, json.dumps(normalize(raw), ensure_ascii=False, indent=1) + "\n")
                atomic(marker, json.dumps(dict(window=w["window_id"], status="saved", at=now())) + "\n")
                row = json.dumps(status_row(k, w, raw), ensure_ascii=False) + "\n"
                with lock, (OUT / "window_status.jsonl").open("a") as f:
                    f.write(row)
                return "ok", float(w["duration_seconds"])
            except HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:500]
                if e.code == 429 or "concurrent" in body or "rate" in body.lower():
                    atomic(marker, json.dumps(dict(window=w["window_id"], status="rate_limited", at=now())) + "\n")
                    time.sleep(min(60, 3 * 2 ** attempt))     # rejected before processing -> safe to resend
                    continue
                atomic(marker, json.dumps(dict(window=w["window_id"], status=f"http_{e.code}", body=body, at=now())) + "\n")
                if "quota" in body or "credits" in body or e.code in (401, 402):
                    with lock:
                        if cur[0] == ki and ki + 1 < len(keys):
                            cur[0] = ki + 1
                            L.warning("key %d out (%s); switching to key %d", ki + 1, body[:120], ki + 2)
                    if cur[0] != ki:
                        continue                                  # retry this window on the next key
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
