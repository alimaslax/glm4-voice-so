"""Track A data prep: transcripts bucket + clean.flac -> cleaned segment/turn/pair manifests.

Input : $SO_DATA/transcripts/<channel>/<episode>/diarized/{transcripts.diarized.json, window_XXXX.response.json}
        $SO_DATA/processed/<channel>/<episode>/clean.flac   (24 kHz mono)
Output: $SO_WORK/somali/manifest.jsonl   one row per clip (segment or merged turn), absolute times in clean.flac
        $SO_WORK/somali/pairs.jsonl      dialogue pairs (turn A -> next turn B, different speakers)
        $SO_WORK/somali/stats.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from common import PROCESSED_DIR, TRANSCRIPTS_DIR, WORK, log, split_of, write_jsonl

L = log("prepare_somali")
LETTERS = re.compile(r"[A-Za-z]")


def overlap(a, b):
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def dedupe(segs):
    """The transcriber sometimes re-emits a time span with corrected text. Keep the later version."""
    kept = []
    for s in segs:
        dur = s["end"] - s["start"]
        kept = [k for k in kept if overlap(k, s) <= 0.5 * min(dur, k["end"] - k["start"])]
        kept.append(s)
    return sorted(kept, key=lambda s: s["start"])


def clean_text(t):
    return re.sub(r"\s+", " ", t or "").strip()


def ok_clip(text, dur, a):
    if not (a.min_dur <= dur <= a.max_dur) or not LETTERS.search(text):
        return False
    cps = len(text) / dur
    return a.min_cps <= cps <= a.max_cps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-dur", type=float, default=0.8)
    p.add_argument("--max-dur", type=float, default=20.0)
    p.add_argument("--min-cps", type=float, default=3.0)
    p.add_argument("--max-cps", type=float, default=30.0)
    p.add_argument("--merge-gap", type=float, default=1.0, help="merge same-speaker segments closer than this")
    p.add_argument("--pair-gap", type=float, default=3.0, help="max silence between turn A and reply B")
    a = p.parse_args()

    clips, pairs, stats = [], [], Counter()
    for idx in sorted(TRANSCRIPTS_DIR.glob("*/*/diarized/transcripts.diarized.json")):
        ep_dir = idx.parent.parent
        channel, episode = ep_dir.parent.name, ep_dir.name
        ep_key = f"{channel}/{episode}"
        audio = PROCESSED_DIR / channel / episode / "clean.flac"
        if not audio.exists():
            stats["episode_missing_audio"] += 1
            continue
        split = split_of(ep_key)
        stats["episodes"] += 1
        meta = json.loads(idx.read_text())
        for w in meta.get("windows", []):
            if w.get("diarization_status") != "ok":
                stats["window_not_ok"] += 1
                continue
            resp = idx.parent / w.get("response_file", f"{w['window_id']}.response.json")
            if not resp.exists():
                stats["window_missing_response"] += 1
                continue
            off = float(w["offset_seconds"])
            raw = [dict(start=float(s["start"]), end=float(s["end"]), text=clean_text(s.get("text")),
                        speaker=s.get("speaker", 0))
                   for s in json.loads(resp.read_text()).get("segments", []) if s.get("end", 0) > s.get("start", 0)]
            segs = dedupe(raw)
            stats["segments_raw"] += len(raw)
            stats["segments_dedup_dropped"] += len(raw) - len(segs)
            stats["windows"] += 1

            # single segments -> ASR / TTS clips
            for i, s in enumerate(segs):
                dur = s["end"] - s["start"]
                if ok_clip(s["text"], dur, a):
                    clips.append(dict(id=f"{ep_key}/{w['window_id']}/s{i:03d}", kind="segment", split=split,
                                      episode=ep_key, audio=str(audio.relative_to(PROCESSED_DIR)),
                                      start=round(off + s["start"], 3), end=round(off + s["end"], 3),
                                      speaker=s["speaker"], text=s["text"]))
                    stats["segments_kept"] += 1
                else:
                    stats["segments_filtered"] += 1

            # merged speaker turns -> dialogue pairs
            turns = []
            for s in segs:
                t = turns[-1] if turns else None
                if (t and t["speaker"] == s["speaker"] and s["start"] - t["end"] <= a.merge_gap
                        and s["end"] - t["start"] <= a.max_dur):
                    t["end"] = max(t["end"], s["end"]); t["text"] += " " + s["text"]
                else:
                    turns.append(dict(s))
            for i in range(len(turns) - 1):
                x, y = turns[i], turns[i + 1]
                if x["speaker"] == y["speaker"] or y["start"] - x["end"] > a.pair_gap:
                    continue
                if not (ok_clip(x["text"], x["end"] - x["start"], a) and ok_clip(y["text"], y["end"] - y["start"], a)):
                    continue
                ids = []
                for j, t in ((i, x), (i + 1, y)):
                    cid = f"{ep_key}/{w['window_id']}/t{j:03d}"
                    ids.append(cid)
                    clips.append(dict(id=cid, kind="turn", split=split, episode=ep_key,
                                      audio=str(audio.relative_to(PROCESSED_DIR)),
                                      start=round(off + t["start"], 3), end=round(off + t["end"], 3),
                                      speaker=t["speaker"], text=t["text"]))
                pairs.append(dict(split=split, episode=ep_key, user=ids[0], reply=ids[1]))

    # a turn can appear in two pairs (as reply, then as prompt): keep one row per id
    uniq = {c["id"]: c for c in clips}
    clips = sorted(uniq.values(), key=lambda c: (c["audio"], c["start"]))
    out = WORK / "somali"
    write_jsonl(out / "manifest.jsonl", clips)
    write_jsonl(out / "pairs.jsonl", pairs)
    hours = Counter()
    for c in clips:
        hours[f"{c['kind']}_{c['split']}_h"] += (c["end"] - c["start"]) / 3600
    stats.update({k: round(v, 2) for k, v in hours.items()})
    stats["clips"], stats["pairs"] = len(clips), len(pairs)
    stats["pairs_by_split"] = dict(Counter(p["split"] for p in pairs))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    L.info("stats: %s", json.dumps(stats))


if __name__ == "__main__":
    main()
