"""Rebuild complete per-episode Scribe indexes (every *.response.json, not just the last run's windows).

The expansion run rewrote transcripts.diarized.json per episode with only its own windows, dropping the round-1
windows of the ~500 episodes both runs touched (their .response.json files are intact). This writes a merged
tree of index files + symlinks to the responses; the source tree is not modified.

  python scribe_reindex.py <scribe dir> <out dir>
Window metadata (offset) per window id, first found: current Scribe index > MAI index ($SO_DATA/transcripts) >
window_status.jsonl (start/duration).
"""
import json
import os
import sys
from pathlib import Path

from common import DATA, log

L = log("scribe_reindex")


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    mai = DATA / "transcripts"
    status = {}
    for r in map(json.loads, (src / "window_status.jsonl").open()):
        status[(r["channel"], r["episode"], r["window"])] = r
    n_meta = {"scribe": 0, "mai": 0, "status": 0}
    n_eps, n_win = 0, 0
    for ep in sorted(p for p in src.glob("*/*") if (p / "diarized").is_dir()):
        ch, name = ep.parent.name, ep.name
        d = ep / "diarized"
        resp = sorted(d.glob("*.response.json"))
        if not resp:
            continue
        known = {}
        for idx in (d / "transcripts.diarized.json", mai / ch / name / "diarized" / "transcripts.diarized.json"):
            if idx.exists():
                for w in json.loads(idx.read_text()).get("windows", []):
                    known.setdefault(w["window_id"], (w, "scribe" if idx.parent == d else "mai"))
        out = dst / ch / name / "diarized"
        out.mkdir(parents=True, exist_ok=True)
        wins = []
        for r in resp:
            wid = r.name[: -len(".response.json")]
            if wid in known:
                w, how = known[wid]
                w = {k: v for k, v in w.items() if k in ("window_id", "kind", "offset_seconds", "duration_seconds",
                                                          "source_start_seconds", "source_end_seconds",
                                                          "sampling_rate")}
            elif (ch, name, wid) in status:
                s, how = status[(ch, name, wid)], "status"
                w = dict(window_id=wid, kind=s.get("kind", "duplex"), offset_seconds=s["start"],
                         duration_seconds=s["duration"])
            else:
                L.warning("no offset for %s/%s/%s, skipped", ch, name, wid)
                continue
            data = json.loads(r.read_text())
            speakers = sorted({x["speaker"] for x in data.get("words", []) if x.get("speaker") is not None})
            w.update(diarization_status="ok" if speakers else "no_speaker_labels", speakers=speakers,
                     language=data.get("language"), response_file=r.name)
            wins.append(w)
            n_meta[how] += 1
            link = out / r.name
            if not link.exists():
                os.symlink(r.resolve(), link)
        (out / "transcripts.diarized.json").write_text(json.dumps(dict(
            recording=f"{ch}/{name}", model="elevenlabs/scribe_v2", diarization=True, windows=wins),
            ensure_ascii=False, indent=2) + "\n")
        n_eps += 1
        n_win += len(wins)
    L.info("episodes %d, windows %d, offsets from %s", n_eps, n_win, n_meta)


if __name__ == "__main__":
    main()
