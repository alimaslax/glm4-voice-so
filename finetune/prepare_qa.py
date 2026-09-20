"""Fold recorded questions + Omar-synthesised answers into the manifest as dialogue pairs.

Inputs: the qa.jsonl from gen_qa.py, a dir of recorded question wavs and a dir of answer wavs,
both named <id>.wav (the id from qa.jsonl).  Rows are appended to an existing manifest, so the
ASR/TTS clips from prepare_somali.py stay in the same build.

  python prepare_qa.py --qa $SO_WORK/qa/qa.jsonl --questions $SO_WORK/qa/q_wav --answers $SO_WORK/qa/a_wav

Clips are written with kind="qa": build_sft.py only makes asr/tts rows from kind=="segment", so the
synthetic Omar audio is used for the dialogue task and nothing else.
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import soundfile as sf

from common import PROCESSED_DIR, SOMALI, log, read_jsonl, split_of, write_jsonl

L = log("prepare_qa")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qa", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--answers", required=True)
    p.add_argument("--max-dur", type=float, default=20.0)
    p.add_argument("--min-dur", type=float, default=0.8)
    p.add_argument("--speaker", default="human", help="who read the questions; becomes the clip speaker")
    a = p.parse_args()

    dest = PROCESSED_DIR / "qa"
    dest.mkdir(parents=True, exist_ok=True)
    clips = list(read_jsonl(SOMALI / "manifest.jsonl")) if (SOMALI / "manifest.jsonl").exists() else []
    pairs = list(read_jsonl(SOMALI / "pairs.jsonl")) if (SOMALI / "pairs.jsonl").exists() else []
    L.info("existing: %d clips, %d pairs", len(clips), len(pairs))

    stats = Counter()
    for r in read_jsonl(Path(a.qa)):
        rid = r["id"]
        got = {}
        for side, root, text in (("q", Path(a.questions), r["question"]), ("a", Path(a.answers), r["answer"])):
            src = root / f"{rid}.wav"
            if not src.is_file():
                stats[f"missing_{side}"] += 1
                break
            info = sf.info(str(src))
            dur = info.frames / info.samplerate
            if not a.min_dur <= dur <= a.max_dur:
                stats[f"bad_dur_{side}"] += 1
                break
            tgt = dest / f"{rid}_{side}.wav"
            if not tgt.exists():
                shutil.copy(src, tgt)
            got[side] = (tgt, dur, text)
        if len(got) != 2:
            continue
        # Split by the pair id so a question and its answer never land on opposite sides.
        split = split_of(rid)
        ids = []
        for side, speaker in (("q", a.speaker), ("a", "omar")):
            tgt, dur, text = got[side]
            cid = f"qa/{rid}/{side}"
            ids.append(cid)
            clips.append(dict(id=cid, kind="qa", split=split, episode=f"qa/{rid}",
                              audio=str(tgt.relative_to(PROCESSED_DIR)),
                              start=0.0, end=round(dur, 3), speaker=speaker, text=text))
        pairs.append(dict(split=split, episode=f"qa/{rid}", user=ids[0], reply=ids[1]))
        stats[f"pair_{split}"] += 1

    uniq = {c["id"]: c for c in clips}
    write_jsonl(SOMALI / "manifest.jsonl", sorted(uniq.values(), key=lambda c: (c["audio"], c["start"])))
    write_jsonl(SOMALI / "pairs.jsonl", pairs)
    L.info("clips %d, pairs %d, stats %s", len(uniq), len(pairs), dict(stats))


if __name__ == "__main__":
    main()
