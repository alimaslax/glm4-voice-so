"""Single-speaker, Somali-forced transcripts (single_speaker.py build) -> MMS ASR datasets.

Input : $SO_WORK/single_speaker/manifest.jsonl (rows with ok=true), clean.flac under $SO_DATA/processed
Output: $SO_WORK/asr/ss_{train,val,test}  (same columns as prepare_asr.py; same episode split)

Agreement filter: a clip is kept only if its single-speaker transcript and the original window's words for the
same span (orig_text) agree after normalization, CER <= --max-cer. Two independent passes agreeing is a cheap
proxy for a correct label; disagreement usually means one of them guessed another language.
"""
import argparse
from collections import Counter, defaultdict

from common import ASR_DIR, WORK, log, read_jsonl
from prepare_asr import clips, normalize

L = log("prepare_asr_ss")


def cer(ref, hyp):
    import jiwer
    return jiwer.cer(ref, hyp) if ref else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-cer", type=float, default=0.15)
    a = p.parse_args()
    from datasets import Dataset, Features, Value
    by_split = {s: defaultdict(list) for s in ("train", "val", "test")}
    stats = Counter()
    for r in read_jsonl(WORK / "single_speaker" / "manifest.jsonl"):
        if not r["ok"]:
            continue
        stats["ok"] += 1
        if cer(normalize(r["orig_text"]), normalize(r["text"])) > a.max_cer:
            stats["disagree"] += 1
            continue
        stats[f"kept_{r['split']}"] += 1
        stats["kept_h"] += r["dur"] / 3600
        by_split[r["split"]][r["audio"]].append(r)
    L.info("agreement filter (CER <= %.2f): %s", a.max_cer, {k: round(v, 2) for k, v in stats.items()})
    feats = Features(id=Value("string"), audio=Value("binary"), text=Value("string"),
                     text_raw=Value("string"), dur=Value("float32"))
    for split, items in by_split.items():
        ds = Dataset.from_generator(clips, gen_kwargs=dict(items_by_audio=items), features=feats,
                                    cache_dir=str(WORK / "cache" / "hf_datasets")) if items else \
            Dataset.from_dict({k: [] for k in feats}, features=feats)
        ds.save_to_disk(str(ASR_DIR / f"ss_{split}"))
        L.info("ss_%s: %d clips, %.1f h", split, len(ds), sum(ds["dur"]) / 3600 if len(ds) else 0)


if __name__ == "__main__":
    main()
