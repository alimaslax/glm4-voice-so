"""Single-speaker, Somali-forced transcripts (single_speaker.py build) -> MMS ASR datasets.

Input : $SO_WORK/single_speaker/manifest.jsonl (rows with ok=true), clean.flac under $SO_DATA/processed
Output: $SO_WORK/asr/ss_{train,val,test}  (same columns as prepare_asr.py; same episode split)
"""
from collections import defaultdict

from common import WORK, log, read_jsonl
from prepare_asr import clips

L = log("prepare_asr_ss")


def main():
    from datasets import Dataset, Features, Value
    by_split = {s: defaultdict(list) for s in ("train", "val", "test")}
    for r in read_jsonl(WORK / "single_speaker" / "manifest.jsonl"):
        if r["ok"]:
            by_split[r["split"]][r["audio"]].append(r)
    feats = Features(id=Value("string"), audio=Value("binary"), text=Value("string"),
                     text_raw=Value("string"), dur=Value("float32"))
    for split, items in by_split.items():
        ds = Dataset.from_generator(clips, gen_kwargs=dict(items_by_audio=items), features=feats,
                                    cache_dir=str(WORK / "cache" / "hf_datasets")) if items else \
            Dataset.from_dict({k: [] for k in feats}, features=feats)
        ds.save_to_disk(str(WORK / "asr" / f"ss_{split}"))
        L.info("ss_%s: %d clips, %.1f h", split, len(ds), sum(ds["dur"]) / 3600 if len(ds) else 0)


if __name__ == "__main__":
    main()
