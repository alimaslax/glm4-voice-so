"""MMS ASR data: Somali segments -> 16 kHz int16 audio + normalized text, as HF datasets.

Input : $SO_WORK/$SO_SOMALI/manifest.jsonl (prepare_somali.py), $SO_DATA/processed/*/*/clean.flac
Output: $SO_WORK/$SO_ASR/{train,val,test}   columns: id, audio (raw int16 bytes @16k), text (normalized), text_raw, dur
        (bytes, not a list column: Arrow writes them ~100x faster; readers use np.frombuffer(..., np.int16))
Text normalization matches what the stock MMS Somali head emits: lowercase, letters/digits/'/-, no punctuation.
"""
import re
import unicodedata
from collections import defaultdict

import numpy as np
import torch

from common import ASR_DIR, PROCESSED_DIR, SOMALI, WORK, load_audio, log, read_jsonl

L = log("prepare_asr")
KEEP = re.compile(r"[^a-z0-9'\- ]")


def normalize(t):
    t = unicodedata.normalize("NFKC", t).lower().replace("’", "'").replace("‘", "'")
    t = KEEP.sub(" ", t)
    return " ".join(t.split())


def clips(items_by_audio):
    """Yields one row per segment; loads/resamples each episode once."""
    import torchaudio.functional as AF
    for i, (audio, items) in enumerate(sorted(items_by_audio.items())):
        full, sr = load_audio(PROCESSED_DIR / audio)
        x16 = AF.resample(torch.from_numpy(full).cuda(), sr, 16000).cpu().numpy()     # GPU: ~50x faster
        for r in items:
            text = normalize(r["text"])
            if len(text.replace(" ", "")) < 2:
                continue
            seg = x16[int(r["start"] * 16000): int(r["end"] * 16000)]
            yield dict(id=r["id"], audio=(np.clip(seg, -1, 1) * 32767).astype(np.int16).tobytes(),
                       text=text, text_raw=r["text"], dur=len(seg) / 16000)
        if i % 50 == 0:
            L.info("%d/%d episodes", i, len(items_by_audio))


def main():
    from datasets import Dataset, Features, Value
    by_split = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    for r in read_jsonl(SOMALI / "manifest.jsonl"):
        if r["kind"] == "segment":
            by_split[r["split"]][r["audio"]].append(r)
    feats = Features(id=Value("string"), audio=Value("binary"), text=Value("string"),
                     text_raw=Value("string"), dur=Value("float32"))
    for split, items in by_split.items():
        # generator -> arrow on disk; never holds a whole split of audio in RAM
        ds = Dataset.from_generator(clips, gen_kwargs=dict(items_by_audio=items), features=feats,
                                    cache_dir=str(WORK / "cache" / "hf_datasets")) if items else \
            Dataset.from_dict({k: [] for k in feats}, features=feats)
        ds.save_to_disk(str(ASR_DIR / split))
        L.info("%s: %d clips, %.1f h", split, len(ds), sum(ds["dur"]) / 3600 if len(ds) else 0)


if __name__ == "__main__":
    main()
