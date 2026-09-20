"""Extra Somali ASR corpora from the HF Hub -> same format as prepare_asr.py.

Datasets and pinned revisions come from configs/asr_mms.yaml (extra_datasets).
Output: $SO_WORK/asr/ext_{train,val,test}   columns: id, audio (raw int16 bytes @16k), text, text_raw, dur
train_asr.py mixes ext_train into training (train_splits); eval_asr.py reports ext_test separately.
"""
import io
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from common import ASR_DIR, WORK, log
from prepare_asr import normalize

L = log("prepare_asr_ext")
SPLITS = {"train": "ext_train", "validation": "ext_val", "val": "ext_val", "test": "ext_test"}


def rows(files, tag):
    import pyarrow.parquet as pq
    import torchaudio.functional as AF
    for f in files:
        t = pq.read_table(f, columns=["id", "audio", "text"])
        for rid, audio, text_raw in zip(t["id"].to_pylist(), t["audio"].to_pylist(), t["text"].to_pylist()):
            text = normalize(text_raw or "")
            if len(text.replace(" ", "")) < 2 or not audio or not audio.get("bytes"):
                continue
            x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
            x = x.mean(1)
            if sr != 16000:
                x = AF.resample(torch.from_numpy(x), sr, 16000).numpy()
            if len(x) < 16000 * 0.5:
                continue
            yield dict(id=f"{tag}:{rid}", audio=(np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes(),
                       text=text, text_raw=text_raw, dur=len(x) / 16000)


def main():
    from datasets import Dataset, Features, Value, concatenate_datasets
    from huggingface_hub import snapshot_download
    cfg = yaml.safe_load(open(Path(__file__).parent / "configs" / "asr_mms.yaml"))
    feats = Features(id=Value("string"), audio=Value("binary"), text=Value("string"),
                     text_raw=Value("string"), dur=Value("float32"))
    parts = {v: [] for v in set(SPLITS.values())}
    for ex in cfg.get("extra_datasets") or []:
        local = Path(snapshot_download(ex["repo"], repo_type="dataset", revision=ex["revision"],
                                       allow_patterns=["data/*.parquet"]))
        tag = ex["repo"].split("/")[-1]
        for hf_split, ours in SPLITS.items():
            files = sorted(str(p) for p in (local / "data").glob(f"{hf_split}-*.parquet"))
            if not files:
                continue
            ds = Dataset.from_generator(rows, gen_kwargs=dict(files=files, tag=tag), features=feats,
                                        cache_dir=str(WORK / "cache" / "hf_datasets"))
            L.info("%s %s -> %s: %d clips, %.2f h", ex["repo"], hf_split, ours, len(ds), sum(ds["dur"]) / 3600)
            parts[ours].append(ds)
    for ours, dss in parts.items():
        ds = concatenate_datasets(dss) if dss else Dataset.from_dict({k: [] for k in feats}, features=feats)
        ds.save_to_disk(str(ASR_DIR / ours))
        L.info("%s: %d clips, %.2f h", ours, len(ds), sum(ds["dur"]) / 3600 if len(ds) else 0)


if __name__ == "__main__":
    main()
