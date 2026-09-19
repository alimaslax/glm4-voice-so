"""Audio -> GLM-4-Voice speech tokens (12.5 Hz, Whisper-VQ). For Omar also 80-bin mels for the flow decoder.

  --corpus somali : $SO_WORK/somali/manifest.jsonl -> $SO_WORK/somali/tokens/<shard>.jsonl  {id, tokens}
  --corpus omar   : $SO_WORK/omar/manifest.jsonl   -> $SO_WORK/omar/feats/<shard>.pt       {id: {tokens, mel}}

Resumable: one shard per source audio file (somali) / video (omar); finished shards are skipped.
"""
import argparse
import hashlib
from collections import defaultdict

import numpy as np
import torch

from common import MEL, OMAR_DIR, PROCESSED_DIR, WORK, load_audio, load_speech_tokenizer, log, read_jsonl, \
    speech_tokens, write_jsonl

L = log("tokenize_audio")


def shard_name(key):
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def mel_22k(x, sr):
    import torchaudio.functional as AF
    from matcha.utils.audio import mel_spectrogram
    y = AF.resample(torch.from_numpy(x).cuda(), sr, MEL["sampling_rate"]).clamp(-1, 1).unsqueeze(0)
    return mel_spectrogram(y, **MEL).squeeze(0).transpose(0, 1)    # [T, 80]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", choices=["somali", "omar"], required=True)
    p.add_argument("--batch", type=int, default=64)
    a = p.parse_args()

    model, fe = load_speech_tokenizer()
    rows = read_jsonl(WORK / a.corpus / "manifest.jsonl")
    groups = defaultdict(list)
    for r in rows:
        groups[r["audio"] if a.corpus == "somali" else r["video"]].append(r)
    out_dir = WORK / a.corpus / ("tokens" if a.corpus == "somali" else "feats")
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for gi, (key, items) in enumerate(sorted(groups.items())):
        out = out_dir / (shard_name(key) + (".jsonl" if a.corpus == "somali" else ".pt"))
        if out.exists():
            done += 1
            continue
        if a.corpus == "somali":
            full, sr = load_audio(PROCESSED_DIR / key)                 # whole episode once, slice clips
            clips = [(full[int(r["start"] * sr): int(r["end"] * sr)], sr) for r in items]
        else:
            clips = [load_audio(OMAR_DIR / r["audio"]) for r in items]
        toks = []
        for i in range(0, len(clips), a.batch):
            toks += speech_tokens(model, fe, clips[i:i + a.batch])
        if a.corpus == "somali":
            write_jsonl(out, [dict(id=r["id"], tokens=t) for r, t in zip(items, toks)])
        else:
            feats = {r["id"]: dict(tokens=torch.tensor(t, dtype=torch.int16),
                                   mel=mel_22k(x, sr).half().cpu())
                     for r, t, (x, sr) in zip(items, toks, clips)}
            tmp = out.with_suffix(".tmp")
            torch.save(feats, tmp)
            tmp.replace(out)
        done += 1
        if gi % 10 == 0:
            L.info("%s: %d/%d shards (%s, %d clips)", a.corpus, done, len(groups), key[:60], len(items))
    L.info("%s: all %d shards done", a.corpus, len(groups))


if __name__ == "__main__":
    main()
