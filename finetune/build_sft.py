"""Track A: tokenized clips -> supervised samples in GLM-4-Voice chat format.

Tasks (all from the same Somali data):
  asr      user = Somali speech            -> assistant = Somali text                (<|assistant|>\\n)
  tts      user = "read this aloud: text"  -> assistant = 13 text / 26 audio interleaved
  dialogue user = speaker A's turn (audio) -> assistant = speaker B's reply, interleaved

Output: $SO_WORK/somali/sft/{train,val,test}  (HF datasets: input_ids, labels, task, length)
"""
import argparse
import random
from collections import Counter

from common import (ASR_SYSTEM, LLM_PATH, SOMALI, SPEECH_SYSTEM, TEXT_SYSTEM, TTS_INSTRUCTION, GLMFormat, log,
                    read_jsonl)

L = log("build_sft")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tasks", default="asr,tts,dialogue", help="comma-separated subset to build")
    a = p.parse_args()
    tasks = {t.strip() for t in a.tasks.split(",") if t.strip()}
    L.info("tasks: %s", ",".join(sorted(tasks)))

    from datasets import Dataset, Features, Sequence, Value
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(LLM_PATH), trust_remote_code=True)
    g = GLMFormat(tok)

    clips = {r["id"]: r for r in read_jsonl(SOMALI / "manifest.jsonl")}
    n_tok = 0
    for f in sorted((SOMALI / "tokens").glob("*.jsonl")):
        for r in read_jsonl(f):
            if r["id"] in clips:
                clips[r["id"]]["tokens"] = r["tokens"]; n_tok += 1
    L.info("clips %d, tokenized %d", len(clips), n_tok)

    rows, stats = [], Counter()

    def add(task, split, prompt, target):
        if task not in tasks:
            return
        ids, labels = g.sample(prompt, target)
        if len(ids) > a.max_len:
            stats[f"{task}_too_long"] += 1
            return
        rows.append(dict(split=split, task=task, input_ids=ids, labels=labels, length=len(ids)))
        stats[f"{task}_{split}"] += 1

    for c in clips.values():
        if c["kind"] != "segment" or not c.get("tokens"):
            continue
        text_ids = g.enc(c["text"])
        add("asr", c["split"], g.prompt(ASR_SYSTEM, g.audio(c["tokens"]), streaming=False), text_ids)
        add("tts", c["split"], g.prompt(TEXT_SYSTEM, g.enc(TTS_INSTRUCTION + c["text"])),
            g.interleave(text_ids, c["tokens"]))
    pairs_file = SOMALI / "pairs.jsonl"
    for pr in (read_jsonl(pairs_file) if "dialogue" in tasks and pairs_file.exists() else []):
        u, r = clips.get(pr["user"]), clips.get(pr["reply"])
        if not (u and r and u.get("tokens") and r.get("tokens")):
            stats["pair_missing_tokens"] += 1
            continue
        add("dialogue", pr["split"], g.prompt(SPEECH_SYSTEM, g.audio(u["tokens"])),
            g.interleave(g.enc(r["text"]), r["tokens"]))

    random.Random(a.seed).shuffle(rows)
    out = SOMALI / "sft"
    for split in ("train", "val", "test"):
        part = [{k: v for k, v in r.items() if k != "split"} for r in rows if r["split"] == split]
        feats = Features(input_ids=Sequence(Value("int32")), labels=Sequence(Value("int32")),
                         task=Value("string"), length=Value("int32"))
        cols = {k: [r[k] for r in part] for k in feats}            # from_dict: works for empty splits too
        Dataset.from_dict(cols, features=feats).save_to_disk(str(out / split))
        L.info("%s: %d samples, %d tokens", split, len(part), sum(r["length"] for r in part))
    L.info("stats: %s", dict(stats))


if __name__ == "__main__":
    main()
