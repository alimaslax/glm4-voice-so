"""Per-task eval loss of a LoRA adapter (or the base model) on the same val subset train_lora.py evaluates.

  python eval_lora_loss.py --adapter $SO_WORK/runs/lora_somali/final [--per-task 300]   (SO_SOMALI picks the data)
  python eval_lora_loss.py --adapter none                                                 (base model)

Lets two adapters trained on different data be compared on ONE val set (e.g. round 1 vs round 2 on the Scribe val).
"""
import argparse
import json

import torch

from common import LLM_PATH, SOMALI, log
from train_lora import Collator

L = log("eval_lora_loss")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--per-task", type=int, default=300)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    from datasets import load_from_disk
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(LLM_PATH), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(LLM_PATH), trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    if a.adapter != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    val, coll, out = load_from_disk(str(SOMALI / "sft" / "val")), Collator(tok.pad_token_id), {}
    for task in ("asr", "tts", "dialogue"):
        sub = val.filter(lambda r, task=task: r["task"] == task)
        sub = sub.select(range(min(a.per_task, len(sub))))
        tot, n = 0.0, 0
        for i in range(0, len(sub), a.batch):
            b = coll([sub[j] for j in range(i, min(i + a.batch, len(sub)))])
            b = {k: v.cuda() for k, v in b.items()}
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).logits[:, :-1]
            tgt = b["labels"][:, 1:]
            sel = tgt != -100
            tot += torch.nn.functional.cross_entropy(logits[sel].float(), tgt[sel], reduction="sum").item()
            n += sel.sum().item()
        out[task] = round(tot / n, 4)
        L.info("%s: loss %.4f (%d samples)", task, out[task], len(sub))
    print(json.dumps(dict(adapter=a.adapter, data=str(SOMALI), **out)))


if __name__ == "__main__":
    main()
