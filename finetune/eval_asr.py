"""CER/WER of an MMS model on the held-out Somali test (and val) split.

  python eval_asr.py --model facebook/mms-1b-all --tag stock
  python eval_asr.py --model $SO_WORK/runs/asr_mms/final --tag finetuned

Output: $SO_WORK/eval_asr/<tag>.json (metrics + 30 example transcripts)
"""
import argparse
import json

import numpy as np
import torch

from common import WORK, log

L = log("eval_asr")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/mms-1b-all")
    p.add_argument("--tag", default="stock")
    p.add_argument("--splits", default="test,val")
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    import jiwer
    from datasets import load_from_disk
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    proc = AutoProcessor.from_pretrained(a.model, target_lang="som")
    model = Wav2Vec2ForCTC.from_pretrained(a.model, target_lang="som", ignore_mismatched_sizes=True).cuda().eval()
    report = {"model": a.model}
    for split in a.splits.split(","):
        ds = load_from_disk(str(WORK / "asr" / split))
        refs, hyps = [], []
        for i in range(0, len(ds), a.batch):
            b = ds[i:i + a.batch]
            audio = [np.asarray(x, dtype=np.float32) / 32767 for x in b["audio"]]
            inp = proc(audio, sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors="pt")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inp.input_values.cuda(), attention_mask=inp.attention_mask.cuda()).logits
            hyps += proc.batch_decode(logits.argmax(-1).cpu())
            refs += b["text"]
        report[split] = dict(n=len(refs), hours=round(sum(ds["dur"]) / 3600, 2),
                             cer=round(jiwer.cer(refs, hyps), 4), wer=round(jiwer.wer(refs, hyps), 4),
                             examples=[dict(ref=r, hyp=h) for r, h in list(zip(refs, hyps))[:30]])
        L.info("%s %s: CER %.4f WER %.4f (%d clips)", a.tag, split, report[split]["cer"], report[split]["wer"],
               len(refs))
    out = WORK / "eval_asr" / f"{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
