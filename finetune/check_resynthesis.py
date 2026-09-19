"""Go/no-go: can the frozen tokenizer + decoder carry Somali? Also validates our mel settings for Track B.

For --n-somali random Somali segments and --n-omar random Omar clips (fixed seed, any split):
  audio -> speech tokens -> flow -> hift -> wav      (stock decoder, or --flow a fine-tuned flow.pt)
  MMS-1b-all with its Somali adapter (facebook/mms-1b-all, target_lang=som) transcribes original and
  resynthesized audio; CER vs reference text. (Whisper-large-v3-turbo was tried first: it can't do Somali -
  it answers in Arabic script / Spanish - so it is useless as a judge here.)
  For Omar: L1 between our 22.05 kHz mel (tokenize_audio.mel_22k) and the stock flow's mel for the same
  tokens. If MEL settings were wrong this is far larger than the Somali-vs-Omar voice difference.

Output: $SO_WORK/resynth/<tag>/{somali,omar}/NN_<id>.{orig,resynth,ab}.wav, report.json, README.md
        *.ab.wav = original, 0.6 s silence, resynthesis (22.05 kHz) - the quickest way to listen.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from common import DECODER_PATH, OMAR_DIR, PROCESSED_DIR, WORK, load_audio, load_speech_tokenizer, log, \
    read_jsonl, speech_tokens

L = log("check_resynthesis")
ASR_MODEL = os.environ.get("SO_ASR_MODEL", "facebook/mms-1b-all")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-somali", type=int, default=50)
    p.add_argument("--n-omar", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-dur", type=float, default=2.0)
    p.add_argument("--max-dur", type=float, default=12.0)
    p.add_argument("--flow", default=None, help="fine-tuned flow.pt (default: stock)")
    p.add_argument("--tag", default="stock")
    a = p.parse_args()

    import jiwer
    import soundfile as sf
    import torchaudio.functional as AF
    from hyperpyyaml import load_hyperpyyaml
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    from tokenize_audio import mel_22k

    with open(DECODER_PATH / "config.yaml") as f:
        cfg = load_hyperpyyaml(f)
    flow, hift = cfg["flow"], cfg["hift"]
    flow.load_state_dict(torch.load(a.flow or DECODER_PATH / "flow.pt", map_location="cpu"))
    hift.load_state_dict(torch.load(DECODER_PATH / "hift.pt", map_location="cpu"))
    flow.cuda().eval(); hift.cuda().eval()
    tokm, fe = load_speech_tokenizer()
    aproc = AutoProcessor.from_pretrained(ASR_MODEL, target_lang="som")
    amod = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL, target_lang="som", ignore_mismatched_sizes=True).cuda().eval()

    @torch.no_grad()
    def asr(x, sr):
        x = AF.resample(torch.from_numpy(x), sr, 16000).numpy()
        f = aproc(x, sampling_rate=16000, return_tensors="pt").input_values.cuda()
        return aproc.decode(amod(f).logits[0].argmax(-1)).strip()

    @torch.no_grad()
    def decode(tokens):
        t = torch.tensor(tokens, dtype=torch.int32).unsqueeze(0).cuda()
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt).cuda()
        mel = flow.inference(token=t, token_len=torch.tensor([t.shape[1]], dtype=torch.int32).cuda(),
                             prompt_token=z(1, 0, dt=torch.int32),
                             prompt_token_len=torch.tensor([0], dtype=torch.int32).cuda(),
                             prompt_feat=z(1, 0, 80), prompt_feat_len=torch.tensor([0], dtype=torch.int32).cuda(),
                             embedding=z(1, 192))
        wav, _ = hift.inference(mel=mel, cache_source=z(1, 1, 0))
        return wav.squeeze(0).float().cpu().numpy(), mel.squeeze(0).transpose(0, 1).float().cpu()

    norm = lambda s: " ".join("".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).split())
    out = WORK / "resynth" / a.tag
    report = {}
    import random
    rng = random.Random(a.seed)
    som = [(r, PROCESSED_DIR / r["audio"], r["start"], r["end"])
           for r in read_jsonl(WORK / "somali" / "manifest.jsonl")
           if r["kind"] == "segment" and a.min_dur <= r["end"] - r["start"] <= a.max_dur]
    om_manifest = WORK / "omar" / "manifest.jsonl"
    om = [(r, OMAR_DIR / r["audio"], None, None) for r in read_jsonl(om_manifest)
          if a.min_dur <= r["dur"] <= a.max_dur + 3] if a.n_omar and om_manifest.exists() else []
    sources = {"somali": rng.sample(som, min(a.n_somali, len(som))),
               "omar": rng.sample(om, min(a.n_omar, len(om)))}
    readme = ["# Resynthesis check (%s decoder)\n" % a.tag,
              "`*.ab.wav`: original, short pause, then the same audio after speech tokenizer -> flow -> HiFT.\n"]
    for name, items in sources.items():
        if not items:
            continue
        (out / name).mkdir(parents=True, exist_ok=True)
        refs, hyp_orig, hyp_resyn, mel_l1 = [], [], [], []
        readme.append(f"\n## {name}\n\n| # | file | reference text | MMS-som on original | MMS-som on resynth |\n|---|---|---|---|---|")
        for k, (r, path, s, e) in enumerate(items):
            x, sr = load_audio(path, s, e)
            toks = speech_tokens(tokm, fe, [(x, sr)])[0]
            y, mel_gen = decode(toks)
            stem = f"{k:02d}_" + r["id"].replace("/", "__")[-80:]
            sf.write(str(out / name / f"{stem}.orig.wav"), x, sr)
            sf.write(str(out / name / f"{stem}.resynth.wav"), y, 22050)
            x22 = AF.resample(torch.from_numpy(x), sr, 22050).numpy()
            sf.write(str(out / name / f"{stem}.ab.wav"), np.concatenate([x22, np.zeros(int(0.6 * 22050)), y]), 22050)
            refs.append(norm(r["text"])); hyp_orig.append(norm(asr(x, sr))); hyp_resyn.append(norm(asr(y, 22050)))
            readme.append(f"| {k} | `{stem}.ab.wav` | {r['text'][:120]} | {hyp_orig[-1][:120]} | {hyp_resyn[-1][:120]} |")
            if name == "omar":
                m = mel_22k(x, sr).float().cpu()
                n = min(len(m), len(mel_gen))
                mel_l1.append(dict(l1=float((m[:n] - mel_gen[:n]).abs().mean()),
                                   mean_ours=float(m.mean()), mean_flow=float(mel_gen.mean()),
                                   frames_ours=len(m), frames_flow=len(mel_gen)))
        report[name] = dict(
            n=len(items),
            asr_cer_original=round(jiwer.cer(refs, hyp_orig), 4),
            asr_cer_resynth=round(jiwer.cer(refs, hyp_resyn), 4),
            examples=[dict(ref=r_, orig=o, resynth=h) for r_, o, h in list(zip(refs, hyp_orig, hyp_resyn))[:5]])
        if mel_l1:
            report[name]["mel_check"] = dict(
                l1=round(float(np.mean([m["l1"] for m in mel_l1])), 4),
                mean_ours=round(float(np.mean([m["mean_ours"] for m in mel_l1])), 3),
                mean_flow=round(float(np.mean([m["mean_flow"] for m in mel_l1])), 3),
                frame_ratio=round(float(np.mean([m["frames_ours"] / m["frames_flow"] for m in mel_l1])), 3))
        L.info("%s: %s", name, json.dumps({k: v for k, v in report[name].items() if k != "examples"}))
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    summary = [f"\n## Summary\n"] + [f"- **{n}**: MMS-som CER original {v['asr_cer_original']:.3f} -> resynth "
                                     f"{v['asr_cer_resynth']:.3f} ({v['n']} clips)" for n, v in report.items()]
    (out / "README.md").write_text("\n".join(readme[:2] + summary + readme[2:]) + "\n")
    L.info("wrote %s", out / "report.json")


if __name__ == "__main__":
    main()
