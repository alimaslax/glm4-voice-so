"""Go/no-go: can the frozen tokenizer + decoder carry Somali? Also validates our mel settings for Track B.

For N Somali val segments and N Omar val clips:
  audio -> speech tokens -> flow -> hift -> wav      (stock decoder, or --flow a fine-tuned flow.pt)
  Whisper-large-v3-turbo (language=so) transcribes original and resynthesized audio; CER vs reference text.
  For Omar: L1 between our 22.05 kHz mel (tokenize_audio.mel_22k) and the stock flow's mel for the same
  tokens. If MEL settings were wrong this is far larger than the Somali-vs-Omar voice difference.

Output: $SO_WORK/resynth/<tag>/{somali,omar}/*.wav, report.json
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
WHISPER = Path(os.environ.get("SO_WHISPER", "/workspace/models/whisper-large-v3-turbo"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--flow", default=None, help="fine-tuned flow.pt (default: stock)")
    p.add_argument("--tag", default="stock")
    a = p.parse_args()

    import jiwer
    import soundfile as sf
    import torchaudio.functional as AF
    from hyperpyyaml import load_hyperpyyaml
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from tokenize_audio import mel_22k

    with open(DECODER_PATH / "config.yaml") as f:
        cfg = load_hyperpyyaml(f)
    flow, hift = cfg["flow"], cfg["hift"]
    flow.load_state_dict(torch.load(a.flow or DECODER_PATH / "flow.pt", map_location="cpu"))
    hift.load_state_dict(torch.load(DECODER_PATH / "hift.pt", map_location="cpu"))
    flow.cuda().eval(); hift.cuda().eval()
    tokm, fe = load_speech_tokenizer()
    wproc = WhisperProcessor.from_pretrained(str(WHISPER))
    wmod = WhisperForConditionalGeneration.from_pretrained(str(WHISPER), torch_dtype=torch.float16).cuda().eval()

    def asr(x, sr):
        x = AF.resample(torch.from_numpy(x), sr, 16000).numpy()
        f = wproc(x, sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
        ids = wmod.generate(f, language="so", task="transcribe", max_new_tokens=200)
        return wproc.batch_decode(ids, skip_special_tokens=True)[0].strip()

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
    sources = {
        "somali": [(r, PROCESSED_DIR / r["audio"], r["start"], r["end"])
                   for r in read_jsonl(WORK / "somali" / "manifest.jsonl")
                   if r["split"] == "val" and r["kind"] == "segment" and 3 <= r["end"] - r["start"] <= 12],
        "omar": [(r, OMAR_DIR / r["audio"], None, None)
                 for r in read_jsonl(WORK / "omar" / "manifest.jsonl") if r["split"] == "val" and r["dur"] <= 15],
    }
    for name, items in sources.items():
        items = items[:: max(1, len(items) // a.n)][: a.n]
        (out / name).mkdir(parents=True, exist_ok=True)
        refs, hyp_orig, hyp_resyn, mel_l1 = [], [], [], []
        for r, path, s, e in items:
            x, sr = load_audio(path, s, e)
            toks = speech_tokens(tokm, fe, [(x, sr)])[0]
            y, mel_gen = decode(toks)
            stem = r["id"].replace("/", "__")[-120:]
            sf.write(str(out / name / f"{stem}.orig.wav"), x, sr)
            sf.write(str(out / name / f"{stem}.resynth.wav"), y, 22050)
            refs.append(norm(r["text"])); hyp_orig.append(norm(asr(x, sr))); hyp_resyn.append(norm(asr(y, 22050)))
            if name == "omar":
                m = mel_22k(x, sr).float().cpu()
                n = min(len(m), len(mel_gen))
                mel_l1.append(dict(l1=float((m[:n] - mel_gen[:n]).abs().mean()),
                                   mean_ours=float(m.mean()), mean_flow=float(mel_gen.mean()),
                                   frames_ours=len(m), frames_flow=len(mel_gen)))
        report[name] = dict(
            n=len(items),
            whisper_cer_original=round(jiwer.cer(refs, hyp_orig), 4),
            whisper_cer_resynth=round(jiwer.cer(refs, hyp_resyn), 4),
            examples=[dict(ref=r_, orig=o, resynth=h) for r_, o, h in list(zip(refs, hyp_orig, hyp_resyn))[:5]])
        if mel_l1:
            report[name]["mel_check"] = dict(
                l1=round(float(np.mean([m["l1"] for m in mel_l1])), 4),
                mean_ours=round(float(np.mean([m["mean_ours"] for m in mel_l1])), 3),
                mean_flow=round(float(np.mean([m["mean_flow"] for m in mel_l1])), 3),
                frame_ratio=round(float(np.mean([m["frames_ours"] / m["frames_flow"] for m in mel_l1])), 3))
        L.info("%s: %s", name, json.dumps({k: v for k, v in report[name].items() if k != "examples"}))
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    L.info("wrote %s", out / "report.json")


if __name__ == "__main__":
    main()
