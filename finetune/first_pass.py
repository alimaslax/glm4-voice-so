"""First listenable pass: Somali LoRA + Omar decoder on held-out (test-split) episodes, with a stock A/B.

For each sample the same prompt runs twice, with the adapter disabled (stock GLM-4-Voice-9B) and enabled
(Somali LoRA); the audio tokens are rendered with Omar's decoder (flow_omar + stock HiFT).
  dialogue : Somali speech turn in  -> spoken + written reply   (the actual voice-assistant use)
  tts      : "read this aloud" + Somali text -> speech
  asr      : Somali speech -> Somali text (CER vs reference)

  python first_pass.py [--lora $SO_WORK/runs/lora_somali/final] [--flow .../flow_omar/latest/flow.pt] [--n 8]
Output: $SO_WORK/first_pass/<tag>/  ALL_dialogue_AB.wav, ALL_tts_AB.wav, <task>/NN_*.wav, README.md, report.json
  ALL_*_AB.wav per sample: [user speech / nothing] . stock reply . beep . LoRA reply
"""
import argparse
import json
import random
import shutil

import numpy as np
import torch

from common import (ASR_SYSTEM, DECODER_PATH, LLM_PATH, PROCESSED_DIR, SPEECH_SYSTEM, TEXT_SYSTEM,
                    TTS_INSTRUCTION, WORK, GLMFormat, load_audio, log, read_jsonl)

L = log("first_pass")
SR = 22050


def resample(x, sr):
    import torchaudio.functional as AF
    return AF.resample(torch.from_numpy(x), sr, SR).numpy() if sr != SR else x


def beep(sec=0.25, hz=880):
    t = np.arange(int(sec * SR)) / SR
    return (0.2 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def silence(sec):
    return np.zeros(int(sec * SR), np.float32)


def cer(ref, hyp):
    import jiwer
    from prepare_asr import normalize
    r, h = normalize(ref), normalize(hyp)
    return round(jiwer.cer(r, h), 4) if r else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora", default=str(WORK / "runs" / "lora_somali" / "final"))
    p.add_argument("--flow", default=str(WORK / "runs" / "flow_omar" / "latest" / "flow.pt"))
    p.add_argument("--n", type=int, default=8, help="samples per task")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--temperature", type=float, default=0.2)   # web_demo defaults
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=700)
    p.add_argument("--tag", default="lora_somali")
    a = p.parse_args()

    import soundfile as sf
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer
    from flow_inference import AudioDecoder

    tok = AutoTokenizer.from_pretrained(str(LLM_PATH), trust_remote_code=True)
    g = GLMFormat(tok)
    base = AutoModel.from_pretrained(str(LLM_PATH), trust_remote_code=True, torch_dtype=torch.bfloat16,
                                     device_map={"": 0}).eval()
    model = PeftModel.from_pretrained(base, a.lora).eval()
    decoder = AudioDecoder(config_path=str(DECODER_PATH / "config.yaml"), flow_ckpt_path=a.flow,
                           hift_ckpt_path=str(DECODER_PATH / "hift.pt"), device="cuda")
    L.info("LLM + LoRA %s, decoder flow %s", a.lora, a.flow)

    @torch.inference_mode()
    def generate(prompt_ids, lora):
        ids = torch.tensor([prompt_ids], device="cuda")
        kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.max_new_tokens,
                  do_sample=True, temperature=a.temperature, top_p=a.top_p, eos_token_id=[g.user_id])
        torch.manual_seed(a.seed)
        if lora:
            out = model.generate(**kw)
        else:
            with model.disable_adapter():
                out = model.generate(**kw)
        new = out[0, len(prompt_ids):].tolist()
        new = new[:new.index(g.user_id)] if g.user_id in new else new
        text = tok.decode([t for t in new if t < g.audio_offset], skip_special_tokens=True)
        audio = [t - g.audio_offset for t in new if t >= g.audio_offset and t != g.eoa and t != g.boa]
        wav = decoder.offline_inference(torch.tensor([audio], dtype=torch.int64))[0].numpy() if audio \
            else np.zeros(0, np.float32)
        return text.strip(), wav, len(audio)

    # held-out material: test-split clips (with speech tokens) and dialogue pairs
    clips = {r["id"]: r for r in read_jsonl(WORK / "somali" / "manifest.jsonl") if r["split"] == "test"}
    for f in sorted((WORK / "somali" / "tokens").glob("*.jsonl")):
        for r in read_jsonl(f):
            if r["id"] in clips:
                clips[r["id"]]["tokens"] = r["tokens"]
    segs = sorted([c for c in clips.values() if c["kind"] == "segment" and c.get("tokens") and 2 <= c["end"] - c["start"] <= 10],
                  key=lambda c: c["id"])
    pairs = sorted([pr for pr in read_jsonl(WORK / "somali" / "pairs.jsonl") if pr["split"] == "test"
                    and clips.get(pr["user"], {}).get("tokens") and 1.5 <= clips[pr["user"]]["end"] - clips[pr["user"]]["start"] <= 10],
                   key=lambda pr: pr["user"])
    rng = random.Random(a.seed)
    segs, pairs = rng.sample(segs, min(a.n, len(segs))), rng.sample(pairs, min(a.n, len(pairs)))

    out = WORK / "first_pass" / a.tag
    shutil.rmtree(out, ignore_errors=True)
    report = dict(lora=a.lora, flow=a.flow, temperature=a.temperature, top_p=a.top_p, dialogue=[], tts=[], asr=[])
    ab = {"dialogue": [], "tts": []}

    def clip_audio(c):
        x, sr = load_audio(PROCESSED_DIR / c["audio"], c["start"], c["end"])
        return resample(x.astype(np.float32), sr)

    for i, pr in enumerate(pairs, 1):
        u, r = clips[pr["user"]], clips.get(pr["reply"], {})
        prompt = g.prompt(SPEECH_SYSTEM, g.audio(u["tokens"]))
        (out / "dialogue").mkdir(parents=True, exist_ok=True)
        user = clip_audio(u)
        sf.write(out / "dialogue" / f"{i:02d}_user.wav", user, SR)
        row = dict(n=i, user_id=u["id"], user_text=u["text"], real_reply_text=r.get("text"))
        for name, lora in (("stock", False), ("lora", True)):
            text, wav, na = generate(prompt, lora)
            sf.write(out / "dialogue" / f"{i:02d}_{name}.wav", wav, SR)
            row[f"{name}_text"], row[f"{name}_audio_tokens"] = text, na
            ab["dialogue"].append((i, name, wav))
        ab["dialogue"].insert(len(ab["dialogue"]) - 2, (i, "user", user))
        report["dialogue"].append(row)
        L.info("dialogue %d | user: %s | lora: %s", i, u["text"][:60], row["lora_text"][:80])

    for i, c in enumerate(segs, 1):
        (out / "tts").mkdir(parents=True, exist_ok=True)
        (out / "asr").mkdir(parents=True, exist_ok=True)
        tts_prompt = g.prompt(TEXT_SYSTEM, g.enc(TTS_INSTRUCTION + c["text"]))
        asr_prompt = g.prompt(ASR_SYSTEM, g.audio(c["tokens"]), streaming=False)
        trow, arow = dict(n=i, id=c["id"], text=c["text"]), dict(n=i, id=c["id"], ref=c["text"])
        for name, lora in (("stock", False), ("lora", True)):
            text, wav, na = generate(tts_prompt, lora)
            sf.write(out / "tts" / f"{i:02d}_{name}.wav", wav, SR)
            trow[f"{name}_text"], trow[f"{name}_audio_tokens"] = text, na
            ab["tts"].append((i, name, wav))
            hyp, _, _ = generate(asr_prompt, lora)
            arow[f"{name}_hyp"], arow[f"{name}_cer"] = hyp, cer(c["text"], hyp)
        report["tts"].append(trow)
        report["asr"].append(arow)
        L.info("asr %d | ref: %s | lora: %s (cer %s)", i, c["text"][:60], arow["lora_hyp"][:60], arow["lora_cer"])

    for task in ("stock", "lora"):
        vals = [r[f"{task}_cer"] for r in report["asr"] if r[f"{task}_cer"] is not None]
        report[f"asr_mean_cer_{task}"] = round(float(np.mean(vals)), 4) if vals else None

    for task, items in ab.items():
        parts = []
        for _, name, wav in items:
            parts += [wav, beep() if name == "stock" else silence(0.6)]
        if parts:
            sf.write(out / f"ALL_{task}_AB.wav", np.concatenate(parts), SR)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))

    md = [f"# First pass: Somali LoRA + Omar decoder ({a.tag})", "",
          f"LoRA `{a.lora}`, flow `{a.flow}`, temperature {a.temperature}, top_p {a.top_p}. Held-out test episodes.",
          "`ALL_dialogue_AB.wav`: user speech, stock reply, beep, LoRA reply. `ALL_tts_AB.wav`: stock, beep, LoRA.", "",
          f"ASR mean CER: stock {report['asr_mean_cer_stock']} -> LoRA {report['asr_mean_cer_lora']}", "",
          "## Dialogue", "", "| # | user said | real reply | stock reply | LoRA reply |", "|---|---|---|---|---|"]
    md += [f"| {r['n']} | {r['user_text']} | {r['real_reply_text']} | {r['stock_text']} | {r['lora_text']} |"
           for r in report["dialogue"]]
    md += ["", "## TTS (text shown = what the model wrote while speaking)", "", "| # | input | stock | LoRA |", "|---|---|---|---|"]
    md += [f"| {r['n']} | {r['text']} | {r['stock_text']} | {r['lora_text']} |" for r in report["tts"]]
    md += ["", "## ASR", "", "| # | reference | stock (CER) | LoRA (CER) |", "|---|---|---|---|"]
    md += [f"| {r['n']} | {r['ref']} | {r['stock_hyp']} ({r['stock_cer']}) | {r['lora_hyp']} ({r['lora_cer']}) |"
           for r in report["asr"]]
    (out / "README.md").write_text("\n".join(md).replace("\n|", "\n|") + "\n")
    L.info("wrote %s", out)


if __name__ == "__main__":
    main()
