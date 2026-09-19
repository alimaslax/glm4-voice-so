"""Track B: fine-tune the flow decoder (MaskedDiffWithXvec) so audio tokens are rendered in Omar's voice.

  python train_flow.py --config configs/flow_omar.yaml [--total-steps 1000] [--run-name smoke]

Input : $SO_WORK/omar/feats/*.pt (tokenize_audio.py --corpus omar), $SO_WORK/omar/manifest.jsonl (splits)
Output: $SO_WORK/runs/<run_name>/step_XXXXXX/{flow.pt, state.pt, samples/*.wav}, latest -> newest step,
        tb/ (tensorboard). flow.pt is a drop-in replacement for glm-4-voice-decoder/flow.pt.
Re-running resumes from `latest`.
"""
import argparse
import math
import random
import shutil
from pathlib import Path

import torch
import yaml

from common import DECODER_PATH, WORK, log, read_jsonl

L = log("train_flow")


def load_split():
    split = {r["id"]: r["split"] for r in read_jsonl(WORK / "omar" / "manifest.jsonl")}
    data = {"train": [], "val": []}
    for f in sorted((WORK / "omar" / "feats").glob("*.pt")):
        for cid, d in torch.load(f).items():
            if cid in split and len(d["tokens"]) > 0:
                data["train" if split[cid] == "train" else "val"].append((cid, d["tokens"], d["mel"]))
    return data


def batches(items, max_frames, shuffle, rng):
    """Length-bucketed batches with a total mel-frame budget."""
    order = sorted(range(len(items)), key=lambda i: items[i][2].shape[0])
    out, cur, longest = [], [], 0
    for i in order:
        n = items[i][2].shape[0]
        if cur and max(longest, n) * (len(cur) + 1) > max_frames:
            out.append(cur); cur, longest = [], 0
        cur.append(i); longest = max(longest, n)
    if cur:
        out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out


def collate(items, idx):
    sel = [items[i] for i in idx]
    tl = torch.tensor([len(t) for _, t, _ in sel], dtype=torch.int32)
    fl = torch.tensor([m.shape[0] for _, _, m in sel], dtype=torch.int32)
    tok = torch.zeros(len(sel), int(tl.max()), dtype=torch.int32)
    mel = torch.zeros(len(sel), int(fl.max()), 80)
    for j, (_, t, m) in enumerate(sel):
        tok[j, : len(t)] = t.int()
        mel[j, : m.shape[0]] = m.float()
    return dict(speech_token=tok, speech_token_len=tl, speech_feat=mel, speech_feat_len=fl,
                embedding=torch.zeros(len(sel), 192))


def lr_at(step, c):
    if step < c["warmup_steps"]:
        return c["lr"] * (step + 1) / c["warmup_steps"]
    prog = min(1.0, (step - c["warmup_steps"]) / max(1, c["total_steps"] - c["warmup_steps"]))
    return c["lr"] * (c["min_lr_ratio"] + (1 - c["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def evaluate(flow, val, c, dev):
    flow.eval()
    state = torch.random.get_rng_state(), torch.cuda.get_rng_state(), random.getstate()
    torch.manual_seed(0); torch.cuda.manual_seed(0); random.seed(0)   # same noise every eval -> comparable
    tot, n = 0.0, 0
    for idx in batches(val, c["max_frames_per_batch"], False, None):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=c["amp"] == "bf16"):
            loss = flow(collate(val, idx), dev)["loss"]
        tot += float(loss) * len(idx); n += len(idx)
    torch.random.set_rng_state(state[0]); torch.cuda.set_rng_state(state[1]); random.setstate(state[2])
    flow.train()
    return tot / max(n, 1)


@torch.no_grad()
def render(flow, hift, val, out_dir, k):
    import soundfile as sf
    flow.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    for cid, tok, _ in val[:k]:
        t = tok.int().unsqueeze(0).cuda()
        mel = flow.inference(token=t, token_len=torch.tensor([t.shape[1]], dtype=torch.int32).cuda(),
                             prompt_token=torch.zeros(1, 0, dtype=torch.int32).cuda(),
                             prompt_token_len=torch.tensor([0], dtype=torch.int32).cuda(),
                             prompt_feat=torch.zeros(1, 0, 80).cuda(),
                             prompt_feat_len=torch.tensor([0], dtype=torch.int32).cuda(),
                             embedding=torch.zeros(1, 192).cuda())
        wav, _ = hift.inference(mel=mel, cache_source=torch.zeros(1, 1, 0).cuda())
        sf.write(str(out_dir / (cid.replace("/", "__") + ".wav")), wav.squeeze(0).float().cpu().numpy(), 22050)
    flow.train()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).parent / "configs" / "flow_omar.yaml"))
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--run-name", default=None)
    a = p.parse_args()
    c = yaml.safe_load(open(a.config))
    if a.total_steps:
        c["total_steps"] = a.total_steps
        c["save_every"] = min(c["save_every"], a.total_steps)
        c["eval_every"] = min(c["eval_every"], a.total_steps)
    run_dir = WORK / "runs" / (a.run_name or c["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(c["seed"]); torch.manual_seed(c["seed"])
    dev = torch.device("cuda")

    from hyperpyyaml import load_hyperpyyaml
    from torch.utils.tensorboard import SummaryWriter
    with open(DECODER_PATH / "config.yaml") as f:
        cfg = load_hyperpyyaml(f)
    flow, hift = cfg["flow"], cfg["hift"]
    flow.load_state_dict(torch.load(DECODER_PATH / "flow.pt", map_location="cpu"))
    hift.load_state_dict(torch.load(DECODER_PATH / "hift.pt", map_location="cpu"))
    flow.to(dev).train()
    hift.to(dev).eval()
    opt = torch.optim.AdamW(flow.parameters(), lr=c["lr"], betas=(0.9, 0.98), weight_decay=0.01)
    step = 0
    latest = run_dir / "latest"
    if latest.exists():
        st = torch.load(latest / "state.pt", map_location="cpu")
        flow.load_state_dict(torch.load(latest / "flow.pt", map_location="cpu"))
        opt.load_state_dict(st["opt"])
        step = st["step"]
        L.info("resumed from %s (step %d)", latest.resolve(), step)

    data = load_split()
    train, val = data["train"], data["val"]
    L.info("train clips %d (%.1f h), val clips %d", len(train),
           sum(m.shape[0] for _, _, m in train) * 256 / 22050 / 3600, len(val))
    tb = SummaryWriter(str(run_dir / "tb"))
    if step == 0:
        base = evaluate(flow, val, c, dev)
        tb.add_scalar("val/loss", base, 0)
        L.info("step 0 (stock decoder) val loss %.4f", base)
        render(flow, hift, val, run_dir / "step_000000" / "samples", c["render_clips"])

    rng = random.Random(c["seed"] + step)
    emb = flow.input_embedding
    while step < c["total_steps"]:
        for idx in batches(train, c["max_frames_per_batch"], True, rng):
            if step >= c["total_steps"]:
                break
            emb.weight.requires_grad_(step >= c["freeze_embedding_steps"])
            for g in opt.param_groups:
                g["lr"] = lr_at(step, c)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=c["amp"] == "bf16"):
                loss = flow(collate(train, idx), dev)["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(flow.parameters(), c["grad_clip"])
            opt.step()
            step += 1
            if step % c["log_every"] == 0:
                tb.add_scalar("train/loss", float(loss), step)
                tb.add_scalar("train/grad_norm", float(gn), step)
                tb.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
                L.info("step %d loss %.4f gn %.2f lr %.2e bs %d", step, float(loss), float(gn),
                       opt.param_groups[0]["lr"], len(idx))
            if step % c["eval_every"] == 0:
                v = evaluate(flow, val, c, dev)
                tb.add_scalar("val/loss", v, step)
                L.info("step %d val loss %.4f", step, v)
            if step % c["save_every"] == 0 or step == c["total_steps"]:
                d = run_dir / f"step_{step:06d}"
                d.mkdir(parents=True, exist_ok=True)
                torch.save(flow.state_dict(), d / "flow.pt")
                torch.save(dict(opt=opt.state_dict(), step=step, config=c), d / "state.pt")
                render(flow, hift, val, d / "samples", c["render_clips"])
                tmp = run_dir / "latest.tmp"
                if tmp.exists() or tmp.is_symlink():
                    tmp.unlink()
                tmp.symlink_to(d.name)
                tmp.replace(latest)
                steps = sorted(run_dir.glob("step_*[0-9]"))
                for old in steps[1:-c["keep_checkpoints"]]:          # keep step_000000 samples + last N
                    shutil.rmtree(old)
                L.info("saved %s", d)
    L.info("done at step %d", step)


if __name__ == "__main__":
    main()
