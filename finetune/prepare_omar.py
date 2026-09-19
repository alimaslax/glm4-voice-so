"""Track B data prep: Omar clips -> single-speaker, speaker-verified manifest.

Input : $SO_DATA/processed/omar/<video>/NNNN.wav + transcripts/NNNN.json (word-level speaker_id)
Output: $SO_WORK/omar/manifest.jsonl   {id, video, split, audio, dur, text, sim}
        $SO_WORK/omar/stats.json
Filtering:
  1. every word in the transcript has the same speaker_id
  2. duration within [min_dur, max_dur]
  3. ECAPA speaker embedding close to Omar's centroid (median of all clips); videos whose median
     similarity is low are dropped whole (they are usually clips of someone else talking).
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import torch

from common import OMAR_DIR, WORK, load_audio, log, split_of, write_jsonl

L = log("prepare_omar")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-dur", type=float, default=1.0)
    p.add_argument("--max-dur", type=float, default=30.0)
    p.add_argument("--clip-sim", type=float, default=0.55, help="min cosine sim to centroid per clip")
    p.add_argument("--video-sim", type=float, default=0.60, help="min median sim per video")
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()

    stats, rows = Counter(), []
    for wav in sorted(OMAR_DIR.glob("*/*.wav")):
        tj = wav.parent / "transcripts" / (wav.stem + ".json")
        if not tj.exists():
            stats["no_transcript"] += 1
            continue
        t = json.loads(tj.read_text())
        words = [w for w in t.get("words", []) if w.get("type", "word") == "word"]
        spk = {w.get("speaker_id") for w in words}
        text = " ".join((t.get("text") or "").split())
        if len(spk) != 1 or not text:
            stats["multi_speaker_or_empty"] += 1
            continue
        import soundfile as sf
        dur = sf.info(str(wav)).duration
        if not (a.min_dur <= dur <= a.max_dur):
            stats["bad_duration"] += 1
            continue
        video = wav.parent.name
        rows.append(dict(id=f"{video}/{wav.stem}", video=video, split=split_of(video, val=0.05, test=0.0),
                         audio=str(wav.relative_to(OMAR_DIR)), dur=round(dur, 3), text=text))
    L.info("candidates after transcript filters: %d (%s)", len(rows), dict(stats))

    # ECAPA speaker verification (speechbrain, 16 kHz)
    import torchaudio
    import torchaudio.functional as AF
    if not hasattr(torchaudio, "list_audio_backends"):      # removed in torchaudio 2.9; speechbrain calls it
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # at import. We load audio ourselves.
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                         savedir=str(WORK / "cache" / "ecapa"), run_opts={"device": "cuda"})
    embs = []
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        wavs = []
        for r in chunk:
            x, sr = load_audio(OMAR_DIR / r["audio"])
            x = AF.resample(torch.from_numpy(x), sr, 16000)[: 16000 * 12]   # 12 s is plenty for identity
            wavs.append(x)
        n = max(len(w) for w in wavs)
        batch = torch.zeros(len(wavs), n)
        lens = torch.tensor([len(w) / n for w in wavs])
        for j, w in enumerate(wavs):
            batch[j, : len(w)] = w
        with torch.no_grad():
            e = enc.encode_batch(batch.cuda(), lens.cuda()).squeeze(1)
        embs.append(torch.nn.functional.normalize(e, dim=-1).cpu())
        if i % (a.batch * 50) == 0:
            L.info("ecapa %d/%d", i, len(rows))
    E = torch.cat(embs).numpy()
    centroid = np.median(E, axis=0)
    centroid /= np.linalg.norm(centroid)
    sims = E @ centroid
    by_video = defaultdict(list)
    for r, s in zip(rows, sims):
        r["sim"] = round(float(s), 4)
        by_video[r["video"]].append(float(s))
    bad_videos = {v for v, s in by_video.items() if np.median(s) < a.video_sim}
    kept = [r for r in rows if r["video"] not in bad_videos and r["sim"] >= a.clip_sim]
    kept_ids = {r["id"] for r in kept}
    stats["dropped_videos"] = len(bad_videos)
    stats["dropped_low_sim_clips"] = len(rows) - len(kept) - sum(len(by_video[v]) for v in bad_videos)
    stats["kept_clips"] = len(kept)
    stats["kept_hours"] = round(sum(r["dur"] for r in kept) / 3600, 2)
    stats["val_clips"] = sum(r["split"] == "val" for r in kept)
    out = WORK / "omar"
    write_jsonl(out / "manifest.jsonl", kept)
    write_jsonl(out / "rejected.jsonl", [r for r in rows if r["id"] not in kept_ids])
    (out / "stats.json").write_text(json.dumps(dict(stats, bad_videos=sorted(bad_videos)), indent=2,
                                               ensure_ascii=False))
    L.info("stats: %s", dict(stats))


if __name__ == "__main__":
    main()
