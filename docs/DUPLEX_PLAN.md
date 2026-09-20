# Somali full-duplex: GLM-4-Voice (via BayLing-Duplex) vs Nemotron VoiceChat

Goal: a Somali voice agent in Omar's voice where **latency and interruptions matter most**.
This doc compares the three duplex-capable stacks already cloned in `~/ai`, explains what
"duplex-trainable" means for GLM-4-Voice, and lays out a gated plan.

---

## 1. TL;DR

| | GLM-4-Voice + VAD (today) | **BayLing-Duplex** (GLM-4-Voice, made duplex) | **Nemotron VoiceChat 11B** | PersonaPlex (Moshi) |
|---|---|---|---|---|
| Truly listens while speaking | no (half-duplex) | yes | yes | yes |
| Decision cadence | per turn | **800 ms blocks** (10:5:10) | **80 ms frames** | 80 ms frames |
| Reaction floor (before compute) | VAD endpoint ≈ 0.3–0.6 s | 0.8–1.6 s (block + 1-block offset) | ~0.1–0.2 s | ~0.2 s |
| Interruption handling | VAD cuts playback (can't tell "haa" from a real interrupt) | learned (`[EPAD]` stops speech) | learned + RNN-T turn/barge-in head | learned, best backchannels |
| Reuses our Somali work (Whisper-VQ tokens, LoRA data, Omar flow decoder) | 100% | **~100%**: same tokenizer + decoder, same 9B family | ~0%: new encoder, LLM, codec, voice | ~0%: Mimi codec, Helium LM |
| Somali starting point | our LoRA | base is English/Chinese; our LoRA recipe re-applies | English only (README: "conversations in English") | English |
| Official training code | ours | **none** (inference only; we write it) | **yes** (NeMo `speechlm2`, configs in repo) | community (moshi-finetune) |
| Data it needs | what we have | per-block user-audio / text / assistant-audio timeline | **separate user and assistant tracks** (16 kHz / 22.05 kHz) + timed turns | two tracks |

**Recommendation: do both, in order, with measured gates.**

1. **Phase 1: BayLing-Duplex Somali** (cheapest path to a working Somali duplex in Omar's voice).
   Everything we built carries over. Then try to **shrink its blocks** (10:5:10 → 5:3:5 = 400 ms)
   to attack latency.
2. **Phase 2: Nemotron VoiceChat Somali**, only if Phase 1 misses the latency/interruption
   targets in §6. Architecturally it's the latency winner, but it's close to a from-scratch
   Somali effort (perception, LLM, TTS and Omar's voice all retrained).

The **duplex dataset** (§4) is shared by both phases, so none of that data work is wasted.

---

## 2. What "duplex" means and why GLM-4-Voice isn't duplex today

GLM-4-Voice is **turn-based**: the user finishes, then the model writes 13 text tokens, 26 audio tokens,
13 text, and so on until `<|user|>`. It never hears you while it talks. Something outside the model (a VAD)
has to decide "the user stopped", and "the user interrupted, stop playback".

A full-duplex model runs **one continuous timeline**. At every step it consumes the next slice
of user audio **and** emits its own next slice (speech or silence). Turn-taking, back-channels
and stopping on interruption become ordinary next-token predictions.

### How BayLing-Duplex did it on top of GLM-4-Voice

(`~/ai/BayLing-Duplex`, arXiv 2606.14528; details in `docs/SOMALI_LLM_FLOW_TRAINING_PLAN.md` there.)

- Same frozen **Whisper-VQ tokenizer** (12.5 tok/s) and the same **flow + HiFT decoder**.
- One sequence of **blocks**. Each block = `10 user-audio tokens | 5 text/state tokens | 10 assistant-audio tokens`
  = 0.8 s of time. Assistant targets lag the user by one block (causal).
- The text channel carries state tokens: `[SILENCE]` (listening), `<|assistant|>` (start talking),
  `[PAD]` (talking, no new text), `[EPAD]` (stop talking / yielded).
- Loss: user audio ignored; `[SILENCE]` weight 0.1; `<|assistant|>` and `[EPAD]` **weight 10**
  (those are the timing decisions); everything else weight 1.
- Trained: full SFT of the 9B, 400k **synthesized** dialogues, 1 epoch, lr 1e-5; then 200 DPO steps for timing.

**So "making GLM-4-Voice duplex-trainable" = adopting exactly this format.** We can either:

- **(a) LoRA on BayLing's weights** with Somali duplex data (recommended; it already knows *when* to talk,
  we teach it *Somali*), or
- (b) re-implement BayLing's SFT on our Somali GLM (merged LoRA). That's more work and repeats the timing training they already paid for.

Our round-1/round-2 LoRA adapters **don't load directly** onto BayLing (its 9B weights were fully
fine-tuned away from GLM-4-Voice-9B). The **data and recipe** transfer, not the adapter file.
We retrain the adapter on BayLing, mixing in our existing ASR/TTS/dialogue SFT so it keeps Somali.

**Omar's flow decoder drops straight in.** Same tokens, same `flow.pt`/`hift.pt` layout: point
`--decoder-path` at `lewenberg/glm-4-voice-decoder-omar`.

---

## 3. Latency and interruptions, concretely

What you feel as a user:

- **Response latency**: you stop talking → first audio from the agent.
- **Barge-in stop time**: you start talking over the agent → the agent goes quiet.
- **False interruptions**: the agent stops or cuts in when it shouldn't (e.g. you said "haa", or coughed).

| Stack | Where the floor comes from | Response latency floor | Barge-in floor |
|---|---|---|---|
| GLM + VAD | VAD silence timeout (300–600 ms) + first ~10 audio tokens + decoder | ~0.6–1.0 s | VAD speech onset (~100–200 ms), but dumb |
| BayLing 10:5:10 | model only decides once per 0.8 s block, plus 1-block offset | ~0.8–1.6 s + compute | same |
| BayLing 5:3:5 (experiment) | 0.4 s blocks | ~0.4–0.8 s + compute | same |
| Nemotron | 80 ms frames, 2-frame speech delay, RNN-T end-of-turn head | ~0.2 s + compute | ~0.1–0.2 s |
| Moshi/PersonaPlex | 80 ms frames | ~0.2 s | ~0.2 s |

**Honest read:** at native 10:5:10, BayLing will feel slower than Nemotron. Whether 5:3:5 works is
an open question. Smaller blocks mean less text per block (3 tokens per 0.4 s = 7.5 tok/s), and Somali
may need more GLM text tokens per second than that. **Check first** by measuring Somali GLM-tokens-per-second
on our Scribe data (cheap, CPU).

Block size can't just be changed at inference. The model was trained on 10:5:10, so a new ratio
needs fine-tuning with that ratio. We fine-tune anyway, so it's an experiment, not a blocker.

---

## 4. The duplex dataset (shared by both phases)

### What each example must contain

- **Common timeline**: user channel and assistant channel covering the same interval (60–140 s windows).
- **Words with speaker + start/end**: exactly what Scribe v2 already gives us (`.response.json`).
- **Turn events**: assistant start, assistant stop (yield vs. interrupted), user back-channels.
- **Overlap policy per window**: mono sources mix voices during overlap (see below).
- **Split by source recording** (never by window), the same sha1-by-episode split we already use.

### The mono-overlap problem, and the fix

Podcasts are mono. Diarization says *who* talks *when*, but during overlap both voices share
the same samples. Copying mono into both channels leaks the assistant's voice into the user
channel, which teaches the model to hear itself.

Our recommended fix, **hybrid resynthesis**, is also how we get Omar's voice:

1. Pick a real conversation window; choose one speaker as "assistant" and the rest as "user".
2. **User channel** = the original mono audio with the assistant's segments muted
   (small overlaps stay mixed and get flagged `mixed_overlap=true`).
3. **Assistant channel** = the assistant's Scribe text **re-spoken in Omar's voice** by our own
   Somali TTS (round-2 LoRA TTS task + Omar flow decoder), placed at the **original timestamps**.
4. Real timing, real interruptions, real back-channels, a clean assistant track, and every assistant turn in Omar's voice.

BayLing itself was trained on synthesized dialogue, so synthetic assistant tracks are proven for this recipe.
Also drop windows with heavy overlap from the first run, and add them back later as a measured experiment.

### Sources and volume

- Scribe-labelled windows: ~100 h done (round 1) plus ~98 h from the current Mac run; `so-duplex-processed` holds ~2,000 h unlabelled.
- **Scribe-lite** (our own labeler: Sortformer v2.1 diarization + Omnilingual/MMS CTC ASR + CTC alignment,
  trained to copy Scribe) is what unlocks the other ~1,800 h for free.
- **Interruption examples**: mine real ones (the user starts talking while the assistant is speaking and the
  assistant stops within ~1 s). Also synthesize negatives, per BayLing: the assistant keeps talking 3–5 s after being interrupted, used as the rejected side for DPO.

### Format

Human-readable JSONL is the source of truth; tensors/Parquet shards are compiled from it.

```json
{"id": "biletv/ep123/w0007", "split": "train", "duration": 96.0,
 "user_audio": "…/w0007.user.flac", "assistant_audio": "…/w0007.asst_omar.flac",
 "overlap_policy": "mute_assistant_keep_small_overlap",
 "turns": [
   {"role": "user", "start": 0.40, "end": 3.10, "text": "Sidee tahay?", "words": [...]},
   {"role": "assistant", "start": 3.45, "end": 6.90, "text": "Waan fiicanahay…", "end_reason": "yield"},
   {"role": "user", "start": 6.20, "end": 6.55, "text": "haa", "kind": "backchannel"},
   {"role": "assistant", "start": 7.10, "end": 9.00, "end_reason": "interrupted"}
 ]}
```

- **BayLing builder**: tokenize both tracks with Whisper-VQ, place text/state tokens per 0.8 s (or 0.4 s) block.
- **Nemotron builder**: Lhotse/SHAR cuts with 16 kHz user + 22.05 kHz `target_audio` + supervisions
  (`nemotron-voice-sm/nemo/collections/speechlm2/data/s2s_dataset.py`; processing spec in
  `nemotron-voice-sm/SOMALI_AUDIO_PROCESSING_PLAN.md` and `nemotron-voice-process/DUPLEX_PROCESSING.md`).

---

## 5. Phase plans

### Phase 1: BayLing-Duplex Somali (GLM family)

| Step | What | Cost / notes |
|---|---|---|
| 1.0 | Zero-shot: run BayLing with the **Omar decoder** on Somali input. Measure latency, barge-in, and whether it answers in Somali (expect not). | an hour; baseline numbers |
| 1.1 | Measure Somali GLM text-tokens/second on Scribe data → is 5:3:5 feasible? | CPU, minutes |
| 1.2 | Build the duplex set (§4): start with ~20–50 h of clean windows, hybrid resynthesis. | TTS render on the GPU; mostly compute |
| 1.3 | Write `train_duplex.py`: our `train_lora.py` + BayLing block builder + weighted loss (0.1 / 10 / 1). Mix ~30% of our existing ASR/TTS/dialogue SFT so Somali isn't forgotten. | code; ~1 day |
| 1.4 | LoRA on BayLing, 10:5:10, 1 epoch. Evaluate (§6). | similar to our round-2 LoRA run |
| 1.5 | Same with **5:3:5** blocks. Compare latency vs quality. | one more run |
| 1.6 | Optional DPO on timing (interrupted-late negatives). | small |

### Phase 2: Nemotron VoiceChat Somali (only if Phase 1 misses targets)

- Train on the same duplex set, exported as Nemotron cuts (separate tracks are mandatory; hybrid resynthesis gives them to us).
- Needs, roughly in order:
  1. Somali in the perception encoder / RNN-T head (our Scribe labels);
  2. Somali in the Nemotron-Nano-9B backbone;
  3. an EARTTS decoder retrained for Somali **and** Omar's voice. Our GLM flow decoder can't be reused, since it's a different codec.
- Training code and configs already exist (`examples/speechlm2/conf/nemotron-labs-voicechat.yaml`, lr 4e-5,
  codec frozen), and the model is licensed OpenMDW-1.1.
- Much bigger compute and risk. Start with a 10–20 h pilot and check whether it learns Somali at all before scaling.

PersonaPlex/Moshi sits in the same bucket as Nemotron (great timing, Somali from scratch, a new voice codec).
It's a fallback if Nemotron's NeMo training proves painful.

---

## 6. How we decide: eval and gates

Held-out Somali conversations (test split), measured automatically:

| Metric | Tool | Target |
|---|---|---|
| Response latency p50 / p90 | timestamps: user speech end (Sortformer/VAD) → first assistant audio | p50 ≤ 0.6 s, p90 ≤ 1.0 s |
| Barge-in stop time p50 | user onset during agent speech → agent silent | ≤ 0.4 s |
| False-interrupt rate | agent stops on back-channel/noise | ≤ 10% |
| Assistant Somali intelligibility | **MMS / Omnilingual CER** on assistant track | ≤ round-2 TTS CER + small margin |
| Voice is Omar | **ECAPA** similarity to Omar reference | ≥ current Omar decoder level |
| One voice per reply, timing sanity | **Sortformer** on the assistant track | 1 speaker, no self-overlap |

(Public reference: Full-Duplex-Bench measures the same pause / back-channel / turn-taking / interruption behaviours.)

**Gate after 1.5:** if the best BayLing variant meets the latency and barge-in targets, ship it and improve the data.
If it's close but slow, try 1.6 (DPO) and serving optimizations (bf16, warm worker, streaming decoder).
If it's far off, start Phase 2 with the same dataset.

---

## 7. What to do right now (no GPU conflict)

1. Keep the Scribe run going; keep `.response.json` (word + speaker + timestamps) untouched.
2. Step 1.1: Somali tokens/second on Scribe text (CPU).
3. After round-2 LoRA finishes: step 1.0 (BayLing zero-shot + Omar decoder, latency numbers).
4. Round-2 LoRA's TTS quality determines hybrid-resynthesis quality. Check its TTS CER first.
