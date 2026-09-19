# LoRA Fine-Tuning GLM-4-Voice-9B for Somali — Implementation Plan

Status: plan only, no training code yet.

**Goal:** a GLM-4-Voice that understands and speaks Somali, **in Omar's voice**.
- **Track A, language:** QLoRA on the 9B LLM using the multi-speaker Somali corpus (`transcripts/` + `so-duplex-processed/`, about 110 h, 5 channels).
- **Track B, voice:** fine-tune the flow decoder on `~/hf/omar` (about 120 h, single speaker) so every generated audio token is rendered as Omar.

The two tracks are independent: separate models, separate data, and they can run in parallel. They meet only at inference time.

## 0. What we're working with

**Model (unchanged pieces)**
- `glm-4-voice-tokenizer`: Whisper-VQ encoder. 16 kHz audio → 12.5 tokens/s from a 16,384-entry codebook. These map to `<|audio_N|>` tokens that are already in the LLM vocab, so **no vocab resize is needed**.
- `glm-4-voice-9b`: a ChatGLM-architecture LLM (`trust_remote_code`). **This is the only thing we LoRA.**
- `glm-4-voice-decoder`: CosyVoice flow-matching (tokens → 80-bin mel @ 22.05 kHz) + HiFT vocoder (mel → wav).
  - The voice is **baked into the flow weights**: `n_spks: 1`, and `flow_inference.py` always passes `embedding=torch.zeros(1,192)`, so there is no speaker embedding to swap.
  - Changing the voice to Omar therefore means **fine-tuning the flow model** (Track B).
  - The training forward pass already exists (`cosyvoice/flow/flow.py: forward → decoder.compute_loss`), and so does a trainer (`cosyvoice/bin/train.py`, `cosyvoice/utils/executor.py`).

**Inference prompt format** (from `web_demo.py`). Training has to match it exactly:
```
<|system|>\n{system_prompt}<|user|>\n{user_input}<|assistant|>streaming_transcription\n{13 text tok, 26 audio tok, 13 text, 26 audio, ...}<|user|>
```
- `user_input` is either text or `<|begin_of_audio|><|audio_…|>…<|end_of_audio|>`.
- The assistant response alternates 13 text tokens with 26 audio tokens. When the text runs out, the remaining audio tokens are emitted back to back.
- Generation stops at `<|user|>`, so labels must include that token.
- The tokenizer call adds the `[gMASK]<sop>` prefix. Training must tokenize through the same `tokenizer(...)` path.

**Data** (checked on disk)
| Source | Path | Content |
|---|---|---|
| Transcripts (text source of truth) | `~/hf/transcripts/<channel>/<episode>/diarized/` | `transcripts.diarized.json` (window index), `window_XXXX.response.json` (segments: `start`, `end`, `text`, `speaker`), `window_XXXX.txt` |
| Audio | `~/hf/so-duplex-processed/<channel>/<episode>/` | `clean.flac` (24 kHz mono, DeepFilterNet3-denoised, −23 LUFS), `windows/windows.json` + per-window pyannote turns |

- Coverage: 522 episodes with transcripts, 9,941 windows, **about 110 h**. Channels: AbukarMahdi, adnachannel, arimaheena, as-podcast, astaanmedia. This corpus is for Somali language only (Track A).

**Omar (target voice, Track B)**: `~/hf/omar/<video>/`
- 322 videos, 15,799 wavs, **about 120 h**, 24 kHz mono.
- Per clip:
  - `NNNN.wav`
  - `transcripts/NNNN.{json,srt,txt}` (json has `text` plus word-level `start`/`end`/`speaker_id`, `language_code: som`)
  - some folders also have a `manifest.jsonl`
- About 99% of sampled clips have a single `speaker_id`; about 1% have 2–3 speakers (guests or clips of other people). Some videos are clearly not Omar talking (for example an Arabic poetry recitation, or a clip of the Namibian president). We filter those out (§3b).
- Alignment: the transcript `window_id`, `offset_seconds`, and `source_start_sample` match `so-duplex-processed/.../windows/windows.json` exactly. Absolute time for a segment is `offset_seconds + segment.start`, cut from `clean.flac`.
- Known noise we have to handle:
  - **Duplicate or overlapping segments** inside a window. Example: `astaanmedia/barnaamijka_qaraaxo…/window_0005` repeats 25.84–29.9 s with different spellings. Dedupe by time overlap.
  - The ASR `language` field is sometimes wrong (for example `"bs"`). Ignore it; don't filter on it.
  - Speaker IDs are per window (`0,1,2…`) and come from the transcriber, not from pyannote. Use the transcript speakers. They are **not** consistent across windows.
  - Very short backchannels ("Soco.", "Haa") and cross-talk.

**Hardware reality**
- The local machine is an Apple M5 with 24 GB RAM. It can't train: `speech_tokenizer/utils.py` hard-codes `.cuda()`, bitsandbytes 4-bit needs CUDA, and 9B at bf16 is 18 GB before activations.
- Plan: **QLoRA (4-bit NF4 base + bf16 LoRA) on one rented CUDA GPU.**
  - A 24 GB card (4090 / A10 / L4) works with gradient checkpointing and sequences of 2k or less.
  - An A100 40/80 GB is faster and less fiddly.
- Rough budget: about 20M training tokens per epoch (see §2). At ~2k tok/s on QLoRA that's about 3 h per epoch on an A100. Two epochs plus preprocessing come to **roughly $10–30 of GPU time**.

## 1. Step zero — go/no-go on the audio stack

LoRA on the LLM can't fix a tokenizer or decoder that loses Somali phonetics. Before writing any training code:

1. Take about 50 random Somali segments. Run each through `extract_speech_token` and then `AudioDecoder.token2wav`. Listen.
2. If resynthesized Somali is intelligible, and it should be because the VQ is fairly phonetic, go ahead with LLM-only LoRA.
3. If it isn't intelligible: Track B (flow fine-tune on Omar's Somali audio) is already the fix for decoder-side problems. If the **tokenizer** itself loses Somali contrasts (check by listening for c/x/q/dh distinctions), that's a much bigger project, out of scope.
4. Also resynthesize about 20 **Omar** clips with the stock decoder. This gives the "before" sample for Track B.

Script: `finetune/check_resynthesis.py`

## 2. Training tasks (built from the same segments)

The data is TV and podcast conversation, not instruction following. So we teach three skills in the model's native format and mix them:

The audio tokens are mostly content with little timbre (the flow sets timbre), so the LLM learns *what* to say in Somali from many speakers, and Track B decides *who* it sounds like.

| Task | user_input | assistant target | Why | Mix |
|---|---|---|---|---|
| **ASR** | Somali audio tokens (one turn) | Somali transcript text only, then `<\|user\|>` | Model understands Somali speech | 30% |
| **TTS** | Somali text ("Read this aloud: …") | 13:26 interleaved: the transcript text and that turn's audio tokens | Model can *speak* Somali (text→audio token mapping) | 30% |
| **Spoken dialogue** | Speaker A's turn as audio tokens | Speaker B's next turn, 13:26 interleaved (text + audio) | The real voice-chat path: hear Somali, answer in spoken Somali | 40% |

- System prompts are the two strings already used in `web_demo.py`, so behavior carries over to inference unchanged. ASR gets its own short system prompt.
- **Dialogue pairs:** inside a window, merge consecutive segments from the same speaker into turns. Emit (turn_i, turn_{i+1}) when the speakers differ and each turn is 1–20 s. Optionally prepend one earlier exchange as history.
- **Interleaving:** text tokens come from `glm_tokenizer.encode(transcript)`, and audio tokens come from that turn's audio. Pack them as `text[0:13] audio[0:26] text[13:26] audio[26:52] …`, then the remaining text or audio. At 12.5 Hz, 26 audio tokens is about 2.1 s. Somali text takes more BPE tokens per second than Chinese, so text will usually lead. That's fine and matches how inference behaves.
- **Loss masking:** labels are `-100` everywhere except assistant tokens and the final `<|user|>`.
- **Anti-forgetting (optional, recommended):** mix in about 5–10% English/Chinese samples, for example tokenized clips from a small public set, so the adapter doesn't erase the base model's abilities.
- **Size:** 110 h is about 5M audio tokens plus about 1.5M text tokens. With the three tasks reusing the same segments, that's about 15–20M tokens per epoch.
- **Optional, stage 2:** a short extra pass that swaps TTS targets to Omar clips (text → Omar's audio tokens). This pulls the LLM's output prosody and pacing toward Omar's calm monologue style, which suits an assistant voice better than TV cross-talk. Only do it if Track A plus Track B still sound "off" in rhythm. Keep Omar clips out of Track A otherwise, so the language data stays multi-speaker.

## 3. Data prep pipeline (new code, `finetune/`)

1. **`prepare_manifest.py`** (runs locally on CPU)
   - Walks `~/hf/transcripts/*/*/diarized/transcripts.diarized.json`.
   - For each window with `diarization_status == ok`, loads `window_XXXX.response.json` segments.
   - Resolves audio to `~/hf/so-duplex-processed/<channel>/<episode>/clean.flac` and asserts the offset and sample rate match `windows/windows.json`.
   - Cleans:
     - Dedupes overlapping segments: for >50% time overlap, keep the later, more-corrected one.
     - Drops segments shorter than 0.8 s or longer than 25 s.
     - Drops empty text and segments outside a chars-per-second band (about 3–30).
     - Normalizes whitespace. Keeps the Somali orthography as-is, including c/x/q.
   - Builds turns and dialogue pairs.
   - **Split by episode** (not by segment): about 95% train, 3% val, 2% test, stratified by channel.
   - Output: `data/somali/manifest_{train,val,test}.jsonl`, one row per segment with `{id, channel, episode, audio, start_s, end_s, speaker, text}` and absolute times, plus `pairs_*.jsonl`.
2. **`tokenize_audio.py`** (CUDA box)
   - Loads each segment slice from `clean.flac` (24 kHz, resampled to 16 kHz inside `extract_speech_token`).
   - Batches through `WhisperVQEncoder` and writes `audio_tokens: [int]` into `data/somali/tokens_*.jsonl` (or Arrow).
   - One pass over 110 h is minutes to tens of minutes on one GPU.
   - Only the segment slices plus manifests get uploaded to the GPU box, not the 157 GB processed tree. Alternatively, tokenize while streaming from the HF bucket `lewenberg/so-duplex-processed`.
3. **`build_samples.py`**
   - Turns the tokenized segments and pairs into final samples per §2: `input_ids`, `labels`, `task`.
   - Truncates or drops anything over `max_len` (2048).
   - Writes the HF `datasets` format to `data/somali/sft/`.

### 3b. Omar voice data (Track B)

1. **`prepare_omar.py`** (local CPU)
   - Walks `~/hf/omar/*/NNNN.wav` + `transcripts/NNNN.json`.
   - Keeps clips where all words have one `speaker_id`.
   - **Speaker verification:**
     - Compute an ECAPA embedding (`speechbrain/spkrec-ecapa-voxceleb`) per clip and take the median of all clips as Omar's centroid.
     - Drop clips with cosine similarity below about 0.6.
     - Also drop whole videos whose median similarity is low (those are clips of other people).
   - Duration 1–30 s. Drop clips with music or heavy noise: estimate SNR, and if many are noisy run DeepFilterNet3 exactly as the processed corpus did. Loudness-normalize to −23 LUFS.
   - Hold out 20 videos for eval.
   - Output: `data/omar/manifest_{train,val}.jsonl`.
2. **`tokenize_omar.py`** (CUDA)
   - Whisper-VQ speech tokens at 12.5 Hz (same `extract_speech_token`).
   - 80-bin mel **exactly matching the decoder's feature config**: 22.05 kHz, n_fft 1024, hop 256, fmin/fmax from `glm-4-voice-decoder/config.yaml`. Read those values from the yaml; don't hard-code them.
   - Speaker `embedding` = zeros(192), the same as inference.
   - Output fields per sample: `speech_token`, `speech_token_len`, `speech_feat`, `speech_feat_len`, `embedding`. That's what `MaskedDiffWithXvec.forward` consumes.

## 4. Training (new code, `finetune/`)

**`train_lora.py`** uses HF `Trainer` and PEFT:
- Base: `THUDM/glm-4-voice-9b`, `load_in_4bit` NF4 + double quant, bf16 compute (reuse the `BitsAndBytesConfig` already in `model_server.py`).
- `prepare_model_for_kbit_training`, `gradient_checkpointing_enable()`, `enable_input_require_grads()`.
- LoRA settings:
  - `r=64`, `alpha=128`, `dropout=0.05`. Higher rank than usual because this is a new language plus a new text↔audio mapping, not a style tweak.
  - `target_modules=["query_key_value","dense","dense_h_to_4h","dense_4h_to_h"]` (ChatGLM naming; confirm by printing `named_modules()`).
  - Embeddings and `output_layer` stay frozen at first. If Somali TTS stalls, try adding `modules_to_save=["output_layer"]` in a second run.
- Hyperparameters:
  - LR `1e-4`, cosine schedule, 3% warmup.
  - Effective batch of about 64 sequences.
  - 2 epochs.
  - Eval every ~500 steps on the val split, with loss reported per task.
- Collator: pad with the tokenizer pad id, and pad labels with -100.
- Checkpoints save the adapter only (about 200–400 MB). Push to an HF repo such as `alimaslax/glm-4-voice-9b-somali-lora`.
- Config lives in `finetune/configs/lora_somali.yaml`.

### 4b. Flow decoder fine-tune on Omar (Track B)

**`train_flow_omar.py`**
- Load the flow from `glm-4-voice-decoder/flow.pt` using the decoder `config.yaml` (hyperpyyaml, as in `flow_inference.py`) and put it in train mode.
- **Full fine-tune, not LoRA.**
  - The flow is small (roughly 100M params), 120 h of one speaker is plenty, and we want the whole voice changed.
  - Freeze `input_embedding` (the token embedding) for the first ~20% of steps so the token→content mapping doesn't drift, then unfreeze.
- Loss: the existing conditional-flow-matching loss (`decoder.compute_loss`).
  - Keep `training_cfg_rate: 0.2`: classifier-free guidance at inference (`inference_cfg_rate: 0.7`) relies on it.
  - Train with random prompt-prefix masking (`only_mask_loss: True` already supports it). This is required because `stream_inference` feeds the previous chunk's tokens and mel as the prompt, so the model must handle both "no prompt" and "Omar prompt".
- AdamW, LR 1e-5 → 5e-5 with warmup, bf16, and **fp32 for the ODE solver/mel**. Batch by total frames.
- 1 GPU (24 GB is enough), about 20–50k steps. Checkpoint every 2k steps, and generate the same 10 eval sentences each time for listening.
- **HiFT vocoder** stays frozen at first, since HiFT generalizes across speakers well. Only fine-tune it if Omar's timbre has buzzy or metallic artifacts. That needs the HiFiGAN discriminators and adversarial losses, which this repo doesn't include (`cosyvoice/hifigan` is generator-only), so it would come from upstream CosyVoice. Treat it as a stretch goal.
- Output: `outputs/flow_omar/flow.pt`. It's a drop-in replacement for `glm-4-voice-decoder/flow.pt`.
- Streaming: the flow trains on whole clips, while inference uses chunked prompt/overlap (`token_overlap_len`, `fade_in_out`). Test streaming with the same `block_size_list` as `web_demo.py` to catch chunk-boundary artifacts.

## 5. Eval

**`finetune/eval.py`** on the held-out episodes:
- **ASR:** WER/CER of the model's transcription against the transcripts.
- **TTS / dialogue:**
  - Generate audio, decode it with the frozen decoder, then transcribe it back (the model's own ASR task, or `facebook/mms-1b-all` with `som`). Report CER against the target text.
  - Also save 20 wav samples for listening.
- **Baseline:** the same metrics on the base model with no adapter. That's the number to beat.
- **Regression:** a few English and Chinese prompts from the original demo, to check the adapter didn't wreck them.
- **Voice (Track B):**
  - Speaker similarity: ECAPA cosine of generated audio against Omar's centroid. Compare stock decoder vs. fine-tuned, on both held-out Omar tokens (resynthesis) and LLM-generated tokens.
  - Intelligibility: CER via MMS-`som` on resynthesized held-out Omar clips. It must not get worse than the stock decoder.
  - Listening A/B against real Omar clips.

## 6. Changes to existing files

| File | Change |
|---|---|
| `flow_inference.py` | Add an optional `flow_ckpt_path` override (already a ctor arg). Nothing else, since the speaker embedding stays zeros. |
| `model_server.py` | Add a `--lora-path` arg. After loading, `PeftModel.from_pretrained(glm_model, lora_path)`. For bf16, optionally `merge_and_unload()`. For int4, keep the adapter unmerged. |
| `web_demo.py` | Add `--flow-path` (defaults to the stock `flow.pt`; point it at `outputs/flow_omar/flow.pt`) and an optional Somali system prompt. |
| `speech_tokenizer/utils.py` | Replace hard-coded `.cuda()` with a `device` param (default `cuda`) so data prep can also run on `mps`/CPU for small tests. |
| `requirements.txt` → new `requirements-train.txt` | Pinned versions compatible with `transformers==4.44.1` / `torch==2.3.0`: `peft` (~0.12), `bitsandbytes` (~0.43), `accelerate` (~0.33), `datasets`, `jiwer`, `speechbrain` (speaker verification), `pyloudnorm`. |
| `.gitignore` | `data/`, `outputs/`, `*.safetensors` |
| `README.md` | Short "Somali LoRA" section: prep, train, and serve commands. |

New directory:
```
finetune/
  check_resynthesis.py
  prepare_manifest.py
  tokenize_audio.py
  build_samples.py
  dataset.py          # collator + masking helpers
  train_lora.py
  eval.py
  prepare_omar.py     # Track B
  tokenize_omar.py    # Track B
  train_flow_omar.py  # Track B
  configs/lora_somali.yaml
  configs/flow_omar.yaml
  README.md
```

## 7. Order of work

1. Write `check_resynthesis.py`, rent a GPU, and listen to Somali plus Omar resynthesis. **Go/no-go.**
2. Local, CPU, in parallel:
   - `prepare_manifest.py` (Track A)
   - `prepare_omar.py` with speaker filtering (Track B)
   - Spot-check 30 samples from each.
3. On the GPU:
   - Tokenize both corpora: Omar also needs mels.
   - `build_samples.py` for Track A.
   - Base-model `eval.py` for baseline numbers.
4. **Track B first**, since it's cheap and fast and gives the earliest audible win:
   - `train_flow_omar.py` smoke run (1k steps; the voice should already shift).
   - Then the full run.
   - Eval similarity and CER.
5. **Track A:**
   - `train_lora.py` smoke run (200 steps; loss must drop on all 3 tasks).
   - Then the full 2-epoch run.
   - Eval.
6. Combine: LLM + LoRA → audio tokens → Omar flow → HiFT. Test end to end in `web_demo.py` with streaming.
7. Optional: stage-2 Omar TTS pass on the LLM (§2), and/or HiFT fine-tune.

## 8. Risks

- **Tokenizer phonetics.** The VQ tokenizer was trained on Chinese and English, so Somali-specific sounds (pharyngeals c/x, q, dh) may be merged. Step 1 checks this. Track B can't fix it.
- **Omar-data contamination.** Guests, quoted clips, and background music would leak other voices into the flow. The ECAPA filter and per-video rejection handle this. Listen to the rejected and kept boundaries.
- **Flow overfitting or drift.** Too high an LR or too many steps and the flow starts ignoring tokens and mumbling Omar-like sounds, so CER rises. Pick the checkpoint by CER + similarity, not by loss.
- **Transcript noise.** Machine transcripts with duplicated overlaps and misspellings mean ASR WER has a floor. Dedupe and filtering help, and a hand-checked test set of about 200 segments would make evals trustworthy.
- **Conversational, not assistant, style.** Dialogue pairs teach the model to talk Somali like TV and podcast hosts, not to follow instructions. If it needs to act as a helpful assistant, a later stage would add Somali instruction data (translated text instructions, spoken through the TTS path).
- **Duplex data used half-duplex.** GLM-4-Voice is turn-based, so the overlap and backchannel structure in `so-duplex` is collapsed into turns.
