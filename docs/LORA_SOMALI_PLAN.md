# LoRA Fine-Tuning GLM-4-Voice-9B for Somali — Implementation Plan

Status: plan only, no training code yet.

## 0. What we're working with

**Model (unchanged pieces)**
- `glm-4-voice-tokenizer`: Whisper-VQ encoder. 16 kHz audio → 12.5 tokens/s from a 16,384-entry codebook. These map to `<|audio_N|>` tokens that are already in the LLM vocab, so **no vocab resize is needed**.
- `glm-4-voice-9b`: a ChatGLM-architecture LLM (`trust_remote_code`). **This is the only thing we LoRA.**
- `glm-4-voice-decoder`: CosyVoice flow-matching + HiFT. Turns audio tokens into a waveform in one fixed voice. Stays frozen.

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

- Coverage: 522 episodes with transcripts, 9,941 windows, **about 110 h**. Channels: AbukarMahdi, adnachannel, arimaheena, as-podcast, astaanmedia. `omar/` is not in `transcripts/`, so it's skipped for now.
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

## 1. Step zero — go/no-go on the frozen audio stack

LoRA on the LLM can't fix a tokenizer or decoder that loses Somali phonetics. Before writing any training code:

1. Take about 50 random Somali segments. Run each through `extract_speech_token` and then `AudioDecoder.token2wav`. Listen.
2. If resynthesized Somali is intelligible, and it should be because the VQ is fairly phonetic, go ahead with LLM-only LoRA.
3. If it isn't intelligible, the decoder or tokenizer needs its own fine-tune. That's a separate project and this plan doesn't cover it.

Script: `finetune/check_resynthesis.py`

## 2. Training tasks (built from the same segments)

The data is TV and podcast conversation, not instruction following. So we teach three skills in the model's native format and mix them:

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

## 5. Eval

**`finetune/eval.py`** on the held-out episodes:
- **ASR:** WER/CER of the model's transcription against the transcripts.
- **TTS / dialogue:**
  - Generate audio, decode it with the frozen decoder, then transcribe it back (the model's own ASR task, or `facebook/mms-1b-all` with `som`). Report CER against the target text.
  - Also save 20 wav samples for listening.
- **Baseline:** the same metrics on the base model with no adapter. That's the number to beat.
- **Regression:** a few English and Chinese prompts from the original demo, to check the adapter didn't wreck them.

## 6. Changes to existing files

| File | Change |
|---|---|
| `model_server.py` | Add a `--lora-path` arg. After loading, `PeftModel.from_pretrained(glm_model, lora_path)`. For bf16, optionally `merge_and_unload()`. For int4, keep the adapter unmerged. |
| `web_demo.py` | Add an optional Somali system prompt ("Respond in Somali…") selectable in the UI. Otherwise unchanged, since the server handles the adapter. |
| `speech_tokenizer/utils.py` | Replace hard-coded `.cuda()` with a `device` param (default `cuda`) so data prep can also run on `mps`/CPU for small tests. |
| `requirements.txt` → new `requirements-train.txt` | Pinned versions compatible with `transformers==4.44.1` / `torch==2.3.0`: `peft` (~0.12), `bitsandbytes` (~0.43), `accelerate` (~0.33), `datasets`, `jiwer`. |
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
  configs/lora_somali.yaml
  README.md
```

## 7. Order of work

1. Write `check_resynthesis.py`, rent a GPU, and listen. **Go/no-go.**
2. Write `prepare_manifest.py` and run it locally. Spot-check 30 random segments: text vs. audio cut.
3. Run `tokenize_audio.py` and `build_samples.py` on the GPU. Sanity-decode 5 samples back to text and wav.
4. Run `eval.py` on the base model for baseline numbers.
5. Smoke-train with `train_lora.py` (200 steps; loss must drop on all 3 tasks), then do the full 2-epoch run.
6. Eval, listen, and iterate on the mix and rank.
7. Wire `--lora-path` into `model_server.py` and demo it.

## 8. Risks

- **Decoder voice/phonetics.** Output uses the decoder's single built-in voice, and Somali-specific sounds (pharyngeals c/x, q) may be blurred by a VQ and decoder trained on Chinese and English. Step 1 checks this.
- **Transcript noise.** Machine transcripts with duplicated overlaps and misspellings mean ASR WER has a floor. Dedupe and filtering help, and a hand-checked test set of about 200 segments would make evals trustworthy.
- **Conversational, not assistant, style.** Dialogue pairs teach the model to talk Somali like TV and podcast hosts, not to follow instructions. If it needs to act as a helpful assistant, a later stage would add Somali instruction data (translated text instructions, spoken through the TTS path).
- **Duplex data used half-duplex.** GLM-4-Voice is turn-based, so the overlap and backchannel structure in `so-duplex` is collapsed into turns.
