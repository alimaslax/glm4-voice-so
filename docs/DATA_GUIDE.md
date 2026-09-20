# Somali GLM-4-Voice: data, formats, and why

A guide to what data goes into each part of the Somali GLM-4-Voice project, what format each part needs,
and the reasoning behind every choice. Numbers are from the runs on 2026-09-19.

For the commands, see [finetune/README.md](../finetune/README.md). For the MMS fine-tune details, see
[finetune/mms/README.md](../finetune/mms/README.md).

---

## 1. The big picture

GLM-4-Voice is not one model. It is three models in a chain, and each one needs different data:

```
 you speak ──► [1] SPEECH TOKENIZER ──► audio tokens ──► [2] LLM (9B) ──► text + audio tokens ──► [3] DECODER ──► speech
               (Whisper-VQ, frozen)     12.5 per second    GLM-4-Voice        interleaved            flow + HiFT vocoder
                                                           + Somali LoRA                             + Omar's voice
```

| Part | What it does | Trained? | Our change | Data it needs |
|---|---|---|---|---|
| **Speech tokenizer** | Turns audio into discrete tokens (16,384 possible, 12.5 per second) | No, frozen | none | none; we only *use* it to convert our audio into tokens |
| **LLM, 9B** | Reads audio or text tokens, writes text and audio tokens | Yes, **LoRA** | teaches it Somali | Somali speech tokens paired with Somali text |
| **Flow decoder** | Turns audio tokens into a mel spectrogram (the sound of a voice) | Yes, **full fine-tune** | makes it sound like Omar | Omar's speech tokens paired with Omar's mel spectrograms |
| **HiFT vocoder** | Turns the mel spectrogram into a waveform | No | none | none |

Separate from the chain there is also:

| Part | What it does | Trained? | Data it needs |
|---|---|---|---|
| **MMS-1b-all (Somali)** | Speech-to-text model used as a **judge**: it transcribes generated speech so we can measure whether it is intelligible Somali (CER) | Yes, full fine-tune | Somali audio paired with Somali text |

**Why split the work this way:** the LLM decides *what* is said (language, content), and the decoder decides
*how it sounds* (voice). Because the audio tokens carry content but mostly not speaker identity, we can teach
Somali from many speakers (the LLM) and teach one voice (the decoder) from Omar alone.

---

## 2. The raw data (Hugging Face, all private)

### 2a. Audio: bucket `lewenberg/so-duplex-processed`

| Folder | What | Used by |
|---|---|---|
| `<channel>/<episode>/clean.flac` | Full episodes from 5 Somali channels (AbukarMahdi, adnachannel, arimaheena, as-podcast, astaanmedia), 24 kHz mono, background cleaned | LLM (LoRA), MMS |
| `omar/<video>/NNNN.wav` | Omar's podcast, already cut into short clips (170 videos) | Flow decoder |

We never copy or re-upload audio. Every clip in the pipeline is just **a start and end time inside a
`clean.flac`**, cut in memory when needed. That keeps disk small and means one audio file serves every
purpose.

### 2b. Transcripts: bucket `lewenberg/so-duplex-transcripts`

The episodes were split into **windows** of about 42 seconds, and each window was transcribed with speaker
labels (diarization). There are two transcriptions of the same windows, plus Omar's:

| Folder | Transcriber | Windows | Notes |
|---|---|---|---|
| `mai/` | **MAI-Transcribe-2** (Microsoft, via OpenRouter) | ~9,900 | MAI **does not support Somali**. With auto language it often guessed Urdu, Estonian, Arabic, English... Some windows are good Somali, many are not |
| `scribe/` | **ElevenLabs Scribe v2**, language forced to Somali (`som`) | 9,941 | Much cleaner Somali. 7,983 pass the Somali check (88.5 h), 1,066 came back empty (music/effects-heavy), 892 fail the check. See `scribe/window_status.jsonl` |
| `omar/` | ElevenLabs Scribe (per clip) | 170 videos | Omar's clips, one `.json/.srt/.txt` per `.wav` |

**Window file format** (same schema for MAI and Scribe, so the pipeline can read either one):

```
<channel>/<episode>/diarized/
  transcripts.diarized.json     index: every window, its offset in clean.flac, status, response file
  window_0001.response.json     {"text": ..., "segments": [{start, end, text, speaker}], "words": [{word, start, end, speaker}]}
  window_0001.scribe.json       (Scribe only) the raw ElevenLabs reply, kept for re-processing
```

Times inside a window are relative to the window; `offset_seconds` in the index converts them to absolute
times in `clean.flac`.

**Why we relabeled with Scribe:** we tested the "failed" MAI windows with Muse (no Somali, answered in
Arabic script) and Scribe v2 (real Somali when forced to `som`). See [outputs/muse/README.md](../outputs/muse/README.md).
Cost: about 140,000 credits for 110 h (about $25, or about 23 cents per audio hour).

### 2c. Single-speaker clips: bucket `lewenberg/so-single-speaker-transcripts`

Clips cut from the MAI diarization so each holds **one speaker only** (2 to 20 s), each re-sent to MAI on its
own. 44,241 clips sent, 27,304 kept (48.1 h). Code: [finetune/single_speaker.py](../finetune/single_speaker.py).
Meant as extra, cleaner MMS training data.

### 2d. Outside dataset: `IbrahimDayax/somali-combined-asr-stt-dataset`

Public Somali ASR set (read speech plus synthetic TTS), pinned to one revision for repeatability. About 4.8 h
of training audio, plus its own test split, which we report as `ext_test`. Used only for MMS.

---

## 3. Splits: how we keep test data honest

```python
split_of(key)  # finetune/common.py: sha1 hash of the EPISODE name -> 95% train, 3% val, 2% test
```

- The split is by **whole episode**, never by clip. If clips from one episode landed in both train and test,
  the model could "recognize" the episode (same speakers, same topic, same room) and the test score would
  look better than it really is.
- It is a **hash**, not random: the same episode always lands in the same split, on any machine, in every
  run. So the MAI build and the Scribe build share the same test episodes, and results stay comparable.
- Omar uses the same function on the video name.

---

## 4. From windows to clips: `prepare_somali.py`

Both the LoRA data and the MMS data start here. Input: one set of window transcripts (MAI or Scribe).
Output: `$SO_WORK/<somali dir>/manifest.jsonl` and `pairs.jsonl`.

It produces two kinds of clips:

| Kind | How it is made | Used for |
|---|---|---|
| `segment` | One transcript segment (one speaker, one sentence-ish) | ASR and TTS samples, MMS |
| `turn` | Consecutive segments from the same speaker merged (gap ≤ 1 s, ≤ 20 s total) | Dialogue pairs |

A **dialogue pair** is turn A followed by turn B, where B is a *different* speaker who starts within 3 s:
"someone says something, someone else answers". This is the only place the model learns to reply.

Filters on every clip, and why:

| Filter | Value | Why |
|---|---|---|
| Duration | 0.8 to 20 s | Too short carries no content; too long exceeds the LLM sample length |
| Characters per second | 3 to 30 | Catches transcripts that don't match the audio (hallucinated or missing text) |
| Has letters | yes | Drops `...` and number-only segments |
| Window status | `ok` | Windows with no speaker labels can't be split into turns |
| Dedupe | overlap > 50% | MAI sometimes re-emitted a time span with corrected text; keep the later one |

Example row (Scribe build):

```json
{"id": "AbukarMahdi/ahmed_naji_.../window_0001/s000", "kind": "segment", "split": "train",
 "audio": "AbukarMahdi/ahmed_naji_.../clean.flac", "start": 56.38, "end": 63.98, "speaker": 0,
 "text": "Waxyaabaha uu ka hadlayo in ay ahayd waxyaabaha tan bulshada Soomaaliyeed u baahan yihiin, ..."}
```

| Build | Segment hours (train) | Turn hours (train) | Dialogue pairs (train) |
|---|---|---|---|
| MAI (`somali/`, round 1) | 57.2 | 38.4 | 25,795 |
| Scribe (`somali_scribe/`, round 2) | 51.5 | 33.5 | 22,498 |
| Full Scribe set (`somali_scribe4/`) | 116.1 | 53.5 | 35,364 |
| **Round 3 (`somali_tts6/`, `--no-pairs`)** | **116.1** | none | **0** |

Scribe gives slightly less than MAI per window, because it returned nothing for music-heavy windows, but the
text is actually Somali.

**Round 3 builds no turns and no pairs at all** (`prepare_somali.py --no-pairs`). Why is in
[section 5d](#5d-round-3-asr--tts-only-and-why-the-dialogue-task-was-removed).

Two flags control this:

| Flag | Effect |
|---|---|
| `--no-pairs` | Skip turn merging and dialogue pairs entirely; `pairs.jsonl` is written empty |
| `--max-speakers N` | Drop any window the diarizer found more than N speakers in |

---

## 5. LoRA (the 9B LLM): format and choices

### 5a. Step 1: audio to tokens (`tokenize_audio.py --corpus somali`)

Every clip is cut from `clean.flac` and run through the frozen Whisper-VQ tokenizer:

```json
{"id": "as-podcast/.../window_0001/s000", "tokens": [10815, 10057, 7988, 8342, 2946, ...]}
```

12.5 tokens per second, so a 10 s clip is 125 tokens. Stored as one `.jsonl` per episode (resumable).

### 5b. Step 2: tokens to chat samples (`build_sft.py`)

The model is trained on **exactly the chat format the demo uses at inference**. If training and inference
formats differ even slightly, the model behaves unpredictably. `selftest` checks that our token ids equal
what the demo's own string tokenization produces.

Every sample is one conversation turn:

```
[gMASK]<sop><|system|>\n{SYSTEM}<|user|>\n{USER}<|assistant|>{mode}\n{TARGET}<|user|>
└──────────────────── prompt: loss masked (-100) ─────────────────────┘└─ learned ─┘
```

The final `<|user|>` is how GLM-4-Voice marks "my turn is over". The model must learn to emit it, or it
never stops talking.

**Three tasks built from the same Somali clips:**

| Task | System prompt | User turn | Assistant target | Teaches |
|---|---|---|---|---|
| `asr` | "User will provide you with a speech in Somali. Transcribe it into Somali text." | `<\|begin_of_audio\|>` + audio tokens + `<\|end_of_audio\|>` | Somali text | **Understanding** Somali speech |
| `tts` | GLM's text-mode system prompt | "Si cod ah u akhri qoraalkan: *text*" ("Read this aloud: ...") | Interleaved: 13 text tokens, 26 audio tokens, repeat | **Speaking** Somali: which sounds go with which words |
| `dialogue` | GLM's speech-mode system prompt | Speaker A's audio tokens | Speaker B's reply, interleaved text and audio | **Conversation** in Somali |

**Why interleaved 13 : 26?** GLM-4-Voice streams: it writes a little text, then the audio for it, then more
text. 13 text tokens followed by 26 audio tokens is the ratio it was pre-trained with (26 audio tokens is about
2 s of speech). Using the same ratio means we teach Somali without re-teaching the streaming mechanism.

**Why ASR and TTS too, not only dialogue?** Dialogue is the goal, but it is the hardest task and has the
least data (22k pairs). ASR and TTS use every clip (85k each) and teach the two halves dialogue needs:
hear Somali, and produce Somali. The first pass showed exactly this: ASR and TTS work, dialogue lags.

**Round 3 drops the dialogue task**: `build_sft.py --tasks asr,tts`. The task selector makes it impossible
for dialogue samples to reach the dataset even if `pairs.jsonl` is non-empty. See
[section 5d](#5d-round-3-asr--tts-only-and-why-the-dialogue-task-was-removed).

Stored as HF datasets (`input_ids`, `labels`, `task`, `length`), max 1,024 tokens per sample.

| Build | Train samples | Train tokens |
|---|---|---|
| Round 1 (MAI) | 165,405 (asr 69,805, tts 69,805, dialogue 25,795) | 20.3 M |
| Round 2 (Scribe) | 191,636 (asr 84,569, tts 84,569, dialogue 22,498) | 20.6 M |
| **Round 3 (full Scribe, asr+tts)** | **337,284 (asr 168,642, tts 168,642, dialogue 0)** | **36.0 M** |

### 5c. Step 3: the LoRA (`train_lora.py`, `configs/lora_somali*.yaml`)

| Choice | Value | Why |
|---|---|---|
| Method | **LoRA**, not a full fine-tune | A full fine-tune of 9B needs optimizer state for all 9B weights, far more than one GPU holds. LoRA trains 1.7% of the weights (169 M) and fits on one 96 GB GPU in bf16 |
| Rank / alpha | r = 64, α = 128 | Learning a new language is a big change, so a larger rank than the usual 8 to 16 |
| Target layers | `query_key_value`, `dense`, `dense_h_to_4h`, `dense_4h_to_h` | All attention and MLP layers: language knowledge lives mostly in the MLPs |
| Precision | bf16 base, not 4-bit | 4-bit (QLoRA) saves memory but costs quality; the GPU has room for bf16 |
| Batch | 8 × 8 accumulation = 64 | Stable gradients across mixed task lengths |
| LR | 1e-4 cosine (rounds 1 and 3), 5e-5 (round 2) | Round 2 refines an adapter that already knows Somali, so it uses half; round 3 starts from the base model again, so it goes back to 1e-4 |
| Epochs | 2 (rounds 1 and 3), 1 (round 2) | |
| Gradient checkpointing | `SO_GRAD_CKPT` (default on) | On = recompute activations, ~30% slower but much less memory. Measured on round 3: **2.27 s/step on, 1.54 s/step off**, 31 GB vs 77 GB of the 97 GB card. Leave it off whenever the GPU has room |
| Loss | cross-entropy only on target positions | Never computes fp32 logits over the whole 168,960-token vocabulary for the prompt; saves a lot of memory |
| Eval | 300 val samples **per task** | One overall loss would hide that dialogue is lagging |

**Round 1 → round 2:** round 2 loads round 1's adapter (`init_adapter: runs/lora_somali/final`) and
continues on the Scribe data. Round 1 learned Somali from noisy MAI labels; round 2 corrects it with clean
ones instead of starting over.

**Output:** `adapter_model.safetensors` (678 MB) + `adapter_config.json`. At inference:
`PeftModel.from_pretrained(base_9b, adapter)`. Published to `lewenberg/glm-4-voice-9b-somali-lora`.

### 5d. Round 3: ASR + TTS only, and why the dialogue task was removed

Round 2 sounded right and said nothing useful. In the live demo it spoke fluent Somali with correct
captions, but it **echoed the user** and produced non-sequiturs.

The cause is the dialogue task, not the base model. The base 9B is already an instruction-tuned assistant:
it answers questions, follows a system prompt, keeps turns short. Our fine-tune pushed that out of the way.
The dialogue objective was literally *"given what one podcast guest just said, produce what the other guest
said next"* — 22–35 k pairs of it, with r=64 adapters on every attention and MLP projection, for two rounds.
Podcast guests agree, interrupt, repeat the last phrase back and change the subject. That is exactly the
behaviour the demo showed. It learned what we taught it.

So round 3 changes three things:

| Change | Round 2 | Round 3 | Why |
|---|---|---|---|
| Dialogue task | 22,498 pairs | **removed** | It taught conversation *continuation*, which the model mistook for answering |
| Starting point | round 1's adapter | **stock 9B** (`init_adapter` unset) | Round 1 and 2 both trained on dialogue; continuing from them would carry the habit forward |
| Data | Scribe relabel, 51.5 h | **full Scribe set, 116.1 h** | All 1,097 episodes, after the index repair |

What survives is what the LLM actually needs from us: **ASR** teaches it to hear Somali, **TTS** teaches it
to speak Somali. Neither says anything about how to behave in a conversation, so the base model's own
assistant behaviour is left alone.

**On "multi-speaker" data.** We first built round 3 with `--max-speakers 1`, dropping every window the
diarizer found two voices in. That produced 49,701 clips / 42.2 h — a third of the data. Checking what those
windows actually contain showed the filter was wrong: **Scribe emits no overlapping segments at all**
(0.0 h of overlap across the whole corpus). "2 speakers" means two people taking turns inside the window,
not talking over each other, and every individual clip is still one voice. That exclusion was throwing away
**78.1 h of clean single-voice audio** for no benefit, because ASR and TTS never use the speaker label. The
final build keeps all windows and just drops the pairs.

**Commands:**

```bash
export SO_TRANSCRIPTS=$SO_WORK/scribe_all SO_SOMALI=somali_tts6
finetune/run.sh prepare_somali --no-pairs
finetune/run.sh tokenize_somali
finetune/run.sh build_sft --tasks asr,tts
SO_GRAD_CKPT=0 finetune/run.sh train_lora --config configs/lora_somali_tts.yaml
```

**Run:** 337,284 samples, 10,540 steps (2 epochs, effective batch 64), fresh adapter, lr 1e-4 cosine.
Per-task validation loss:

| Step | ASR | TTS |
|---|---|---|
| 500 | 2.414 | 5.164 |
| 1000 | 2.164 | 4.384 |
| 1500 | 2.057 | 4.153 |
| 5500 | 1.735 | 3.475 |
| 6000 | 1.717 | 3.455 |
| 6500 | 1.692 | 3.441 |
| (round 1 final, for reference) | 2.375 | 3.622 |
| (round 2 final, for reference) | 1.787 | 3.420 |

Cross-round numbers are indicative only: round 3 evaluates on `somali_tts6`'s validation split, which is a
different, larger set of clips than rounds 1 and 2 used.

**The run is unfinished.** It was stopped at **checkpoint-6500 of 10,540** (62%, into epoch 2) to listen to
it, and the host was shut down before the remaining 4,040 steps ran. The checkpoint keeps optimizer state,
so `bash /workspace/so-train/tts6c.sh` resumes it to a full 2 epochs and then merges. Back
`runs/lora_somali_tts/checkpoint-6500` up to the bucket first — training stopped between sync passes, so the
bucket holds only 3000-6000.

### 5f. What round 3 proved: audio input now means "transcribe"

Listening to checkpoint-6500 through the orb demo, the model replies to speech with a **transcript of what
was just said**, and emits no audio at all. Probing the same Somali test clip in four prompt shapes shows why:

| Prompt | Output |
|---|---|
| speech in, Somali assistant system prompt | 22 text, 0 audio — a transcript |
| speech in, GLM's own speech system prompt | 15 text, 0 audio — a transcript |
| speech in, our ASR system prompt | 23 text, 0 audio — a transcript |
| **text in, our TTS instruction** | 16 text + **26 audio** — speaks it correctly |

No system prompt changes the behaviour, and merging the adapter at 0.3 or 0.6 only moves how strongly it
does it. TTS is unaffected and good.

The cause is the task mix, not the training length. 168,642 ASR samples teach *audio in → text out*. The only
task that ever mapped *audio in → audio out* was dialogue, and round 3 removed it. With nothing else
claiming speech input, ASR owns it, so speaking to the model triggers transcription.

This is the real lesson of the round, and it constrains the next one: **removing the dialogue task removes
the conversational shape entirely.** The fix is not a different adapter strength or more steps — it is data
shaped like the target behaviour (a spoken Somali question paired with a spoken Somali answer), or keeping
the conversational path on the base model and using the LoRA only to speak.

### 5e. Merging for inference, and adapter strength

`model_server.py` and the demos load a plain model directory, not an adapter, so the LoRA is folded into the
base weights first:

```bash
finetune/run.sh merge_lora --lora $SO_WORK/runs/lora_somali_tts/final \
                           --out $SO_WORK/merged/lora_somali_tts_s03 --scale 0.3
```

`--scale` multiplies each LoRA module's `scaling` (the α/r factor) before the merge, so the fine-tune is
**blended** with the base weights instead of replacing them: 1.0 is the full adapter, 0.3 is a third of the
delta. Lower values keep more of the base model's original behaviour at the cost of a weaker Somali accent.
Verified numerically: merging at 0.3 gives a weight delta with exactly 0.30× the norm of the full merge.

This is a knob for the symptom, not the cause. If a fine-tune has to be dialled down to behave, the training
data is usually the real problem — which is what round 3 addresses.

---

## 6. Flow decoder (Omar's voice): format and choices

### 6a. Choosing Omar's clips (`prepare_omar.py`)

| Filter | Value | Why |
|---|---|---|
| Single speaker | every word has the same `speaker_id` | A guest's voice would blur Omar's |
| Duration | 1 to 30 s | |
| **Speaker verification (ECAPA)** | each clip ≥ 0.55 cosine similarity to Omar's average voice; each video ≥ 0.60 median | Catches clips of someone else (2 whole videos were dropped: a clip of a Namibian president and one of another host) |

Result: **8,163 clips, 61.1 h** (7,499 train / 664 val).

### 6b. Format (`tokenize_audio.py --corpus omar`)

For every clip, **two views of the same audio**:

```python
{clip_id: {"tokens": int16 tensor  (12.5 Hz audio tokens: the input),
           "mel":    fp16 tensor   (80 × frames at 22,050 Hz, hop 256: the target)}}
```

The decoder learns: "given these audio tokens (what is said), produce this mel spectrogram (how Omar says
it)". Mel settings must match CosyVoice exactly (`MEL` in `common.py`), because the frozen HiFT vocoder
only understands mels made that way.

### 6c. Training (`train_flow.py`, `configs/flow_omar.yaml`)

| Choice | Value | Why |
|---|---|---|
| What trains | `flow.pt` only, fully | The flow is small enough to fully train; the vocoder is generic and stays frozen |
| Steps | 30,000, LR 3e-5, warmup 500 | Low LR: adapting a voice, not learning from scratch |
| Frozen embedding for first 6,000 steps | | Keeps "which token means which sound" fixed while the voice adapts, so content doesn't drift |
| Batch by frames | 24,000 mel frames (~280 s of audio) | Clip lengths vary a lot; batching by frames keeps memory constant |

Result: val loss 0.950 → 0.573. Resynthesis intelligibility (MMS CER) improved from 0.320 (stock voice) to
0.281 (Omar). Published to `lewenberg/glm-4-voice-decoder-omar`. Drop-in: replace `flow.pt`.

---

## 7. MMS (the Somali judge): format and choices

### 7a. Format (`prepare_asr.py`)

HF datasets on disk, `$SO_WORK/asr/<split>/`:

| Column | Example | Why this form |
|---|---|---|
| `id` | `AbukarMahdi/<episode>/window_0012/s004` | trace any row back to its window |
| `audio` | raw **int16 bytes**, 16 kHz mono | MMS expects 16 kHz. Bytes instead of a float list: Arrow writes them ~100× faster, and half the size |
| `text` | `haa shakin waaye meeshan waal buuxaya` | **normalized**: lowercase, letters/digits/`'`/`-` only. Matches what the MMS Somali head can output (it has no punctuation or capitals), so the loss never punishes it for something it can't produce |
| `text_raw` | `Haa shakin waaye, meeshan waal buuxaya.` | kept for reading |
| `dur` | seconds | length filtering and hour counts |

Splits: `train/val/test` (our episodes), `ext_train/ext_val/ext_test` (IbrahimDayax), `ss_*` (single-speaker).

### 7b. Training (`train_asr.py`, `configs/asr_mms.yaml`)

| Choice | Value | Why |
|---|---|---|
| Base | `facebook/mms-1b-all` + its Somali adapter | Already has a Somali head that knows some Somali (CER 0.34 on our test), so it starts far from zero |
| Mode | **full fine-tune**, CNN feature encoder frozen (961 M trainable) | The adapter alone (2 M params) is too small to absorb this much new data; the CNN is generic audio features |
| Loss | CTC | MMS's native loss: per-frame letters, no language model |
| LR / epochs | 3e-5, 3 epochs | about 65 min on 62 h |
| Keep | best val CER checkpoint | |

Results (CER / WER):

| Test set | Stock | Fine-tuned |
|---|---|---|
| Our test episodes | 0.341 / 0.788 | **0.208 / 0.604** |
| IbrahimDayax test | 0.100 / 0.396 | **0.056 / 0.226** |

Caveat: our test labels are MAI text, which is itself noisy, so the real accuracy on our episodes is uncertain.
The next step is retraining on the Scribe labels and scoring against Scribe test labels.

---

## 8. Quality checks along the way

| Check | Where | What it catches |
|---|---|---|
| `selftest` | before any LoRA data | our token ids ≠ the demo's (would silently break training) |
| Somali word check | `single_speaker.py: somali_check` | non-Latin script > 2%, English > 30%, or too few common Somali words. Recorded, not a filter |
| Agreement filter | `prepare_asr_ss.py` | single-speaker text vs original window text CER > 0.15 → drop (two passes disagreeing usually means a wrong-language guess) |
| Resynthesis | `check_resynthesis.py` | real audio → tokens → decoder → MMS. Tells whether the *decoder* keeps speech intelligible, separately from the LLM |
| First pass | `first_pass.py` | held-out episodes: stock vs LoRA on dialogue, TTS and ASR, voiced by Omar's decoder, A/B wavs to listen to |

**Judge the model through the demo, not only the loss.** Two failures in round 2 were invisible in the
training curves:

1. **Greedy decoding.** `model_server.py` called `generate()` without `do_sample=True`, so the
   `temperature` and `top_p` it was passed were silently ignored. Every reply was the argmax path, and a
   model fine-tuned on short conversational turns falls into loops that way ("Waa yahay. Waa yahay. ..."). It
   now passes `do_sample` and a `repetition_penalty` (default 1.1, `GLM_REPETITION_PENALTY` to override).
2. **The wrong system prompt.** `orb_demo.py`'s built-in prompt is English, so the Somali model answered in
   English. `GLM_LANG=so` selects a Somali prompt; `GLM_SYSTEM_PROMPT` overrides it entirely.

Neither of these changes a single loss number. Listen to the model before drawing conclusions about the
training data.

---

## 9. Where everything lives

| What | Where |
|---|---|
| Code | `finetune/` (one script per stage; `run.sh <stage>` runs it in Docker) |
| Configs | `finetune/configs/*.yaml` |
| Working data on the VM | `/workspace/so-train/` (`somali/`, `somali_scribe/`, `somali_scribe4/`, `somali_tts6/`, `omar/`, `asr*/`, `single_speaker/`, `scribe_all/`, `runs/`) |
| Merged models for the demo | `/workspace/so-train/merged/` (`lora_somali_scribe`, `lora_somali_tts_s03`, `lora_somali_tts_full`; ~18 GB each) |
| Raw downloads on the VM | `/workspace/so-data/` |
| Models | `lewenberg/mms-1b-somali`, `lewenberg/glm-4-voice-decoder-omar`, `lewenberg/glm-4-voice-9b-somali-lora` |
| Checkpoint backup | bucket `lewenberg/so-train-checkpoints` (every 10 min) |
| Listening files | `outputs/first_pass/`, `outputs/resynth_*`, `outputs/muse/` |
