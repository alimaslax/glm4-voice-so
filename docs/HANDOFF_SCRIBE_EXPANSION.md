# Handoff: Scribe expansion (Mac) → audio follow (VM) → round-3 training data

State as of **2026-09-19 18:06 UTC**. Read this top to bottom before touching anything that is running.

Background reading: [docs/DATA_GUIDE.md](DATA_GUIDE.md) (data, formats and why) and [finetune/README.md](../finetune/README.md) (stages).

---

## 1. Goal

Label as much *new* Somali audio as the remaining ElevenLabs credits allow, spread evenly across channels, and
get the matching audio onto the training VM so a bigger round-3 dataset can be built for the Somali LoRA and MMS.

- **Labeler:** ElevenLabs **Scribe v2**, language forced to Somali (`som`), diarization on.
- **Audio source:** bucket `lewenberg/so-duplex-processed`, mirrored locally on the Mac at `~/hf/so-duplex-processed`.
- **Results:** bucket `lewenberg/so-duplex-transcripts/scribe/` (private).

---

## 2. What is running right now (don't restart blindly)

### Mac (`/Users/mali`)

| tmux | What | Log |
|---|---|---|
| `scribe-new` | `~/hf/so-work/run_scribe_local.sh --workers 16` → `finetune/scribe_windows.py --source processed --single` | `~/hf/so-work/scribe_new.log` |
| `scribe-push` | `~/hf/so-work/push_scribe_local.sh`: every 10 min, `hf buckets sync ~/hf/so-work/scribe → hf://buckets/lewenberg/so-duplex-transcripts/scribe` (excludes `*.request.json`, `*.tmp`; **never deletes**) | `~/hf/so-work/scribe_push.log` |
| `hf-so-duplex-sync` | The user's own download of `so-duplex-processed` to `~/hf/so-duplex-processed` (186 GB+). **Not ours, leave it alone** | — |
| `hf-sync-status`, `codex-duplex-health`, `codex_shutdown`, `openrouter-diverse-100h` | Older idle shells. **Not ours, leave them alone** | — |

Progress at handoff: 1,300 windows / 7.9 h of audio in the new run, all `ok`, about 2.3 windows/s.

### VM (`ssh -i ~/.ssh/verda_cpu_runner_20260830 root@86.38.182.93`)

| tmux | What | Log |
|---|---|---|
| `so-r2` | `/workspace/so-train/r2.sh`: **round-2 LoRA training** (`train_lora --config configs/lora_somali_scribe.yaml`, from the `/workspace/glm4-voice-so-r2` checkout). Step ~1,500 / 2,994, finishing about 19:10–19:30 UTC | `/workspace/so-train/logs/train_lora.20260919T170808Z.log`, `/workspace/so-train/r2.log` |
| `so-scribe-audio` | `finetune/scribe_audio_sync.sh` in the training image: every 10 min, pull `scribe/` + `window_status.jsonl`, then download the `clean.flac` of every episode listed in it | `/workspace/so-train/logs/scribe_audio_sync.log` |
| `so-ckpt` | Backs up `/workspace/so-train/runs` to bucket `lewenberg/so-train-checkpoints` every 10 min | `/workspace/so-train/logs/ckpt_sync.log` |

At handoff: 972 episodes listed, 790 `clean.flac` on disk (was 522), 236 GB free.

---

## 3. Credits

| Key (`.env`) | State |
|---|---|
| `ELEVENLABS_API_KEY` | **Empty** (quota 130,136 used). The script tries it first, gets `quota_exceeded`, logs `switching to key 2` |
| `ELEVENLABS_API_KEY2` | Creator tier, **~125,000 credits** at start of this run. Has `user_read`, so the balance is readable |

Cost: about **21.2 credits per audio minute** (~1,275 per audio hour), $1.82 per 10,000 credits, so about **98 h of
audio** for key 2. When key 2 runs out the script logs `stopping: ...quota_exceeded...` and exits cleanly.

Check the balance:
```bash
K=$(grep -E '^ELEVENLABS_API_KEY2=' .env | cut -d= -f2- | tr -d '"'"'"' \r' | tail -1)
curl -s -H "xi-api-key: $K" https://api.elevenlabs.io/v1/user/subscription | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["character_count"], "/", d["character_limit"])'
```

---

## 4. What the Mac run sends and in what order

`scribe_windows.py --source processed --single` plans from `~/hf/so-duplex-processed` directly (not from the MAI
transcript index, which is exhausted):

1. **Duplex windows:** every window in `<channel>/<episode>/windows/windows.json` (two-speaker, ~42 s,
   `selection_mode: two_person_stitched_dialogue`).
2. **New single-speaker windows** (`--single`), from `<episode>/diarization.json`:
   - one speaker's consecutive turns, gap ≤ 1 s, 3–30 s long;
   - no other speaker's turn overlaps;
   - 0.2 s padding that never reaches a neighbouring speaker;
   - never overlapping an existing duplex window (so no audio is paid for twice);
   - IDs `single_0001`, …
3. **Skipped:** every window already in `window_status.jsonl` (the 9,941 original windows, the 20 probe
   windows, and everything this run has done), `omar` (already ElevenLabs), episodes without `clean.flac`.
   Unreadable audio (e.g. still downloading) → `bad_audio`, skipped, not sent.

Plan size at start: 336k windows / ~1,988 h (1,247 h duplex + 741 h single). **Credits cover ~5%.**

**Order:**
- **Channels round-robin** (AbukarMahdi, adnachannel, arimaheena, as-podcast, astaanmedia, audio, biletv).
- **Inside a channel, one episode is finished before the next**, starting with episodes already in
  `window_status.jsonl`.
- **Inside an episode, duplex and single windows alternate.**

Why episode-deep: the first version rotated episodes too, and touched 527 episodes after 600 windows. At ~50 MB
per `clean.flac`, that would have needed ~300 GB on the VM. Episode-deep keeps the total to roughly 1,000–1,300
episodes (~60 GB).

Dry run of the plan (sends nothing): `~/hf/so-work/run_scribe_local.sh --plan-only`.

---

## 5. Output format

Per window, under `~/hf/so-work/scribe/<channel>/<episode>/diarized/` (and the same paths in the bucket):

| File | What |
|---|---|
| `<window>.scribe.json` | raw ElevenLabs reply |
| `<window>.response.json` | same schema as the MAI files: `text`, `segments[{start,end,text,speaker}]`, `words[{word,start,end,speaker}]` |
| `<window>.request.json` | local marker (dispatching / saved / http_*), never uploaded |
| `transcripts.diarized.json` | per-episode index (window, `kind`, offsets in `clean.flac`, speakers, status), written when an episode's queued windows finish and at the end of the run |

**No audio is written anywhere.** Windows are cut from `clean.flac` in memory. Output is ~35 KB per window.

`window_status.jsonl` (one line per window ever attempted, appended as results arrive):
`channel, episode, window, kind (duplex|single), start, duration, status (somali|empty|not_somali_check),
language, language_forced, chars, words, speakers, somali_check{...}, text_preview, at`.
The 9,941 original rows have no `kind` (all duplex).

---

## 6. Rules (from the user)

- **Never `git commit` or `git push` without the user saying so.** The repo `alimaslax/glm4-voice-so` is PUBLIC.
  All changes below are uncommitted on the Mac and copied to the VM checkouts.
- Never commit `.env` or any key (scan diffs for `hf_`, `sk-or-`, `sk_`).
- HF repos and buckets stay **private**.
- Don't save window audio anywhere; the user has no space for it.
- Keep ≥ 25 GB free on the VM `/workspace` (the audio loop stops at 40 GB).
- Don't touch `lewenberg/so-duplex-transcripts/mai/` or `omar/`. Only add to `scribe/`.
- Don't `git pull` or overwrite files in a VM checkout whose script is currently running (bash reads scripts
  as it goes). `so-r2` runs from `/workspace/glm4-voice-so-r2`, `so-scribe-audio` from `/workspace/glm4-voice-so`.
- tmux doesn't pass your shell's variables into a new session: load `.env` *inside* the tmux command
  (`set -a; . ./.env; set +a; docker run -e HF_TOKEN ...`), or the Hub returns 401.

---

## 7. Code changed (uncommitted)

| File | Change |
|---|---|
| `finetune/scribe_windows.py` | `--source processed`, `--single`, `--plan-only`; `processed_episodes`, `single_windows`, `round_robin`; episode-deep ordering; per-result `window_status.jsonl` rows (`status_row`); `bad_audio` guard; key 1 → key 2 fallback; `request_once`/`normalize` take `language` (None = auto) |
| `finetune/common.py` | `SO_PROCESSED` (audio root), `SO_SOMALI` (Track A work dir) |
| `finetune/scribe_audio_sync.sh` | new: the VM follow loop |
| `finetune/scribe_retry_empty.py` | new: probe-first resend of empty windows (probe was 0/20 Somali, rest not sent) |
| `finetune/eval_lora_loss.py` | new: per-task val loss of any adapter on one val set |
| `finetune/{prepare_somali,tokenize_audio,build_sft,first_pass,train_lora}.py`, `run.sh`, `configs/lora_somali_scribe.yaml` | round-2 LoRA: `SO_SOMALI` work dir, `init_adapter` |
| `finetune/prepare_asr_ss.py`, `configs/asr_mms_ss.yaml` | MMS single-speaker agreement filter (CER ≤ 0.15) and config |
| `deploy/train/push_env.sh` | also sends `ELEVENLABS_API_KEY2` |
| `~/hf/so-work/run_scribe_local.sh`, `push_scribe_local.sh` | Mac launch scripts (outside the repo) |
| `~/hf/.venv-scribe` | Mac Python venv (soundfile, numpy) |

---

## 8. How to check on it

```bash
# Mac
tail -2 ~/hf/so-work/scribe_new.log                  # N/total, hours done, win/s
grep -E "stopping|switching|Traceback" ~/hf/so-work/scribe_new.log | tail
tail -2 ~/hf/so-work/scribe_push.log
wc -l ~/hf/so-work/scribe/window_status.jsonl
```
```bash
# VM
ssh -i ~/.ssh/verda_cpu_runner_20260830 root@86.38.182.93 'tail -3 /workspace/so-train/logs/scribe_audio_sync.log; find /workspace/so-data/processed -name clean.flac | wc -l; df -h /workspace | tail -1'
```

Status breakdown of what came back:
```bash
python3 -c "import json,collections; c=collections.Counter((r.get('kind','duplex'),r['status']) for r in map(json.loads,open('$HOME/hf/so-work/scribe/window_status.jsonl'))); print(c)"
```

---

## 9. When the credits run out

1. The Mac log shows `stopping: ... quota_exceeded`, and the script writes all episode indexes and exits.
2. Force one final upload: `ONCE=1 ~/hf/so-work/push_scribe_local.sh`, then stop tmux `scribe-push`.
3. Let the VM loop run one more cycle (or run it once with `ONCE=1`) so every listed episode's `clean.flac` is on
   disk. Then stop tmux `so-scribe-audio`.
4. Check coverage: every `(channel, episode)` in `window_status.jsonl` has
   `/workspace/so-data/processed/<channel>/<episode>/clean.flac`.

---

## 10. Next: round-3 training data (not started)

The Scribe output uses the same schema as MAI, so the existing stages read it with env vars only:

```bash
# on the VM, from a checkout that has the changes above, after round-2 LoRA has finished
export SO_TRANSCRIPTS=/workspace/so-train/scribe SO_SOMALI=somali_scribe3
finetune/run.sh prepare_somali     # windows -> segments / turns / dialogue pairs (single windows give segments only)
finetune/run.sh tokenize_somali    # ~25 min per ~500 episodes
finetune/run.sh build_sft
```

Then either continue the round-2 adapter (a `lora_somali_scribe3.yaml` with `init_adapter: runs/lora_somali_scribe/final`)
or retrain MMS on the Scribe labels. Decisions still open with the user:
- **MMS:** retrain from `facebook/mms-1b-all` (recommended: the current model learned MAI's wrong-language labels)
  or continue from `asr_mms`. Score against **Scribe** test labels, not MAI.
- Whether to drop `not_somali_check` windows (currently kept; the check is informational).
- Whether round 2 beats round 1 in the first-pass listening test (pending, after training finishes).

Round 2 so far (Scribe val loss, lower is better): ASR 2.375 → 1.941, TTS 3.622 → 3.637, dialogue 5.328 → 5.219
(round-1 adapter → round-2 step 1,000).
