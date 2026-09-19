# Somali fine-tuning — runbook

Three trainings, run in this order (design and rationale: [docs/LORA_SOMALI_PLAN.md](../docs/LORA_SOMALI_PLAN.md)):

| Track | What trains | Data | Output |
|---|---|---|---|
| **ASR** | full CTC fine-tune of `facebook/mms-1b-all` (Somali adapter) | 74k Somali segments, ~57 h train | `$SO_WORK/runs/asr_mms/final/` → private HF `lewenberg/mms-1b-somali` |
| **A — language** | bf16 LoRA on `glm-4-voice-9b` | 5 Somali channels, ~110 h (transcripts bucket + episode `clean.flac`) | `$SO_WORK/runs/lora_somali/final/` (PEFT adapter) |
| **B — voice** | full fine-tune of the flow decoder | Omar, ~8.7k transcribed clips | `$SO_WORK/runs/flow_omar/latest/flow.pt` (drop-in for `glm-4-voice-decoder/flow.pt`) → private HF `lewenberg/glm-4-voice-decoder-omar` |

Everything runs in Docker on the GPU VM (RTX PRO 6000 Blackwell, 96 GB; see `deploy/README.md`).
The same commands rebuild everything from scratch on a fresh VM with the `/workspace` volume.

## From zero

```bash
# on the VM (once per VM): clone/pull, build the training image, stop the voice demo
curl -fsSL https://raw.githubusercontent.com/alimaslax/glm4-voice-so/main/deploy/train/bootstrap.sh | bash
```
```bash
# on your laptop: send ONLY the HF token from your local .env to the VM
deploy/train/push_env.sh <vm-ip>
```
```bash
# on your laptop: pull the latest code on the VM and run the pipeline in tmux (session "so-train")
deploy/train/start.sh <vm-ip>
```

Watch it: `ssh -i ~/.ssh/verda_cpu_runner_20260830 root@<vm-ip> tmux attach -t so-train`
(detach with `Ctrl-b d`). Every stage also logs to `$SO_WORK/logs/<stage>.<utc>.log`.

## Stages

`finetune/pipeline.sh [first-stage]` runs these in order and records finished ones in `$SO_WORK/.done/`,
so re-running skips completed work. `FORCE=1` redoes from the given stage. Any single stage:
`finetune/run.sh <stage> [args]`.

| Stage | Does | Writes |
|---|---|---|
| `download` | syncs `lewenberg/so-duplex-transcripts`, `omar/` from `lewenberg/so-duplex-processed`, and only the `clean.flac` of transcribed episodes (windows are just offsets into these) | `$SO_DATA/{transcripts,processed}` |
| `selftest` | checks our prompt ids == the demo's string tokenization | — |
| `prepare_somali` | dedupes overlapping ASR segments, filters, builds speaker turns + dialogue pairs, splits by episode | `somali/manifest.jsonl`, `pairs.jsonl`, `stats.json` |
| `prepare_asr` | Somali segments → 16 kHz int16 audio + normalized text (lowercase, no punctuation) | `asr/{train,val,test}` |
| `prepare_asr_ext` | extra HF Somali ASR sets listed in `configs/asr_mms.yaml` (`extra_datasets`, pinned revisions) → same format; `ext_train` is mixed into training via `train_splits` | `asr/ext_{train,val,test}` |
| `eval_asr_stock` | CER/WER of stock MMS-som on held-out episodes | `eval_asr/stock.json` |
| `train_asr` | MMS full CTC fine-tune (CNN frozen), keeps the best-CER checkpoint | `runs/asr_mms/final/` |
| `eval_asr_ft` | CER/WER of the fine-tuned MMS | `eval_asr/finetuned.json` |
| `publish_asr` | private repo `lewenberg/mms-1b-somali`: commit 1 = untouched base (tag `base`), commit 2 = fine-tune + evals (tag `asr_mms`) | HF |
| `prepare_omar` | single-speaker clips, ECAPA speaker verification against Omar's centroid (waits for the download) | `omar/manifest.jsonl`, `rejected.jsonl`, `stats.json` |
| `tokenize_omar` | Whisper-VQ speech tokens + 22.05 kHz mels | `omar/feats/` |
| `resynth` | go/no-go: 50 Somali + 20 Omar clips through tokenizer → **stock** decoder; A/B wavs; MMS CER (fine-tuned MMS as judge once it exists) | `resynth/stock/` |
| `train_flow` | Track B, renders sample wavs at every checkpoint | `runs/flow_omar/` |
| `resynth_flow` | the same clips through the **Omar** decoder, same judge → compare with `resynth/stock` | `resynth/flow_omar/` |
| `publish_flow` | private repo `lewenberg/glm-4-voice-decoder-omar`: base commit, then `flow.pt` + samples + eval | HF |
| `tokenize_somali` / `build_sft` / `train_lora` | Track A (Somali LoRA on the 9B) | `somali/tokens/`, `somali/sft/`, `runs/lora_somali/` |

Ad hoc: `finetune/run.sh eval_asr --model <hf id or dir> --tag <name>`.
Smoke runs: `finetune/run.sh train_lora --max-steps 50 --run-name lora_smoke`,
`finetune/run.sh train_flow --total-steps 200 --run-name flow_smoke`.
Hyperparameters live in `finetune/configs/*.yaml`.

## Paths (VM defaults)

| Var | Default | Holds |
|---|---|---|
| `SO_DATA` | `/workspace/so-data` | bucket downloads (~28 GB) |
| `SO_WORK` | `/workspace/so-train` | manifests, tokens, mels, datasets, runs, logs |
| `SO_MODELS` | `/workspace/glm-4-voice/models` | stock `glm-4-voice-{9b,tokenizer,decoder}` |

Secrets: `.env` at the repo root (`HF_TOKEN`), git-ignored. Image: `glm4voice-train-blackwell:0.1` =
`glm4voice-runtime-blackwell:0.2` + `finetune/requirements.txt` (pinned by `finetune/constraints.txt`)
+ an isolated `hf` CLI (buckets need huggingface_hub 1.x; transformers 4.44 needs 0.25).

## Listening / pulling results to the laptop

```bash
rsync -av -e "ssh -i ~/.ssh/verda_cpu_runner_20260830" root@<vm-ip>:/workspace/so-train/resynth ./outputs/   # */ALL_*_AB.wav, README.md
rsync -av -e "ssh -i ~/.ssh/verda_cpu_runner_20260830" root@<vm-ip>:/workspace/so-train/runs/flow_omar/latest/samples ./outputs/flow_samples/
```
