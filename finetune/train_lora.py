"""Track A: bf16 LoRA fine-tune of glm-4-voice-9b on the Somali SFT set.

  python train_lora.py --config configs/lora_somali.yaml [--max-steps 200] [--run-name smoke]

Output: $SO_WORK/runs/<run_name>/  (Trainer checkpoints = adapter only, tensorboard logs, final/)
Re-running resumes from the last checkpoint automatically.
"""
import argparse
from pathlib import Path

import torch
import yaml

from common import LLM_PATH, WORK, log

L = log("train_lora")


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids = torch.full((len(feats), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(feats), n), -100, dtype=torch.long)
        mask = torch.zeros((len(feats), n), dtype=torch.long)
        for i, f in enumerate(feats):
            k = len(f["input_ids"])
            ids[i, :k] = torch.tensor(f["input_ids"])
            labels[i, :k] = torch.tensor(f["labels"])
            mask[i, :k] = 1
        return dict(input_ids=ids, labels=labels, attention_mask=mask)


def build_trainer_cls():
    from transformers import Trainer

    class SFTTrainer(Trainer):
        # Loss only over supervised positions: never upcasts the full [B, T, 168960] logits to fp32.
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
            logits = out.logits[:, :-1]
            tgt = labels[:, 1:]
            sel = tgt != -100
            loss = torch.nn.functional.cross_entropy(logits[sel].float(), tgt[sel])
            return (loss, out) if return_outputs else loss

    return SFTTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).parent / "configs" / "lora_somali.yaml"))
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--run-name", default=None)
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.max_steps is not None:
        cfg["train"]["max_steps"] = a.max_steps
    run = a.run_name or cfg["run_name"]
    out_dir = WORK / "runs" / run

    from datasets import load_from_disk
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer, TrainingArguments, set_seed
    set_seed(cfg["seed"])

    tok = AutoTokenizer.from_pretrained(str(LLM_PATH), trust_remote_code=True)
    kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
    try:
        model = AutoModel.from_pretrained(str(LLM_PATH), attn_implementation=cfg["attn_implementation"], **kw)
    except (ValueError, ImportError) as e:
        L.warning("attn_implementation=%s rejected (%s); using eager", cfg["attn_implementation"], e)
        model = AutoModel.from_pretrained(str(LLM_PATH), **kw)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], bias="none",
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    sft = WORK / "somali" / "sft"
    train = load_from_disk(str(sft / "train"))
    val = load_from_disk(str(sft / "val"))
    evals = {t: val.filter(lambda r, t=t: r["task"] == t).select(
        range(min(cfg["eval_per_task"], sum(1 for x in val["task"] if x == t)))) for t in ("asr", "tts", "dialogue")}
    L.info("train %d samples; eval %s", len(train), {k: len(v) for k, v in evals.items()})

    t = cfg["train"]
    args = TrainingArguments(
        output_dir=str(out_dir), run_name=run, seed=cfg["seed"], bf16=True,
        num_train_epochs=t["num_train_epochs"], max_steps=t["max_steps"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"], lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"], weight_decay=t["weight_decay"], max_grad_norm=t["max_grad_norm"],
        logging_steps=t["logging_steps"], eval_strategy="steps", eval_steps=t["eval_steps"],
        save_strategy="steps", save_steps=t["save_steps"], save_total_limit=t["save_total_limit"],
        dataloader_num_workers=t["dataloader_num_workers"], group_by_length=True, length_column_name="length",
        remove_unused_columns=False, report_to=["tensorboard"], logging_dir=str(out_dir / "tb"),
        prediction_loss_only=True,      # never gather [B, T, 168960] logits during eval
        gradient_checkpointing=False,   # enabled on the model above (remote code handles it itself)
    )
    trainer = build_trainer_cls()(model=model, args=args, train_dataset=train, eval_dataset=evals,
                                  data_collator=Collator(tok.pad_token_id))
    has_ckpt = any(out_dir.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    trainer.save_model(str(out_dir / "final"))
    tok.save_pretrained(str(out_dir / "final"))
    L.info("saved adapter to %s", out_dir / "final")


if __name__ == "__main__":
    main()
