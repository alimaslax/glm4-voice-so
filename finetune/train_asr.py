"""Fine-tune MMS-1b-all (Somali adapter) on the Somali segments with CTC.

  python train_asr.py --config configs/asr_mms.yaml [--max-steps 50] [--run-name smoke]

Output: $SO_WORK/runs/<run_name>/ (checkpoints, tb/), final/ = model + processor + adapter.som.safetensors
Re-running resumes from the last checkpoint.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from common import WORK, log

L = log("train_asr")


class Collator:
    def __init__(self, proc):
        self.proc = proc

    def __call__(self, feats):
        audio = [np.frombuffer(f["audio"], dtype=np.int16).astype(np.float32) / 32767 for f in feats]
        batch = self.proc.feature_extractor(audio, sampling_rate=16000, padding=True, return_attention_mask=True,
                                            return_tensors="pt")
        lab = self.proc.tokenizer([f["text"] for f in feats], padding=True, return_tensors="pt")
        batch["labels"] = lab["input_ids"].masked_fill(lab["attention_mask"] == 0, -100)
        return batch


def load_base(cfg):
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    proc = AutoProcessor.from_pretrained(cfg["base_model"], target_lang=cfg["target_lang"])
    model = Wav2Vec2ForCTC.from_pretrained(cfg["base_model"], target_lang=cfg["target_lang"],
                                           ignore_mismatched_sizes=True)
    return proc, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).parent / "configs" / "asr_mms.yaml"))
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--run-name", default=None)
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.max_steps is not None:
        cfg["train"]["max_steps"] = a.max_steps
    run = a.run_name or cfg["run_name"]
    out_dir = WORK / "runs" / run

    import jiwer
    from datasets import load_from_disk
    from safetensors.torch import save_file
    from transformers import Trainer, TrainingArguments, set_seed
    set_seed(cfg["seed"])
    proc, model = load_base(cfg)
    model.config.ctc_loss_reduction = "mean"
    model.config.ctc_zero_infinity = True
    model.freeze_feature_encoder()
    if cfg["mode"] == "adapter":
        model.init_adapter_layers()
        for n, prm in model.named_parameters():
            prm.requires_grad = "adapter" in n or "lm_head" in n
        model.load_adapter(cfg["target_lang"])       # re-load the pretrained Somali adapter as the start point
    L.info("trainable params: %d", sum(x.numel() for x in model.parameters() if x.requires_grad))

    train = load_from_disk(str(WORK / "asr" / "train")).filter(lambda r: r["dur"] <= cfg["max_dur"])
    val = load_from_disk(str(WORK / "asr" / "val"))
    val = val.shuffle(seed=0).select(range(min(cfg["eval_samples"], len(val)))) if len(val) else None
    L.info("train %d clips (%.1f h), val %s", len(train), sum(train["dur"]) / 3600, len(val) if val else 0)

    def metrics(pred):
        ids = pred.predictions
        ids[ids == -100] = proc.tokenizer.pad_token_id      # Trainer pads ragged batches with -100
        lab = pred.label_ids.copy()
        lab[lab == -100] = proc.tokenizer.pad_token_id
        hyp = proc.batch_decode(ids)
        ref = proc.batch_decode(lab, group_tokens=False)
        return {"cer": jiwer.cer(ref, hyp), "wer": jiwer.wer(ref, hyp)}

    t = cfg["train"]
    args = TrainingArguments(
        output_dir=str(out_dir), run_name=run, seed=cfg["seed"], bf16=True, gradient_checkpointing=True,
        num_train_epochs=t["num_train_epochs"], max_steps=t["max_steps"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"], learning_rate=t["learning_rate"],
        warmup_steps=t["warmup_steps"], lr_scheduler_type=t["lr_scheduler_type"], weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"], eval_strategy="steps" if val else "no", eval_steps=t["eval_steps"],
        save_strategy="steps", save_steps=t["save_steps"], save_total_limit=t["save_total_limit"],
        load_best_model_at_end=bool(val), metric_for_best_model="cer", greater_is_better=False,
        group_by_length=True, length_column_name="dur", remove_unused_columns=False,
        dataloader_num_workers=t["dataloader_num_workers"], report_to=["tensorboard"],
        logging_dir=str(out_dir / "tb"),
    )
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val, data_collator=Collator(proc),
                      compute_metrics=metrics,
                      preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(-1))
    trainer.train(resume_from_checkpoint=True if any(out_dir.glob("checkpoint-*")) else None)
    final = out_dir / "final"
    trainer.save_model(str(final))
    proc.save_pretrained(str(final))
    # adapter file too, so from_pretrained(final, target_lang="som") reloads the same (fine-tuned) adapter
    save_file({k: v.detach().cpu().contiguous() for k, v in model._get_adapters().items()},
              str(final / f"adapter.{cfg['target_lang']}.safetensors"))
    L.info("saved %s", final)


if __name__ == "__main__":
    main()
