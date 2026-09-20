"""Fold a LoRA adapter into glm-4-voice-9b and save a plain model dir (model_server.py / web_demo.py can't load adapters).

  python merge_lora.py --lora $SO_WORK/runs/lora_somali_scribe/final --out $SO_WORK/merged/lora_somali_scribe
Runs on CPU (~20 GB RAM); output ~18 GB bf16 safetensors + the base's tokenizer/remote code.
"""
import argparse
import shutil
from pathlib import Path

import torch

from common import LLM_PATH, log

L = log("merge_lora")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=float, default=1.0,
                   help="fold the adapter in at this strength (0.3 = weaker Somali accent, more of the base model)")
    a = p.parse_args()
    from peft import PeftModel
    from transformers import AutoModel
    out = Path(a.out)
    base = AutoModel.from_pretrained(str(LLM_PATH), trust_remote_code=True, torch_dtype=torch.bfloat16,
                                     low_cpu_mem_usage=True)
    peft_model = PeftModel.from_pretrained(base, a.lora)
    if a.scale != 1.0:
        # A LoRA delta is scaled by alpha/r before it is added; dialling that down blends the fine-tune
        # with the base weights instead of replacing them.
        n = 0
        for mod in peft_model.modules():
            if hasattr(mod, "scaling") and isinstance(getattr(mod, "scaling"), dict):
                for k in mod.scaling:
                    mod.scaling[k] *= a.scale
                n += 1
        L.info("scaled %d LoRA modules by %.2f", n, a.scale)
    model = peft_model.merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="5GB")
    for f in LLM_PATH.iterdir():            # tokenizer + remote code, exactly as upstream
        if f.is_file() and f.suffix in (".py", ".model", ".json") and "index" not in f.name and f.name != "config.json":
            shutil.copy(f, out / f.name)
    L.info("merged %s -> %s", a.lora, out)


if __name__ == "__main__":
    main()
