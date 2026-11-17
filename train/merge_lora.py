#!/usr/bin/env python3
"""Fold a LoRA adapter into the full-precision base model.

The exporter needs an ordinary checkpoint, not a 4-bit one with an adapter
beside it. This loads the base model in float32 (not the 4-bit copy used for
training), adds the adapter's weights into it, and saves the result in the
base model's own format (bf16 safetensors, about 2.3 GB).

    python merge_lora.py --adapter out/lora --output out/merged

Runs on the CPU (needs about 10 GB of RAM) or on the GPU with --device cuda.
"""

import argparse
import json
import pathlib
import shutil
import sys

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Copied from the base checkpoint as they are, so the exporter sees the
# original config and tokenizer, not a re-serialised version of them.
BASE_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "LICENSE")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="the folder finetune_qlora.py wrote")
    p.add_argument("--output", default="out/merged")
    p.add_argument("--base", help="base model (default: the one recorded in the adapter)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    adapter = pathlib.Path(args.adapter)
    base = args.base
    if not base:
        base = json.loads((adapter / "adapter_config.json").read_text()).get("base_model_name_or_path")
    if not base:
        sys.exit("Could not tell the base model from the adapter; pass --base.")

    print(f"Loading {base} in float32 on {args.device}")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    model.to(args.device)

    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    leftover = [name for name, _ in model.named_parameters() if "lora_" in name]
    if leftover:
        sys.exit(f"Merging left adapter weights behind: {leftover[:3]}")

    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Same storage format as the base checkpoint.
    model.to(torch.bfloat16).save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(output)

    source = pathlib.Path(base) if pathlib.Path(base).is_dir() else pathlib.Path(snapshot_download(base, allow_patterns=list(BASE_FILES)))
    for name in BASE_FILES:
        if (source / name).exists():
            shutil.copy2(source / name, output / name)

    config = json.loads((output / "config.json").read_text())
    if "quantization_config" in config:
        sys.exit("The merged config.json has a quantization_config; the exporter will refuse it.")
    print(f"Merged model saved to {output}")
    print(f"Next: export it to ONNX with Liquid AI's exporter (README.md, step 3)")


if __name__ == "__main__":
    main()
