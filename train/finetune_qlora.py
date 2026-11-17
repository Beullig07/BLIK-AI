#!/usr/bin/env python3
"""QLoRA fine-tune of LiquidAI/LFM2.5-1.2B-Instruct on chat-format JSONL.

The base model is loaded in 4-bit (bitsandbytes), frozen, and small LoRA
matrices are trained on top of it. Defaults are sized for one 16 GB GPU
(written for an RTX 5070 Ti); see README.md for what to lower if memory runs out.

Data: one conversation per line, in the format the model's chat template takes:

    {"messages": [{"role": "system", "content": "..."},
                  {"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

The system message is optional and a conversation may have several turns.
Only the assistant's words are trained on; the rest is context.

    python finetune_qlora.py --data data/train.jsonl --output out/lora
    python finetune_qlora.py --data data/train.jsonl --eval-data data/eval.jsonl --epochs 2

The result (out/lora) is a LoRA adapter, a few tens of MB. merge_lora.py
folds it into the full-precision model for export.
"""

import argparse
import json
import math
import pathlib
import sys

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

BASE_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"

# Every linear layer inside LFM2's blocks: attention (q/k/v/out), the gated
# convolution blocks' projections (in/out) and the MLPs (w1/w2/w3).
# check_lora_targets() fails loudly if a model has none of them.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]

IGNORE = -100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="training conversations, JSONL")
    p.add_argument("--eval-data", help="held-out conversations, JSONL (optional)")
    p.add_argument("--output", default="out/lora", help="where the adapter is written")
    p.add_argument("--model", default=BASE_MODEL, help=f"base model (default: {BASE_MODEL})")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4, help="conversations per step on the GPU")
    p.add_argument("--grad-accum", type=int, default=4, help="steps added up before each update (effective batch = batch-size x grad-accum)")
    p.add_argument("--max-length", type=int, default=1024, help="longest conversation in tokens; longer ones are skipped")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.03, help="share of the steps spent raising the learning rate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-4bit", action="store_true", help="plain LoRA on the bf16 model instead of QLoRA (needs no bitsandbytes; about 1.5 GB more memory)")
    p.add_argument("--no-gradient-checkpointing", action="store_true", help="faster, but uses more memory")
    p.add_argument("--max-steps", type=int, default=-1, help="stop after this many updates (for a quick trial)")
    p.add_argument("--dry-run", action="store_true", help="check the data, load the model, attach LoRA, then stop before any training step")
    p.add_argument("--cpu", action="store_true", help="run without a GPU (meant for --dry-run: implies --no-4bit, and training this way is very slow)")
    return p.parse_args()


def read_jsonl(path: str) -> list[list[dict]]:
    conversations = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                messages = json.loads(line)["messages"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                sys.exit(f"{path} line {number}: expected {{\"messages\": [...]}} ({error})")
            roles = [m.get("role") for m in messages]
            if not messages or roles[-1] != "assistant" or any(r not in ("system", "user", "assistant") for r in roles):
                sys.exit(f"{path} line {number}: roles must be system/user/assistant and the last message must be the assistant's (got {roles})")
            if any(not isinstance(m.get("content"), str) for m in messages):
                sys.exit(f"{path} line {number}: every message needs a string \"content\"")
            conversations.append(messages)
    if not conversations:
        sys.exit(f"{path} has no conversations.")
    return conversations


def encode(tokenizer, messages: list[dict]) -> dict | None:
    """Token ids for a whole conversation, and labels that are IGNORE everywhere
    except the assistant's turns (their text and the end-of-turn token)."""
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    ids = list(ids)
    labels = [IGNORE] * len(ids)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        # Tokens up to the start of this reply, then up to its end.
        before = tokenizer.apply_chat_template(messages[:index], tokenize=True, add_generation_prompt=True)
        through = tokenizer.apply_chat_template(messages[: index + 1], tokenize=True, add_generation_prompt=False)
        before = list(before["input_ids"] if hasattr(before, "keys") else before)
        through = list(through["input_ids"] if hasattr(through, "keys") else through)
        if ids[: len(before)] != before or ids[: len(through)] != through:
            # The template does not render this conversation prefix by prefix
            # (it rewrites earlier turns). Train on nothing rather than on the wrong tokens.
            return None
        end = len(through)
        # Stop at the end-of-turn token; whatever the template adds after it is not the model's to write.
        while end > len(before) and ids[end - 1] != tokenizer.eos_token_id:
            end -= 1
        labels[len(before) : end] = ids[len(before) : end]
    return {"input_ids": ids, "labels": labels}


class Conversations(torch.utils.data.Dataset):
    def __init__(self, tokenizer, conversations: list[list[dict]], max_length: int, name: str):
        self.items = []
        skipped_long = skipped_template = 0
        for messages in conversations:
            item = encode(tokenizer, messages)
            if item is None:
                skipped_template += 1
            elif len(item["input_ids"]) > max_length:
                skipped_long += 1
            else:
                self.items.append(item)
        trained = sum(sum(1 for label in item["labels"] if label != IGNORE) for item in self.items)
        total = sum(len(item["input_ids"]) for item in self.items)
        print(f"{name}: {len(self.items)} conversations, {total} tokens, {trained} of them trained on")
        if skipped_long:
            print(f"{name}: skipped {skipped_long} longer than --max-length {max_length} tokens")
        if skipped_template:
            print(f"{name}: skipped {skipped_template} the chat template does not render prefix by prefix")
        if not self.items:
            sys.exit(f"{name}: nothing left to train on.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def collate(pad_id: int):
    def pad(batch: list[dict]) -> dict:
        width = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), IGNORE, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, item in enumerate(batch):
            n = len(item["input_ids"])
            input_ids[row, :n] = torch.tensor(item["input_ids"])
            labels[row, :n] = torch.tensor(item["labels"])
            attention_mask[row, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return pad


def check_lora_targets(model) -> list[str]:
    """The LoRA target names this model really has linear layers for."""
    present = sorted({name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)})
    targets = [name for name in LORA_TARGETS if name in present]
    if not targets:
        sys.exit(f"None of the expected LoRA target layers {LORA_TARGETS} exist in this model. Its linear layers are: {present}")
    return targets


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    on_gpu = torch.cuda.is_available() and not args.cpu
    if not on_gpu and not args.cpu:
        sys.exit("No CUDA GPU found. This script is meant for a GPU; pass --cpu only for a smoke test.")
    four_bit = on_gpu and not args.no_4bit

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train = Conversations(tokenizer, read_jsonl(args.data), args.max_length, "train")
    evaluation = Conversations(tokenizer, read_jsonl(args.eval_data), args.max_length, "eval") if args.eval_data else None

    load = {"dtype": torch.bfloat16 if on_gpu else torch.float32}
    if four_bit:
        from transformers import BitsAndBytesConfig

        load["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if on_gpu:
        load["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(args.model, **load)
    targets = check_lora_targets(model)

    if four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=not args.no_gradient_checkpointing)
    elif not args.no_gradient_checkpointing:
        # transformers 5 renamed this; 4.x has only the old name.
        (getattr(model, "activation_checkpointing_enable", None) or model.gradient_checkpointing_enable)()
        model.enable_input_require_grads()
    model.config.use_cache = False

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    print(f"LoRA on: {', '.join(targets)}")
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    training = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(total_steps * args.warmup_ratio)),
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=on_gpu,
        optim="paged_adamw_8bit" if four_bit else "adamw_torch",
        logging_steps=max(1, min(10, total_steps // 10 or 1)),
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch" if evaluation else "no",
        report_to="none",
        seed=args.seed,
        use_cpu=not on_gpu,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=train,
        eval_dataset=evaluation,
        data_collator=collate(tokenizer.pad_token_id),
    )
    if args.dry_run:
        batch = collate(tokenizer.pad_token_id)([train[i] for i in range(min(args.batch_size, len(train)))])
        print(f"Dry run: {total_steps} updates planned ({steps_per_epoch} per epoch), first batch {tuple(batch['input_ids'].shape)}.")
        print("Nothing was trained and nothing was saved.")
        return
    trainer.train()

    output = pathlib.Path(args.output)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "blik_training.json").write_text(
        json.dumps({"base_model": args.model, "lora_targets": targets, "four_bit": four_bit, "args": vars(args)}, indent=2)
    )
    if on_gpu:
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print(f"Adapter saved to {output}. Next: python merge_lora.py --adapter {output} --output out/merged")


if __name__ == "__main__":
    main()
