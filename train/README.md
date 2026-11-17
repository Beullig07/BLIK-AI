# train/: fine-tune LFM2.5-1.2B and put your weights in BLIK llm

Everything here runs on your own machine (written for an RTX 5070 Ti, 16 GB).
Nothing in this folder was trained by the person who wrote it: see
[What was and was not checked](#what-was-and-was-not-checked) before trusting a step.

The path, start to finish:

```
finetune_qlora.py   chat JSONL  ->  LoRA adapter               (GPU, the long step)
merge_lora.py       adapter     ->  full-precision checkpoint
Liquid AI's exporter            ->  ONNX, q4 (int4 weights)
to_q4f16.py         q4          ->  q4f16 (int4 weights, fp16 activations)
prepare_repo.py                 ->  adds num_logits_to_keep, writes the upload folder, checks it
upload_to_hub.py                ->  your Hugging Face repo, and the lines for js/config.js
```

## Setup

Python 3.12 or newer.

```sh
# PyTorch first, from https://pytorch.org, in a CUDA 12.8+ build (RTX 50-series needs it).
pip install -r requirements.txt
```

The model prints a note that `causal_conv1d` is not installed and that it is
using a slower built-in path. That is fine; installing `causal_conv1d` only
makes training faster.

## 1. Data

One conversation per line, in the chat format:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The system message is optional. A conversation can have several turns and
must end with the assistant. `data/example.jsonl` has three lines to copy from;
it is far too small to train on.

Only the assistant's words (and the end-of-turn token after them) are trained
on. The system and user text is context.

## 2. Fine-tune

Check the data and the setup first. This loads the model and stops before
any training step:

```sh
python finetune_qlora.py --data data/train.jsonl --dry-run
```

Then train:

```sh
python finetune_qlora.py --data data/train.jsonl --eval-data data/eval.jsonl --output out/lora
```

Defaults: 4-bit base model (QLoRA), LoRA rank 16 on every linear layer inside
the blocks, 3 epochs, learning rate 2e-4, batch 4 with 4 accumulation steps
(effective batch 16), conversations up to 1,024 tokens, gradient
checkpointing on.

These defaults are chosen to be safe on 16 GB, not measured on one. The
script prints the peak GPU memory when it finishes. If it runs out of memory,
lower `--batch-size` (and raise `--grad-accum` by the same factor), then
`--max-length`. If there is plenty of room, `--no-gradient-checkpointing`
is faster.

`--no-4bit` trains plain LoRA on the bf16 model instead. It needs no
bitsandbytes and a 1.2B model in bf16 is only about 2.4 GB, so it also fits.

## 3. Merge

```sh
python merge_lora.py --adapter out/lora --output out/merged
```

Loads the base model in float32 (not the 4-bit copy), adds the adapter into
it, and saves a normal checkpoint in the base model's format, with the base
model's own `config.json`, tokenizer and chat template copied in unchanged.

## 4. Export to ONNX (q4)

Use Liquid AI's exporter, [Liquid4All/onnx-export](https://github.com/Liquid4All/onnx-export),
**at commit `5ae7589`**, not at its latest version:

```sh
git clone https://github.com/Liquid4All/onnx-export
git -C onnx-export checkout 5ae758991464f2acd77c4ba5b654a9f32dbd60ea
pip install ./onnx-export
lfm2-export out/merged --precision q4 --output-dir out
```

This writes `out/exports/merged-ONNX/` with `onnx/model.onnx` (float32, about
4.7 GB for this model; you can delete it afterwards) and `onnx/model_q4.onnx`.

Why that commit: see [Which exporter, and why](#which-exporter-and-why).

## 5. q4 to q4f16

```sh
python to_q4f16.py out/exports/merged-ONNX/onnx/model_q4.onnx
```

Same int4 weights, float16 activations and KV caches, float32 logits. This is
the precision BLIK loads (`dtype: 'q4f16'`).

## 6. Prepare the upload folder

```sh
python prepare_repo.py --export out/exports/merged-ONNX --output out/hf-repo
```

It:

- adds the `num_logits_to_keep` input to the graph (`add_num_logits_to_keep.py`);
- adds the `transformers.js_config` block to `config.json`;
- copies the tokenizer files, the chat template and the weights;
- adds the LFM Open License and a short model card;
- compares the graph's inputs and outputs with the file BLIK ships with, and
  says so if they differ (`check_onnx.py` does the same on any graph).

Expect one accepted difference: an extra `position_ids` input, which
Transformers.js fills in.

## 7. Upload, then point BLIK at it

```sh
hf auth login
python upload_to_hub.py --folder out/hf-repo --repo your-name/your-model
```

The repo must be **public**: the browser downloads the weights without a
token. The script prints the block to paste into `js/config.js`:

```js
export const MODEL = {
  repo: 'your-name/your-model',
  revision: '<the commit it printed>',
  dtype: 'q4f16',
};
```

Update the text in `MODEL_CARD` just below it too (name and build line). A
fine-tune of LFM2.5 stays under the LFM Open License v1.0, so the license
lines stay as they are.

## Which exporter, and why

Liquid AI's published q4f16 file (`LiquidAI/LFM2.5-1.2B-Instruct-ONNX`), which
BLIK's default model is built from, was not made by any version of Liquid AI's
public exporter: its `producer` field says `huggingface`, and Liquid AI's
README says its Hugging Face ONNX repos "come from other export pipelines".
So no public command reproduces that file exactly. What exists:

| Exporter version | Inputs and outputs, against the published q4f16 | On WebGPU in BLIK |
|---|---|---|
| **Commit `5ae7589`** (24 Sep 2026, the last version before the switch below) | Same names, types and ranks, plus a `position_ids` input. Same operators plus `RotaryEmbedding`. Has q4 but no q4f16, hence step 5. | Loads and runs. First-token probabilities match ONNX Runtime on the CPU (96.7% against 96.4% for the top token). |
| **Latest** (`26857c2`, builds on onnxruntime-genai) | Different: conv caches are named `past.N.conv` instead of `past_conv.N` and hold 2 positions instead of 3; a new operator, `CausalConvWithState`; an 8-bit LM head; opset 22. Liquid AI's README says these exports "are not loadable by Transformers.js". | After renaming the caches it loads and writes text, but its probabilities are wrong: 48% for a wrong first token where the CPU gives 89% for the right one. ONNX Runtime's `enableMatmulFp32Accumulation` option improves it but does not make it match. |

So the older commit is the one to use. `prepare_repo.py` can still rename the
newer exporter's caches, but nothing here supports relying on that output.

## What was and was not checked

All checks ran on a machine with no GPU, with the small sibling model
`LiquidAI/LFM2.5-350M` (the same architecture, a quarter of the size), and
with the software WebGPU adapter described in `../tests/README.md`.

**Checked by running it**

- `add_num_logits_to_keep.py`: applied to Liquid AI's published q4f16 graph,
  it gives the graph BLIK ships with: the same nodes in the same order, the
  same inputs and outputs. On the CPU, with `num_logits_to_keep` 0 the logits
  are identical to the original's at every position; with 1 the last position
  differs by at most 0.004 (the LM head sums one row instead of 22) with the
  same top token, and the caches are identical.
- `finetune_qlora.py --dry-run`: the data is read, labels cover exactly the
  assistant's tokens, all eight LoRA target layer names exist in the LFM2
  architecture, and the trainer is built (transformers 5.23.0, peft 0.21.3,
  torch 2.15.0).
- `merge_lora.py`: with an adapter filled with random numbers, the merged
  weights equal base plus the adapter's product, no adapter weights are left,
  and the output has the base model's config.
- Steps 4 to 6 on LFM2.5-350M, from the merged folder and from the plain
  model: the exporter at `5ae7589`, `to_q4f16.py` (same graph and weights as
  Liquid AI's own conversion function gives), `prepare_repo.py`.
- That export in BLIK, in headless Chromium on WebGPU: it loads, chats,
  reuses the KV cache between turns, reads a 316-token prompt in chunks with
  the same result as in one pass, and stops on request.

**Not checked**

- **Any training.** No training step was run, on purpose. The 4-bit
  (bitsandbytes) path needs a CUDA GPU and was not executed at all; neither
  was memory use on 16 GB.
- **The 1.2B model through steps 4 to 6.** They were run on the 350M model
  only. The exporter's memory and time for 1.2B are unknown, and so is the
  size of the resulting q4f16 file. If it is much larger than the 725 MiB
  default, it may not fit on the phone.
- **`pip install ./onnx-export` at that commit.** The same source was run with
  its dependencies already installed, through
  `PYTHONPATH=onnx-export/src python -m liquidonnx.lfm2.export`. If the
  install gives trouble, use that form.
- **Quality after quantization.** The older exporter's q4 is plain 4-bit
  rounding. On the 350M model it moved the first-token probabilities
  noticeably away from the float32 model's (which the newer exporter's
  quantizer does less). Nothing here measures what it costs a 1.2B fine-tune.
- **Upload** (`upload_to_hub.py` was not run: no account here) and **the
  result on an iPhone**.

Before spending a day on a real fine-tune, it is worth running steps 3 to 7
once on the unmodified `LiquidAI/LFM2.5-1.2B-Instruct` (skip the adapter:
`lfm2-export LiquidAI/LFM2.5-1.2B-Instruct --precision q4 --output-dir out`)
and loading that in BLIK on the phone.
