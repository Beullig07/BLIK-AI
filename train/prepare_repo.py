#!/usr/bin/env python3
"""Turn an ONNX export folder into a Hugging Face repo BLIK llm can load.

Takes the folder Liquid AI's exporter wrote (exports/<name>-ONNX) and writes
a clean folder with only what the browser needs:

    config.json                 with a "transformers.js_config" block added
    generation_config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja
    onnx/model_<dtype>.onnx     with a num_logits_to_keep input added
    onnx/model_<dtype>.onnx_data*
    LICENSE, README.md          the LFM Open License and a short model card

Then it checks the graph's inputs and outputs against the file BLIK ships with.

    python prepare_repo.py --export exports/merged-ONNX --output out/hf-repo

Needs: pip install onnx huggingface_hub
"""

import argparse
import json
import pathlib
import re
import shutil
import sys

import onnx

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from add_num_logits_to_keep import add_num_logits_to_keep  # noqa: E402
from check_onnx import (  # noqa: E402
    REFERENCE_FILE,
    REFERENCE_REPO,
    REFERENCE_REVISION,
    accepted_extras,
    compare,
    fixed_size_differences,
    signature,
)

TEXT_FILES = ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
# Liquid AI's current exporter names LFM2's conv caches `past.N.conv` /
# `present.N.conv`. Transformers.js feeds them as `past_conv.N` and reads
# `present_conv.N`, the names of Liquid AI's published ONNX files.
CONV_CACHE = re.compile(r"^(past|present)\.(\d+)\.conv$")
BASE_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"

MODEL_CARD = """---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model:
  - {base}
library_name: transformers.js
pipeline_tag: text-generation
tags:
  - onnx
  - webgpu
  - lfm2.5
---

# {name}

A fine-tune of [{base}](https://huggingface.co/{base}), exported to ONNX
({dtype}) for in-browser use with Transformers.js on WebGPU.

The graph has a `num_logits_to_keep` input (int64 scalar): 1 for generation,
0 for logits at every position.

## License

LFM2.5 is released under the [LFM Open License v1.0](LICENSE), which also
covers this fine-tune. Commercial use is licensed only for entities with
annual revenue under $10 million. Original model (c) Liquid AI, Inc.
"""


def rename_conv_caches(model: onnx.ModelProto) -> int:
    """Renames `past.N.conv` -> `past_conv.N` and `present.N.conv` -> `present_conv.N`
    everywhere in the graph. Returns how many graph inputs and outputs were renamed."""

    def fixed(name: str) -> str:
        match = CONV_CACHE.match(name)
        return f"{match.group(1)}_conv.{match.group(2)}" if match else name

    graph = model.graph
    renamed = 0
    for value in list(graph.input) + list(graph.output):
        if fixed(value.name) != value.name:
            value.name = fixed(value.name)
            renamed += 1
    for node in graph.node:
        for names in (node.input, node.output):
            for index, name in enumerate(names):
                names[index] = fixed(name)
    for value in graph.value_info:
        value.name = fixed(value.name)
    return renamed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", required=True, help="the exporter's output folder (has config.json and onnx/)")
    p.add_argument("--output", default="out/hf-repo")
    p.add_argument("--dtype", default="q4f16", help="which precision to publish (default: q4f16)")
    p.add_argument("--base", default=BASE_MODEL, help="base model named in the model card, and the source of LICENSE")
    p.add_argument("--reference", help="reference graph for the check (default: downloaded)")
    args = p.parse_args()

    source = pathlib.Path(args.export)
    output = pathlib.Path(args.output)
    graph_name = f"model_{args.dtype}.onnx"
    graph_path = source / "onnx" / graph_name
    if not graph_path.exists():
        sys.exit(f"{graph_path} does not exist. Export that precision first (README.md, step 3).")
    (output / "onnx").mkdir(parents=True, exist_ok=True)

    # The graph, with num_logits_to_keep.
    model = onnx.load(graph_path, load_external_data=False)
    renamed = rename_conv_caches(model)
    added = add_num_logits_to_keep(model)
    onnx.save(model, output / "onnx" / graph_name)
    print(f"onnx/{graph_name}: {'added' if added else 'already has'} num_logits_to_keep" + (f"; renamed {renamed} conv cache inputs and outputs" if renamed else ""))

    # The weights files the graph points at, exactly as named there.
    locations = sorted({e.value for i in model.graph.initializer for e in i.external_data if e.key == "location"})
    for name in locations:
        if not (source / "onnx" / name).exists():
            sys.exit(f"The graph points at onnx/{name}, which is not in {source / 'onnx'}.")
        shutil.copy2(source / "onnx" / name, output / "onnx" / name)
        print(f"onnx/{name}: {(output / 'onnx' / name).stat().st_size / 2**20:.1f} MiB")
    # Transformers.js names them model_x.onnx_data, model_x.onnx_data_1, ... and needs their count.
    expected = [f"{graph_name}_data" + ("" if i == 0 else f"_{i}") for i in range(len(locations))]
    if locations and sorted(locations) != sorted(expected):
        sys.exit(f"The weights files are {locations}; Transformers.js expects {expected}.")

    # config.json, with the block Transformers.js reads.
    config = json.loads((source / "config.json").read_text())
    js = config.setdefault("transformers.js_config", {})
    # Per file, as a count of weights files. (An exporter may have written a bare true here.)
    if not isinstance(js.get("use_external_data_format"), dict):
        js["use_external_data_format"] = {}
    js["use_external_data_format"][graph_name] = len(locations)
    if not isinstance(js.get("kv_cache_dtype"), dict):
        js["kv_cache_dtype"] = {}
    kv = [i for i in model.graph.input if i.name.startswith("past_key_values.")]
    if kv:
        elem = kv[0].type.tensor_type.elem_type
        js["kv_cache_dtype"][args.dtype] = "float16" if elem == onnx.TensorProto.FLOAT16 else "float32"
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for name in TEXT_FILES:
        if (source / name).exists():
            shutil.copy2(source / name, output / name)
        elif name != "chat_template.jinja":
            sys.exit(f"{source / name} is missing.")

    # The license travels with the weights.
    if not (output / "LICENSE").exists():
        if (source / "LICENSE").exists():
            shutil.copy2(source / "LICENSE", output / "LICENSE")
        else:
            from huggingface_hub import hf_hub_download

            shutil.copy2(hf_hub_download(args.base, "LICENSE"), output / "LICENSE")
    if not (output / "README.md").exists():
        (output / "README.md").write_text(MODEL_CARD.format(name=output.resolve().name, base=args.base, dtype=args.dtype))

    # Same inputs and outputs as the file BLIK ships with?
    reference = args.reference
    if not reference:
        from huggingface_hub import hf_hub_download

        reference = hf_hub_download(REFERENCE_REPO, REFERENCE_FILE, revision=REFERENCE_REVISION)
    my_in, my_out, *_ = signature(str(output / "onnx" / graph_name))
    ref_in, ref_out, *_ = signature(reference)
    problems = compare("input", my_in, ref_in) + compare("output", my_out, ref_out)
    if problems:
        print(f"\nNOT THE SAME inputs and outputs as the reference ({len(problems)} differences):")
        for problem in problems[:20]:
            print(f"  - {problem}")
        print("\nBLIK llm may not be able to run this file. See README.md.")
        return 1
    extras = accepted_extras(my_in, ref_in)
    sizes = fixed_size_differences(my_in, ref_in) + fixed_size_differences(my_out, ref_out)
    total = sum(f.stat().st_size for f in output.rglob("*") if f.is_file())
    print("\nSame inputs and outputs as the reference (names, types and ranks)" + (f", plus {', '.join(extras)}." if extras else "."))
    if sizes:
        print(f"{len(sizes)} fixed sizes differ, e.g. {sizes[0]}. Expected for a different base model, not for a fine-tune of {BASE_MODEL}.")
    print(f"{output} is ready to upload ({total / 2**20:.0f} MiB).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
