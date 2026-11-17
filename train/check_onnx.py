#!/usr/bin/env python3
"""Check that an exported decoder has the inputs and outputs BLIK llm expects.

It compares your graph with a reference graph (by default the file BLIK ships
with) name by name: every input and output, its type and its shape. Sizes
that depend on the model (hidden size, heads, vocabulary) may differ between
two different models; names, types and ranks may not.

    python check_onnx.py my-export/onnx/model_q4f16.onnx
    python check_onnx.py my.onnx --reference other.onnx

With no --reference, the reference graph (about 180 kB, no weights) is
downloaded from Hugging Face.

Needs: pip install onnx huggingface_hub
"""

import argparse
import sys

import onnx

REFERENCE_REPO = "daveycodez/LFM2.5-1.2B-Web-ONNX"
REFERENCE_REVISION = "f8df717df1ee01ccb899484d7f25cddfa78dd49f"
REFERENCE_FILE = "onnx/model_q4f16.onnx"

# Inputs a graph may have beyond the reference's. Transformers.js fills
# `position_ids` in from the attention mask when a graph asks for it; graphs
# from Liquid AI's exporter have it, the published files do not.
ACCEPTED_EXTRA_INPUTS = {"position_ids": "Transformers.js supplies it"}


def signature(path: str):
    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    initializers = {i.name for i in graph.initializer}

    def describe(value):
        tensor = value.type.tensor_type
        dims = [d.dim_param if d.dim_param else d.dim_value for d in tensor.shape.dim]
        return onnx.TensorProto.DataType.Name(tensor.elem_type), dims

    inputs = {v.name: describe(v) for v in graph.input if v.name not in initializers}
    outputs = {v.name: describe(v) for v in graph.output}
    external = sorted({e.value for i in graph.initializer for e in i.external_data if e.key == "location"})
    ops = sorted({(n.domain or "ai.onnx", n.op_type) for n in graph.node})
    opsets = sorted((o.domain or "ai.onnx", o.version) for o in model.opset_import)
    return inputs, outputs, external, ops, opsets


def compare(kind: str, mine: dict, reference: dict) -> list[str]:
    problems = []
    for name in reference:
        if name not in mine:
            problems.append(f"missing {kind} `{name}`")
    for name in mine:
        if name not in reference and not (kind == "input" and name in ACCEPTED_EXTRA_INPUTS):
            problems.append(f"extra {kind} `{name}` (the reference has none)")
    for name in sorted(set(mine) & set(reference)):
        (my_type, my_dims), (ref_type, ref_dims) = mine[name], reference[name]
        if my_type != ref_type:
            problems.append(f"{kind} `{name}` is {my_type}, the reference has {ref_type}")
        if len(my_dims) != len(ref_dims):
            problems.append(f"{kind} `{name}` has {len(my_dims)} dimensions, the reference has {len(ref_dims)}")
            continue
        for index, (a, b) in enumerate(zip(my_dims, ref_dims)):
            # A named (dynamic) dimension must be dynamic in both; its name is free.
            if isinstance(a, str) != isinstance(b, str):
                problems.append(f"{kind} `{name}` dimension {index} is {a!r}, the reference has {b!r}")
    return problems


def accepted_extras(mine: dict, reference: dict) -> list[str]:
    return [f"`{name}` ({ACCEPTED_EXTRA_INPUTS[name]})" for name in mine if name not in reference and name in ACCEPTED_EXTRA_INPUTS]


def fixed_size_differences(mine: dict, reference: dict) -> list[str]:
    """Fixed (not dynamic) dimensions that differ, e.g. the hidden size of a different model."""
    lines = []
    for name in sorted(set(mine) & set(reference)):
        a, b = mine[name][1], reference[name][1]
        if len(a) == len(b) and any(isinstance(x, int) and isinstance(y, int) and x != y for x, y in zip(a, b)):
            lines.append(f"{name}: {a} here, {b} in the reference")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="your .onnx graph file")
    parser.add_argument("--reference", help="a reference .onnx graph (default: the file BLIK ships with)")
    args = parser.parse_args()

    reference = args.reference
    if not reference:
        from huggingface_hub import hf_hub_download

        reference = hf_hub_download(REFERENCE_REPO, REFERENCE_FILE, revision=REFERENCE_REVISION)

    my_in, my_out, my_external, my_ops, my_opsets = signature(args.model)
    ref_in, ref_out, _, ref_ops, ref_opsets = signature(reference)

    print(f"{args.model}: {len(my_in)} inputs, {len(my_out)} outputs, weights in {my_external or 'the graph file'}")
    print(f"reference:    {len(ref_in)} inputs, {len(ref_out)} outputs")
    problems = compare("input", my_in, ref_in) + compare("output", my_out, ref_out)

    extras = accepted_extras(my_in, ref_in)
    if extras:
        print(f"\nExtra inputs that are fine: {', '.join(extras)}")
    sizes = fixed_size_differences(my_in, ref_in) + fixed_size_differences(my_out, ref_out)
    if sizes:
        print(f"\nFixed sizes that differ ({len(sizes)}; expected between different models, not for a fine-tune of the same one):")
        for line in sizes[:6]:
            print(f"  {line}")
        if len(sizes) > 6:
            print(f"  ... and {len(sizes) - 6} more")

    new_ops = [op for op in my_ops if op not in ref_ops]
    if new_ops:
        print(f"\nOperators the reference does not use (they may or may not run on WebGPU): {new_ops}")
    if my_opsets != ref_opsets:
        print(f"\nOpsets: {my_opsets} here, {ref_opsets} in the reference")

    if problems:
        print(f"\nNOT THE SAME inputs and outputs ({len(problems)} differences):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nSame inputs and outputs as the reference (names, types and ranks)" + (", plus the accepted extra input." if extras else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
