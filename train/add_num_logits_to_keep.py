#!/usr/bin/env python3
"""Add a `num_logits_to_keep` input to an LFM2 ONNX decoder.

Without it the graph returns logits for every prompt token: for an 800-token
prompt and a 65,536-token vocabulary that is a 210 MB float32 tensor, made on
the GPU and copied to the page, which is enough to get an iPhone Safari tab
killed. With it, Transformers.js passes 1 during generate() and the graph
computes one row.

The edit is the one daveycodez/LFM2.5-1.2B-Web-ONNX made to Liquid AI's
official file: the hidden states that feed the LM head are sliced to their
last `num_logits_to_keep` positions (0 keeps every position, as before).
Three nodes (Neg, Unsqueeze, Slice), three small int64 constants and the new
input. The weights file (.onnx_data) is not read or changed.

    python add_num_logits_to_keep.py onnx/model_q4f16.onnx            # in place
    python add_num_logits_to_keep.py in.onnx --output out.onnx

Needs: pip install onnx
"""

import argparse
import sys

import onnx
from onnx import TensorProto, helper

INPUT = "num_logits_to_keep"
PREFIX = "/lm_head/num_logits_to_keep"
LM_HEAD_OPS = {"MatMul", "MatMulNBits", "Gemm"}
# Nodes that may sit between the LM head and the `logits` output.
PASS_THROUGH = {"Cast", "Identity"}


def find_lm_head(graph: onnx.GraphProto) -> onnx.NodeProto:
    """The node that turns hidden states into logits: walk back from the `logits` output."""
    producers = {output: node for node in graph.node for output in node.output}
    if "logits" not in {o.name for o in graph.output}:
        raise SystemExit("This graph has no `logits` output.")
    name = "logits"
    for _ in range(8):
        node = producers.get(name)
        if node is None:
            break
        if node.op_type in LM_HEAD_OPS:
            return node
        if node.op_type not in PASS_THROUGH:
            break
        name = node.input[0]
    raise SystemExit(
        "Could not find the LM head: expected `logits` to come from a MatMul, MatMulNBits or "
        f"Gemm (through Cast/Identity at most), but found {node.op_type if node else 'nothing'}."
    )


def add_num_logits_to_keep(model: onnx.ModelProto) -> bool:
    """Edits the model in place. Returns False if it already has the input."""
    graph = model.graph
    if INPUT in {i.name for i in graph.input}:
        return False

    head = find_lm_head(graph)
    hidden = head.input[0]
    sliced = f"{PREFIX}/Slice/output_0"

    graph.input.append(helper.make_tensor_value_info(INPUT, TensorProto.INT64, []))
    graph.initializer.extend(
        [
            helper.make_tensor(f"{PREFIX}/axes", TensorProto.INT64, [1], [0]),
            helper.make_tensor(f"{PREFIX}/ends", TensorProto.INT64, [1], [2**31 - 1]),
            helper.make_tensor(f"{PREFIX}/slice_axes", TensorProto.INT64, [1], [1]),
        ]
    )
    # starts = [-num_logits_to_keep]; -0 is 0, so 0 keeps the whole sequence.
    nodes = [
        helper.make_node("Neg", [INPUT], [f"{PREFIX}/Neg/output_0"], name=f"{PREFIX}/Neg"),
        helper.make_node(
            "Unsqueeze",
            [f"{PREFIX}/Neg/output_0", f"{PREFIX}/axes"],
            [f"{PREFIX}/Unsqueeze/output_0"],
            name=f"{PREFIX}/Unsqueeze",
        ),
        helper.make_node(
            "Slice",
            [hidden, f"{PREFIX}/Unsqueeze/output_0", f"{PREFIX}/ends", f"{PREFIX}/slice_axes"],
            [sliced],
            name=f"{PREFIX}/Slice",
        ),
    ]
    position = list(graph.node).index(head)
    for offset, node in enumerate(nodes):
        graph.node.insert(position + offset, node)
    head.input[0] = sliced

    # The logits' sequence dimension is no longer the input's.
    for output in graph.output:
        if output.name == "logits":
            dims = output.type.tensor_type.shape.dim
            if len(dims) == 3:
                dims[1].ClearField("dim_value")
                dims[1].dim_param = "logits_sequence_length"
    # Shape notes about tensors after the slice would now be wrong; drop them.
    stale = {head.output[0], sliced}
    kept = [v for v in graph.value_info if v.name not in stale]
    del graph.value_info[:]
    graph.value_info.extend(kept)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="the .onnx graph file, e.g. onnx/model_q4f16.onnx")
    parser.add_argument("--output", help="where to write the result (default: over the input)")
    args = parser.parse_args()

    # The graph only: the weights stay in their .onnx_data file, untouched.
    model = onnx.load(args.model, load_external_data=False)
    if not add_num_logits_to_keep(model):
        print(f"{args.model} already has a {INPUT} input; nothing to do.")
        return
    onnx.save(model, args.output or args.model)
    print(f"Added {INPUT} to {args.output or args.model}")


if __name__ == "__main__":
    sys.exit(main())
