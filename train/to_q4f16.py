#!/usr/bin/env python3
"""Turn a q4 ONNX decoder (int4 weights, float32 activations) into q4f16:
the same int4 weights with float16 activations and KV caches. The logits
stay float32, as in Liquid AI's published q4f16 file.

    python to_q4f16.py exports/merged-ONNX/onnx/model_q4.onnx

writes model_q4f16.onnx and model_q4f16.onnx_data beside it. This is the call
Liquid AI's own exporter makes for its fp16 files (onnxruntime's float16
converter, with the logits kept as float32).

Needs: pip install onnx onnxruntime
"""

import argparse
import pathlib
import sys

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="the q4 graph, e.g. exports/<name>-ONNX/onnx/model_q4.onnx")
    p.add_argument("--output", help="where to write (default: model_q4f16.onnx beside the input)")
    args = p.parse_args()

    source = pathlib.Path(args.model)
    output = pathlib.Path(args.output) if args.output else source.with_name("model_q4f16.onnx")
    data = output.with_name(output.name + "_data")

    model = onnx.load(str(source))  # with its weights
    converted = convert_float_to_float16(
        model,
        keep_io_types=["logits"],
        force_fp16_initializers=True,
        disable_shape_infer=True,
    )
    if data.exists():
        data.unlink()
    onnx.save_model(
        converted,
        str(output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data.name,
        size_threshold=1024,
    )
    kinds = {
        value.name.split(".")[0]: onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)
        for value in list(converted.graph.input) + list(converted.graph.output)
    }
    print(f"Wrote {output} and {data.name} ({data.stat().st_size / 2**20:.1f} MiB)")
    print("Input and output types: " + ", ".join(f"{name} {kind}" for name, kind in sorted(kinds.items())))


if __name__ == "__main__":
    sys.exit(main())
