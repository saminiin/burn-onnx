#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate models:
# - one_hot_encoder.onnx
# - one_hot_encoder_2d.onnx
# - one_hot_encoder_zeros0.onnx

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

ML_OPSET = 1
DEFAULT_OPSET = 17


def make_model(node, input_info, output_info, name):
    graph = helper.make_graph([node], name, [input_info], [output_info])
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("ai.onnx.ml", ML_OPSET),
            helper.make_operatorsetid("", DEFAULT_OPSET),
        ],
    )


def gen_one_hot_encoder_1d():
    # Baseline: 1D int64 input, integer categories.
    x = np.array([1, 0, 2, 3], dtype=np.int64)

    node = helper.make_node(
        "OneHotEncoder",
        ["input"],
        ["output"],
        domain="ai.onnx.ml",
        cats_int64s=[0, 1, 2, 3],
        zeros=1,
    )

    model = make_model(
        node,
        helper.make_tensor_value_info("input", TensorProto.INT64, [4]),
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 4]),
        "one_hot_encoder_1d",
    )

    onnx.save(model, "one_hot_encoder.onnx")
    out = ReferenceEvaluator(model).run(None, {"input": x})[0]
    print("one_hot_encoder.onnx input:\n", x)
    print("one_hot_encoder.onnx output:\n", out)


def gen_one_hot_encoder_2d():
    # Non-default shape path: rank-2 input -> rank-3 output.
    x = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int64)

    node = helper.make_node(
        "OneHotEncoder",
        ["input"],
        ["output"],
        domain="ai.onnx.ml",
        cats_int64s=[0, 1, 2],
        zeros=1,
    )

    model = make_model(
        node,
        helper.make_tensor_value_info("input", TensorProto.INT64, [2, 3]),
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3, 3]),
        "one_hot_encoder_2d",
    )

    onnx.save(model, "one_hot_encoder_2d.onnx")
    out = ReferenceEvaluator(model).run(None, {"input": x})[0]
    print("one_hot_encoder_2d.onnx input:\n", x)
    print("one_hot_encoder_2d.onnx output:\n", out)


def gen_one_hot_encoder_zeros0():
    # Non-default attribute path: zeros=0. Inputs only contain known categories,
    # so behavior matches zeros=1 while still exercising config extraction.
    x = np.array([2, 1, 0], dtype=np.int64)

    node = helper.make_node(
        "OneHotEncoder",
        ["input"],
        ["output"],
        domain="ai.onnx.ml",
        cats_int64s=[0, 1, 2],
        zeros=0,
    )

    model = make_model(
        node,
        helper.make_tensor_value_info("input", TensorProto.INT64, [3]),
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 3]),
        "one_hot_encoder_zeros0",
    )

    onnx.save(model, "one_hot_encoder_zeros0.onnx")
    out = ReferenceEvaluator(model).run(None, {"input": x})[0]
    print("one_hot_encoder_zeros0.onnx input:\n", x)
    print("one_hot_encoder_zeros0.onnx output:\n", out)


def main():
    np.random.seed(42)
    gen_one_hot_encoder_1d()
    gen_one_hot_encoder_2d()
    gen_one_hot_encoder_zeros0()


if __name__ == "__main__":
    main()
