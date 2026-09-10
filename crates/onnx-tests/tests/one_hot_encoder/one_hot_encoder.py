#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = ["numpy", "onnx"]
# ///
"""
Generate OneHotEncoder ONNX models for float32 and float64 inputs.
Expected outputs are printed from onnx.reference.ReferenceEvaluator.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def make_model(name: str, input_dtype: int) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("X", input_dtype, [None])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 3])

    node = helper.make_node(
        "OneHotEncoder",
        inputs=["X"],
        outputs=["Y"],
        domain="ai.onnx.ml",
        cats_int64s=[1, 2, 4],
        zeros=1,
    )

    graph = helper.make_graph([node], name, [x], [y])
    return helper.make_model(
        graph,
        producer_name="one_hot_encoder-test",
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("ai.onnx.ml", 1),
        ],
    )


def main() -> None:
    model_f32 = make_model("one_hot_encoder_f32", TensorProto.FLOAT)
    onnx.save(model_f32, "one_hot_encoder_f32.onnx")
    x_f32 = np.array([1.0, 4.0, 2.0, 1.0], dtype=np.float32)
    (y_f32,) = ReferenceEvaluator(model_f32).run(None, {"X": x_f32})
    print("one_hot_encoder_f32 expected:")
    print(y_f32.tolist())

    model_f64 = make_model("one_hot_encoder_f64", TensorProto.DOUBLE)
    onnx.save(model_f64, "one_hot_encoder_f64.onnx")
    x_f64 = np.array([4.0, 2.0, 3.0, 1.0], dtype=np.float64)
    (y_f64,) = ReferenceEvaluator(model_f64).run(None, {"X": x_f64})
    print("one_hot_encoder_f64 expected:")
    print(y_f64.tolist())


if __name__ == "__main__":
    main()
