#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: imputer.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 1


def main():
    # Test case 1: Replace NaN with 0.0
    np.random.seed(42)
    
    # Create input with some NaN values
    input_data = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    
    # Define Imputer node - replaces NaN with 0.0
    node = helper.make_node(
        "Imputer",
        ["input"],
        ["output"],
        domain="ai.onnx.ml",
        imputed_value_floats=[0.0],
    )
    
    # Create graph
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])
    
    graph = helper.make_graph(
        [node],
        "imputer_test",
        [input_tensor],
        [output_tensor],
    )
    
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("ai.onnx.ml", OPSET_VERSION)]
    )
    
    onnx.save(model, "imputer.onnx")
    print(f"Finished exporting model to imputer.onnx")
    
    # Validate using ReferenceEvaluator
    sess = ReferenceEvaluator(model)
    result = sess.run(None, {"input": input_data})
    
    print("\nInput:")
    print(input_data)
    print("\nOutput (NaN replaced with 0.0):")
    print(result[0])
    
    # Save test data
    np.save("input.npy", input_data)
    np.save("output.npy", result[0])


if __name__ == "__main__":
    main()
