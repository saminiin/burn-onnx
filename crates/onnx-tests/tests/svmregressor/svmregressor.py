#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = ["numpy", "onnx"]
# ///
"""
ONNX SVMRegressor operator test model generator.

Generates a simple SVMRegressor model with linear kernel for testing.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper

def main():
    np.random.seed(42)
    
    # Define input shape
    batch_size = 3
    n_features = 2
    
    # Create a simple SVM regressor with linear kernel
    # Support vectors: 2 vectors with 2 features each
    support_vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).flatten()
    coefficients = np.array([1.0, -0.5], dtype=np.float32)
    rho = np.array([0.5], dtype=np.float32)
    
    # Create input tensor
    input_data = np.random.randn(batch_size, n_features).astype(np.float32)
    
    # Define the SVMRegressor node
    svm_node = helper.make_node(
        'SVMRegressor',
        inputs=['X'],
        outputs=['Y'],
        domain='ai.onnx.ml',
        coefficients=coefficients.tolist(),
        kernel_type='LINEAR',
        n_supports=2,
        rho=rho.tolist(),
        support_vectors=support_vectors.tolist(),
    )
    
    # Create the graph
    graph = helper.make_graph(
        [svm_node],
        'svmregressor_test',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [batch_size, n_features])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [batch_size])],
    )
    
    # Create the model
    model = helper.make_model(
        graph,
        producer_name='svmregressor-test',
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("ai.onnx.ml", 1)
        ]
    )
    
    # Save model and input
    onnx.save(model, 'svmregressor.onnx')
    input_data.tofile('input.bin')
    
    # Compute expected output using reference implementation
    # For LINEAR kernel: prediction = (X @ SV^T) @ coefficients + rho
    sv_matrix = support_vectors.reshape(2, n_features)
    kernel_values = input_data @ sv_matrix.T  # [batch_size, n_supports]
    output_data = (kernel_values @ coefficients + rho[0]).astype(np.float32)
    output_data.tofile('output.bin')
    
    print(f"Generated SVMRegressor test model")
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output_data.shape}")
    print(f"Support vectors: {sv_matrix}")
    print(f"Coefficients: {coefficients}")
    print(f"Rho: {rho[0]}")

if __name__ == '__main__':
    main()
