#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = ["numpy", "onnx"]
# ///
"""
ONNX TreeEnsembleClassifier operator test model generator.

Generates a simple TreeEnsembleClassifier model with 2 classes for testing.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper

def main():
    np.random.seed(42)
    
    # Define input shape
    batch_size = 3
    n_features = 2
    
    # Create a simple tree classifier with 2 classes
    # Tree structure: root node splits on feature 0 at threshold 0.5
    # Left child (node 1) predicts class 0, right child (node 2) predicts class 1
    
    # Node structure for one tree
    nodes_treeids = [0, 0, 0]  # All nodes in tree 0
    nodes_nodeids = [0, 1, 2]  # Root (0), left child (1), right child (2)
    nodes_featureids = [0, 0, 0]  # Feature to split on
    nodes_modes = ["BRANCH_LEQ", "LEAF", "LEAF"]  # Root branches, children are leaves
    nodes_values = [0.5, 0.0, 0.0]  # Split threshold for root
    nodes_truenodeids = [1, 0, 0]  # Go to node 1 if true
    nodes_falsenodeids = [2, 0, 0]  # Go to node 2 if false
    
    # Class information for leaf nodes
    class_treeids = [0, 0]  # Both in tree 0
    class_nodeids = [1, 2]  # Left and right leaves
    class_ids = [0, 1]  # Classes 0 and 1
    class_weights = [1.0, 1.0]  # Weights for each class
    classlabels_int64s = [0, 1]  # Class labels
    
    # Create input tensor
    input_data = np.random.randn(batch_size, n_features).astype(np.float32)
    
    # Define the TreeEnsembleClassifier node
    tree_node = helper.make_node(
        'TreeEnsembleClassifier',
        inputs=['X'],
        outputs=['Y', 'Z'],
        domain='ai.onnx.ml',
        nodes_treeids=nodes_treeids,
        nodes_nodeids=nodes_nodeids,
        nodes_featureids=nodes_featureids,
        nodes_modes=nodes_modes,
        nodes_values=nodes_values,
        nodes_truenodeids=nodes_truenodeids,
        nodes_falsenodeids=nodes_falsenodeids,
        class_treeids=class_treeids,
        class_nodeids=class_nodeids,
        class_ids=class_ids,
        class_weights=class_weights,
        classlabels_int64s=classlabels_int64s,
        post_transform='NONE',
    )
    
    # Create the graph
    graph = helper.make_graph(
        [tree_node],
        'tree_ensemble_classifier_test',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [batch_size, n_features])],
        [
            helper.make_tensor_value_info('Y', TensorProto.INT64, [batch_size]),
            helper.make_tensor_value_info('Z', TensorProto.FLOAT, [batch_size, 2]),
        ],
    )
    
    # Create the model
    model = helper.make_model(
        graph,
        producer_name='tree-ensemble-classifier-test',
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("ai.onnx.ml", 1)
        ]
    )
    
    # Save model and input
    onnx.save(model, 'tree_ensemble_classifier.onnx')
    input_data.tofile('input.bin')
    
    # Compute expected output manually
    labels = []
    probs = []
    for i in range(batch_size):
        feature_0 = input_data[i, 0]
        # Simple tree logic: if feature[0] <= 0.5, predict class 0, else class 1
        if feature_0 <= 0.5:
            labels.append(0)
            probs.extend([1.0, 0.0])  # Class 0 probability
        else:
            labels.append(1)
            probs.extend([0.0, 1.0])  # Class 1 probability
    
    output_labels = np.array(labels, dtype=np.int64)
    output_probs = np.array(probs, dtype=np.float32)
    
    output_labels.tofile('output_labels.bin')
    output_probs.tofile('output_probs.bin')
    
    print(f"Generated TreeEnsembleClassifier test model")
    print(f"Input shape: {input_data.shape}")
    print(f"Output labels shape: {output_labels.shape}")
    print(f"Output probs shape: {output_probs.shape}")

if __name__ == '__main__':
    main()
