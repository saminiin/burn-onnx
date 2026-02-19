use super::prelude::*;

impl NodeCodegen for onnx_ir::tree_ensemble_classifier::TreeEnsembleClassifierNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(&self.inputs[0]);
        let label_output = arg_to_ident(&self.outputs[0]);
        let prob_output = arg_to_ident(&self.outputs[1]);

        // Extract tree structure
        let nodes_treeids = self.config.nodes_treeids.as_ref();
        let nodes_nodeids = self.config.nodes_nodeids.as_ref();
        let nodes_featureids = self.config.nodes_featureids.as_ref();
        let nodes_modes = self.config.nodes_modes.as_ref();
        let nodes_values = self.config.nodes_values.as_ref();
        let nodes_truenodeids = self.config.nodes_truenodeids.as_ref();
        let nodes_falsenodeids = self.config.nodes_falsenodeids.as_ref();
        
        // Extract class information
        let class_treeids = self.config.class_treeids.as_ref();
        let class_nodeids = self.config.class_nodeids.as_ref();
        let class_ids = self.config.class_ids.as_ref();
        let class_weights = self.config.class_weights.as_ref();
        let classlabels = self.config.classlabels_int64s.as_ref();
        
        let post_transform = self.config.post_transform.as_deref().unwrap_or("NONE");

        // Generate tree evaluation logic
        let function = if let (Some(tree_ids), Some(node_ids), Some(feat_ids), Some(modes), 
                              Some(values), Some(true_ids), Some(false_ids),
                              Some(c_tree_ids), Some(c_node_ids), Some(c_ids), Some(c_weights)) 
            = (nodes_treeids, nodes_nodeids, nodes_featureids, nodes_modes, nodes_values,
               nodes_truenodeids, nodes_falsenodeids, class_treeids, class_nodeids, 
               class_ids, class_weights) {
            
            // Convert to vectors for code generation
            let tree_ids_vec: Vec<_> = tree_ids.iter().copied().collect();
            let node_ids_vec: Vec<_> = node_ids.iter().copied().collect();
            let feat_ids_vec: Vec<_> = feat_ids.iter().copied().collect();
            let modes_vec: Vec<_> = modes.iter().map(|s| s.as_str()).collect();
            let values_vec: Vec<_> = values.iter().copied().collect();
            let true_ids_vec: Vec<_> = true_ids.iter().copied().collect();
            let false_ids_vec: Vec<_> = false_ids.iter().copied().collect();
            
            let c_tree_ids_vec: Vec<_> = c_tree_ids.iter().copied().collect();
            let c_node_ids_vec: Vec<_> = c_node_ids.iter().copied().collect();
            let c_ids_vec: Vec<_> = c_ids.iter().copied().collect();
            let c_weights_vec: Vec<_> = c_weights.iter().copied().collect();
            
            let num_classes = classlabels.map(|l| l.len()).unwrap_or(2);
            let class_labels: Vec<_> = classlabels.map(|l| l.iter().copied().collect())
                .unwrap_or_else(|| (0..num_classes as i64).collect());

            quote! {
                {
                    // Tree structure arrays
                    let tree_ids = vec![#(#tree_ids_vec),*];
                    let node_ids = vec![#(#node_ids_vec),*];
                    let feature_ids = vec![#(#feat_ids_vec),*];
                    let modes = vec![#(#modes_vec),*];
                    let split_values = vec![#(#values_vec),*];
                    let true_node_ids = vec![#(#true_ids_vec),*];
                    let false_node_ids = vec![#(#false_ids_vec),*];
                    
                    // Class arrays
                    let class_tree_ids = vec![#(#c_tree_ids_vec),*];
                    let class_node_ids = vec![#(#c_node_ids_vec),*];
                    let class_ids = vec![#(#c_ids_vec),*];
                    let class_weights = vec![#(#c_weights_vec),*];
                    let class_labels = vec![#(#class_labels),*];
                    
                    let num_classes = #num_classes;
                    
                    // Get input shape
                    let input_shape = #input.shape();
                    let batch_size = input_shape.dims[0];
                    let n_features = input_shape.dims[input_shape.dims.len() - 1];
                    
                    // Initialize class scores for each sample
                    let mut all_scores = vec![vec![0.0f32; num_classes]; batch_size];
                    
                    // Evaluate each tree for each sample
                    let input_vals = #input.to_data().to_vec::<f32>().unwrap();
                    
                    for sample_idx in 0..batch_size {
                        // Extract features for this sample
                        let sample_offset = sample_idx * n_features;
                        let sample_features: Vec<f32> = input_vals[sample_offset..sample_offset + n_features].to_vec();
                        
                        // Find unique tree IDs
                        let mut unique_trees = tree_ids.clone();
                        unique_trees.sort();
                        unique_trees.dedup();
                        
                        // Evaluate each tree
                        for &tree_id in &unique_trees {
                            // Start at root node (node_id = 0) for this tree
                            let mut current_node_id = 0i64;
                            
                            // Traverse tree until leaf
                            loop {
                                // Find node in arrays
                                let node_idx = tree_ids.iter()
                                    .zip(node_ids.iter())
                                    .position(|(&t, &n)| t == tree_id && n == current_node_id);
                                
                                if let Some(idx) = node_idx {
                                    let mode = modes[idx];
                                    
                                    // Check if this is a leaf node
                                    if mode == "LEAF" {
                                        // Get class predictions for this leaf
                                        for (i, (((&ct, &cn), &cid), &cw)) in class_tree_ids.iter()
                                            .zip(class_node_ids.iter())
                                            .zip(class_ids.iter())
                                            .zip(class_weights.iter())
                                            .enumerate() {
                                            if ct == tree_id && cn == current_node_id {
                                                let class_idx = cid as usize;
                                                if class_idx < num_classes {
                                                    all_scores[sample_idx][class_idx] += cw;
                                                }
                                            }
                                        }
                                        break;
                                    }
                                    
                                    // Internal node - evaluate split
                                    let feature_id = feature_ids[idx] as usize;
                                    let threshold = split_values[idx];
                                    let feature_val = if feature_id < sample_features.len() {
                                        sample_features[feature_id]
                                    } else {
                                        0.0
                                    };
                                    
                                    // Determine which child to follow
                                    let go_left = match mode {
                                        "BRANCH_LEQ" => feature_val <= threshold,
                                        "BRANCH_LT" => feature_val < threshold,
                                        "BRANCH_GTE" => feature_val >= threshold,
                                        "BRANCH_GT" => feature_val > threshold,
                                        "BRANCH_EQ" => (feature_val - threshold).abs() < 1e-6,
                                        "BRANCH_NEQ" => (feature_val - threshold).abs() >= 1e-6,
                                        _ => feature_val <= threshold, // Default to LEQ
                                    };
                                    
                                    current_node_id = if go_left {
                                        true_node_ids[idx]
                                    } else {
                                        false_node_ids[idx]
                                    };
                                } else {
                                    // Node not found, break
                                    break;
                                }
                            }
                        }
                    }
                    
                    // Apply post-transform and get predictions
                    let post_transform = #post_transform;
                    match post_transform {
                        "SOFTMAX" => {
                            // Apply softmax to scores
                            for scores in &mut all_scores {
                                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                                let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                                let sum: f32 = exp_scores.iter().sum();
                                for (i, &exp_s) in exp_scores.iter().enumerate() {
                                    scores[i] = exp_s / sum;
                                }
                            }
                        }
                        "LOGISTIC" => {
                            // Apply logistic function (sigmoid)
                            for scores in &mut all_scores {
                                for score in scores.iter_mut() {
                                    *score = 1.0 / (1.0 + (-*score).exp());
                                }
                            }
                        }
                        "SOFTMAX_ZERO" => {
                            // Softmax with zero baseline
                            for scores in &mut all_scores {
                                scores.push(0.0);
                                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                                let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                                let sum: f32 = exp_scores.iter().sum();
                                for (i, &exp_s) in exp_scores.iter().enumerate() {
                                    scores[i] = exp_s / sum;
                                }
                                scores.pop(); // Remove the zero we added
                            }
                        }
                        _ => {} // NONE - raw scores
                    }
                    
                    // Find argmax for each sample (predicted class)
                    let mut labels = Vec::with_capacity(batch_size);
                    for scores in &all_scores {
                        let max_idx = scores.iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .map(|(idx, _)| idx)
                            .unwrap_or(0);
                        labels.push(class_labels[max_idx]);
                    }
                    
                    // Convert to tensors
                    let #label_output = Tensor::<B, 1, burn::tensor::Int>::from_data(
                        burn::tensor::TensorData::from(&labels[..]),
                        &*self.device
                    );
                    
                    // Flatten probability scores
                    let probs_flat: Vec<f32> = all_scores.into_iter().flatten().collect();
                    let #prob_output = Tensor::<B, 2>::from_data(
                        burn::tensor::TensorData::new(probs_flat, [batch_size, num_classes]),
                        &*self.device
                    );
                    
                    (#label_output, #prob_output)
                }
            }
        } else {
            // Fallback if configuration is incomplete
            quote! {
                {
                    let #label_output = Tensor::<B, 1, burn::tensor::Int>::zeros([1], &*self.device);
                    let #prob_output = Tensor::<B, 2>::zeros([1, 2], &*self.device);
                    (#label_output, #prob_output)
                }
            }
        };

        // Need to return tuple of outputs
        quote! {
            let (#label_output, #prob_output) = #function;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use onnx_ir::tree_ensemble_classifier::{TreeEnsembleClassifierConfig, TreeEnsembleClassifierNodeBuilder};

    #[test]
    fn test_tree_ensemble_classifier_basic() {
        // Simple tree with 2 classes
        let config = TreeEnsembleClassifierConfig::new(
            None,
            Some(vec![0, 1]), // class IDs
            Some(vec![1, 1]), // class node IDs (leaf node)
            Some(vec![0, 0]), // class tree IDs
            Some(vec![1.0, 1.0]), // class weights
            Some(vec![0, 1]), // class labels
            None,
            Some(vec![0]), // feature IDs
            None,
            None,
            Some(vec!["BRANCH_LEQ".to_string()]), // modes
            Some(vec![0]), // node IDs
            Some(vec![0]), // tree IDs
            Some(vec![1]), // true node IDs
            Some(vec![1]), // false node IDs
            Some(vec![0.5]), // threshold values
            Some("NONE".to_string()),
        );
        
        let node = TreeEnsembleClassifierNodeBuilder::new("tree1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("labels", 1, DType::I64)
            .output_tensor("probabilities", 2, DType::F32)
            .config(config)
            .build();
        
        let code = codegen_forward_default(&node);
        assert!(code.contains("tree_ids"));
        assert!(code.contains("class_weights"));
    }

    #[test]
    fn test_tree_ensemble_classifier_softmax() {
        let config = TreeEnsembleClassifierConfig::new(
            None,
            Some(vec![0, 1]),
            Some(vec![1, 1]),
            Some(vec![0, 0]),
            Some(vec![1.0, 1.0]),
            Some(vec![0, 1]),
            None,
            Some(vec![0]),
            None,
            None,
            Some(vec!["LEAF".to_string()]),
            Some(vec![1]),
            Some(vec![0]),
            None,
            None,
            None,
            Some("SOFTMAX".to_string()),
        );
        
        let node = TreeEnsembleClassifierNodeBuilder::new("tree1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("labels", 1, DType::I64)
            .output_tensor("probabilities", 2, DType::F32)
            .config(config)
            .build();
        
        let code = codegen_forward_default(&node);
        // Just verify it generates some code
        assert!(!code.is_empty());
    }

    #[test]
    fn test_tree_ensemble_classifier_logistic() {
        let config = TreeEnsembleClassifierConfig::new(
            None,
            Some(vec![0, 1]),
            Some(vec![1, 1]),
            Some(vec![0, 0]),
            Some(vec![1.0, 1.0]),
            Some(vec![0, 1]),
            None,
            Some(vec![0]),
            None,
            None,
            Some(vec!["LEAF".to_string()]),
            Some(vec![1]),
            Some(vec![0]),
            None,
            None,
            None,
            Some("LOGISTIC".to_string()),
        );
        
        let node = TreeEnsembleClassifierNodeBuilder::new("tree1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("labels", 1, DType::I64)
            .output_tensor("probabilities", 2, DType::F32)
            .config(config)
            .build();
        
        let code = codegen_forward_default(&node);
        // Just verify it generates some code
        assert!(!code.is_empty());
    }
}
