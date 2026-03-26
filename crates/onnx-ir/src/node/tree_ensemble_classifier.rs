use crate::{
    ir::{Argument, AttributeValue, Node, RawNode},
    processor::{InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError},
};
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

/// Configuration for the TreeEnsembleClassifier operator.
///
/// Performs classification using a tree ensemble (e.g., random forest, gradient boosted trees).
#[allow(clippy::too_many_arguments)]
#[derive(Debug, Clone, Default, new)]
pub struct TreeEnsembleClassifierConfig {
    /// Base values for each class (optional)
    pub base_values: Option<Vec<f32>>,
    /// Class ID associated with each leaf node
    pub class_ids: Option<Vec<i64>>,
    /// Node ID where each class weight is defined
    pub class_nodeids: Option<Vec<i64>>,
    /// Tree ID where each class weight is defined
    pub class_treeids: Option<Vec<i64>>,
    /// Class weight for each leaf
    pub class_weights: Option<Vec<f32>>,
    /// Integer class labels (alternative to string labels)
    pub classlabels_int64s: Option<Vec<i64>>,
    /// String class labels (alternative to int labels)
    pub classlabels_strings: Option<Vec<String>>,
    /// Feature ID used by each node for branching
    pub nodes_featureids: Option<Vec<i64>>,
    /// Hitrates for each node (optional)
    pub nodes_hitrates: Option<Vec<f32>>,
    /// Missing value tracks for each node (optional)
    pub nodes_missing_value_tracks_true: Option<Vec<i64>>,
    /// Mode of each node (BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF)
    pub nodes_modes: Option<Vec<String>>,
    /// Node IDs
    pub nodes_nodeids: Option<Vec<i64>>,
    /// Tree IDs for each node
    pub nodes_treeids: Option<Vec<i64>>,
    /// Child node ID if condition is true
    pub nodes_truenodeids: Option<Vec<i64>>,
    /// Child node ID if condition is false
    pub nodes_falsenodeids: Option<Vec<i64>>,
    /// Threshold/value for each node's branching condition
    pub nodes_values: Option<Vec<f32>>,
    /// Post-transform function: NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT
    pub post_transform: Option<String>,
}

/// TreeEnsembleClassifier ONNX operator.
///
/// Performs classification using tree ensembles (random forest, gradient boosted trees, etc.).
/// Outputs both class labels and class probabilities.
#[derive(Debug, Clone, new, NodeBuilder)]
pub struct TreeEnsembleClassifierNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: TreeEnsembleClassifierConfig,
}

pub(crate) struct TreeEnsembleClassifierProcessor;

impl NodeProcessor for TreeEnsembleClassifierProcessor {
    type Config = TreeEnsembleClassifierConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(2), // labels and probabilities
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        use crate::ir::{ArgType, DType, TensorType};

        // First output is labels (int64 tensor with rank 1: [batch_size])
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: DType::I64,
            rank: 1,
            static_shape: None,
        });

        // Second output is probabilities (float tensor with rank 2: [batch_size, num_classes])
        node.outputs[1].ty = ArgType::Tensor(TensorType {
            dtype: DType::F32,
            rank: 2,
            static_shape: None,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut base_values: Option<Vec<f32>> = None;
        let mut class_ids: Option<Vec<i64>> = None;
        let mut class_nodeids: Option<Vec<i64>> = None;
        let mut class_treeids: Option<Vec<i64>> = None;
        let mut class_weights: Option<Vec<f32>> = None;
        let mut classlabels_int64s: Option<Vec<i64>> = None;
        let mut classlabels_strings: Option<Vec<String>> = None;
        let mut nodes_featureids: Option<Vec<i64>> = None;
        let mut nodes_hitrates: Option<Vec<f32>> = None;
        let mut nodes_missing_value_tracks_true: Option<Vec<i64>> = None;
        let mut nodes_modes: Option<Vec<String>> = None;
        let mut nodes_nodeids: Option<Vec<i64>> = None;
        let mut nodes_treeids: Option<Vec<i64>> = None;
        let mut nodes_truenodeids: Option<Vec<i64>> = None;
        let mut nodes_falsenodeids: Option<Vec<i64>> = None;
        let mut nodes_values: Option<Vec<f32>> = None;
        let mut post_transform: Option<String> = None;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "base_values" => {
                    if let AttributeValue::Float32s(floats) = value {
                        base_values = Some(floats.clone());
                    }
                }
                "class_ids" => {
                    if let AttributeValue::Int64s(ints) = value {
                        class_ids = Some(ints.clone());
                    }
                }
                "class_nodeids" => {
                    if let AttributeValue::Int64s(ints) = value {
                        class_nodeids = Some(ints.clone());
                    }
                }
                "class_treeids" => {
                    if let AttributeValue::Int64s(ints) = value {
                        class_treeids = Some(ints.clone());
                    }
                }
                "class_weights" => {
                    if let AttributeValue::Float32s(floats) = value {
                        class_weights = Some(floats.clone());
                    }
                }
                "classlabels_int64s" => {
                    if let AttributeValue::Int64s(ints) = value {
                        classlabels_int64s = Some(ints.clone());
                    }
                }
                "classlabels_strings" => {
                    if let AttributeValue::Strings(strings) = value {
                        classlabels_strings = Some(strings.clone());
                    }
                }
                "nodes_featureids" => {
                    if let AttributeValue::Int64s(ints) = value {
                        nodes_featureids = Some(ints.clone());
                    }
                }
                "nodes_hitrates" => {
                    if let AttributeValue::Float32s(floats) = value {
                        nodes_hitrates = Some(floats.clone());
                    }
                }
                "nodes_missing_value_tracks_true" => {
                    if let AttributeValue::Int64s(ints) = value {
                        nodes_missing_value_tracks_true = Some(ints.clone());
                    }
                }
                "nodes_modes" => {
                    if let AttributeValue::Strings(strings) = value {
                        nodes_modes = Some(strings.clone());
                    }
                }
                "nodes_nodeids" => {
                    if let AttributeValue::Int64s(ints) = value {
                        nodes_nodeids = Some(ints.clone());
                    }
                }
                "nodes_treeids" => {
                    if let AttributeValue::Int64s(ints) = value {
                        nodes_treeids = Some(ints.clone());
                    }
                }
                "nodes_truenodeids" => {
                    if let AttributeValue::Int64s(ints) = value {
                        nodes_truenodeids = Some(ints.clone());
                    }
                }
                "nodes_falsenodeids" => {
                    if let AttributeValue::Int64s(ints) = value {
                        nodes_falsenodeids = Some(ints.clone());
                    }
                }
                "nodes_values" => {
                    if let AttributeValue::Float32s(floats) = value {
                        nodes_values = Some(floats.clone());
                    }
                }
                "post_transform" => {
                    if let AttributeValue::String(s) = value {
                        post_transform = Some(s.clone());
                    }
                }
                _ => {}
            }
        }

        Ok(TreeEnsembleClassifierConfig::new(
            base_values,
            class_ids,
            class_nodeids,
            class_treeids,
            class_weights,
            classlabels_int64s,
            classlabels_strings,
            nodes_featureids,
            nodes_hitrates,
            nodes_missing_value_tracks_true,
            nodes_modes,
            nodes_nodeids,
            nodes_treeids,
            nodes_truenodeids,
            nodes_falsenodeids,
            nodes_values,
            post_transform,
        ))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self.extract_config(&builder, opset).unwrap();
        Node::TreeEnsembleClassifier(TreeEnsembleClassifierNode::new(
            builder.name,
            builder.inputs,
            builder.outputs,
            config,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_ensemble_classifier_config() {
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
            Some(vec!["BRANCH_LEQ".to_string()]),
            Some(vec![0]),
            Some(vec![0]),
            Some(vec![1]),
            Some(vec![1]),
            Some(vec![0.5]),
            Some("NONE".to_string()),
        );
        assert_eq!(config.class_ids, Some(vec![0, 1]));
        assert_eq!(config.classlabels_int64s, Some(vec![0, 1]));
        assert_eq!(config.post_transform, Some("NONE".to_string()));
    }

    #[test]
    fn test_tree_ensemble_classifier_node_builder() {
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
            None,
        );
        let node = TreeEnsembleClassifierNode::new("test_tree".to_string(), vec![], vec![], config);

        assert_eq!(node.name, "test_tree");
        assert!(node.config.class_ids.is_some());
        assert!(node.config.nodes_modes.is_some());
    }
}
