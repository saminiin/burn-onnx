use crate::{
    ir::{Argument, AttributeValue, Node, RawNode},
    processor::{
        InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
        same_as_input,
    },
};
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

/// Configuration for the SVMRegressor operator.
///
/// Performs regression using Support Vector Machine (SVM) with various kernel types.
#[allow(clippy::too_many_arguments)]
#[derive(Debug, Clone, Default, new)]
pub struct SVMRegressorConfig {
    /// Coefficients for the support vector in the decision function.
    pub coefficients: Option<Vec<f32>>,
    /// Parameters for the kernel function (gamma, coef0, degree).
    pub kernel_params: Option<Vec<f32>>,
    /// Type of kernel function: LINEAR, POLY, RBF, SIGMOID.
    pub kernel_type: Option<String>,
    /// Number of support vectors.
    pub n_supports: Option<i64>,
    /// Whether to predict using one class per regression target or one class for all targets.
    pub one_class: Option<i64>,
    /// How to transform the output: NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT.
    pub post_transform: Option<String>,
    /// Bias term(s) in decision function.
    pub rho: Option<Vec<f32>>,
    /// Support vectors.
    pub support_vectors: Option<Vec<f32>>,
}

/// SVMRegressor ONNX operator.
///
/// Performs regression using Support Vector Machine (SVM) with configurable kernel types:
/// - LINEAR: K(x, sv) = x · sv
/// - RBF: K(x, sv) = exp(-gamma * ||x - sv||^2)
/// - POLY: K(x, sv) = (gamma * x · sv + coef0)^degree
/// - SIGMOID: K(x, sv) = tanh(gamma * x · sv + coef0)
#[derive(Debug, Clone, new, NodeBuilder)]
pub struct SVMRegressorNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: SVMRegressorConfig,
}

pub(crate) struct SVMRegressorProcessor;

impl NodeProcessor for SVMRegressorProcessor {
    type Config = SVMRegressorConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Output has the same type as input
        same_as_input(node);

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut coefficients: Option<Vec<f32>> = None;
        let mut kernel_params: Option<Vec<f32>> = None;
        let mut kernel_type: Option<String> = None;
        let mut n_supports: Option<i64> = None;
        let mut one_class: Option<i64> = None;
        let mut post_transform: Option<String> = None;
        let mut rho: Option<Vec<f32>> = None;
        let mut support_vectors: Option<Vec<f32>> = None;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "coefficients" => {
                    if let AttributeValue::Float32s(floats) = value {
                        coefficients = Some(floats.clone());
                    }
                }
                "kernel_params" => {
                    if let AttributeValue::Float32s(floats) = value {
                        kernel_params = Some(floats.clone());
                    }
                }
                "kernel_type" => {
                    if let AttributeValue::String(s) = value {
                        kernel_type = Some(s.clone());
                    }
                }
                "n_supports" => {
                    if let AttributeValue::Int64(n) = value {
                        n_supports = Some(*n);
                    }
                }
                "one_class" => {
                    if let AttributeValue::Int64(n) = value {
                        one_class = Some(*n);
                    }
                }
                "post_transform" => {
                    if let AttributeValue::String(s) = value {
                        post_transform = Some(s.clone());
                    }
                }
                "rho" => {
                    if let AttributeValue::Float32s(floats) = value {
                        rho = Some(floats.clone());
                    }
                }
                "support_vectors" => {
                    if let AttributeValue::Float32s(floats) = value {
                        support_vectors = Some(floats.clone());
                    }
                }
                _ => {}
            }
        }

        Ok(SVMRegressorConfig::new(
            coefficients,
            kernel_params,
            kernel_type,
            n_supports,
            one_class,
            post_transform,
            rho,
            support_vectors,
        ))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self.extract_config(&builder, opset).unwrap();
        Node::SVMRegressor(SVMRegressorNode::new(
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
    fn test_svmregressor_config() {
        let config = SVMRegressorConfig::new(
            Some(vec![1.0, -0.5]),
            None,
            Some("LINEAR".to_string()),
            Some(2),
            None,
            None,
            Some(vec![0.5]),
            Some(vec![1.0, 2.0, 3.0, 4.0]),
        );
        assert_eq!(config.coefficients, Some(vec![1.0, -0.5]));
        assert_eq!(config.kernel_type, Some("LINEAR".to_string()));
        assert_eq!(config.n_supports, Some(2));
        assert_eq!(config.rho, Some(vec![0.5]));
    }

    #[test]
    fn test_svmregressor_node_builder() {
        let config = SVMRegressorConfig::new(
            Some(vec![1.0]),
            None,
            Some("LINEAR".to_string()),
            Some(1),
            None,
            None,
            Some(vec![0.0]),
            Some(vec![1.0, 2.0]),
        );
        let node = SVMRegressorNode::new("test_svm".to_string(), vec![], vec![], config);

        assert_eq!(node.name, "test_svm");
        assert_eq!(node.inputs.len(), 0);
        assert_eq!(node.outputs.len(), 0);
        assert!(node.config.coefficients.is_some());
        assert!(node.config.kernel_type.is_some());
    }
}
