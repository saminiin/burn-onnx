//! # OneHotEncoder
//!
//! Sklearn's OneHotEncoder operator (non-standard ONNX extension)
//! Takes a single input and produces one-hot encoded output

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for OneHotEncoder operation (sklearn variant)
#[derive(Debug, Clone, new)]
pub struct OneHotEncoderConfig {
    pub cats_int64s: Option<Vec<i64>>,
    pub cats_strings: Option<Vec<String>>,
    pub zeros: Option<i64>,
}

/// Node representation for OneHotEncoder operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct OneHotEncoderNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: OneHotEncoderConfig,
}

/// Processor for OneHotEncoder (sklearn variant)
pub struct OneHotEncoderProcessor;

impl NodeProcessor for OneHotEncoderProcessor {
    type Config = OneHotEncoderConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(1), // OneHotEncoder takes 1 input
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Input validation
        let input_ty = match &node.inputs[0].ty {
            ArgType::Tensor(t) => t,
            ArgType::Scalar(dtype) => {
                // Input is scalar, output will be scalar too
                node.outputs[0].ty = ArgType::Scalar(*dtype);
                return Ok(());
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Scalar".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // OneHotEncoder transforms [batch_size, 1] -> [batch_size, num_categories]
        // The output dtype stays the same as input
        let output_rank = input_ty.rank;

        node.outputs[0].ty = if output_rank == 0 {
            // Rank-0 should be Scalar
            ArgType::Scalar(input_ty.dtype)
        } else {
            ArgType::Tensor(TensorType {
                dtype: input_ty.dtype,
                rank: output_rank,
                static_shape: None,
            })
        };

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut cats_int64s: Option<Vec<i64>> = None;
        let mut cats_strings: Option<Vec<String>> = None;
        let mut zeros: Option<i64> = None;

        log::debug!("OneHotEncoder '{}' has {} attributes", node.name, node.attrs.len());
        
        for (key, value) in node.attrs.iter() {
            log::debug!("  Attribute '{}': {:?}", key, value);
            
            match key.as_str() {
                "cats_int64s" => {
                    if let crate::ir::AttributeValue::Int64s(ints) = value {
                        cats_int64s = Some(ints.clone());
                    }
                }
                "cats_strings" => {
                    if let crate::ir::AttributeValue::Strings(strings) = value {
                        cats_strings = Some(strings.clone());
                    }
                }
                "zeros" => {
                    if let crate::ir::AttributeValue::Int64(val) = value {
                        zeros = Some(*val);
                    }
                }
                _ => {}
            }
        }

        // Validate: need either cats_int64s OR cats_strings
        if cats_int64s.is_none() && cats_strings.is_none() {
            return Err(ProcessError::MissingAttribute(
                format!("cats_int64s or cats_strings (required for OneHotEncoder in node '{}')", node.name)
            ));
        }

        // If we have cats_strings but not cats_int64s, try to parse strings as integers
        let cats_int64s = if cats_int64s.is_none() && cats_strings.is_some() {
            let strings = cats_strings.as_ref().unwrap();
            let parsed: Result<Vec<i64>, _> = strings.iter()
                .map(|s| s.parse::<i64>())
                .collect();
            
            match parsed {
                Ok(ints) => {
                    log::debug!("Parsed cats_strings as integers: {:?}", ints);
                    Some(ints)
                }
                Err(e) => {
                    log::warn!("cats_strings contains non-numeric values, string categories not yet supported: {}", e);
                    None
                }
            }
        } else {
            cats_int64s
        };

        let config = OneHotEncoderConfig::new(cats_int64s, cats_strings, zeros);
        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .unwrap_or_else(|e| {
                log::error!(
                    "OneHotEncoder node '{}' configuration failed. \
                     Requires 'cats_int64s' or 'cats_strings' attribute. \
                     Error: {}",
                    builder.name,
                    e
                );
                panic!(
                    "OneHotEncoder node '{}' requires 'cats_int64s' or 'cats_strings' attribute",
                    builder.name
                )
            });

        Node::OneHotEncoder(OneHotEncoderNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}
