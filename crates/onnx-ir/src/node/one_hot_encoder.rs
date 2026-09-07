//! # OneHotEncoder
//!
//! Replace each input element with an array of ones and zeros, where a single
//! one is placed at the index of the category that was passed in.
//! The output has one more dimension than the input (categories appended as last dim).
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx_aionnxml_OneHotEncoder.html>
//!
//! ## Type Constraints
//!
//! - T: tensor(float), tensor(double), tensor(int32), tensor(int64)
//!   (tensor(string) is not supported in Burn)
//!
//! ## Opset Versions
//!
//! - **Opset 1**: Initial version with cats_int64s, cats_strings, and zeros attributes

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, AttributeValue, DType, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for OneHotEncoder operation
#[derive(Debug, Clone, Default, new)]
pub struct OneHotEncoderConfig {
    /// List of integer categories for lookup
    pub cats_int64s: Option<Vec<i64>>,
    /// List of string categories for lookup (not supported in Burn codegen)
    pub cats_strings: Option<Vec<String>>,
    /// If true (1, default) and category is not present, return all zeros;
    /// if false (0), the operator will fail on unknown categories.
    pub zeros: Option<i64>,
}

/// Node representation for OneHotEncoder operation
#[derive(Debug, Clone, new, NodeBuilder)]
pub struct OneHotEncoderNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: OneHotEncoderConfig,
}

pub(crate) struct OneHotEncoderProcessor;

impl NodeProcessor for OneHotEncoderProcessor {
    type Config = OneHotEncoderConfig;

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
        // Output Y is always tensor(float) with rank = input_rank + 1.
        // The extra dimension is the number of categories.
        let (input_rank, num_categories) = match &node.inputs[0].ty {
            ArgType::Tensor(t) => {
                match t.dtype {
                    DType::F32 | DType::F64 | DType::I32 | DType::I64 => {}
                    other => {
                        return Err(ProcessError::TypeMismatch {
                            expected: "tensor(float | double | int32 | int64)".to_string(),
                            actual: format!("tensor({other:?})"),
                        });
                    }
                }

                // Determine number of categories from attributes
                let num_cats = self.get_num_categories(node)?;

                (t.rank, num_cats)
            }
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "tensor(float | double | int32 | int64)".to_string(),
                    actual: format!("{other:?}"),
                });
            }
        };

        let output_rank = input_rank + 1;

        // Build static shape: input dims + [num_categories]
        let static_shape = match &node.inputs[0].ty {
            ArgType::Tensor(t) => {
                if let Some(input_shape) = &t.static_shape {
                    let mut shape = input_shape.clone();
                    shape.push(Some(num_categories));
                    Some(shape)
                } else {
                    // We know the last dim is num_categories even if input shape is unknown
                    let mut shape: Vec<Option<usize>> = vec![None; input_rank];
                    shape.push(Some(num_categories));
                    Some(shape)
                }
            }
            _ => None,
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: DType::F32,
            rank: output_rank,
            static_shape,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut cats_int64s: Option<Vec<i64>> = None;
        let mut cats_strings: Option<Vec<String>> = None;
        let mut zeros: Option<i64> = None;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "cats_int64s" => {
                    if let AttributeValue::Int64s(ints) = value {
                        cats_int64s = Some(ints.clone());
                    } else {
                        return Err(ProcessError::InvalidAttribute {
                            name: "cats_int64s".to_string(),
                            reason: format!("expected Int64s, got {value:?}"),
                        });
                    }
                }
                "cats_strings" => {
                    if let AttributeValue::Strings(strings) = value {
                        cats_strings = Some(strings.clone());
                    } else {
                        return Err(ProcessError::InvalidAttribute {
                            name: "cats_strings".to_string(),
                            reason: format!("expected Strings, got {value:?}"),
                        });
                    }
                }
                "zeros" => {
                    if let AttributeValue::Int64(i) = value {
                        zeros = Some(*i);
                    } else {
                        return Err(ProcessError::InvalidAttribute {
                            name: "zeros".to_string(),
                            reason: format!("expected Int64, got {value:?}"),
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(OneHotEncoderConfig::new(cats_int64s, cats_strings, zeros))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self.extract_config(&builder, opset).unwrap();
        Node::OneHotEncoder(OneHotEncoderNode::new(
            builder.name,
            builder.inputs,
            builder.outputs,
            config,
        ))
    }
}

impl OneHotEncoderProcessor {
    /// Get the number of categories from attributes
    fn get_num_categories(&self, node: &RawNode) -> Result<usize, ProcessError> {
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "cats_int64s" => {
                    if let AttributeValue::Int64s(ints) = value {
                        return Ok(ints.len());
                    }
                }
                "cats_strings" => {
                    if let AttributeValue::Strings(strings) = value {
                        return Ok(strings.len());
                    }
                }
                _ => {}
            }
        }

        Err(ProcessError::MissingAttribute(
            "cats_int64s or cats_strings".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onehotencoder_config_int_categories() {
        let config =
            OneHotEncoderConfig::new(Some(vec![0, 1, 2, 3]), None, Some(1));
        assert_eq!(config.cats_int64s.as_ref().unwrap().len(), 4);
        assert!(config.cats_strings.is_none());
        assert_eq!(config.zeros, Some(1));
    }

    #[test]
    fn test_onehotencoder_config_default() {
        let config = OneHotEncoderConfig::default();
        assert!(config.cats_int64s.is_none());
        assert!(config.cats_strings.is_none());
        assert!(config.zeros.is_none());
    }

    #[test]
    fn test_onehotencoder_node_builder() {
        let config =
            OneHotEncoderConfig::new(Some(vec![0, 1, 2]), None, None);
        let node = OneHotEncoderNode::new(
            "test_ohe".to_string(),
            vec![],
            vec![],
            config,
        );
        assert_eq!(node.name, "test_ohe");
        assert_eq!(node.config.cats_int64s.as_ref().unwrap().len(), 3);
    }
}
