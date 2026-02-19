//! # ZipMap
//!
//! Deprecated ONNX operator that creates a map from two sequences.
//! Commonly found in scikit-learn and ML.NET models.
//!
//! **ONNX Spec**: https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ZipMap-1
//!
//! ## Note
//! This operator is deprecated and typically used for metadata transformation.
//! In burn-onnx, it's treated as a pass-through operator.

use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Node representation for ZipMap operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ZipMapNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

pub(crate) struct ZipMapProcessor;

impl NodeProcessor for ZipMapProcessor {
    type Config = ();

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
        // ZipMap passes input type through
        // In practice, it converts sequence<map> but we treat it as pass-through
        node.outputs[0].ty = node.inputs[0].ty.clone();

        Ok(())
    }

    fn is_noop(&self, _node: &RawNode) -> bool {
        true // Treated as no-op in burn
    }

    fn extract_config(&self, _node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::ZipMap(ZipMapNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, ArgumentInfo, NodeType, Ty};

    #[test]
    fn test_zipmap_inference() {
        let input_ty = Ty::Tensor(ArgType::F32);
        
        let mut node = RawNode {
            node_type: NodeType::ZipMap,
            name: "zipmap_test".to_string(),
            inputs: vec![Argument {
                name: "input".to_string(),
                ty: input_ty.clone(),
                info: ArgumentInfo::default(),
            }],
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: Ty::Unknown,
                info: ArgumentInfo::default(),
            }],
            attrs: Default::default(),
        };

        let processor = ZipMapProcessor;
        processor
            .infer_types(&mut node, 1, &OutputPreferences::default())
            .unwrap();

        assert_eq!(node.outputs[0].ty, input_ty);
    }
}
