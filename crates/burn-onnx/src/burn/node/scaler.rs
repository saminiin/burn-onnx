use super::prelude::*;

impl NodeCodegen for onnx_ir::scaler::ScalerNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = &self.inputs[0];
        let output = arg_to_ident(&self.outputs[0]);
        let input = scope.arg(input_arg);

        let scale = self.config.scale.as_ref().and_then(|s| s.first()).copied();
        let offset = self.config.offset.as_ref().and_then(|o| o.first()).copied();

        // Generate the transformation based on input type
        let function = match &input_arg.ty {
            ArgType::Scalar(_) => {
                // Scalar case: apply formula directly
                match (offset, scale) {
                    (Some(offset), Some(scale)) => quote! { (#input - #offset) * #scale },
                    (Some(offset), None) => quote! { #input - #offset },
                    (None, Some(scale)) => quote! { #input * #scale },
                    (None, None) => quote! { #input },
                }
            }
            ArgType::Tensor(_) => {
                // Tensor case: use tensor operations
                match (offset, scale) {
                    (Some(offset), Some(scale)) => quote! { (#input.clone() - #offset) * #scale },
                    (Some(offset), None) => quote! { #input.clone() - #offset },
                    (None, Some(scale)) => quote! { #input.clone() * #scale },
                    (None, None) => quote! { #input.clone() },
                }
            }
            _ => {
                // Unsupported type, just pass through
                quote! { #input.clone() }
            }
        };

        quote! {
            let #output = #function;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::scaler::ScalerConfig;
    use onnx_ir::ir::{ArgType, TensorType};
    use onnx_ir::scaler::ScalerNode;

    #[test]
    fn test_scaler_scale_only() {
        let config = ScalerConfig::new(Some(vec![2.0]), None);
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ScalerNode::new("scaler1".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.clone() * 2f32;
            output
        }
        "###);
    }

    #[test]
    fn test_scaler_offset_only() {
        let config = ScalerConfig::new(None, Some(vec![1.0]));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ScalerNode::new("scaler2".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.clone() - 1f32;
            output
        }
        "###);
    }

    #[test]
    fn test_scaler_both_scale_and_offset() {
        let config = ScalerConfig::new(Some(vec![2.0]), Some(vec![1.0]));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ScalerNode::new("scaler3".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = (input.clone() - 1f32) * 2f32;
            output
        }
        "###);
    }

    #[test]
    fn test_scaler_no_transform() {
        let config = ScalerConfig::new(None, None);
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ScalerNode::new("scaler4".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.clone();
            output
        }
        "###);
    }
}
