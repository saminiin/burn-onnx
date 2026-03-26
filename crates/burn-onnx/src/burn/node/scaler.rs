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

        // Generate the transformation based on input type
        let function = match &input_arg.ty {
            ArgType::ScalarNative(_) => {
                // Scalar case: use first element of scale/offset arrays
                let scale = self.config.scale.as_ref().and_then(|s| s.first()).copied();
                let offset = self.config.offset.as_ref().and_then(|o| o.first()).copied();

                // Convert f32 values to tokens with proper type suffix
                use crate::burn::codegen::f32_to_tokens;
                let scale_tokens = scale.map(f32_to_tokens);
                let offset_tokens = offset.map(f32_to_tokens);

                match (offset_tokens, scale_tokens) {
                    (Some(offset), Some(scale)) => quote! { (#input - #offset) * #scale },
                    (Some(offset), None) => quote! { #input - #offset },
                    (None, Some(scale)) => quote! { #input * #scale },
                    (None, None) => quote! { #input },
                }
            }
            ArgType::Tensor(tensor_type) => {
                // Tensor case: per-feature scaling
                // Formula: Y = (X - offset) * scale
                // Scale and offset are applied element-wise along the last dimension (feature dimension)

                let has_offset = self.config.offset.is_some();
                let has_scale = self.config.scale.is_some();
                let input_rank = tensor_type.rank;

                // Helper to create reshape dimensions: [1, 1, ..., 1, num_features]
                let create_reshape_dims = |num_features: usize| -> Vec<TokenStream> {
                    (0..input_rank.saturating_sub(1))
                        .map(|_| quote! { 1usize })
                        .chain(std::iter::once(quote! { #num_features }))
                        .collect()
                };

                match (has_offset, has_scale) {
                    (true, true) => {
                        // Both offset and scale
                        let offset_values: Vec<_> = self.config.offset.as_ref().unwrap().to_vec();
                        let scale_values: Vec<_> = self.config.scale.as_ref().unwrap().to_vec();
                        let num_features = offset_values.len();
                        let reshape_dims = create_reshape_dims(num_features);

                        quote! {
                            {
                                // Create offset and scale tensors, reshape to broadcast along feature dimension
                                let offset_tensor = Tensor::<B, 1>::from_floats([#(#offset_values),*], &*self.device)
                                    .reshape([#(#reshape_dims),*]);
                                let scale_tensor = Tensor::<B, 1>::from_floats([#(#scale_values),*], &*self.device)
                                    .reshape([#(#reshape_dims),*]);

                                // Apply formula: (input - offset) * scale with broadcasting
                                (#input.clone() - offset_tensor) * scale_tensor
                            }
                        }
                    }
                    (true, false) => {
                        // Only offset
                        let offset_values: Vec<_> = self.config.offset.as_ref().unwrap().to_vec();
                        let num_features = offset_values.len();
                        let reshape_dims = create_reshape_dims(num_features);

                        quote! {
                            {
                                let offset_tensor = Tensor::<B, 1>::from_floats([#(#offset_values),*], &*self.device)
                                    .reshape([#(#reshape_dims),*]);
                                #input.clone() - offset_tensor
                            }
                        }
                    }
                    (false, true) => {
                        // Only scale
                        let scale_values: Vec<_> = self.config.scale.as_ref().unwrap().to_vec();
                        let num_features = scale_values.len();
                        let reshape_dims = create_reshape_dims(num_features);

                        quote! {
                            {
                                let scale_tensor = Tensor::<B, 1>::from_floats([#(#scale_values),*], &*self.device)
                                    .reshape([#(#reshape_dims),*]);
                                #input.clone() * scale_tensor
                            }
                        }
                    }
                    (false, false) => {
                        // No transformation
                        quote! { #input.clone() }
                    }
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
    use onnx_ir::ir::{ArgType, TensorType};
    use onnx_ir::scaler::ScalerConfig;
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
            let output = {
                let scale_tensor = Tensor::<B, 1>::from_floats([2f32], &*self.device)
                    .reshape([1usize, 1usize]);
                input.clone() * scale_tensor
            };
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
            let output = {
                let offset_tensor = Tensor::<B, 1>::from_floats([1f32], &*self.device)
                    .reshape([1usize, 1usize]);
                input.clone() - offset_tensor
            };
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
            let output = {
                let offset_tensor = Tensor::<B, 1>::from_floats([1f32], &*self.device)
                    .reshape([1usize, 1usize]);
                let scale_tensor = Tensor::<B, 1>::from_floats([2f32], &*self.device)
                    .reshape([1usize, 1usize]);
                (input.clone() - offset_tensor) * scale_tensor
            };
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

    #[test]
    fn test_scaler_per_feature_scaling() {
        // Test with different scale/offset per feature
        let config = ScalerConfig::new(Some(vec![1.0, 2.0, 3.0]), Some(vec![0.5, 1.0, 1.5]));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ScalerNode::new("scaler5".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let offset_tensor = Tensor::<
                    B,
                    1,
                >::from_floats([0.5f32, 1f32, 1.5f32], &*self.device)
                    .reshape([1usize, 3usize]);
                let scale_tensor = Tensor::<B, 1>::from_floats([1f32, 2f32, 3f32], &*self.device)
                    .reshape([1usize, 3usize]);
                (input.clone() - offset_tensor) * scale_tensor
            };
            output
        }
        "###);
    }
}
