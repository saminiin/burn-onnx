use super::prelude::*;

impl NodeCodegen for onnx_ir::imputer::ImputerNode {
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

        let function = match &input_arg.ty {
            ArgType::Scalar(scalar_ty) => {
                // Handle scalar inputs
                match scalar_ty {
                    DType::F32 | DType::F64 => {
                        if let Some(imputed_floats) = &self.config.imputed_value_floats {
                            if let Some(first_value) = imputed_floats.first() {
                                let replaced_value = self.config.replaced_value_float.unwrap_or(f32::NAN);
                                quote! { if #input.clone().is_nan() || #input == #replaced_value { #first_value } else { #input } }
                            } else {
                                quote! { #input }
                            }
                        } else {
                            quote! { #input }
                        }
                    }
                    _ => quote! { #input }
                }
            }
            ArgType::Tensor(tensor_ty) => {
                // Handle tensor inputs based on dtype
                match tensor_ty.dtype {
                    DType::F32 | DType::F64 => {
                        if let Some(imputed_floats) = &self.config.imputed_value_floats {
                            let replaced_value = self.config.replaced_value_float.unwrap_or(f32::NAN);
                            
                            if imputed_floats.len() == 1 {
                                let imputed_value = imputed_floats[0];
                                if replaced_value.is_nan() {
                                    quote! {
                                        {
                                            let mask = #input.clone().is_nan();
                                            #input.clone().mask_fill(mask, #imputed_value)
                                        }
                                    }
                                } else {
                                    quote! {
                                        {
                                            let mask = #input.clone().equal_elem(#replaced_value);
                                            #input.clone().mask_fill(mask, #imputed_value)
                                        }
                                    }
                                }
                            } else {
                                // Multiple imputed values per feature
                                let imputed_values_vec: Vec<_> = imputed_floats.iter().copied().collect();
                                if replaced_value.is_nan() {
                                    quote! {
                                        {
                                            let imputed_values = vec![#(#imputed_values_vec),*];
                                            let mask = #input.clone().is_nan();
                                            let mut result = #input.clone();
                                            let shape = result.shape();
                                            let num_features = shape.dims[shape.dims.len() - 1];
                                            
                                            for feature_idx in 0..num_features.min(imputed_values.len()) {
                                                let feature_mask = mask.clone().narrow(shape.dims.len() - 1, feature_idx, 1);
                                                let imputed_val = imputed_values[feature_idx];
                                                result = result.mask_fill(feature_mask, imputed_val);
                                            }
                                            result
                                        }
                                    }
                                } else {
                                    quote! {
                                        {
                                            let imputed_values = vec![#(#imputed_values_vec),*];
                                            let mask = #input.clone().equal_elem(#replaced_value);
                                            let mut result = #input.clone();
                                            let shape = result.shape();
                                            let num_features = shape.dims[shape.dims.len() - 1];
                                            
                                            for feature_idx in 0..num_features.min(imputed_values.len()) {
                                                let feature_mask = mask.clone().narrow(shape.dims.len() - 1, feature_idx, 1);
                                                let imputed_val = imputed_values[feature_idx];
                                                result = result.mask_fill(feature_mask, imputed_val);
                                            }
                                            result
                                        }
                                    }
                                }
                            }
                        } else {
                            quote! { #input.clone() }
                        }
                    }
                    DType::I32 | DType::I64 | DType::I8 | DType::I16 => {
                        if let Some(imputed_ints) = &self.config.imputed_value_ints {
                            if let Some(replaced_value) = self.config.replaced_value_float {
                                let replaced_int = replaced_value as i64;
                                
                                if imputed_ints.len() == 1 {
                                    let imputed_value = imputed_ints[0];
                                    quote! {
                                        {
                                            let mask = #input.clone().equal_elem(#replaced_int);
                                            #input.clone().mask_fill(mask, #imputed_value)
                                        }
                                    }
                                } else {
                                    // Multiple imputed values per feature
                                    let imputed_values_vec: Vec<_> = imputed_ints.iter().copied().collect();
                                    quote! {
                                        {
                                            let imputed_values = vec![#(#imputed_values_vec),*];
                                            let mask = #input.clone().equal_elem(#replaced_int);
                                            let mut result = #input.clone();
                                            let shape = result.shape();
                                            let num_features = shape.dims[shape.dims.len() - 1];
                                            
                                            for feature_idx in 0..num_features.min(imputed_values.len()) {
                                                let feature_mask = mask.clone().narrow(shape.dims.len() - 1, feature_idx, 1);
                                                let imputed_val = imputed_values[feature_idx];
                                                result = result.mask_fill(feature_mask, imputed_val);
                                            }
                                            result
                                        }
                                    }
                                }
                            } else {
                                quote! { #input.clone() }
                            }
                        } else {
                            quote! { #input.clone() }
                        }
                    }
                    _ => quote! { #input.clone() }
                }
            }
            _ => quote! { #input.clone() }
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
    use onnx_ir::imputer::ImputerConfig;
    use onnx_ir::ir::{ArgType, TensorType};
    use onnx_ir::imputer::ImputerNode;

    #[test]
    fn test_imputer_single_float() {
        let config = ImputerConfig::new(Some(vec![0.0]), None, None);
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ImputerNode::new("imputer1".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let mask = input.is_nan();
                input.mask_fill(mask, 0f32)
            };
            output
        }
        "###);
    }

    #[test]
    fn test_imputer_single_float_with_replaced_value() {
        let config = ImputerConfig::new(Some(vec![1.0]), None, Some(-999.0));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ImputerNode::new("imputer2".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let mask = input.clone().equal_elem(-999f32);
                input.mask_fill(mask, 1f32)
            };
            output
        }
        "###);
    }

    #[test]
    fn test_imputer_multiple_floats() {
        let config = ImputerConfig::new(Some(vec![0.0, 1.0, 2.0]), None, None);
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = ImputerNode::new("imputer3".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let imputed_values = vec![0f32, 1f32, 2f32];
                let mask = input.is_nan();
                let mut result = input.clone();
                let shape = result.shape();
                let num_features = shape.dims[shape.dims.len() - 1];
                for feature_idx in 0..num_features.min(imputed_values.len()) {
                    let feature_mask = mask.clone().narrow(shape.dims.len() - 1, feature_idx, 1);
                    let imputed_val = imputed_values[feature_idx];
                    result = result.mask_fill(feature_mask, imputed_val);
                }
                result
            };
            output
        }
        "###);
    }

    #[test]
    fn test_imputer_single_int() {
        let config = ImputerConfig::new(None, Some(vec![0]), Some(-1.0));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::I64, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::I64, 2, None)),
        );
        let node = ImputerNode::new("imputer4".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
            let output = {
                let mask = input.clone().equal_elem(-1i64);
                input.mask_fill(mask, 0i64)
            };
            output
        }
        "###);
    }
}
