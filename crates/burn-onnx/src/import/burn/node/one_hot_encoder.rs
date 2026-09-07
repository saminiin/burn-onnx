use super::prelude::*;

/// Get categories as f32 values.
/// Prefer cats_int64s; fall back to cats_strings parsed as numbers.
fn cats_f32(config: &onnx_ir::one_hot_encoder::OneHotEncoderConfig) -> Vec<f32> {
    if let Some(cats) = &config.cats_int64s {
        cats.iter().map(|&c| c as f32).collect()
    } else if let Some(strings) = &config.cats_strings {
        strings
            .iter()
            .map(|s| {
                s.parse::<f64>().unwrap_or_else(|_| {
                    log::error!(
                        "OneHotEncoder: cannot parse cats_strings value '{}' as a number. \
                         String categories without numeric representation are not supported in Burn codegen.",
                        s
                    );
                    0.0
                }) as f32
            })
            .collect()
    } else {
        log::error!("OneHotEncoder: neither cats_int64s nor cats_strings provided");
        vec![]
    }
}

impl NodeCodegen for onnx_ir::one_hot_encoder::OneHotEncoderNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        // Store the category lookup table once at construction time instead of
        // rebuilding it on every forward() call — category lists can be large
        // (thousands of entries), and forward() may be called per-inference.
        let cats = cats_f32(&self.config);
        let name = Ident::new(&self.name, Span::call_site());

        Some(Field::new(
            &self.name,
            quote! { Tensor<1> },
            quote! {
                let #name: Tensor<1> = Tensor::<1>::from_data(
                    [#(#cats),*],
                    (device, burn::tensor::DType::F32),
                );
            },
        ))
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = &self.inputs[0];
        let output = arg_to_ident(&self.outputs[0]);
        let input = scope.arg(input_arg);
        let field_name = Ident::new(&self.name, Span::call_site());

        let function = match &input_arg.ty {
            ArgType::Tensor(tensor_type) => {
                let input_rank = tensor_type.rank;

                // Cast input to F32 (all inputs get cast to int for lookup, but we compare as F32)
                let input_expr = match tensor_type.dtype {
                    DType::F32 => quote! { #input.clone() },
                    DType::F64 => quote! { #input.cast(burn::tensor::DType::F32) },
                    _ => quote! { #input.float().cast(burn::tensor::DType::F32) },
                };

                // Mirrors the derivation in field() exactly.
                let num_categories = cats_f32(&self.config).len();

                // Build reshape dims for the category tensor to broadcast:
                // input shape: [d0, d1, ..., d_{r-1}]
                // category tensor shape: [1, 1, ..., 1, num_categories]
                // The category tensor has rank = input_rank + 1, with 1s for all input dims
                // and num_categories for the last dim.
                let ones: Vec<TokenStream> = (0..input_rank).map(|_| quote! { 1usize }).collect();

                // Strategy: unsqueeze input to add a trailing dim, reshape the stored
                // category tensor with shape [1,...,1,num_cats], broadcast-compare via
                // equal, then float-cast. This is efficient and uses native tensor ops.

                quote! {
                    {
                        let x = #input_expr;
                        // Unsqueeze input: [d0,...,d_{r-1}] -> [d0,...,d_{r-1}, 1]
                        let x_unsqueezed = x.unsqueeze_dim(#input_rank);
                        // Reshape the precomputed category tensor: [num_categories] -> [1,...,1, num_categories]
                        let cats = self.#field_name.clone().reshape([#(#ones,)* #num_categories]);
                        // Broadcast compare: [d0,...,d_{r-1}, 1] == [1,...,1, num_cats]
                        // -> [d0,...,d_{r-1}, num_cats] (bool)
                        // Convert bool to float
                        x_unsqueezed.equal(cats).float().cast(burn::tensor::DType::F32)
                    }
                }
            }
            ty => {
                unreachable!(
                    "OneHotEncoder input is always a tensor (validated in onnx-ir), got {ty:?}"
                )
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
    use onnx_ir::one_hot_encoder::OneHotEncoderConfig;
    use onnx_ir::one_hot_encoder::OneHotEncoderNode;

    #[test]
    fn test_onehotencoder_1d_input() {
        let config = OneHotEncoderConfig::new(Some(vec![0, 1, 2, 3]), None, Some(1));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 1, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = OneHotEncoderNode::new("ohe1".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<1>) -> Tensor<2> {
            let output = {
                let x = input.clone();
                let x_unsqueezed = x.unsqueeze_dim(1usize);
                let cats = self.ohe1.clone().reshape([1usize, 4usize]);
                x_unsqueezed.equal(cats).float().cast(burn::tensor::DType::F32)
            };
            output
        }
        ");
    }

    #[test]
    fn test_onehotencoder_2d_input() {
        let config = OneHotEncoderConfig::new(Some(vec![0, 1, 2]), None, Some(1));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 3, None)),
        );
        let node = OneHotEncoderNode::new("ohe2".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<3> {
            let output = {
                let x = input.clone();
                let x_unsqueezed = x.unsqueeze_dim(2usize);
                let cats = self.ohe2.clone().reshape([1usize, 1usize, 3usize]);
                x_unsqueezed.equal(cats).float().cast(burn::tensor::DType::F32)
            };
            output
        }
        ");
    }

    #[test]
    fn test_onehotencoder_int_input() {
        let config = OneHotEncoderConfig::new(Some(vec![0, 1, 2, 3, 4]), None, Some(1));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::I64, 1, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = OneHotEncoderNode::new("ohe3".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<1, Int>) -> Tensor<2> {
            let output = {
                let x = input.float().cast(burn::tensor::DType::F32);
                let x_unsqueezed = x.unsqueeze_dim(1usize);
                let cats = self.ohe3.clone().reshape([1usize, 5usize]);
                x_unsqueezed.equal(cats).float().cast(burn::tensor::DType::F32)
            };
            output
        }
        ");
    }

    #[test]
    fn test_onehotencoder_cats_strings() {
        // cats_strings with numeric string values should work
        let config = OneHotEncoderConfig::new(
            None,
            Some(vec!["0".to_string(), "1".to_string(), "2".to_string()]),
            Some(1),
        );
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 1, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = OneHotEncoderNode::new("ohe4".to_string(), vec![input], vec![output], config);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<1>) -> Tensor<2> {
            let output = {
                let x = input.clone();
                let x_unsqueezed = x.unsqueeze_dim(1usize);
                let cats = self.ohe4.clone().reshape([1usize, 3usize]);
                x_unsqueezed.equal(cats).float().cast(burn::tensor::DType::F32)
            };
            output
        }
        ");
    }

    #[test]
    fn test_onehotencoder_field_init() {
        let config = OneHotEncoderConfig::new(Some(vec![0, 1, 2, 3]), None, Some(1));
        let input = onnx_ir::ir::Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 1, None)),
        );
        let output = onnx_ir::ir::Argument::new(
            "output",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let node = OneHotEncoderNode::new("ohe1".to_string(), vec![input], vec![output], config);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @"
        let ohe1: Tensor<1> = Tensor::<
            1,
        >::from_data([0f32, 1f32, 2f32, 3f32], (device, burn::tensor::DType::F32));
        ");
    }
}
