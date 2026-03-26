use super::prelude::*;

impl NodeCodegen for onnx_ir::one_hot_encoder::OneHotEncoderNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(&self.inputs[0]);
        let output = arg_to_ident(&self.outputs[0]);

        // Extract categories from config (validated in onnx-ir extract_config)
        let cats_int64s = self.config.cats_int64s.as_ref().unwrap();

        let num_categories = cats_int64s.len();

        // Generate category mapping as a lookup array
        let cat_values: Vec<_> = cats_int64s.to_vec();

        quote! {
            // OneHotEncoder: Map categorical values to one-hot encoded output
            // Input: [batch_size, 1] with categorical values
            // Output: [batch_size, num_categories] with one-hot encoding
            let #output = {
                let categories = [#(#cat_values),*];
                let num_categories = #num_categories;

                // Get input shape and flatten to 1D
                let input_shape = #input.shape();
                let batch_size = input_shape.dims[0];

                // Flatten input to 1D vector and convert to i64
                let input_flat = #input.clone().reshape([batch_size]).into_data();
                let input_vec: Vec<i64> = input_flat.iter::<i64>()
                    .map(|x| x)
                    .collect();

                // Create output tensor initialized with zeros
                let mut output_data = Vec::with_capacity(batch_size * num_categories);

                // For each input value, find its index in categories and set one-hot
                for &val in input_vec.iter() {
                    let mut row = vec![0.0f32; num_categories];

                    // Find the index of this value in categories
                    if let Some(idx) = categories.iter().position(|&c| c == val) {
                        row[idx] = 1.0;
                    }
                    // If value not found in categories, row remains all zeros

                    output_data.extend_from_slice(&row);
                }

                // Convert to tensor
                Tensor::<B, 2>::from_data(
                    burn::tensor::TensorData::new(output_data, [batch_size, num_categories].into()).convert(),
                    &*self.device
                )
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::one_hot_encoder::{OneHotEncoderConfig, OneHotEncoderNodeBuilder};

    #[test]
    fn test_one_hot_encoder_basic() {
        // Test basic one-hot encoding with integer categories
        let config = OneHotEncoderConfig::new(
            Some(vec![1, 2, 3, 4, 5, 6, 7]), // 7 categories
            None,
            None,
        );

        let node = OneHotEncoderNodeBuilder::new("encoder1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();

        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let categories = [1i64, 2i64, 3i64, 4i64, 5i64, 6i64, 7i64];
                let num_categories = 7usize;
                let input_shape = input.shape();
                let batch_size = input_shape.dims[0];
                let input_flat = input.clone().reshape([batch_size]).into_data();
                let input_vec: Vec<i64> = input_flat.iter::<i64>().map(|x| x).collect();
                let mut output_data = Vec::with_capacity(batch_size * num_categories);
                for &val in input_vec.iter() {
                    let mut row = vec![0.0f32; num_categories];
                    if let Some(idx) = categories.iter().position(|&c| c == val) {
                        row[idx] = 1.0;
                    }
                    output_data.extend_from_slice(&row);
                }
                Tensor::<
                    B,
                    2,
                >::from_data(
                    burn::tensor::TensorData::new(
                            output_data,
                            [batch_size, num_categories].into(),
                        )
                        .convert(),
                    &*self.device,
                )
            };
            output
        }
        "###);
    }

    #[test]
    fn test_one_hot_encoder_fewer_categories() {
        // Test with only 3 categories
        let config = OneHotEncoderConfig::new(Some(vec![0, 1, 2]), None, None);

        let node = OneHotEncoderNodeBuilder::new("encoder2")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();

        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let categories = [0i64, 1i64, 2i64];
                let num_categories = 3usize;
                let input_shape = input.shape();
                let batch_size = input_shape.dims[0];
                let input_flat = input.clone().reshape([batch_size]).into_data();
                let input_vec: Vec<i64> = input_flat.iter::<i64>().map(|x| x).collect();
                let mut output_data = Vec::with_capacity(batch_size * num_categories);
                for &val in input_vec.iter() {
                    let mut row = vec![0.0f32; num_categories];
                    if let Some(idx) = categories.iter().position(|&c| c == val) {
                        row[idx] = 1.0;
                    }
                    output_data.extend_from_slice(&row);
                }
                Tensor::<
                    B,
                    2,
                >::from_data(
                    burn::tensor::TensorData::new(
                            output_data,
                            [batch_size, num_categories].into(),
                        )
                        .convert(),
                    &*self.device,
                )
            };
            output
        }
        "###);
    }

    #[test]
    fn test_one_hot_encoder_with_gaps() {
        // Test with non-contiguous category values (e.g., protocol numbers)
        let config = OneHotEncoderConfig::new(
            Some(vec![6, 17, 1]), // TCP, UDP, ICMP
            None,
            None,
        );

        let node = OneHotEncoderNodeBuilder::new("protocol_encoder")
            .input_tensor("data", 2, DType::F32)
            .output_tensor("encoded", 2, DType::F32)
            .config(config)
            .build();

        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r###"
        pub fn forward(&self, data: Tensor<B, 2>) -> Tensor<B, 2> {
            let encoded = {
                let categories = [6i64, 17i64, 1i64];
                let num_categories = 3usize;
                let input_shape = data.shape();
                let batch_size = input_shape.dims[0];
                let input_flat = data.clone().reshape([batch_size]).into_data();
                let input_vec: Vec<i64> = input_flat.iter::<i64>().map(|x| x).collect();
                let mut output_data = Vec::with_capacity(batch_size * num_categories);
                for &val in input_vec.iter() {
                    let mut row = vec![0.0f32; num_categories];
                    if let Some(idx) = categories.iter().position(|&c| c == val) {
                        row[idx] = 1.0;
                    }
                    output_data.extend_from_slice(&row);
                }
                Tensor::<
                    B,
                    2,
                >::from_data(
                    burn::tensor::TensorData::new(
                            output_data,
                            [batch_size, num_categories].into(),
                        )
                        .convert(),
                    &*self.device,
                )
            };
            encoded
        }
        "###);
    }
}
