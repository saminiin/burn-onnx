use super::prelude::*;

impl NodeCodegen for onnx_ir::svmregressor::SVMRegressorNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(&self.inputs[0]);
        let output = arg_to_ident(&self.outputs[0]);

        // Extract configuration
        let kernel_type = self.config.kernel_type.as_deref().unwrap_or("LINEAR");
        let post_transform = self.config.post_transform.as_deref().unwrap_or("NONE");
        
        let coefficients = self.config.coefficients.as_ref();
        let support_vectors = self.config.support_vectors.as_ref();
        let rho = self.config.rho.as_ref().and_then(|r| r.first()).copied().unwrap_or(0.0);
        let n_supports = self.config.n_supports.unwrap_or(0) as usize;

        // Generate kernel computation based on kernel type
        let kernel_computation = match kernel_type {
            "LINEAR" => {
                if let (Some(coef), Some(sv)) = (coefficients, support_vectors) {
                    let coef_vec: Vec<_> = coef.iter().copied().collect();
                    let sv_vec: Vec<_> = sv.iter().copied().collect();
                    
                    quote! {
                        {
                            let coefficients = vec![#(#coef_vec),*];
                            let support_vectors = vec![#(#sv_vec),*];
                            let n_supports = #n_supports;
                            let rho = #rho;
                            
                            // Compute linear kernel: K(x, sv) = x · sv
                            let input_shape = #input.shape();
                            let n_features = input_shape.dims[input_shape.dims.len() - 1];
                            let batch_size = input_shape.dims[0];
                            
                            // Reshape support vectors into matrix [n_supports, n_features]
                            let sv_data = burn::tensor::TensorData::new(support_vectors.clone(), [n_supports, n_features]);
                            let sv_tensor = Tensor::<B, 2>::from_data(sv_data, &*self.device);
                            
                            // Compute kernel matrix: input @ sv^T [batch, n_supports]
                            let kernel_values = #input.matmul(sv_tensor.transpose());
                            
                            // Apply coefficients and sum
                            let coef_data = burn::tensor::TensorData::from(&coefficients[..n_supports]);
                            let coef_tensor = Tensor::<B, 1>::from_data(coef_data, &*self.device);
                            
                            // prediction = kernel_values @ coefficients - rho
                            let result = kernel_values.matmul(coef_tensor.unsqueeze()) - rho;
                            result
                        }
                    }
                } else {
                    quote! { #input.clone() }
                }
            }
            "RBF" => {
                if let (Some(coef), Some(sv), Some(params)) = (coefficients, support_vectors, &self.config.kernel_params) {
                    let coef_vec: Vec<_> = coef.iter().copied().collect();
                    let sv_vec: Vec<_> = sv.iter().copied().collect();
                    let gamma = params.first().copied().unwrap_or(0.1);
                    
                    quote! {
                        {
                            let coefficients = vec![#(#coef_vec),*];
                            let support_vectors = vec![#(#sv_vec),*];
                            let n_supports = #n_supports;
                            let rho = #rho;
                            let gamma = #gamma;
                            
                            // Compute RBF kernel: K(x, sv) = exp(-gamma * ||x - sv||^2)
                            let input_shape = #input.shape();
                            let n_features = input_shape.dims[input_shape.dims.len() - 1];
                            let batch_size = input_shape.dims[0];
                            
                            let sv_data = burn::tensor::TensorData::new(support_vectors.clone(), [n_supports, n_features]);
                            let sv_tensor = Tensor::<B, 2>::from_data(sv_data, &*self.device);
                            
                            // Compute squared distances
                            let mut kernel_values = Tensor::<B, 2>::zeros([batch_size, n_supports], &*self.device);
                            for i in 0..n_supports {
                                let sv_i = sv_tensor.clone().narrow(0, i, 1).squeeze_dims::<1>(&[0]);
                                let diff = #input.clone() - sv_i.unsqueeze();
                                let sq_dist = diff.clone().powf_scalar(2.0).sum_dim(1);
                                let kernel_val = (-gamma * sq_dist).exp();
                                kernel_values = kernel_values.slice_assign([0..batch_size as i64, i as i64..(i + 1) as i64], kernel_val.unsqueeze_dim(1));
                            }
                            
                            let coef_data = burn::tensor::TensorData::from(&coefficients[..n_supports]);
                            let coef_tensor = Tensor::<B, 1>::from_data(coef_data, &*self.device);
                            
                            let result = kernel_values.matmul(coef_tensor.unsqueeze()) - rho;
                            result
                        }
                    }
                } else {
                    quote! { #input.clone() }
                }
            }
            "POLY" => {
                if let (Some(coef), Some(sv), Some(params)) = (coefficients, support_vectors, &self.config.kernel_params) {
                    let coef_vec: Vec<_> = coef.iter().copied().collect();
                    let sv_vec: Vec<_> = sv.iter().copied().collect();
                    let gamma = params.get(0).copied().unwrap_or(1.0);
                    let coef0 = params.get(1).copied().unwrap_or(0.0);
                    let degree = params.get(2).copied().unwrap_or(3.0);
                    
                    quote! {
                        {
                            let coefficients = vec![#(#coef_vec),*];
                            let support_vectors = vec![#(#sv_vec),*];
                            let n_supports = #n_supports;
                            let rho = #rho;
                            let gamma = #gamma;
                            let coef0 = #coef0;
                            let degree = #degree;
                            
                            // Compute polynomial kernel: K(x, sv) = (gamma * x · sv + coef0)^degree
                            let input_shape = #input.shape();
                            let n_features = input_shape.dims[input_shape.dims.len() - 1];
                            
                            let sv_data = burn::tensor::TensorData::new(support_vectors.clone(), [n_supports, n_features]);
                            let sv_tensor = Tensor::<B, 2>::from_data(sv_data, &*self.device);
                            
                            let dot_products = #input.matmul(sv_tensor.transpose());
                            let kernel_values = (dot_products * gamma + coef0).powf_scalar(degree);
                            
                            let coef_data = burn::tensor::TensorData::from(&coefficients[..n_supports]);
                            let coef_tensor = Tensor::<B, 1>::from_data(coef_data, &*self.device);
                            
                            let result = kernel_values.matmul(coef_tensor.unsqueeze()) - rho;
                            result
                        }
                    }
                } else {
                    quote! { #input.clone() }
                }
            }
            "SIGMOID" => {
                if let (Some(coef), Some(sv), Some(params)) = (coefficients, support_vectors, &self.config.kernel_params) {
                    let coef_vec: Vec<_> = coef.iter().copied().collect();
                    let sv_vec: Vec<_> = sv.iter().copied().collect();
                    let gamma = params.get(0).copied().unwrap_or(1.0);
                    let coef0 = params.get(1).copied().unwrap_or(0.0);
                    
                    quote! {
                        {
                            let coefficients = vec![#(#coef_vec),*];
                            let support_vectors = vec![#(#sv_vec),*];
                            let n_supports = #n_supports;
                            let rho = #rho;
                            let gamma = #gamma;
                            let coef0 = #coef0;
                            
                            // Compute sigmoid kernel: K(x, sv) = tanh(gamma * x · sv + coef0)
                            let input_shape = #input.shape();
                            let n_features = input_shape.dims[input_shape.dims.len() - 1];
                            
                            let sv_data = burn::tensor::TensorData::new(support_vectors.clone(), [n_supports, n_features]);
                            let sv_tensor = Tensor::<B, 2>::from_data(sv_data, &*self.device);
                            
                            let dot_products = #input.matmul(sv_tensor.transpose());
                            let kernel_values = (dot_products * gamma + coef0).tanh();
                            
                            let coef_data = burn::tensor::TensorData::from(&coefficients[..n_supports]);
                            let coef_tensor = Tensor::<B, 1>::from_data(coef_data, &*self.device);
                            
                            let result = kernel_values.matmul(coef_tensor.unsqueeze()) - rho;
                            result
                        }
                    }
                } else {
                    quote! { #input.clone() }
                }
            }
            _ => quote! { #input.clone() }
        };

        // Apply post-transform if needed
        let function = match post_transform {
            "NONE" => kernel_computation,
            "LOGISTIC" => quote! { { let y = #kernel_computation; (y.neg().exp() + 1.0).recip() } },
            "SOFTMAX" => quote! { { let y = #kernel_computation; y.exp() / y.exp().sum_dim(0) } },
            "SOFTMAX_ZERO" => quote! { { 
                let y = #kernel_computation;
                let zeros = y.zeros_like();
                let combined = Tensor::cat(vec![y, zeros], 0);
                combined.exp() / combined.exp().sum_dim(0)
            } },
            "PROBIT" => quote! { { let y = #kernel_computation; y } }, // Probit requires special functions
            _ => kernel_computation,
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
    use onnx_ir::svmregressor::{SVMRegressorConfig, SVMRegressorNodeBuilder};

    #[test]
    fn test_svm_regressor_linear() {
        let config = SVMRegressorConfig {
            coefficients: Some(vec![1.0, -0.5]),
            kernel_params: None,
            kernel_type: Some("LINEAR".to_string()),
            n_supports: Some(2),
            one_class: None,
            post_transform: None,
            rho: Some(vec![0.5]),
            support_vectors: Some(vec![1.0, 2.0, 3.0, 4.0]), // 2 support vectors with 2 features each
        };
        let node = SVMRegressorNodeBuilder::new("svm1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1> {
            let output = {
                let coefficients = vec![1f32, - 0.5f32];
                let support_vectors = vec![1f32, 2f32, 3f32, 4f32];
                let n_supports = 2usize;
                let rho = 0.5f32;
                let input_shape = input.shape();
                let n_features = input_shape.dims[input_shape.dims.len() - 1];
                let batch_size = input_shape.dims[0];
                let sv_data = burn::tensor::TensorData::new(
                    support_vectors.clone(),
                    [n_supports, n_features],
                );
                let sv_tensor = Tensor::<B, 2>::from_data(sv_data, &*self.device);
                let kernel_values = input.matmul(sv_tensor.transpose());
                let coef_data = burn::tensor::TensorData::from(&coefficients[..n_supports]);
                let coef_tensor = Tensor::<B, 1>::from_data(coef_data, &*self.device);
                let result = kernel_values.matmul(coef_tensor.unsqueeze()) - rho;
                result
            };
            output
        }
        "#);
    }

    #[test]
    fn test_svm_regressor_rbf() {
        let config = SVMRegressorConfig {
            coefficients: Some(vec![1.0, -0.5]),
            kernel_params: Some(vec![0.1]), // gamma
            kernel_type: Some("RBF".to_string()),
            n_supports: Some(2),
            one_class: None,
            post_transform: None,
            rho: Some(vec![0.0]),
            support_vectors: Some(vec![1.0, 2.0, 3.0, 4.0]),
        };
        let node = SVMRegressorNodeBuilder::new("svm1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert!(code.contains("gamma"));
        assert!(code.contains("exp()"));
    }

    #[test]
    fn test_svm_regressor_poly() {
        let config = SVMRegressorConfig {
            coefficients: Some(vec![1.0, -0.5]),
            kernel_params: Some(vec![1.0, 0.0, 3.0]), // gamma, coef0, degree
            kernel_type: Some("POLY".to_string()),
            n_supports: Some(2),
            one_class: None,
            post_transform: None,
            rho: Some(vec![0.0]),
            support_vectors: Some(vec![1.0, 2.0, 3.0, 4.0]),
        };
        let node = SVMRegressorNodeBuilder::new("svm1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert!(code.contains("degree"));
        assert!(code.contains("powf_scalar"));
    }

    #[test]
    fn test_svm_regressor_with_logistic() {
        let config = SVMRegressorConfig {
            coefficients: Some(vec![1.0]),
            kernel_params: None,
            kernel_type: Some("LINEAR".to_string()),
            n_supports: Some(1),
            one_class: None,
            post_transform: Some("LOGISTIC".to_string()),
            rho: Some(vec![0.0]),
            support_vectors: Some(vec![1.0, 2.0]),
        };
        let node = SVMRegressorNodeBuilder::new("svm1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert!(code.contains("neg"));
        assert!(code.contains("recip"));
    }
}
