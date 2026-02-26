use super::prelude::*;

impl NodeCodegen for onnx_ir::svmregressor::SVMRegressorNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        // Store coefficients and support vectors as a tuple of tensors
        let coefficients = self.config.coefficients.as_ref()?;
        let support_vectors = self.config.support_vectors.as_ref()?;
        let n_supports = self.config.n_supports.unwrap_or(0) as usize;
        
        if coefficients.is_empty() || support_vectors.is_empty() || n_supports == 0 {
            return None;
        }
        
        let n_features = support_vectors.len() / n_supports;
        let name = Ident::new(&self.name, Span::call_site());
        
        let coef_data: Vec<_> = coefficients.iter().copied().collect();
        let sv_data: Vec<_> = support_vectors.iter().copied().collect();
        
        // Store as a tuple (coefficients, support_vectors)
        // Store coefficients as [n_supports, 1] for direct matmul compatibility
        Some(Field::new(
            &self.name,
            quote! { (Tensor<B, 2>, Tensor<B, 2>) },
            quote! {
                let #name = (
                    Tensor::<B, 2>::from_data(
                        burn::tensor::TensorData::new(vec![#(#coef_data),*], [#n_supports, 1]),
                        device
                    ),
                    Tensor::<B, 2>::from_data(
                        burn::tensor::TensorData::new(vec![#(#sv_data),*], [#n_supports, #n_features]),
                        device
                    )
                );
            },
        ))
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(&self.inputs[0]);
        let output = arg_to_ident(&self.outputs[0]);

        // Extract configuration
        let kernel_type = self.config.kernel_type.as_deref().unwrap_or("LINEAR");
        let post_transform = self.config.post_transform.as_deref().unwrap_or("NONE");
        
        let has_data = self.config.coefficients.is_some() && self.config.support_vectors.is_some();
        let rho = self.config.rho.as_ref().and_then(|r| r.first()).copied().unwrap_or(0.0);
        let n_supports = self.config.n_supports.unwrap_or(0) as usize;

        // Reference the stored tensors (tuple access .0 and .1)
        let field_name = Ident::new(&self.name, Span::call_site());

        // Generate kernel computation based on kernel type
        let kernel_computation = if !has_data {
            quote! { #input.clone() }
        } else {
            match kernel_type {
                "LINEAR" => {
                    quote! {
                        {
                            let rho = #rho;
                            let (coef, sv) = &self.#field_name;
                            
                            // Compute linear kernel: K(x, sv) = x · sv
                            // Compute kernel matrix: input @ sv^T [batch, n_supports]
                            let kernel_values = #input.matmul(sv.clone().transpose());
                            
                            // prediction = kernel_values @ coefficients - rho
                            // coef is already [n_supports, 1]
                            let result = kernel_values.matmul(coef.clone()) - rho;
                            result
                        }
                    }
                }
                "RBF" => {
                    let gamma = self.config.kernel_params.as_ref()
                        .and_then(|p| p.first()).copied().unwrap_or(0.1);
                    
                    quote! {
                        {
                            let rho = #rho;
                            let gamma = #gamma;
                            let n_supports = #n_supports;
                            let (coef, sv) = &self.#field_name;
                            
                            // Compute RBF kernel: K(x, sv) = exp(-gamma * ||x - sv||^2)
                            let input_shape = #input.shape();
                            let batch_size = input_shape.dims[0];
                            
                            // Compute squared distances
                            let mut kernel_values = Tensor::<B, 2>::zeros([batch_size, n_supports], &*self.device);
                            for i in 0..n_supports {
                                let sv_i = sv.clone().narrow(0, i, 1); // Shape: [1, n_features]
                                let diff = #input.clone() - sv_i; // Broadcasting: [batch, n_features] - [1, n_features]
                                let sq_dist = diff.clone().powf_scalar(2.0).sum_dim(1);
                                let kernel_val = (-gamma * sq_dist).exp().reshape([batch_size, 1]);
                                kernel_values = kernel_values.slice_assign([0..batch_size as i64, i as i64..(i + 1) as i64], kernel_val);
                            }
                            
                            // coef is already [n_supports, 1]
                            let result = kernel_values.matmul(coef.clone()) - rho;
                            result
                        }
                    }
                }
                "POLY" => {
                    let params = self.config.kernel_params.as_ref();
                    let gamma = params.and_then(|p| p.get(0)).copied().unwrap_or(1.0);
                    let coef0 = params.and_then(|p| p.get(1)).copied().unwrap_or(0.0);
                    let degree = params.and_then(|p| p.get(2)).copied().unwrap_or(3.0);
                    
                    quote! {
                        {
                            let rho = #rho;
                            let gamma = #gamma;
                            let coef0 = #coef0;
                            let degree = #degree;
                            let (coef, sv) = &self.#field_name;
                            
                            // Compute polynomial kernel: K(x, sv) = (gamma * x · sv + coef0)^degree
                            let dot_products = #input.matmul(sv.clone().transpose());
                            let kernel_values = (dot_products * gamma + coef0).powf_scalar(degree);
                            
                            // coef is already [n_supports, 1]
                            let result = kernel_values.matmul(coef.clone()) - rho;
                            result
                        }
                    }
                }
                "SIGMOID" => {
                    let params = self.config.kernel_params.as_ref();
                    let gamma = params.and_then(|p| p.get(0)).copied().unwrap_or(1.0);
                    let coef0 = params.and_then(|p| p.get(1)).copied().unwrap_or(0.0);
                    
                    quote! {
                        {
                            let rho = #rho;
                            let gamma = #gamma;
                            let coef0 = #coef0;
                            let (coef, sv) = &self.#field_name;
                            
                            // Compute sigmoid kernel: K(x, sv) = tanh(gamma * x · sv + coef0)
                            let dot_products = #input.matmul(sv.clone().transpose());
                            let kernel_values = (dot_products * gamma + coef0).tanh();
                            
                            // coef is already [n_supports, 1]
                            let result = kernel_values.matmul(coef.clone()) - rho;
                            result
                        }
                    }
                }
                _ => quote! { #input.clone() }
            }
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
    use onnx_ir::svmregressor::{SVMRegressorConfig, SVMRegressorNodeBuilder};

    #[test]
    fn test_svm_regressor_linear() {
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
        let node = SVMRegressorNodeBuilder::new("svm1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert!(code.contains("svm1"));
    }

    #[test]
    fn test_svm_regressor_rbf() {
        let config = SVMRegressorConfig::new(
            Some(vec![1.0, -0.5]),
            Some(vec![0.1]),
            Some("RBF".to_string()),
            Some(2),
            None,
            None,
            Some(vec![0.0]),
            Some(vec![1.0, 2.0, 3.0, 4.0]),
        );
        let node = SVMRegressorNodeBuilder::new("svm1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert!(code.contains("gamma"));
        assert!(code.contains("exp()"));
    }

    #[test]
    fn test_svm_regressor_poly() {
        let config = SVMRegressorConfig::new(
            Some(vec![1.0, -0.5]),
            Some(vec![1.0, 0.0, 3.0]),
            Some("POLY".to_string()),
            Some(2),
            None,
            None,
            Some(vec![0.0]),
            Some(vec![1.0, 2.0, 3.0, 4.0]),
        );
        let node = SVMRegressorNodeBuilder::new("svm1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert!(code.contains("degree"));
        assert!(code.contains("powf_scalar"));
    }

    #[test]
    fn test_svm_regressor_with_logistic() {
        let config = SVMRegressorConfig::new(
            Some(vec![1.0]),
            None,
            Some("LINEAR".to_string()),
            Some(1),
            None,
            Some("LOGISTIC".to_string()),
            Some(vec![0.0]),
            Some(vec![1.0, 2.0]),
        );
        let node = SVMRegressorNodeBuilder::new("svm1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert!(code.contains("neg"));
        assert!(code.contains("recip"));
    }
}
