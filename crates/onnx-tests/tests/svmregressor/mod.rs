// Import the shared macro
use crate::include_models;
include_models!(svmregressor);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn svmregressor_linear() {
        let device = Default::default();
        let model: svmregressor::Model<TestBackend> = svmregressor::Model::new(&device);

        // Input: [3, 2]
        let input = Tensor::<TestBackend, 2>::from_floats(
            burn_ndarray::from_slice(&svmregressor::INPUT),
            &device,
        )
        .reshape([3, 2]);

        let output = model.forward(input);
        let expected = Tensor::<TestBackend, 1>::from_floats(
            burn_ndarray::from_slice(&svmregressor::OUTPUT),
            &device,
        )
        .reshape([3]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::default());
    }
}
