// Import the shared macro
use crate::include_models;
include_models!(svmregressor);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn svmregressor_linear() {
        let device = Default::default();
        let model: svmregressor::Model<TestBackend> = svmregressor::Model::new(&device);

        // Input: [3, 2]
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.49671414f32, -0.13826430],
                [0.64768857,  1.52302980],
                [-0.23415338, -0.23413695],
            ],
            &device,
        );

        let output = model.forward(input);
        let expected = Tensor::<TestBackend, 1>::from_floats(
            [0.25164291f32, 0.17615581, 0.61707670],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::default());
    }
}
