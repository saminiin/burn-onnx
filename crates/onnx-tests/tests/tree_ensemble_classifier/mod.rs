use crate::include_models;
include_models!(tree_ensemble_classifier);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn tree_ensemble_classifier_basic() {
        let device = Default::default();
        let model: tree_ensemble_classifier::Model<TestBackend> =
            tree_ensemble_classifier::Model::new(&device);

        // Input: [3, 2]
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.49671414f32, -0.13826430],
                [0.64768857,  1.52302980],
                [-0.23415338, -0.23413695],
            ],
            &device,
        );

        let (labels, probs) = model.forward(input);

        let expected_labels = Tensor::<TestBackend, 1, burn::tensor::Int>::from_ints(
            [0i64, 1, 0],
            &device,
        );
        let expected_probs = Tensor::<TestBackend, 2>::from_floats(
            [[1.0f32, 0.0], [0.0, 1.0], [1.0, 0.0]],
            &device,
        );

        labels.to_data().assert_eq(&expected_labels.to_data(), true);
        probs
            .to_data()
            .assert_approx_eq::<FT>(&expected_probs.to_data(), Tolerance::default());
    }
}
