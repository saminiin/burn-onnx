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
            burn_ndarray::from_slice(&tree_ensemble_classifier::INPUT),
            &device,
        )
        .reshape([3, 2]);

        let (labels, probs) = model.forward(input);

        let expected_labels = Tensor::<TestBackend, 1, burn::tensor::Int>::from_ints(
            burn_ndarray::from_slice(&tree_ensemble_classifier::OUTPUT_LABELS),
            &device,
        )
        .reshape([3]);
        let expected_probs = Tensor::<TestBackend, 2>::from_floats(
            burn_ndarray::from_slice(&tree_ensemble_classifier::OUTPUT_PROBS),
            &device,
        )
        .reshape([3, 2]);

        labels.to_data().assert_eq(&expected_labels.to_data(), true);
        probs
            .to_data()
            .assert_approx_eq::<FT>(&expected_probs.to_data(), Tolerance::default());
    }
}
