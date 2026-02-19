include_model!("tree_ensemble_classifier");

#[test]
fn tree_ensemble_classifier_basic() {
    let device = Default::default();
    let model: tree_ensemble_classifier::Model<Backend> =
        tree_ensemble_classifier::Model::new(&device);

    // Input: [3, 2]
    let input = TestTensor::<3>::from_floats(burn_ndarray::from_slice(&INPUT), &device)
        .reshape([3, 2]);

    let (labels, probs) = model.forward(input);

    let expected_labels =
        TestTensor::<3, burn::tensor::Int>::from_ints(burn_ndarray::from_slice(&OUTPUT_LABELS), &device)
            .reshape([3]);
    let expected_probs =
        TestTensor::<3>::from_floats(burn_ndarray::from_slice(&OUTPUT_PROBS), &device)
            .reshape([3, 2]);

    labels.to_data().assert_eq(&expected_labels.to_data(), true);
    probs.to_data().assert_approx_eq(&expected_probs.to_data(), 3);
}
