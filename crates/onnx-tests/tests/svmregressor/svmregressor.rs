include_model!("svmregressor");

#[test]
fn svmregressor_linear() {
    let device = Default::default();
    let model: svmregressor::Model<Backend> = svmregressor::Model::new(&device);

    // Input: [3, 2]
    let input = TestTensor::<3>::from_floats(burn_ndarray::from_slice(&INPUT), &device)
        .reshape([3, 2]);

    let output = model.forward(input);
    let expected =
        TestTensor::<3>::from_floats(burn_ndarray::from_slice(&OUTPUT), &device).reshape([3]);

    output.to_data().assert_approx_eq(&expected.to_data(), 3);
}
