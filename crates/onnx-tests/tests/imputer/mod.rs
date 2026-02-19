use crate::utils::*;

#[test]
fn imputer_nan_replacement() {
    let device = Default::default();
    let model: BurnModel<Backend> = BurnModel::from_file("imputer.onnx", &device);

    // Load test data
    let input = load_tensor_from_npy::<2>("input.npy", &device);
    let expected = load_tensor_from_npy::<2>("output.npy", &device);

    let output = model.forward(input);
    assert_approx_equal_tensor(expected, output, 3);
}
