use crate::include_models;
include_models!(one_hot_encoder, one_hot_encoder_2d, one_hot_encoder_zeros0);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Device, Int, Tensor, TensorData, Tolerance};

    #[test]
    fn one_hot_encoder_1d() {
        let device = Default::default();
        let model = one_hot_encoder::Model::new(&device);

        let input: Tensor<1, Int> =
            Tensor::from_data(TensorData::from([1i64, 0, 2, 3]), (&device, DType::I64));
        let output: Tensor<2> = model.forward(input);

        let expected = TensorData::from([
            [0.0f32, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn one_hot_encoder_2d() {
        let device = Default::default();
        let model = one_hot_encoder_2d::Model::new(&device);

        let input: Tensor<2, Int> = Tensor::from_data(
            TensorData::from([[0i64, 1, 2], [2, 0, 1]]),
            (&device, DType::I64),
        );
        let output: Tensor<3> = model.forward(input);

        let expected = TensorData::from([
            [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn one_hot_encoder_zeros0_non_default_attr() {
        let device = Default::default();
        let model = one_hot_encoder_zeros0::Model::new(&device);

        let input: Tensor<1, Int> =
            Tensor::from_data(TensorData::from([2i64, 1, 0]), (&device, DType::I64));
        let output: Tensor<2> = model.forward(input);

        let expected = TensorData::from([
            [0.0f32, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
