use crate::include_models;
include_models!(one_hot_encoder_f32, one_hot_encoder_f64);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Device, Tensor, TensorData};

    #[test]
    fn one_hot_encoder_f32_input() {
        let device = Default::default();
        let model = one_hot_encoder_f32::Model::new(&device);

        let input: Tensor<1> = Tensor::from_data(
            TensorData::from([1.0f32, 4.0, 2.0, 1.0]),
            (&device, DType::F32),
        );
        let output: Tensor<2> = model.forward(input);

        let expected = TensorData::from([
            [1.0f32, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn one_hot_encoder_f64_input() {
        let device = Default::default();
        let model = one_hot_encoder_f64::Model::new(&device);

        let input: Tensor<1> = Tensor::from_data(
            TensorData::from([4.0f64, 2.0, 3.0, 1.0]),
            (&device, DType::F64),
        );
        let output: Tensor<2> = model.forward(input);

        let expected = TensorData::from([
            [0.0f32, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]);
        output.to_data().assert_eq(&expected, true);
    }
}
