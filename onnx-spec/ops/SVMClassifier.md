# SVMClassifier

Domain: **ai.onnx.ml**

First introduced in opset **1**

## Description

Support Vector Machine classifier

## Attributes

- **classlabels_ints** (INTS, optional): Class labels if using integer labels.<br>One and only one of the 'classlabels_*' attributes must be defined.
- **classlabels_strings** (STRINGS, optional): Class labels if using string labels.<br>One and only one of the 'classlabels_*' attributes must be defined.
- **coefficients** (FLOATS, optional)
- **kernel_params** (FLOATS, optional): List of 3 elements containing gamma, coef0, and degree, in that order. Zero if unused for the kernel.
- **kernel_type** (STRING, optional): The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'.
- **post_transform** (STRING, optional): Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'
- **prob_a** (FLOATS, optional): First set of probability coefficients.
- **prob_b** (FLOATS, optional): Second set of probability coefficients. This array must be same size as prob_a.<br>If these are provided then output Z are probability estimates, otherwise they are raw scores.
- **rho** (FLOATS, optional)
- **support_vectors** (FLOATS, optional)
- **vectors_per_class** (INTS, optional)

## Inputs (1 - 1)

- **X** (T1): Data to be classified.

## Outputs (2 - 2)

- **Y** (T2): Classification outputs (one class per example).
- **Z** (tensor(float)): Class scores (one per class per example), if prob_a and prob_b are provided they are probabilities for each class, otherwise they are raw scores.

## Type Constraints

- **T1**: tensor(double), tensor(float), tensor(int32), tensor(int64)
  The input must be a tensor of a numeric type, either [C] or [N,C].
- **T2**: tensor(int64), tensor(string)
  The output type will be a tensor of strings or integers, depending on which of the classlabels_* attributes is used. Its size will match the batch size of the input.
