# OneHotEncoder

Domain: **ai.onnx.ml**

First introduced in opset **1**

## Description

Replace each input element with an array of ones and zeros, where a single one is placed at the index of the category that was passed in. The total category count determines the size of the extra dimension of the output array Y. This operator assumes every input feature is from the same set of categories.

If the input is a tensor of float, int32, or double, the data is cast to integers and the `cats_int64s` category list is used for the lookups.

## Attributes

- **cats_int64s** (INTS, optional): List of integer categories. One and only one of the `cats_*` attributes must be defined.
- **cats_strings** (STRINGS, optional): List of string categories. One and only one of the `cats_*` attributes must be defined.
- **zeros** (INT, optional, default is `1`): If true and category is not present, returns all zeros; if false and a category is not found, the operator fails.

## Inputs (1 - 1)

- **X** (T): Data to be encoded.

## Outputs (1 - 1)

- **Y** (tensor(float)): Encoded output data, having one more dimension than X.

## Type Constraints

- **T**: tensor(double), tensor(float), tensor(int32), tensor(int64), tensor(string)
  The input must be a tensor of a supported type.
