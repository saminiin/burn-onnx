# ZipMap

Domain: **ai.onnx.ml**

First introduced in opset **1**

## Description

Creates a map from the input and the attributes.<br>
    The values are provided by the input tensor, while the keys are specified by the attributes.
    Must provide keys in either classlabels_strings or classlabels_int64s (but not both).<br>
    The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.<br>

## Attributes

- **classlabels_int64s** (INTS, optional): The keys when using int keys.<br>One and only one of the 'classlabels_*' attributes must be defined.
- **classlabels_strings** (STRINGS, optional): The keys when using string keys.<br>One and only one of the 'classlabels_*' attributes must be defined.

## Inputs (1 - 1)

- **X** (tensor(float)): The input values

## Outputs (1 - 1)

- **Z** (T): The output map

## Type Constraints

- **T**: seq(map(int64, float)), seq(map(string, float))
  The output will be a sequence of string or integer maps to float.
