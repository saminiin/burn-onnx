# TreeEnsembleClassifier

Domain: **ai.onnx.ml**

First introduced in opset **1**

All versions: 1, 3, 5

## Description

This operator is DEPRECATED. Please use TreeEnsemble with provides similar functionality.
    In order to determine the top class, the ArgMax node can be applied to the output of TreeEnsemble.
    To encode class labels, use a LabelEncoder operator.
    Tree Ensemble classifier. Returns the top class for each of N inputs.<br>
    The attributes named 'nodes_X' form a sequence of tuples, associated by
    index into the sequences, which must all be of equal length. These tuples
    define the nodes.<br>
    Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
    A leaf may have multiple votes, where each vote is weighted by
    the associated class_weights index.<br>
    One and only one of classlabels_strings or classlabels_int64s
    will be defined. The class_ids are indices into this list.
    All fields ending with <i>_as_tensor</i> can be used instead of the
    same parameter without the suffix if the element type is double and not float.

## Attributes

- **base_values** (FLOATS, optional): Base values for classification, added to final class score; the size must be the same as the classes or can be left unassigned (assumed 0)
- **base_values_as_tensor** (TENSOR, optional): Base values for classification, added to final class score; the size must be the same as the classes or can be left unassigned (assumed 0)
- **class_ids** (INTS, optional): The index of the class list that each weight is for.
- **class_nodeids** (INTS, optional): node id that this weight is for.
- **class_treeids** (INTS, optional): The id of the tree that this node is in.
- **class_weights** (FLOATS, optional): The weight for the class in class_id.
- **class_weights_as_tensor** (TENSOR, optional): The weight for the class in class_id.
- **classlabels_int64s** (INTS, optional): Class labels if using integer labels.<br>One and only one of the 'classlabels_*' attributes must be defined.
- **classlabels_strings** (STRINGS, optional): Class labels if using string labels.<br>One and only one of the 'classlabels_*' attributes must be defined.
- **nodes_falsenodeids** (INTS, optional): Child node if expression is false.
- **nodes_featureids** (INTS, optional): Feature id for each node.
- **nodes_hitrates** (FLOATS, optional): Popularity of each node, used for performance and may be omitted.
- **nodes_hitrates_as_tensor** (TENSOR, optional): Popularity of each node, used for performance and may be omitted.
- **nodes_missing_value_tracks_true** (INTS, optional): For each node, define what to do in the presence of a missing value: if a value is missing (NaN), use the 'true' or 'false' branch based on the value in this array.<br>This attribute may be left undefined, and the default value is false (0) for all nodes.
- **nodes_modes** (STRINGS, optional): The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf node.<br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'
- **nodes_nodeids** (INTS, optional): Node id for each node. Ids may restart at zero for each tree, but it not required to.
- **nodes_treeids** (INTS, optional): Tree id for each node.
- **nodes_truenodeids** (INTS, optional): Child node if expression is true.
- **nodes_values** (FLOATS, optional): Thresholds to do the splitting on for each node.
- **nodes_values_as_tensor** (TENSOR, optional): Thresholds to do the splitting on for each node.
- **post_transform** (STRING, optional): Indicates the transform to apply to the score. <br> One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT.'

## Inputs (1 - 1)

- **X** (T1): Input of shape [N,F]

## Outputs (2 - 2)

- **Y** (T2): N, Top class for each point
- **Z** (tensor(float)): The class score for each class, for each point, a tensor of shape [N,E].

## Type Constraints

- **T1**: tensor(double), tensor(float), tensor(int32), tensor(int64)
  The input type must be a tensor of a numeric type.
- **T2**: tensor(int64), tensor(string)
  The output type will be a tensor of strings or integers, depending on which of the classlabels_* attributes is used.

## Version History

- **Opset 5**: Types: tensor(double), tensor(float), tensor(int32), tensor(int64)
- **Opset 3**: Types: tensor(double), tensor(float), tensor(int32), tensor(int64)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(int32), tensor(int64)
