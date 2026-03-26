# TreeEnsembleRegressor

Domain: **ai.onnx.ml**

First introduced in opset **1**

All versions: 1, 3, 5

## Description

This operator is DEPRECATED. Please use TreeEnsemble instead which provides the same
    functionality.<br>
    Tree Ensemble regressor.  Returns the regressed values for each input in N.<br>
    All args with nodes_ are fields of a tuple of tree nodes, and
    it is assumed they are the same length, and an index i will decode the
    tuple across these inputs.  Each node id can appear only once
    for each tree id.<br>
    All fields prefixed with target_ are tuples of votes at the leaves.<br>
    A leaf may have multiple votes, where each vote is weighted by
    the associated target_weights index.<br>
    All fields ending with <i>_as_tensor</i> can be used instead of the
    same parameter without the suffix if the element type is double and not float.
    All trees must have their node ids start at 0 and increment by 1.<br>
    Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF

## Attributes

- **aggregate_function** (STRING, optional): Defines how to aggregate leaf values within a target. <br>One of 'AVERAGE,' 'SUM,' 'MIN,' 'MAX.'
- **base_values** (FLOATS, optional): Base values for regression, added to final prediction after applying aggregate_function; the size must be the same as the classes or can be left unassigned (assumed 0)
- **base_values_as_tensor** (TENSOR, optional): Base values for regression, added to final prediction after applying aggregate_function; the size must be the same as the classes or can be left unassigned (assumed 0)
- **n_targets** (INT, optional): The total number of targets.
- **nodes_falsenodeids** (INTS, optional): Child node if expression is false
- **nodes_featureids** (INTS, optional): Feature id for each node.
- **nodes_hitrates** (FLOATS, optional): Popularity of each node, used for performance and may be omitted.
- **nodes_hitrates_as_tensor** (TENSOR, optional): Popularity of each node, used for performance and may be omitted.
- **nodes_missing_value_tracks_true** (INTS, optional): For each node, define what to do in the presence of a NaN: use the 'true' (if the attribute value is 1) or 'false' (if the attribute value is 0) branch based on the value in this array.<br>This attribute may be left undefined and the default value is false (0) for all nodes.
- **nodes_modes** (STRINGS, optional): The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf node.<br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'
- **nodes_nodeids** (INTS, optional): Node id for each node. Node ids must restart at zero for each tree and increase sequentially.
- **nodes_treeids** (INTS, optional): Tree id for each node.
- **nodes_truenodeids** (INTS, optional): Child node if expression is true
- **nodes_values** (FLOATS, optional): Thresholds to do the splitting on for each node.
- **nodes_values_as_tensor** (TENSOR, optional): Thresholds to do the splitting on for each node.
- **post_transform** (STRING, optional): Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'
- **target_ids** (INTS, optional): The index of the target that each weight is for
- **target_nodeids** (INTS, optional): The node id of each weight
- **target_treeids** (INTS, optional): The id of the tree that each node is in.
- **target_weights** (FLOATS, optional): The weight for each target
- **target_weights_as_tensor** (TENSOR, optional): The weight for each target

## Inputs (1 - 1)

- **X** (T): Input of shape [N,F]

## Outputs (1 - 1)

- **Y** (tensor(float)): N classes

## Type Constraints

- **T**: tensor(double), tensor(float), tensor(int32), tensor(int64)
  The input type must be a tensor of a numeric type.

## Version History

- **Opset 5**: Types: tensor(double), tensor(float), tensor(int32), tensor(int64)
- **Opset 3**: Types: tensor(double), tensor(float), tensor(int32), tensor(int64)
- **Opset 1**: Types: tensor(double), tensor(float), tensor(int32), tensor(int64)
