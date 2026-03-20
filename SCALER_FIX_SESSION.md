# Scaler Bug Fix Session - March 19, 2026

## Bug Identified

The Scaler operator implementation in [crates/burn-onnx/src/burn/node/scaler.rs](crates/burn-onnx/src/burn/node/scaler.rs) had a critical bug where it only used the **first value** from the scale and offset arrays, applying the same scaling to all features in the tensor.

### Symptom
```
DEBUG: First preprocessed sample (first 10 features): 
[-1.140047, -1.1400471, -1.1400471, -1.1400471, -1.1400471, -1.1400471, -1.1400471, -1.1400471, -1.1400471, -1.1400471]
```

All values were identical, indicating uniform scaling instead of per-feature scaling.

## Root Cause

Lines 17-18 of the original implementation:
```rust
let scale = self.config.scale.as_ref().and_then(|s| s.first()).copied();
let offset = self.config.offset.as_ref().and_then(|o| o.first()).copied();
```

This only extracted the first element, then applied it as a scalar to the entire tensor.

## Expected Behavior

The ONNX Scaler operator (from `ai.onnx.ml` domain) applies **per-feature scaling**:
- Formula: `Y = (X - offset) * scale`
- Each feature (column) should get its own scale and offset value
- Arrays broadcast along the feature dimension

### Verification Test
Created `crates/onnx-tests/tests/scaler/test_per_feature.py`:
```python
Input:
[[1. 2. 3.]
 [4. 5. 6.]]

With scale=[1.0, 2.0, 3.0], offset=[0.5, 1.0, 1.5]:

Output:
[[ 0.5  2.   4.5]
 [ 3.5  8.  13.5]]

# Feature 0: (X - 0.5) * 1.0
# Feature 1: (X - 1.0) * 2.0  
# Feature 2: (X - 1.5) * 3.0
```

## Fix Implemented

Replaced scalar operations with tensor-based per-feature scaling:

### For Tensor inputs:
- Create 1D tensors from scale/offset arrays
- Use Burn's broadcasting to apply per-feature scaling
- Generated code example:
```rust
let offset_data = vec![0.5f32, 1f32, 1.5f32];
let scale_data = vec![1f32, 2f32, 3f32];
let offset_tensor = Tensor::<B, 1>::from_floats(offset_data.as_slice(), &*B::Device::default());
let scale_tensor = Tensor::<B, 1>::from_floats(scale_data.as_slice(), &*B::Device::default());
(input.clone() - offset_tensor) * scale_tensor
```

### For Scalar inputs:
- Still uses first element only (correct for scalar case)
- No change needed

## Current State

✅ **Implementation completed** in [crates/burn-onnx/src/burn/node/scaler.rs](crates/burn-onnx/src/burn/node/scaler.rs)
✅ **All unit tests passing** (5/5 tests)
✅ **Code generation verified** - produces correct per-feature scaling code
✅ **ONNX validation** - matches reference implementation output

### Final Implementation

The fix uses **tensor reshaping + broadcasting** instead of element-wise mapping:

```rust
// Per-feature scaling with 3 features:
let offset_tensor = Tensor::<B, 1>::from_floats([0.5f32, 1f32, 1.5f32], &*self.device)
    .reshape([1usize, 3usize]);  // Shape: [1, 3]
let scale_tensor = Tensor::<B, 1>::from_floats([1f32, 2f32, 3f32], &*self.device)
    .reshape([1usize, 3usize]);   // Shape: [1, 3]
(input.clone() - offset_tensor) * scale_tensor  // Broadcasting handles per-feature ops
```

For a 2D input `[batch_size, num_features]`, the reshaped tensors `[1, num_features]` broadcast correctly across the batch dimension, applying each scale/offset value to its corresponding feature column.

### Verification

Created test with different scale/offset per feature:
- Input: `[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]`
- Scale: `[1.0, 2.0, 3.0]`, Offset: `[0.5, 1.0, 1.5]`
- Output: `[[0.5, 2.0, 4.5], [3.5, 8.0, 13.5]]` ✓

## Next Steps

✅ **COMPLETED** - All steps finished!

The bug has been completely fixed and verified:
1. ✅ Unit tests updated and passing
2. ✅ Code generation verified 
3. ✅ ONNX reference implementation validation passed
4. ✅ Per-feature scaling test created and validated

The implementation is ready for use. When applied to real models with per-feature StandardScaler preprocessing, each feature will now be correctly scaled with its own mean/std values instead of using the same values for all features.

## Files Modified

- [crates/burn-onnx/src/burn/node/scaler.rs](crates/burn-onnx/src/burn/node/scaler.rs) - Fixed implementation and updated tests

## Files Created (temporary)

- `crates/onnx-tests/tests/scaler/test_per_feature.py` - Verification script (can be deleted or kept for reference)

## References

- ONNX Scaler spec: `ai.onnx.ml` domain, opset 1
- Original issue: Per-feature scaling not working, all features got same value
- Solution: Use 1D tensor broadcasting instead of scalar operations
