# SVMRegressor Sign Flip Fix - March 20, 2026

## Bug Identified

The SVMRegressor operator implementation had an incorrect sign for the bias term (`rho`). The code was **subtracting** rho when it should have been **adding** it.

### Root Cause

In all kernel implementations (LINEAR, RBF, POLY, SIGMOID), the formula was:
```rust
let result = kernel_values.matmul(coef.clone()) - rho;  // ❌ WRONG
```

This is inconsistent with the ONNX specification and reference implementations, which use:
```
prediction = kernel_values @ coefficients + rho  // ✅ CORRECT
```

### Why This Matters

- The sign error shifts all predictions by `2 * rho`
- For `one_class` SVM anomaly detection, this can flip the decision boundary
- The bug was not caught by integration tests because the test generator used the same incorrect formula

## Fix Applied

### Code Changes

Updated [crates/burn-onnx/src/burn/node/svmregressor.rs](crates/burn-onnx/src/burn/node/svmregressor.rs):
- Changed all occurrences from `- rho` to `+ rho` (lines 78, 111, 135, 157)
- Updated comment from `prediction = kernel_values @ coefficients - rho` to `prediction = kernel_values @ coefficients + rho`

### Test Data Regeneration

Updated [crates/onnx-tests/tests/svmregressor/svmregressor.py](crates/onnx-tests/tests/svmregressor/svmregressor.py):
- Fixed expected output calculation: `output_data = (kernel_values @ coefficients + rho[0])`
- Regenerated `input.bin` and `output.bin` test data files

### Verification

✅ **All unit tests passing** (4/4 tests in burn-onnx)
- `test_svm_regressor_linear`
- `test_svm_regressor_rbf`
- `test_svm_regressor_poly`
- `test_svm_regressor_with_logistic`

✅ **Generated code now correct** - all kernel branches use `+ rho`

✅ **Test data regenerated** with correct formula

## References

- ONNX SVMRegressor spec: https://onnx.ai/onnx/operators/onnx_aionnxml_SVMRegressor.html
- ONNXRuntime implementation uses `+ rho` in `svmregressor.cc`
- Similar to Scaler fix from March 19, 2026 (per-feature scaling bug)

## Files Modified

- [crates/burn-onnx/src/burn/node/svmregressor.rs](crates/burn-onnx/src/burn/node/svmregressor.rs) - Fixed bias sign
- [crates/onnx-tests/tests/svmregressor/svmregressor.py](crates/onnx-tests/tests/svmregressor/svmregressor.py) - Fixed test generator
- [crates/onnx-tests/tests/svmregressor/input.bin](crates/onnx-tests/tests/svmregressor/input.bin) - Regenerated
- [crates/onnx-tests/tests/svmregressor/output.bin](crates/onnx-tests/tests/svmregressor/output.bin) - Regenerated
- [crates/onnx-tests/tests/svmregressor/mod.rs](crates/onnx-tests/tests/svmregressor/mod.rs) - Fixed test module structure

## Impact

This fix ensures SVMRegressor predictions match the ONNX specification and other ONNX runtime implementations. Models using SVM regression (especially one-class SVM for anomaly detection) will now produce correct outputs.
