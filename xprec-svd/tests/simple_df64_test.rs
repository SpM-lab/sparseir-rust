use mdarray::Tensor;
use xprec_svd::precision::Df64Precision;

#[test]
fn test_df64_basic_operations() {
    // Test basic Df64 operations
    let a = Df64Precision::from_f64(1.0);
    let b = Df64Precision::from_f64(2.0);

    let sum = a + b;
    let product = a * b;

    assert!((sum.to_f64() - 3.0).abs() < 1e-15);
    assert!((product.to_f64() - 2.0).abs() < 1e-15);
}

#[test]
fn test_df64_precision_comparison() {
    // Test precision difference between f64 and Df64
    // Use a more significant precision test
    let val_f64 = 1.0 + f64::EPSILON;
    let val_tf = Df64Precision::from_f64(1.0) + Df64Precision::epsilon();

    // Both should be close to 1.0 + EPSILON
    let expected = 1.0_f64 + f64::EPSILON;
    let diff_f64 = (val_f64 - expected).abs();
    let diff_tf = (val_tf.to_f64() - expected).abs();

    println!("f64 difference: {}", diff_f64);
    println!("Df64 difference: {}", diff_tf);

    // Both should be very close to expected value
    assert!(diff_f64 < 100.0 * f64::EPSILON);
    assert!(diff_tf < 100.0 * f64::EPSILON);
}

#[test]
fn test_df64_matrix_conversion() {
    // Test matrix conversion
    let matrix_f64 = Tensor::from_fn((2, 2), |idx| [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]);

    // Convert to Df64
    let matrix_tf = Tensor::from_fn((2, 2), |idx| {
        Df64Precision::from_f64(matrix_f64[[idx[0], idx[1]]])
    });

    // Convert back to f64
    let matrix_back = Tensor::from_fn((2, 2), |idx| matrix_tf[[idx[0], idx[1]]].to_f64());

    // Should be identical
    for i in 0..2 {
        for j in 0..2 {
            assert!((matrix_f64[[i, j]] - matrix_back[[i, j]]).abs() < 1e-15);
        }
    }
}
