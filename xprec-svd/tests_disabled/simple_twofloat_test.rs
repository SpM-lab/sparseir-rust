use xprec_svd::precision::TwoFloatPrecision;
use ndarray::array;

#[test]
fn test_twofloat_basic_operations() {
    // Test basic TwoFloat operations
    let a = TwoFloatPrecision::from_f64(1.0);
    let b = TwoFloatPrecision::from_f64(2.0);
    
    let sum = a + b;
    let product = a * b;
    
    assert!((sum.to_f64() - 3.0).abs() < 1e-15);
    assert!((product.to_f64() - 2.0).abs() < 1e-15);
}

#[test]
fn test_twofloat_precision_comparison() {
    // Test precision difference between f64 and TwoFloat
    // Use a more significant precision test
    let val_f64 = 1.0 + f64::EPSILON;
    let val_tf = TwoFloatPrecision::from_f64(1.0) + TwoFloatPrecision::epsilon();
    
    // Both should be close to 1.0 + EPSILON
    let expected = 1.0_f64 + f64::EPSILON;
    let diff_f64 = (val_f64 - expected).abs();
    let diff_tf = (val_tf.to_f64() - expected).abs();
    
    println!("f64 difference: {}", diff_f64);
    println!("TwoFloat difference: {}", diff_tf);
    
    // Both should be very close to expected value
    assert!(diff_f64 < 100.0 * f64::EPSILON);
    assert!(diff_tf < 100.0 * f64::EPSILON);
}

#[test]
fn test_twofloat_matrix_conversion() {
    // Test matrix conversion
    let matrix_f64 = array![
        [1.0, 2.0],
        [3.0, 4.0]
    ];
    
    // Convert to TwoFloat
    let mut matrix_tf = ndarray::Array2::zeros((2, 2));
    for i in 0..2 {
        for j in 0..2 {
            matrix_tf[[i, j]] = TwoFloatPrecision::from_f64(matrix_f64[[i, j]]);
        }
    }
    
    // Convert back to f64
    let mut matrix_back = ndarray::Array2::zeros((2, 2));
    for i in 0..2 {
        for j in 0..2 {
            matrix_back[[i, j]] = matrix_tf[[i, j]].to_f64();
        }
    }
    
    // Should be identical
    for i in 0..2 {
        for j in 0..2 {
            assert!((matrix_f64[[i, j]] - matrix_back[[i, j]]).abs() < 1e-15);
        }
    }
}
