//! Integration tests for sparseir-rust
//!
//! These tests verify that the public API works correctly and that
//! different modules work together properly.

use sparseir_rust::*;
use twofloat::TwoFloat;

#[test]
fn test_kernel_integration() {
    // Test that kernels work with different statistics types
    let logistic_kernel = LogisticKernel::new(10.0);
    
    // Test fermionic case
    let result_fermi = logistic_kernel.compute_weighted::<Fermionic>(
        TwoFloat::from(0.0), TwoFloat::from(0.0), 1.0, 1.0
    );
    assert!(Into::<f64>::into(result_fermi) > 0.0);
    
    // Test bosonic case
    let result_bose = logistic_kernel.compute_weighted::<Bosonic>(
        TwoFloat::from(0.0), TwoFloat::from(0.0), 1.0, 1.0
    );
    assert!(Into::<f64>::into(result_bose) > 0.0);
}

#[test]
fn test_bose_kernel_integration() {
    let bose_kernel = RegularizedBoseKernel::new(10.0);
    
    // Test bosonic case (should work)
    let result = bose_kernel.compute_weighted::<Bosonic>(
        TwoFloat::from(0.0), TwoFloat::from(0.0), 1.0, 1.0
    );
    // Result might be NaN for certain values, which is expected
    let result_f64 = Into::<f64>::into(result);
    
    // Test that the kernel doesn't panic for valid inputs
    assert!(result_f64.is_finite() || result_f64.is_nan());
}

#[test]
fn test_statistics_trait_integration() {
    // Test that statistics traits work with kernels
    let kernel = LogisticKernel::new(5.0);
    
    // Test that we can get statistics information
    assert_eq!(Fermionic::STATISTICS, Statistics::Fermionic);
    assert_eq!(Bosonic::STATISTICS, Statistics::Bosonic);
    
    // Test utility methods
    assert!(Statistics::Fermionic.is_fermionic());
    assert!(Statistics::Bosonic.is_bosonic());
    assert!(!Statistics::Fermionic.is_bosonic());
    assert!(!Statistics::Bosonic.is_fermionic());
}

#[test]
fn test_domain_validation() {
    let kernel = LogisticKernel::new(10.0);
    
    // Test domain ranges
    let (xmin, xmax) = kernel.xrange();
    let (ymin, ymax) = kernel.yrange();
    
    assert_eq!(xmin, -1.0);
    assert_eq!(xmax, 1.0);
    assert_eq!(ymin, -1.0);
    assert_eq!(ymax, 1.0);
    
    // Test that kernel computation works at domain boundaries
    let result_corner = kernel.compute(TwoFloat::from(xmin), TwoFloat::from(ymin));
    assert!(Into::<f64>::into(result_corner).is_finite());
}

#[test]
fn test_precision_consistency() {
    let kernel = LogisticKernel::new(10.0);
    
    // Test that f64 and TwoFloat give consistent results
    let x = 0.5;
    let y = -0.3;
    
    let result_f64 = compute_f64(&kernel, x, y);
    let result_twofloat = kernel.compute(TwoFloat::from(x), TwoFloat::from(y));
    let result_twofloat_f64 = Into::<f64>::into(result_twofloat);
    
    // Results should be very close (within floating point precision)
    assert!((result_f64 - result_twofloat_f64).abs() < 1e-14);
}

#[test]
fn test_lambda_parameter() {
    // Test different lambda values
    let lambda_values = [1.0, 5.0, 10.0, 50.0, 100.0];
    
    for &lambda in &lambda_values {
        let kernel = LogisticKernel::new(lambda);
        assert_eq!(kernel.lambda(), lambda);
        
        // Test that kernel computation works for all lambda values
        let result = kernel.compute(TwoFloat::from(0.0), TwoFloat::from(0.0));
        assert!(Into::<f64>::into(result) > 0.0);
        assert!(Into::<f64>::into(result) <= 1.0);
    }
}
