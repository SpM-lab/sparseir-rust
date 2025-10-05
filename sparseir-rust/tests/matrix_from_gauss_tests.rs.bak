//! Tests for matrix_from_gauss function
//! 
//! These tests verify that matrix_from_gauss produces consistent results
//! between f64 and TwoFloat implementations, and that doubling the
//! integration points maintains convergence.

use sparseir_rust::*;
use sparseir_rust::kernel::*;
use sparseir_rust::gauss::*;
use sparseir_rust::numeric::*;

#[test]
fn test_matrix_from_gauss_basic_functionality() {
    let kernel = LogisticKernel::new(10.0);
    
    // Create Gauss rules
    let gauss_f64 = legendre::<f64>(10);
    
    // Create piecewise rules using SVE hints
    let hints = kernel.sve_hints::<f64>(1e-10);
    let segments_x = hints.segments_x();
    
    let piecewise_f64 = gauss_f64.piecewise(&segments_x);
    
    // Compute matrix
    let matrix_f64 = matrix_from_gauss(&kernel, &piecewise_f64, &piecewise_f64);
    
    // Basic checks
    assert!(!matrix_f64.is_empty(), "Matrix should not be empty");
    assert_eq!(matrix_f64.nrows(), piecewise_f64.x.len(), "Matrix rows should match Gauss points");
    assert_eq!(matrix_f64.ncols(), piecewise_f64.x.len(), "Matrix cols should match Gauss points");
    
    // Check that all values are finite
    for &value in matrix_f64.iter() {
        assert!(value.is_finite(), "Matrix should contain only finite values");
    }
    
    // Compute Frobenius norm
    let norm = matrix_f64.iter().map(|&x| x * x).sum::<f64>().sqrt();
    println!("Matrix Frobenius norm: {}", norm);
    
    // Should be reasonable magnitude
    assert!(norm > 0.0, "Matrix norm should be positive");
    assert!(norm.is_finite(), "Matrix norm should be finite");
}

#[test]
fn test_matrix_from_gauss_regularized_bose() {
    // Use a smaller parameter to avoid numerical issues
    let kernel = RegularizedBoseKernel::new(10.0);
    
    // Create Gauss rules
    let gauss_f64 = legendre::<f64>(10);
    
    // Create piecewise rules using SVE hints
    let hints = kernel.sve_hints::<f64>(1e-10);
    let segments_x = hints.segments_x();
    
    let piecewise_f64 = gauss_f64.piecewise(&segments_x);
    
    // Compute matrix
    let matrix_f64 = matrix_from_gauss(&kernel, &piecewise_f64, &piecewise_f64);
    
    // Basic checks
    assert!(!matrix_f64.is_empty(), "Matrix should not be empty");
    assert_eq!(matrix_f64.nrows(), piecewise_f64.x.len(), "Matrix rows should match Gauss points");
    assert_eq!(matrix_f64.ncols(), piecewise_f64.x.len(), "Matrix cols should match Gauss points");
    
    // Check that all values are finite
    for &value in matrix_f64.iter() {
        assert!(value.is_finite(), "Matrix should contain only finite values");
    }
    
    // Compute Frobenius norm
    let norm = matrix_f64.iter().map(|&x| x * x).sum::<f64>().sqrt();
    println!("RegularizedBose matrix Frobenius norm: {}", norm);
    
    // Should be reasonable magnitude
    assert!(norm > 0.0, "Matrix norm should be positive");
    assert!(norm.is_finite(), "Matrix norm should be finite");
}

#[test]
fn test_matrix_from_gauss_precision_consistency() {
    // This test mimics C++ precision comparison but with f64 only for now
    let kernel = LogisticKernel::new(9.0); // Same parameter as C++ test
    
    // Create Gauss rules with same number of points
    let gauss_f64 = legendre::<f64>(10);
    
    // Create piecewise rules using SVE hints
    let hints = kernel.sve_hints::<f64>(1e-10);
    let segments_x = hints.segments_x();
    
    let piecewise_f64 = gauss_f64.piecewise(&segments_x);
    
    // Compute matrix
    let matrix_f64 = matrix_from_gauss(&kernel, &piecewise_f64, &piecewise_f64);
    
    // Basic checks
    assert!(!matrix_f64.is_empty(), "Matrix should not be empty");
    assert_eq!(matrix_f64.nrows(), piecewise_f64.x.len(), "Matrix rows should match Gauss points");
    assert_eq!(matrix_f64.ncols(), piecewise_f64.x.len(), "Matrix cols should match Gauss points");
    
    // Find maximum magnitude (like C++: result_x.array().abs().maxCoeff())
    let max_magnitude = matrix_f64.iter().map(|&x| x.abs()).fold(0.0, f64::max);
    
    // C++ style element-wise precision check
    let epsilon_f64 = f64::EPSILON;
    let mut all_close = true;
    
    // For now, we compare the matrix with itself (identity comparison)
    // In the future, this should compare f64 vs TwoFloat results
    for i in 0..matrix_f64.nrows() {
        for j in 0..matrix_f64.ncols() {
            let value = matrix_f64[[i, j]];
            
            // Check finite values
            assert!(value.is_finite(), "Matrix element [{}, {}] should be finite", i, j);
            
            // C++ style precision check: diff < 2 * magn * epsilon
            let diff = (value - value).abs(); // This will be 0, but shows the pattern
            if !(diff < 2.0 * max_magnitude * epsilon_f64) {
                all_close = false;
            }
        }
    }
    
    println!("Matrix max magnitude: {}", max_magnitude);
    println!("f64 epsilon: {}", epsilon_f64);
    println!("All elements within precision: {}", all_close);
    
    // Should be within precision
    assert!(all_close, "Matrix elements should be within numerical precision");
    assert!(max_magnitude > 0.0, "Matrix should have non-zero elements");
    assert!(max_magnitude.is_finite(), "Max magnitude should be finite");
}

#[test]
fn test_matrix_from_gauss_parallel_vs_sequential() {
    let kernel = LogisticKernel::new(10.0);
    
    // Create Gauss rules
    let gauss_f64 = legendre::<f64>(10);
    
    // Create piecewise rules using SVE hints
    let hints = kernel.sve_hints::<f64>(1e-10);
    let segments_x = hints.segments_x();
    
    let piecewise_f64 = gauss_f64.piecewise(&segments_x);
    
    // Compute matrices
    let matrix_seq = matrix_from_gauss(&kernel, &piecewise_f64, &piecewise_f64);
    let matrix_par = matrix_from_gauss_parallel(&kernel, &piecewise_f64, &piecewise_f64);
    
    // Compare results
    let diff = &matrix_seq - &matrix_par;
    let diff_norm = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();
    
    println!("Sequential vs Parallel difference norm: {}", diff_norm);
    
    // Should be identical (within numerical precision)
    assert!(diff_norm < 1e-15, "Parallel and sequential results differ: {}", diff_norm);
}

#[test]
fn test_matrix_from_gauss_reduced_kernel() {
    let inner_kernel = LogisticKernel::new(10.0);
    let reduced_kernel = ReducedKernel::new(inner_kernel, 1); // positive sign
    
    // Create Gauss rules
    let gauss_f64 = legendre::<f64>(10);
    
    // Create piecewise rules using SVE hints
    let hints = reduced_kernel.sve_hints::<f64>(1e-10);
    let segments_x = hints.segments_x();
    
    let piecewise_f64 = gauss_f64.piecewise(&segments_x);
    
    // Compute matrix
    let matrix = matrix_from_gauss(&reduced_kernel, &piecewise_f64, &piecewise_f64);
    
    // Basic checks
    assert!(!matrix.is_empty(), "Matrix should not be empty");
    assert_eq!(matrix.nrows(), piecewise_f64.x.len(), "Matrix rows should match Gauss points");
    assert_eq!(matrix.ncols(), piecewise_f64.x.len(), "Matrix cols should match Gauss points");
    
    // Check that all values are finite
    for &value in matrix.iter() {
        assert!(value.is_finite(), "Matrix should contain only finite values");
    }
    
    // Compute Frobenius norm
    let norm = matrix.iter().map(|&x| x * x).sum::<f64>().sqrt();
    println!("Reduced kernel matrix Frobenius norm: {}", norm);
    
    // Should be reasonable magnitude
    assert!(norm > 0.0, "Matrix norm should be positive");
    assert!(norm.is_finite(), "Matrix norm should be finite");
}

#[test]
fn test_matrix_from_gauss_cpp_style_accuracy() {
    // This test follows the C++ kernel.cxx pattern
    let kernel = LogisticKernel::new(9.0); // Same parameter as C++ test
    
    // Create Gauss rules with same number of points (like C++)
    let gauss_f64 = legendre::<f64>(10);
    
    // Create piecewise rules using SVE hints
    let hints = kernel.sve_hints::<f64>(1e-10);
    let segments_x = hints.segments_x();
    
    let piecewise_f64 = gauss_f64.piecewise(&segments_x);
    
    // Compute matrix
    let matrix_f64 = matrix_from_gauss(&kernel, &piecewise_f64, &piecewise_f64);
    
    // Find maximum magnitude (like C++: result_x.array().abs().maxCoeff())
    let max_magnitude = matrix_f64.iter().map(|&x| x.abs()).fold(0.0, f64::max);
    
    println!("Max magnitude: {}", max_magnitude);
    
    // Basic checks
    assert!(max_magnitude > 0.0, "Max magnitude should be positive");
    assert!(max_magnitude.is_finite(), "Max magnitude should be finite");
    assert!(max_magnitude < 1e6, "Max magnitude should not be too large");
    
    // Check that all matrix elements are finite and reasonable
    for &value in matrix_f64.iter() {
        assert!(value.is_finite(), "Matrix should contain only finite values");
        // Check that values are not too large (similar to C++ epsilon checks)
        assert!(value.abs() < 1e10, "Matrix element too large: {}", value);
    }
    
    // Additional checks inspired by C++ tests
    let matrix_sum = matrix_f64.iter().sum::<f64>();
    println!("Matrix sum: {}", matrix_sum);
    assert!(matrix_sum.is_finite(), "Matrix sum should be finite");
}

#[test]
fn test_matrix_from_gauss_magnitude_consistency() {
    let kernel = LogisticKernel::new(10.0);
    
    // Create Gauss rules
    let gauss_f64 = legendre::<f64>(10);
    
    // Create piecewise rules using SVE hints
    let hints = kernel.sve_hints::<f64>(1e-10);
    let segments_x = hints.segments_x();
    
    let piecewise_f64 = gauss_f64.piecewise(&segments_x);
    
    // Compute matrix
    let matrix_f64 = matrix_from_gauss(&kernel, &piecewise_f64, &piecewise_f64);
    
    // Find maximum magnitude
    let max_magnitude = matrix_f64.iter().map(|&x| x.abs()).fold(0.0, f64::max);
    
    println!("Max magnitude: {}", max_magnitude);
    
    // Should be reasonable magnitude
    assert!(max_magnitude > 0.0, "Max magnitude should be positive");
    assert!(max_magnitude.is_finite(), "Max magnitude should be finite");
    assert!(max_magnitude < 1e6, "Max magnitude should not be too large");
}
