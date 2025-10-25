//! Hilbert matrix reconstruction tests for xprec-svd2
//!
//! Tests SVD reconstruction accuracy using Hilbert matrices, which are known
//! to be ill-conditioned and thus good test cases for SVD accuracy.

use nalgebra::DMatrix;
use xprec_svd2::{tsvd, TSVDConfig};
use num_traits::cast::ToPrimitive;
use xprec::Df64;

/// Create Hilbert matrix of size n x n with generic type
/// H[i,j] = 1 / (i + j + 1)
fn create_hilbert_matrix_generic<T>(n: usize) -> DMatrix<T>
where
    T: nalgebra::RealField + From<f64> + Copy + std::ops::Div<Output = T>,
{
    DMatrix::from_fn(n, n, |i, j| {
        // For high precision types like Df64, we need to do the division in type T
        // to preserve precision, not in f64
        T::one() / T::from((i + j + 1) as f64)
    })
}

/// Reconstruct matrix from SVD with generic type: A = U * S * V^T
fn reconstruct_matrix_generic<T>(
    u: &DMatrix<T>,
    s: &nalgebra::DVector<T>,
    v: &DMatrix<T>,
) -> DMatrix<T>
where
    T: nalgebra::RealField + Copy,
{
    // A = U * S * V^T
    // U: (m × k), S: (k), V: (n × k)
    // Result: (m × n)
    u * &DMatrix::from_diagonal(s) * &v.transpose()
}

/// Calculate Frobenius norm of matrix with generic type
fn frobenius_norm_generic<T>(matrix: &DMatrix<T>) -> f64
where
    T: nalgebra::RealField + Copy + ToPrimitive,
{
    let mut sum = 0.0;
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            let val = matrix[(i, j)].to_f64().unwrap_or(0.0);
            sum += val * val;
        }
    }
    sum.sqrt()
}

/// Generic Hilbert matrix reconstruction test
fn test_hilbert_reconstruction_generic<T>(n: usize, rtol: f64, expected_max_error: f64)
where
    T: nalgebra::RealField + From<f64> + Copy + ToPrimitive + std::fmt::Debug,
{
    let h = create_hilbert_matrix_generic::<T>(n);

    println!("Testing Hilbert {}x{} with type: {}, rtol: {:.2e}", 
             n, n, std::any::type_name::<T>(), rtol);
    println!("Original matrix norm: {:.6e}", frobenius_norm_generic(&h));

    // Compute TSVD with specified tolerance and measure execution time
    let config = TSVDConfig::new(T::from(rtol));
    let start = std::time::Instant::now();
    let result = tsvd(&h, config).unwrap();
    let duration = start.elapsed();
    
    println!("TSVD execution time: {:?}", duration);

    println!("Detected rank: {}", result.rank);
    println!("Singular values: {:?}", result.s);

    // Reconstruct matrix
    let h_reconstructed = reconstruct_matrix_generic(&result.u, &result.s, &result.v);
    
    // Calculate reconstruction error (in the same type T to preserve precision)
    let error_matrix = &h - &h_reconstructed;
    let error_norm = frobenius_norm_generic(&error_matrix);
    let relative_error = error_norm / frobenius_norm_generic(&h);

    println!("Reconstruction error norm: {:.6e}", error_norm);
    println!("Relative reconstruction error: {:.6e}", relative_error);

    // Check that reconstruction error is within expected bounds
    assert!(relative_error <= expected_max_error, 
            "Relative reconstruction error {} exceeds expected maximum {}", 
            relative_error, expected_max_error);
}

#[test]
fn test_hilbert_5x5_f64_reconstruction() {
    test_hilbert_reconstruction_generic::<f64>(5, 1e-12, 1e-14);
}

#[test]
fn test_hilbert_5x5_df64_reconstruction() {
    test_hilbert_reconstruction_generic::<Df64>(5, 1e-28, 1e-28);
}

#[test]
fn test_hilbert_10x10_f64_reconstruction() {
    test_hilbert_reconstruction_generic::<f64>(10, 1e-12, 1e-12);
}

#[test]
fn test_hilbert_10x10_df64_reconstruction() {
    // Note: 10x10 Hilbert matrix has very large condition number (~1e13)
    // Even with Df64, reconstruction is limited by nalgebra's matrix operations
    // which may not fully utilize Df64's precision in intermediate calculations
    test_hilbert_reconstruction_generic::<Df64>(10, 1e-28, 1e-30);
}

#[test]
fn test_hilbert_100x100_f64_reconstruction() {
    // Large matrix test with f64 - expect reasonable performance
    test_hilbert_reconstruction_generic::<f64>(100, 1e-12, 1e-12);
}

#[test]
fn test_hilbert_100x100_df64_reconstruction() {
    // Large matrix test with Df64 - expect high precision but longer execution time
    test_hilbert_reconstruction_generic::<Df64>(100, 1e-28, 1e-28);
}

