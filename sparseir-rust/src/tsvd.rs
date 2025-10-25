//! High-precision truncated SVD implementation using nalgebra
//!
//! This module provides QR + SVD based truncated SVD decomposition
//! with support for extended precision arithmetic.

use nalgebra::{DMatrix, DVector, ComplexField, RealField};
use nalgebra::linalg::ColPivQR;
use num_traits::{Zero, One, ToPrimitive};
use crate::Df64;

/// Result of SVD decomposition
#[derive(Debug, Clone)]
pub struct SVDResult<T> {
    /// Left singular vectors (m × rank)
    pub u: DMatrix<T>,
    /// Singular values (rank)
    pub s: DVector<T>,
    /// Right singular vectors (n × rank)
    pub v: DMatrix<T>,
    /// Effective rank
    pub rank: usize,
}

/// Configuration for TSVD computation
#[derive(Debug, Clone)]
pub struct TSVDConfig<T> {
    /// Relative tolerance for rank determination
    pub rtol: T,
}

impl<T> TSVDConfig<T> {
    pub fn new(rtol: T) -> Self {
        Self { rtol }
    }
}

/// Error types for TSVD computation
#[derive(Debug, thiserror::Error)]
pub enum TSVDError {
    #[error("Matrix is empty")]
    EmptyMatrix,
    #[error("Invalid tolerance: {0}")]
    InvalidTolerance(String),
}

/// Perform SVD decomposition using nalgebra
///
/// # Arguments
/// * `matrix` - Input matrix (m × n)
/// * `rtol` - Relative tolerance for rank determination (default: 1e-12 for f64)
///
/// # Returns
/// * `SVDResult` - SVD decomposition result
pub fn svd_decompose<T>(matrix: &DMatrix<T>, rtol: f64) -> SVDResult<T>
where
    T: ComplexField + RealField + Copy + nalgebra::RealField + ToPrimitive,
{
    let (_m, _n) = matrix.shape();

    // Perform SVD decomposition
    let svd = matrix.clone().svd(true, true);

    // Extract U, S, V matrices
    let u_matrix = svd.u.unwrap();
    let s_vector = svd.singular_values;
    let v_t_matrix = svd.v_t.unwrap();

    // Calculate effective rank (number of non-zero singular values)
    let rank = calculate_rank_from_vector(&s_vector, rtol);

    // Convert to thin SVD (truncate to effective rank)
    let u = DMatrix::from(u_matrix.columns(0, rank));
    let s = DVector::from(s_vector.rows(0, rank));
    let v = DMatrix::from(v_t_matrix.rows(0, rank).transpose());

    SVDResult {
        u,
        s,
        v,
        rank,
    }
}

/// Calculate effective rank from singular values
///
/// # Arguments
/// * `singular_values` - Vector of singular values
/// * `rtol` - Relative tolerance for rank determination
///
/// # Returns
/// * `usize` - Effective rank
fn calculate_rank_from_vector<T>(singular_values: &DVector<T>, rtol: f64) -> usize
where
    T: RealField + Copy + ToPrimitive,
{
    if singular_values.is_empty() {
        return 0;
    }

    let max_sv = singular_values.max();
    let threshold = max_sv * T::from_f64(rtol).unwrap_or(T::zero());

    let mut rank = 0;
    for &sv in singular_values.iter() {
        if sv > threshold {
            rank += 1;
        } else {
            break;
        }
    }

    rank
}

/// Calculate rank from R matrix diagonal elements
fn calculate_rank_from_r<T: RealField>(
    r_matrix: &DMatrix<T>,
    rtol: T,
) -> usize
where
    T: ComplexField + RealField + Copy,
{
    let dim = r_matrix.nrows().min(r_matrix.ncols());
    let mut rank = dim;

    // Find the maximum diagonal element
    let mut max_diag_abs = Zero::zero();
    for i in 0..dim {
        let diag_abs = ComplexField::abs(r_matrix[(i, i)]);
        if diag_abs > max_diag_abs {
            max_diag_abs = diag_abs;
        }
    }

    // If max_diag_abs is zero, rank is zero
    if max_diag_abs == Zero::zero() {
        return 0;
    }

    // Check each diagonal element
    for i in 0..dim {
        let diag_abs = ComplexField::abs(r_matrix[(i, i)]);

        // Check if the diagonal element is too small relative to the maximum diagonal element
        if diag_abs < rtol * max_diag_abs {
            rank = i;
            break;
        }
    }

    rank
}

/// Main TSVD function using QR + SVD approach
///
/// Computes the truncated SVD using the algorithm:
/// 1. Apply QR decomposition to A to get Q and R
/// 2. Compute SVD of R
/// 3. Reconstruct final U and V matrices
///
/// # Arguments
/// * `matrix` - Input matrix (m × n)
/// * `config` - TSVD configuration
///
/// # Returns
/// * `SVDResult` - Truncated SVD result
pub fn tsvd<T>(
    matrix: &DMatrix<T>,
    config: TSVDConfig<T>,
) -> Result<SVDResult<T>, TSVDError>
where
    T: ComplexField + RealField + Copy + nalgebra::RealField + std::fmt::Debug + ToPrimitive,
{
    let (m, n) = matrix.shape();

    if m == 0 || n == 0 {
        return Err(TSVDError::EmptyMatrix);
    }

    if config.rtol <= Zero::zero() || config.rtol >= One::one() {
        return Err(TSVDError::InvalidTolerance(format!(
            "Tolerance must be in (0, 1), got {:?}",
            config.rtol
        )));
    }

    // Step 1: Apply QR decomposition to A using nalgebra
    let qr = ColPivQR::new(matrix.clone());
    let q_matrix = qr.q();
    let r_matrix = qr.r();
    let permutation = qr.p();

    // Step 2: Apply QR-based rank estimation first
    // Use type-specific epsilon for QR diagonal elements (more conservative than rtol)
    let qr_rank = calculate_rank_from_r(&r_matrix, T::default_epsilon());
    
    if qr_rank == 0 {
        // Matrix has zero rank
        return Ok(SVDResult {
            u: DMatrix::zeros(m, 0),
            s: DVector::zeros(0),
            v: DMatrix::zeros(n, 0),
            rank: 0,
        });
    }

    // Step 3: Truncate R to estimated rank and apply SVD
    let r_truncated: DMatrix<T> = r_matrix.rows(0, qr_rank).into();
    // Use rtol directly as T, fallback to type-specific epsilon
    let rtol_t = if config.rtol.to_f64().is_some() {
        config.rtol
    } else {
        T::default_epsilon()
    };
    let rtol_f64 = rtol_t.to_f64().unwrap_or(f64::EPSILON);
    let svd_result = svd_decompose(&r_truncated, rtol_f64);
    
    if svd_result.rank == 0 {
        // Matrix has zero rank
        return Ok(SVDResult {
            u: DMatrix::zeros(m, 0),
            s: DVector::zeros(0),
            v: DMatrix::zeros(n, 0),
            rank: 0,
        });
    }

    // Step 4: Reconstruct full SVD
    // U = Q * U_R (Q is (m, qr_rank), U_R is (qr_rank, svd_result.rank))
    let q_truncated: DMatrix<T> = q_matrix.columns(0, qr_rank).into();
    let u_full = &q_truncated * &svd_result.u;

    // V = P^T * V_R (apply inverse permutation matrix)
    // Since A*P = Q*R, we have A = Q*R*P^T
    // After SVD of R: A = Q*U_R*S_R*V_R^T*P^T = U*S*V^T
    // where V^T = V_R^T*P^T, so V = P^(-1)*V_R = P^T*V_R
    let mut v_full = svd_result.v.clone();
    permutation.inv_permute_rows(&mut v_full);

    // S_full = S_R (already correct size)
    let s_full = svd_result.s.clone();

    Ok(SVDResult {
        u: u_full,
        s: s_full,
        v: v_full,
        rank: svd_result.rank,
    })
}

/// Convenience function for f64 TSVD
pub fn tsvd_f64(matrix: &DMatrix<f64>, rtol: f64) -> Result<SVDResult<f64>, TSVDError> {
    tsvd(matrix, TSVDConfig::new(rtol))
}

/// Convenience function for Df64 TSVD
pub fn tsvd_df64(matrix: &DMatrix<Df64>, rtol: Df64) -> Result<SVDResult<Df64>, TSVDError> {
    tsvd(matrix, TSVDConfig::new(rtol))
}

/// Convenience function for Df64 TSVD from f64 matrix
pub fn tsvd_df64_from_f64(matrix: &DMatrix<f64>, rtol: f64) -> Result<SVDResult<Df64>, TSVDError> {
    let matrix_df64 = DMatrix::from_fn(matrix.nrows(), matrix.ncols(), |i, j| {
        Df64::from(matrix[(i, j)])
    });
    let rtol_df64 = Df64::from(rtol);
    tsvd(&matrix_df64, TSVDConfig::new(rtol_df64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use num_traits::cast::ToPrimitive;

    #[test]
    fn test_svd_identity_matrix() {
        let matrix = DMatrix::<f64>::identity(3, 3);
        let result = svd_decompose(&matrix, 1e-12);
        
        assert_eq!(result.rank, 3);
        assert_eq!(result.s.len(), 3);
        assert_eq!(result.u.nrows(), 3);
        assert_eq!(result.u.ncols(), 3);
        assert_eq!(result.v.nrows(), 3);
        assert_eq!(result.v.ncols(), 3);
    }

    #[test]
    fn test_tsvd_identity_matrix() {
        let matrix = DMatrix::<f64>::identity(3, 3);
        let result = tsvd_f64(&matrix, 1e-12).unwrap();
        
        assert_eq!(result.rank, 3);
        assert_eq!(result.s.len(), 3);
    }

    #[test]
    fn test_tsvd_rank_one() {
        let matrix = DMatrix::<f64>::from_fn(3, 3, |i, j| (i + 1) as f64 * (j + 1) as f64);
        let result = tsvd_f64(&matrix, 1e-12).unwrap();
        
        assert_eq!(result.rank, 1);
    }

    #[test]
    fn test_tsvd_empty_matrix() {
        let matrix = DMatrix::<f64>::zeros(0, 0);
        let result = tsvd_f64(&matrix, 1e-12);
        
        assert!(matches!(result, Err(TSVDError::EmptyMatrix)));
    }

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
}
