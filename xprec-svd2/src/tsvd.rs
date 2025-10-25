//! Truncated SVD (TSVD) implementation using QR + SVD

use crate::svd::{svd_decompose, SVDResult};
use nalgebra::{DMatrix, DVector, ComplexField, RealField};
use nalgebra::linalg::ColPivQR;
use num_traits::{Zero, One, ToPrimitive};
use xprec::Df64;

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

/// Convenience function for f64 precision
pub fn tsvd_f64(
    matrix: &DMatrix<f64>,
    rtol: f64,
) -> Result<SVDResult<f64>, TSVDError> {
    let config = TSVDConfig::new(rtol);
    tsvd(matrix, config)
}

/// Convenience function for Df64 precision
pub fn tsvd_df64(
    matrix: &DMatrix<Df64>,
    rtol: Df64,
) -> Result<SVDResult<Df64>, TSVDError> {
    let config = TSVDConfig::new(rtol);
    tsvd(matrix, config)
}

/// Convenience function to convert f64 matrix to Df64 precision and compute SVD
pub fn tsvd_df64_from_f64(
    matrix: &DMatrix<f64>,
    rtol: f64,
) -> Result<SVDResult<Df64>, TSVDError> {
    // Convert f64 matrix to Df64
    let matrix_df64 = DMatrix::from_fn(matrix.nrows(), matrix.ncols(), |i, j| {
        Df64::from(matrix[(i, j)])
    });

    tsvd_df64(&matrix_df64, Df64::from(rtol))
}

#[cfg(test)]
mod tests {
    use super::*;
    use xprec::Df64;

    fn test_tsvd_identity_matrix_generic<T>(rtol: f64) 
    where
        T: nalgebra::RealField + From<f64> + Copy + ToPrimitive,
    {
        let matrix: DMatrix<T> = DMatrix::identity(3, 3);
        let config = TSVDConfig::new(T::from(rtol));

        let result = tsvd(&matrix, config).unwrap();

        assert_eq!(result.rank, 3);
        assert_eq!(result.u.nrows(), 3);
        assert_eq!(result.u.ncols(), 3);
        assert_eq!(result.s.len(), 3);
        assert_eq!(result.v.nrows(), 3);
        assert_eq!(result.v.ncols(), 3);
    }

    fn test_tsvd_rank_one_generic<T>(rtol: f64) 
    where
        T: nalgebra::RealField + From<f64> + Copy + ToPrimitive,
    {
        let matrix: DMatrix<T> = DMatrix::from_fn(3, 3, |_, _| T::from(1.0));
        let config = TSVDConfig::new(T::from(rtol));

        let result = tsvd(&matrix, config).unwrap();

        // With appropriate rtol, we should get rank 1
        assert_eq!(result.rank, 1);
        assert_eq!(result.u.nrows(), 3);
        assert_eq!(result.u.ncols(), 1);
        assert_eq!(result.s.len(), 1);
        assert_eq!(result.v.nrows(), 3);
        assert_eq!(result.v.ncols(), 1);
    }

    fn test_tsvd_empty_matrix_generic<T>(rtol: f64) 
    where
        T: nalgebra::RealField + From<f64> + Copy + ToPrimitive,
    {
        let matrix: DMatrix<T> = DMatrix::zeros(0, 0);
        let config = TSVDConfig::new(T::from(rtol));
        let result = tsvd(&matrix, config);
        assert!(result.is_err());
    }

    fn test_tsvd_rank_deficient_generic<T>(rtol: f64) 
    where
        T: nalgebra::RealField + From<f64> + Copy + ToPrimitive,
    {
        let matrix: DMatrix<T> = DMatrix::from_fn(3, 3, |i, j| {
            T::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]][i][j])
        });
        let config = TSVDConfig::new(T::from(rtol));

        let result = tsvd(&matrix, config).unwrap();

        assert_eq!(result.rank, 2);
        assert_eq!(result.u.nrows(), 3);
        assert_eq!(result.u.ncols(), 2);
        assert_eq!(result.s.len(), 2);
        assert_eq!(result.v.nrows(), 3);
        assert_eq!(result.v.ncols(), 2);
    }

    #[test]
    fn test_tsvd_identity_matrix_f64() {
        test_tsvd_identity_matrix_generic::<f64>(1e-12);
    }

    #[test]
    fn test_tsvd_identity_matrix_df64() {
        test_tsvd_identity_matrix_generic::<Df64>(1e-28);
    }

    #[test]
    fn test_tsvd_rank_one_f64() {
        test_tsvd_rank_one_generic::<f64>(1e-12);
    }

    #[test]
    fn test_tsvd_rank_one_df64() {
        test_tsvd_rank_one_generic::<Df64>(1e-32);
    }

    #[test]
    fn test_tsvd_empty_matrix_f64() {
        test_tsvd_empty_matrix_generic::<f64>(1e-12);
    }

    #[test]
    fn test_tsvd_empty_matrix_df64() {
        test_tsvd_empty_matrix_generic::<Df64>(1e-28);
    }

    #[test]
    fn test_tsvd_rank_deficient_f64() {
        test_tsvd_rank_deficient_generic::<f64>(1e-12);
    }

    #[test]
    fn test_tsvd_rank_deficient_df64() {
        test_tsvd_rank_deficient_generic::<Df64>(1e-28);
    }
}
