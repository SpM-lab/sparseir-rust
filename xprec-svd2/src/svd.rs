//! SVD decomposition using nalgebra

use nalgebra::{DMatrix, DVector, ComplexField, RealField};

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
    T: ComplexField + RealField + Copy + nalgebra::RealField,
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
    let u = u_matrix.columns(0, rank).into();
    let s = s_vector.rows(0, rank).into();
    let v = v_t_matrix.rows(0, rank).transpose().into();

    SVDResult {
        u,
        s,
        v,
        rank,
    }
}

/// Calculate the effective rank based on singular values
fn calculate_rank_from_vector<T>(singular_values: &DVector<T>, rtol: f64) -> usize
where
    T: RealField + Copy,
{
    let len = singular_values.len();
    if len == 0 {
        return 0;
    }

    // Find the first singular value that is effectively zero
    // We use a relative tolerance based on the largest singular value
    let max_sv = singular_values.iter().fold(T::zero(), |acc, &x| {
        if x > acc { x } else { acc }
    });
    
    // Convert rtol from f64 to T using from_f64 (now works correctly for Df64)
    let rtol_t = T::from_f64(rtol).unwrap_or(T::from_f64(1e-15).unwrap());
    let tolerance = max_sv * rtol_t;

    for i in 0..len {
        if singular_values[i] < tolerance {
            return i;
        }
    }

    len
}

#[cfg(test)]
mod tests {
    use super::*;
    use xprec::Df64;

    fn test_svd_identity_matrix_generic<T>(rtol: f64) 
    where
        T: nalgebra::RealField + From<f64> + Copy,
    {
        let matrix: DMatrix<T> = DMatrix::identity(3, 3);

        let result = svd_decompose(&matrix, rtol);

        assert_eq!(result.rank, 3);
        assert_eq!(result.u.nrows(), 3);
        assert_eq!(result.u.ncols(), 3);
        assert_eq!(result.s.len(), 3);
        assert_eq!(result.v.nrows(), 3);
        assert_eq!(result.v.ncols(), 3);
    }

    fn test_svd_rank_one_generic<T>(rtol: f64) 
    where
        T: nalgebra::RealField + From<f64> + Copy,
    {
        let matrix: DMatrix<T> = DMatrix::from_fn(3, 3, |_, _| T::from(1.0));

        let result = svd_decompose(&matrix, rtol);
        
        // With appropriate rtol, we should get rank 1
        assert_eq!(result.rank, 1);
        assert_eq!(result.u.nrows(), 3);
        assert_eq!(result.u.ncols(), 1);
        assert_eq!(result.s.len(), 1);
        assert_eq!(result.v.nrows(), 3);
        assert_eq!(result.v.ncols(), 1);
    }

    fn test_svd_rectangular_generic<T>(rtol: f64) 
    where
        T: nalgebra::RealField + From<f64> + Copy,
    {
        let matrix: DMatrix<T> = DMatrix::from_fn(4, 2, |i, j| {
            T::from([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]][i][j])
        });

        let result = svd_decompose(&matrix, rtol);

        assert_eq!(result.rank, 2);
        assert_eq!(result.u.nrows(), 4);
        assert_eq!(result.u.ncols(), 2);
        assert_eq!(result.s.len(), 2);
        assert_eq!(result.v.nrows(), 2);
        assert_eq!(result.v.ncols(), 2);
    }

    #[test]
    fn test_svd_identity_matrix_f64() {
        test_svd_identity_matrix_generic::<f64>(1e-12);
    }

    #[test]
    fn test_svd_identity_matrix_df64() {
        test_svd_identity_matrix_generic::<Df64>(1e-28);
    }

    #[test]
    fn test_svd_rank_one_f64() {
        test_svd_rank_one_generic::<f64>(1e-12);
    }

    #[test]
    fn test_svd_rank_one_df64() {
        test_svd_rank_one_generic::<Df64>(1e-10);
    }

    #[test]
    fn test_svd_rectangular_f64() {
        test_svd_rectangular_generic::<f64>(1e-12);
    }

    #[test]
    fn test_svd_rectangular_df64() {
        test_svd_rectangular_generic::<Df64>(1e-28);
    }
}
