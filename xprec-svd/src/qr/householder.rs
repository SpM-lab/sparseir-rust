//! Householder reflection utilities for QR decomposition

use mdarray::Tensor;
use crate::precision::Precision;
// Householder reflection utilities

/// Compute Householder reflection vector and coefficient
/// 
/// Given a vector x, computes a Householder reflection H = I - τvv^T
/// such that Hx = [β, 0, ..., 0]^T where β = ±||x||
/// 
/// Returns (τ, β) where τ is the reflection coefficient and β is the first element
pub fn reflector<T: Precision>(x: &mut Vec<T>) -> (T, T) {
    let n = x.len();
    if n == 0 {
        return (T::zero(), T::zero());
    }
    
    let xi1 = x[0];
    // Compute norm
    let mut sum = T::zero();
    for i in 0..n {
        sum = sum + x[i] * x[i];
    }
    let normu = Precision::sqrt(sum);
    
    if normu == T::zero() {
        return (T::zero(), T::zero());
    }
    
    // Compute ν = sign(xi1) * ||x||
    let nu = if xi1 >= T::zero() { normu } else { -normu };
    let xi1_new = xi1 + nu;
    x[0] = -nu;
    
    // Update remaining elements: x[i] /= xi1_new for i > 0
    if xi1_new != T::zero() {
        for i in 1..n {
            x[i] = x[i] / xi1_new;
        }
    }
    
    // Compute tau = xi1 / nu (like libsparseir)
    let tau = if nu != T::zero() {
        xi1_new / nu
    } else {
        T::zero()
    };
    
    (tau, -nu)
}

/// Apply Householder reflection to a block of matrix
/// 
/// Applies the Householder reflection H = I - τvv^T to columns [col_start..col_end) of matrix
/// starting from row row_start
pub fn reflector_apply_to_block<T: Precision>(
    matrix: &mut Tensor<T, (usize, usize)>,
    row_start: usize,
    v_col: usize,
    tau: T,
    col_start: usize,
    col_end: usize,
) {
    let shape = *matrix.shape();
    let m = shape.0;
    
    if m == 0 || col_start >= col_end {
        return;
    }
    
    let v_len = m - row_start;
    
    // Apply H = I - τvv^T to each column of A (like libsparseir)
    for j in col_start..col_end {
        // Compute vBj = tau * (B(0, j) + v.dot(B_col_j))
        let mut vBj = matrix[[row_start, j]];
        for i in 1..v_len {
            vBj = vBj + matrix[[row_start + i, v_col]] * matrix[[row_start + i, j]];
        }
        vBj = tau * vBj;
        
        // Update column j
        matrix[[row_start, j]] = matrix[[row_start, j]] - vBj;
        for i in 1..v_len {
            matrix[[row_start + i, j]] = matrix[[row_start + i, j]] - vBj * matrix[[row_start + i, v_col]];
        }
    }
}

/// Compute the Q matrix from QR factorization
/// 
/// Given the packed QR factorization (factors, taus), computes the full Q matrix
pub fn compute_q<T: Precision>(
    factors: &Tensor<T, (usize, usize)>,
    taus: &Tensor<T, (usize,)>,
) -> Tensor<T, (usize, usize)> {
    let shape = *factors.shape();
    let m = shape.0;
    let k = taus.len();
    
    let mut q = Tensor::from_fn((m, m), |idx| {
        if idx[0] == idx[1] { T::one() } else { T::zero() }
    });
    
    // Apply Householder reflections in reverse order
    for i in (0..k).rev() {
        if taus[[i]] != T::zero() {
            let tau = taus[[i]];
            
            // Apply H_i to Q from the left
            for j in 0..m {
                let mut vQj = q[[i, j]];
                for row in (i+1)..m {
                    vQj = vQj + factors[[row, i]] * q[[row, j]];
                }
                vQj = tau * vQj;
                
                q[[i, j]] = q[[i, j]] - vQj;
                for row in (i+1)..m {
                    q[[row, j]] = q[[row, j]] - vQj * factors[[row, i]];
                }
            }
        }
    }
    
    q
}

/// Compute the R matrix from QR factorization
/// 
/// Extracts the upper triangular R matrix from the packed QR factorization
pub fn compute_r<T: Precision>(
    factors: &Tensor<T, (usize, usize)>,
) -> Tensor<T, (usize, usize)> {
    let shape = *factors.shape();
    let (m, n) = shape;
    let k = m.min(n);
    
    let mut r = Array2::zeros((k, n));
    
    for i in 0..k {
        for j in i..n {
            r[[i, j]] = factors[[i, j]];
        }
    }
    
    // Ensure lower triangular part is zero (like libsparseir)
    for j in 0..k {
        for i in j + 1..k {
            r[[i, j]] = T::zero();
        }
    }
    
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_reflector() {
        let mut x = array![3.0, 4.0, 0.0];
        let (tau, beta) = reflector(x.view_mut());
        
        // After reflection, x should be [-5, 0.5, 0] (libsparseir implementation)
        assert_abs_diff_eq!(x[0], -5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(x[2], 0.0, epsilon = 1e-10);
        
        // Check that tau is correct (libsparseir: tau = xi1 / nu = 8 / 5 = 1.6)
        assert_abs_diff_eq!(tau, 1.6, epsilon = 1e-10);
        assert_abs_diff_eq!(beta, -5.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_reflector_apply() {
        let v = array![1.0, 0.0, 0.0];
        let tau = 2.0;
        let mut a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        
        reflector_apply(v.view(), tau, a.view_mut());
        
        // After applying H = I - 2vv^T, the first row should be negated
        assert_abs_diff_eq!(a[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[[0, 1]], -2.0, epsilon = 1e-10);
        // Other rows should be unchanged
        assert_abs_diff_eq!(a[[1, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[[2, 0]], 5.0, epsilon = 1e-10);
    }
}
