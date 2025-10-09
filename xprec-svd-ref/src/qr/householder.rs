//! Householder reflection utilities for QR decomposition

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, ArrayViewMut2, s};
use crate::precision::Precision;
// Householder reflection utilities

/// Compute Householder reflection vector and coefficient
/// 
/// Given a vector x, computes a Householder reflection H = I - τvv^T
/// such that Hx = [β, 0, ..., 0]^T where β = ±||x||
/// 
/// Returns (τ, β) where τ is the reflection coefficient and β is the first element
pub fn reflector<T: Precision>(mut x: ArrayViewMut1<T>) -> (T, T) {
    let n = x.len();
    if n == 0 {
        return (T::zero(), T::zero());
    }
    
    let xi1 = x[0];
    let normu = crate::utils::norms::norm_2(x.view());
    
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

/// Apply Householder reflection to a matrix
/// 
/// Applies the Householder reflection H = I - τvv^T to matrix A from the left:
/// A = H * A
pub fn reflector_apply<T: Precision>(
    v: ArrayView1<T>,
    tau: T,
    mut a: ArrayViewMut2<T>,
) {
    let m = a.nrows();
    let n = a.ncols();
    
    if m == 0 || n == 0 {
        return;
    }
    
    // Apply H = I - τvv^T to each column of A (like libsparseir)
    for j in 0..n {
        // Compute vBj = tau * (B(0, j) + xj.dot(Bj))
        // where xj = v[1..] and Bj = B[1.., j]
        let mut vBj = a[[0, j]];
        for i in 1..m {
            vBj = vBj + v[i] * a[[i, j]];
        }
        vBj = tau * vBj;
        
        // Update B(0, j)
        a[[0, j]] = a[[0, j]] - vBj;
        
        // Apply axpy operation: Bj -= vBj * xj
        for i in 1..m {
            a[[i, j]] = a[[i, j]] - vBj * v[i];
        }
    }
}

/// Compute the Q matrix from QR factorization
/// 
/// Given the packed QR factorization (factors, taus), computes the full Q matrix
pub fn compute_q<T: Precision>(
    factors: &Array2<T>,
    taus: &Array1<T>,
) -> Array2<T> {
    let m = factors.nrows();
    let k = taus.len();
    
    let mut q = Array2::eye(m);
    
    // Apply Householder reflections in forward order (like libsparseir)
    // Q = H_1 * H_2 * ... * H_k
    for i in 0..k {
        if taus[i] != T::zero() {
            let v = factors.slice(s![i.., i]);
            let tau = taus[i];
            
            // Apply H_i to Q (left multiplication)
            let q_slice = q.slice_mut(s![i.., ..]);
            reflector_apply(v, tau, q_slice);
        }
        // If tau == 0, no reflection is needed (column is already in upper triangular form)
    }
    
    q
}

/// Compute the R matrix from QR factorization
/// 
/// Extracts the upper triangular R matrix from the packed QR factorization
pub fn compute_r<T: Precision>(
    factors: &Array2<T>,
) -> Array2<T> {
    let m = factors.nrows();
    let n = factors.ncols();
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
