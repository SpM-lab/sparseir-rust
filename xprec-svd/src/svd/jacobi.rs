//! Jacobi SVD implementation

use ndarray::{s, Array1, Array2};
use crate::precision::Precision;
// Jacobi SVD implementation

/// Result of SVD decomposition
#[derive(Debug, Clone)]
pub struct SVDResult<T: Precision> {
    /// Left singular vectors (m × k)
    pub u: Array2<T>,
    /// Singular values (k)
    pub s: Array1<T>,
    /// Right singular vectors (n × k)
    pub v: Array2<T>,
    /// Effective rank
    pub rank: usize,
}

/// Jacobi rotation structure
#[derive(Debug, Clone, Copy)]
struct JacobiRotation<T: Precision> {
    c: T,  // cosine
    s: T,  // sine
}

impl<T: Precision> JacobiRotation<T> {
    fn new(c: T, s: T) -> Self {
        Self { c, s }
    }
    
    fn identity() -> Self {
        Self { c: T::one(), s: T::zero() }
    }
    
    fn transpose(&self) -> Self {
        Self { c: self.c, s: -self.s }
    }
    
    fn compose(&self, other: &Self) -> Self {
        Self {
            c: self.c * other.c - self.s * other.s,
            s: self.s * other.c + self.c * other.s,
        }
    }
}

/// Compute 2×2 Jacobi SVD using Eigen3's approach
/// 
/// This implements the same algorithm as Eigen3's real_2x2_jacobi_svd
/// Returns (left_rotation, right_rotation, singular_values)
fn real_2x2_jacobi_svd<T: Precision>(
    a: T, b: T, c: T, d: T
) -> (JacobiRotation<T>, JacobiRotation<T>, (T, T)) {
    // Create 2x2 matrix
    let mut m = [[a, b], [c, d]];
    
    
    // First rotation to eliminate off-diagonal elements
    let t = m[0][0] + m[1][1];
    let d_val = m[1][0] - m[0][1];
    
    
    let rot1 = if Precision::abs(d_val) < <T as Precision>::epsilon() {
        JacobiRotation::identity()
    } else {
        let u = t / d_val;
        let tmp = Precision::sqrt(T::one() + u * u);
        let r = JacobiRotation::new(u / tmp, T::one() / tmp);
        r
    };
    
    // Apply first rotation (left)
    apply_rotation_left(&mut m, 0, 1, &rot1);
    
    // Compute second rotation (right) from the left-rotated matrix
    let rot2 = make_jacobi(&m, 0, 1);
    
    
    // Note: We don't apply rot2 to m here (unlike what we did before)
    // The right rotation is applied by the caller to the full matrix
    // We only return the rotation parameters
    
    // Compose rotations for j_left
    let left_rot = rot1.compose(&rot2.transpose());
    
    // Extract singular values from the diagonalized matrix
    let s1 = Precision::abs(m[0][0]);
    let s2 = Precision::abs(m[1][1]);
    
    
    // Ensure singular values are in descending order
    if s1 < s2 {
        (left_rot, rot2, (s2, s1))
    } else {
        (left_rot, rot2, (s1, s2))
    }
}

/// Make Jacobi rotation to eliminate off-diagonal element
/// This implements Eigen3's makeJacobi(x, y, z) function
/// For a self-adjoint 2x2 matrix B = [[x, y], [conj(y), z]]
/// Returns rotation J such that J^* B J is diagonal
fn make_jacobi<T: Precision>(m: &[[T; 2]; 2], p: usize, q: usize) -> JacobiRotation<T> {
    let x = m[p][p];
    let y = m[p][q];
    let z = m[q][q];
    
    
    // Check if matrix is already diagonalized
    // For a self-adjoint matrix [[x, y], [conj(y), z]], if x = z and y = 0, it's already diagonal
    let two = <T as From<f64>>::from(2.0);
    let deno = two * Precision::abs(y);
    
    
    if deno < <T as Precision>::epsilon() {
        // If y is too small, return identity rotation
        return JacobiRotation::identity();
    }
    
    // tau = (x - z) / (2 * |y|)
    let tau = (x - z) / deno;
    let w = Precision::sqrt(tau * tau + T::one());
    
    
    // Compute t
    let t = if tau > T::zero() {
        T::one() / (tau + w)
    } else {
        T::one() / (tau - w)
    };
    
    
    // Compute rotation parameters (matching Eigen3's implementation)
    let sign_t = if t > T::zero() { T::one() } else { -T::one() };
    let n = T::one() / Precision::sqrt(t * t + T::one());
    
    // For real numbers: conj(y) / abs(y) = y / abs(y) = sign(y)
    let sign_y = if y >= T::zero() { T::one() } else { -T::one() };
    
    
    let c_val = n;
    let s_val = -sign_t * sign_y * Precision::abs(t) * n;
    
    
    JacobiRotation::new(c_val, s_val)
}

/// Apply left rotation to 2x2 matrix
fn apply_rotation_left<T: Precision>(m: &mut [[T; 2]; 2], p: usize, q: usize, rot: &JacobiRotation<T>) {
    let temp1 = rot.c * m[p][0] - rot.s * m[q][0];
    let temp2 = rot.c * m[p][1] - rot.s * m[q][1];
    m[q][0] = rot.s * m[p][0] + rot.c * m[q][0];
    m[q][1] = rot.s * m[p][1] + rot.c * m[q][1];
    m[p][0] = temp1;
    m[p][1] = temp2;
}

/// Apply right rotation to 2x2 matrix
fn apply_rotation_right<T: Precision>(m: &mut [[T; 2]; 2], p: usize, q: usize, rot: &JacobiRotation<T>) {
    let temp1 = rot.c * m[0][p] - rot.s * m[0][q];
    let temp2 = rot.c * m[1][p] - rot.s * m[1][q];
    m[0][q] = rot.s * m[0][p] + rot.c * m[0][q];
    m[1][q] = rot.s * m[1][p] + rot.c * m[1][q];
    m[0][p] = temp1;
    m[1][p] = temp2;
}

/// Apply Givens rotation to a matrix
/// 
/// Applies the rotation matrix:
/// [c -s] to the left or [c  s] to the right
/// [s  c]               [-s c]
pub fn apply_givens_left<T: Precision>(
    matrix: &mut Array2<T>,
    i: usize,
    j: usize,
    c: T,
    s: T,
) {
    let n = matrix.ncols();
    
    // Left rotation: update rows i and j
    for k in 0..n {
        let xi = matrix[[i, k]];
        let yi = matrix[[j, k]];
        let new_xi = c * xi + s * yi;
        let new_yi = -s * xi + c * yi;
        matrix[[i, k]] = new_xi;
        matrix[[j, k]] = new_yi;
    }
}

pub fn apply_givens_right<T: Precision>(
    matrix: &mut Array2<T>,
    i: usize,
    j: usize,
    c: T,
    s: T,
) {
    let m = matrix.nrows();
    
    
    // Right rotation: update columns i and j
    // Note: Eigen3's applyOnTheRight applies the transpose of the rotation
    // So we apply (c, -s) instead of (c, s)
    for k in 0..m {
        let xi = matrix[[k, i]];
        let yi = matrix[[k, j]];
        
        // Apply transpose: (c, -s)
        // Must use temporary variables to avoid in-place issues
        let new_xi = c * xi - s * yi;
        let new_yi = s * xi + c * yi;
        matrix[[k, i]] = new_xi;
        matrix[[k, j]] = new_yi;
        
    }
}

/// Jacobi SVD algorithm
/// 
/// Computes the SVD using two-sided Jacobi iterations.
/// This is more accurate than bidiagonalization methods but slower.
pub fn jacobi_svd<T: Precision>(
    matrix: &Array2<T>,
) -> SVDResult<T> {
    let m = matrix.nrows();
    let n = matrix.ncols();
    let k = m.min(n);
    
    // QR preconditioner for rectangular matrices (like Eigen3)
    let (mut a, mut u) = if m > n {
        // More rows than columns: A = QR, then SVD(R)
        let mut a_work = matrix.clone();
        // Use rrqr_with_options with pivoting disabled for standard Householder QR
        let qr_result = crate::qr::rrqr_with_options(&mut a_work, T::zero(), false);
        let (q_full, r_full) = crate::qr::truncate_qr_result(&qr_result.0, n);
        
        // Extract R matrix (n x n upper triangular)
        let r = r_full.slice(s![0..n, 0..n]).to_owned();
        
        // U = Q (m x n)
        let u_init = q_full.slice(s![.., 0..n]).to_owned();
        
        (r, u_init)
    } else if n > m {
        // More columns than rows: A^T = QR, then SVD(R^T)
        // TODO: Implement this case if needed
        (matrix.clone(), {
            let mut mat = Array2::zeros((m, k));
            for i in 0..k {
                mat[[i, i]] = T::one();
            }
            mat
        })
    } else {
        // Square matrix: no QR preconditioner needed
        (matrix.clone(), {
            let mut mat = Array2::zeros((m, k));
            for i in 0..k {
                mat[[i, i]] = T::one();
            }
            mat
        })
    };
    
    let mut v = Array2::eye(n); // V is always n x n initially, then we take first k columns
    
    // Use Eigen3's convergence criteria
    let consider_as_zero = <T as From<f64>>::from(std::f64::MIN_POSITIVE);
    let precision = <T as From<f64>>::from(2.0) * <T as Precision>::epsilon();
    let max_iter = 30; // Maximum number of sweeps
    
    // Track maximum diagonal entry for threshold calculation
    // Initialize with maximum absolute value in the matrix to handle non-diagonal matrices
    let mut max_diag_entry = T::zero();
    let a_rows = a.nrows();
    let a_cols = a.ncols();
    for i in 0..a_rows {
        for j in 0..a_cols {
            max_diag_entry = Precision::max(max_diag_entry, Precision::abs(a[[i, j]]));
        }
    }
    
    let mut _iterations = 0;
    let diagsize = a_rows.min(a_cols); // For the working matrix after QR
    
    for iter in 0..max_iter {
        let mut converged = true;
        
        
        // One-sided Jacobi: eliminate off-diagonal elements
        // Use Eigen3's loop order: p from 1 to diagsize, q from 0 to p-1
        for p in 1..diagsize {
            for q in 0..p {
                // Use Eigen3's threshold calculation
                let threshold = Precision::max(consider_as_zero, precision * max_diag_entry);
                
                
                if Precision::abs(a[[p, q]]) <= threshold && Precision::abs(a[[q, p]]) <= threshold {
                    continue;
                }
                
                converged = false;
                _iterations += 1;
                
                // Compute 2×2 Jacobi SVD of the 2×2 submatrix
                let (left_rot, right_rot, (_s1, _s2)) = real_2x2_jacobi_svd(
                    a[[p, p]], a[[p, q]],
                    a[[q, p]], a[[q, q]]
                );
                
                let c = left_rot.c;
                let s = left_rot.s;
                let c2 = right_rot.c;
                let s2_rot = right_rot.s;
                
                
                // Apply left rotation to A
                apply_givens_left(&mut a, p, q, c, s);
                
                // Apply left rotation transpose to U (i.e., right rotation with transposed parameters)
                // Eigen3: m_matrixU.applyOnTheRight(p, q, j_left.transpose())
                // Transpose means we swap c and -s: (c, s) -> (c, -s)
                apply_givens_right(&mut u, p, q, c, -s);
                
                
                // Apply right rotation to A and V
                apply_givens_right(&mut a, p, q, c2, s2_rot);
                apply_givens_right(&mut v, p, q, c2, s2_rot);
                
            }
        }
        
        if converged {
            break;
        }
    }
    
    // Step 3: Extract singular values and ensure they are positive (like Eigen3)
    let mut s = Array1::zeros(diagsize);
    for i in 0..diagsize {
        let diag_val = a[[i, i]];
        s[i] = Precision::abs(diag_val);
        
        // If diagonal entry is negative, flip the corresponding U column
        if diag_val < T::zero() {
            for row in 0..u.nrows() {
                u[[row, i]] = -u[[row, i]];
            }
        }
    }
    
    // Step 4: Sort singular values in descending order (like Eigen3)
    let mut indices: Vec<usize> = (0..diagsize).collect();
    indices.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap());
    
    let mut s_sorted = Array1::zeros(diagsize);
    let mut u_sorted = Array2::zeros((m, diagsize));
    let mut v_sorted = Array2::zeros((n, diagsize));
    
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        s_sorted[new_idx] = s[old_idx];
        u_sorted.column_mut(new_idx).assign(&u.column(old_idx));
        v_sorted.column_mut(new_idx).assign(&v.column(old_idx));
    }
    
    SVDResult {
        u: u_sorted,
        s: s_sorted,
        v: v_sorted,
        rank: diagsize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_svd_2x2() {
        let (left_rot, right_rot, (s1, s2)) = real_2x2_jacobi_svd(3.0, 4.0, 0.0, 5.0);
        let c = left_rot.c;
        let s = left_rot.s;
        let c2 = right_rot.c;
        let s2_rot = right_rot.s;
        
        // Check that singular values are positive
        assert!(s1 > 0.0);
        assert!(s2 > 0.0);
        
        // Check that rotation matrices are orthogonal
        assert_abs_diff_eq!(c * c + s * s, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c2 * c2 + s2_rot * s2_rot, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_jacobi_svd_identity() {
        let a = Array2::eye(3);
        let result: SVDResult<f64> = jacobi_svd(&a);
        
        // Identity matrix should have singular values all equal to 1
        for &s in result.s.iter() {
            assert_abs_diff_eq!(s, 1.0, epsilon = 1e-10);
        }
        
        // U and V should be identity matrices
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(result.u[[i, j]], expected, epsilon = 1e-10);
                assert_abs_diff_eq!(result.v[[i, j]], expected, epsilon = 1e-10);
            }
        }
    }
    
    #[test]
    fn test_jacobi_svd_rank_one() {
        let a = array![
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ];
        
        let result: SVDResult<f64> = jacobi_svd(&a);
        
        // Should have only one non-zero singular value
        assert!(result.s[0] > 1.0); // Should be around 3.0
        assert_abs_diff_eq!(result.s[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.s[2], 0.0, epsilon = 1e-10);
    }
}
