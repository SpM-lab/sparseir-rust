//! Jacobi SVD implementation

use mdarray::Tensor;
use crate::precision::Precision;
// Jacobi SVD implementation

/// Result of SVD decomposition
#[derive(Debug, Clone)]
pub struct SVDResult<T: Precision> {
    /// Left singular vectors (m × k)
    pub u: Tensor<T, (usize, usize)>,
    /// Singular values (k)
    pub s: Tensor<T, (usize,)>,
    /// Right singular vectors (n × k)
    pub v: Tensor<T, (usize, usize)>,
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

/// Jacobi SVD algorithm
/// 
/// Computes the SVD using two-sided Jacobi iterations.
/// This is more accurate than bidiagonalization methods but slower.
pub fn jacobi_svd<T: Precision>(
    matrix: &Tensor<T, (usize, usize)>,
) -> SVDResult<T> {
    let shape = *matrix.shape();
    let (m, n) = shape;
    let k = m.min(n);
    
    // QR preconditioner for rectangular matrices (like Eigen3)
    let (mut a, mut u) = if m > n {
        // More rows than columns: A = QR, then SVD(R)
        let mut a_work = matrix.clone();
        // Use rrqr_with_options with pivoting disabled for standard Householder QR
        let qr_result = crate::qr::rrqr_with_options(&mut a_work, T::zero(), false);
        let (q_full, r_full) = crate::qr::truncate_qr_result(&qr_result.0, n);
        
        // Extract R matrix (n x n upper triangular)
        let r = Tensor::from_fn((n, n), |idx| r_full[[idx[0], idx[1]]]);
        
        // U = Q (m x n)
        let u_init = Tensor::from_fn((m, n), |idx| q_full[[idx[0], idx[1]]]);
        
        (r, u_init)
    } else if n > m {
        // More columns than rows: A^T = QR, then SVD(R^T)
        // TODO: Implement this case if needed
        (matrix.clone(), {
            Tensor::from_fn((m, k), |idx| {
                if idx[0] == idx[1] { T::one() } else { T::zero() }
            })
        })
    } else {
        // Square matrix: no QR preconditioner needed
        (matrix.clone(), {
            Tensor::from_fn((m, k), |idx| {
                if idx[0] == idx[1] { T::one() } else { T::zero() }
            })
        })
    };
    
    // V is identity matrix (n x n)
    let mut v = Tensor::from_fn((n, n), |idx| {
        if idx[0] == idx[1] { T::one() } else { T::zero() }
    });
    
    // Scale matrix like Eigen3 to reduce over/under-flows and improve threshold calculation
    // This ensures max_diag_entry is close to 1, making threshold ~ epsilon (achievable)
    let a_shape = *a.shape();
    let (a_rows, a_cols) = a_shape;
    let mut scale = T::zero();
    for i in 0..a_rows {
        for j in 0..a_cols {
            scale = Precision::max(scale, Precision::abs(a[[i, j]]));
        }
    }
    if scale <= T::zero() || !scale.is_finite() {
        scale = T::one();
    }
    // Scale the matrix
    for i in 0..a_rows {
        for j in 0..a_cols {
            a[[i, j]] = a[[i, j]] / scale;
        }
    }
    
    // Use Eigen3's convergence criteria
    let consider_as_zero = <T as From<f64>>::from(std::f64::MIN_POSITIVE);  // Like Eigen3
    let precision = <T as From<f64>>::from(2.0) * <T as Precision>::epsilon();
    let max_iter = 1000; // Maximum number of sweeps
    
    // Convergence condition: All off-diagonal elements must be below threshold
    // threshold = max(consider_as_zero, precision * max_diag_entry)
    // where:
    //   - consider_as_zero ≈ 2e-308 (smallest positive float)
    //   - precision = 2 * machine_epsilon ≈ 4e-16 for f64
    //   - max_diag_entry = max absolute value in the matrix
    
    // Track maximum diagonal entry for threshold calculation
    // Initialize with maximum absolute value of diagonal elements (like Eigen3)
    let diagsize = a_rows.min(a_cols);
    
    let mut max_diag_entry = T::zero();
    for i in 0..diagsize {
        max_diag_entry = Precision::max(max_diag_entry, Precision::abs(a[[i, i]]));
    }
    
    let mut converged;
    
    // eprintln!("[Jacobi SVD] Starting iterations: matrix {}x{}, diagsize={}, initial max_diag_entry={:.6e}", 
    //           a_rows, a_cols, diagsize, max_diag_entry.to_f64().unwrap_or(0.0));
    
    for iter in 0..max_iter {
        converged = true;
        let mut num_rotations = 0;
        let mut max_off_diag = T::zero();
        
        // One-sided Jacobi: eliminate off-diagonal elements
        // Use Eigen3's loop order: p from 1 to diagsize, q from 0 to p-1
        for p in 1..diagsize {
            for q in 0..p {
                // Use Eigen3's threshold calculation
                let threshold = Precision::max(consider_as_zero, precision * max_diag_entry);
                
                let off_diag = Precision::max(Precision::abs(a[[p, q]]), Precision::abs(a[[q, p]]));
                max_off_diag = Precision::max(max_off_diag, off_diag);
                
                // Check if off-diagonal elements are below threshold (like Eigen3)
                // Eigen3 uses: if (abs(...) > threshold || abs(...) > threshold) { not_finished }
                // So we skip (continue) if both are <= threshold
                if Precision::abs(a[[p, q]]) <= threshold && Precision::abs(a[[q, p]]) <= threshold {
                    continue;
                }
                
                converged = false;
                num_rotations += 1;
                
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
                
                // Update max_diag_entry like Eigen3 does
                max_diag_entry = Precision::max(
                    max_diag_entry,
                    Precision::max(Precision::abs(a[[p, p]]), Precision::abs(a[[q, q]]))
                );
            }
        }
        
        // Log progress (disabled for production)
        // if iter % 10 == 0 || converged || iter == max_iter - 1 {
        //     let threshold = Precision::max(consider_as_zero, precision * max_diag_entry);
        //     eprintln!("[Jacobi SVD] Iter {}: rotations={}, max_off_diag={:.6e}, threshold={:.6e}, max_diag={:.6e}, converged={}", 
        //               iter, num_rotations, max_off_diag.to_f64().unwrap_or(0.0), threshold.to_f64().unwrap_or(0.0), 
        //               max_diag_entry.to_f64().unwrap_or(0.0), converged);
        // }
        
        if converged {
            // eprintln!("[Jacobi SVD] Converged after {} iterations", iter);
            break;
        }
        
        if iter == max_iter - 1 {
            eprintln!("[Jacobi SVD] WARNING: Max iterations ({}) reached without convergence!", max_iter);
        }
    }
    
    // Step 3: Extract singular values and ensure they are positive (like Eigen3)
    // Note: singular values need to be scaled back by the original scale factor
    let mut s = Tensor::from_fn((diagsize,), |idx| {
        let i = idx[0];
        let diag_val = a[[i, i]];
        Precision::abs(diag_val) * scale  // Scale back
    });
    
    // If diagonal entry is negative, flip the corresponding U column
    for i in 0..diagsize {
        let diag_val = a[[i, i]];
        if diag_val < T::zero() {
            let u_shape = *u.shape();
            for row in 0..u_shape.0 {
                u[[row, i]] = -u[[row, i]];
            }
        }
    }
    
    // Step 4: Sort singular values in descending order (like Eigen3)
    let mut indices: Vec<usize> = (0..diagsize).collect();
    indices.sort_by(|&a, &b| s[[b]].partial_cmp(&s[[a]]).unwrap());
    
    let s_sorted = Tensor::from_fn((diagsize,), |idx| s[[indices[idx[0]]]]);
    let u_sorted = Tensor::from_fn((m, diagsize), |idx| u[[idx[0], indices[idx[1]]]]);
    let v_sorted = Tensor::from_fn((n, diagsize), |idx| v[[idx[0], indices[idx[1]]]]);
    
    
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
    use mdarray::tensor;
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
        let a = Tensor::from_fn((3, 3), |idx| {
            if idx[0] == idx[1] { 1.0 } else { 0.0 }
        });
        let result: SVDResult<f64> = jacobi_svd(&a);
        
        // Identity matrix should have singular values all equal to 1
        for i in 0..result.s.len() {
            assert_abs_diff_eq!(result.s[[i]], 1.0, epsilon = 1e-10);
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
        let a = tensor![
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ];
        
        let result: SVDResult<f64> = jacobi_svd(&a);
        
        // Should have only one non-zero singular value
        assert!(result.s[[0]] > 1.0); // Should be around 3.0
        assert_abs_diff_eq!(result.s[[1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.s[[2]], 0.0, epsilon = 1e-10);
    }
}
