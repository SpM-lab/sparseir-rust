//! Fitters for least-squares problems with various matrix types
//!
//! This module provides fitters for solving min ||A * coeffs - values||^2
//! where the matrix A and value types can vary.

use mdarray::DTensor;
use num_complex::Complex;
use std::cell::RefCell;

/// SVD decomposition for real matrices
struct RealSVD {
    u: DTensor<f64, 2>,   // (n_rows, min_dim)
    s: Vec<f64>,          // (min_dim,)
    vt: DTensor<f64, 2>,  // (min_dim, n_cols)
}

/// SVD decomposition for complex matrices
struct ComplexSVD {
    u: DTensor<Complex<f64>, 2>,   // (n_rows, min_dim)
    s: Vec<f64>,                    // (min_dim,) - singular values are real
    vt: DTensor<Complex<f64>, 2>,  // (min_dim, n_cols)
}

/// Fitter for real matrix: A ∈ R^{n×m}
///
/// Solves: min ||A * coeffs - values||^2
/// where A, coeffs, values are all real
///
/// # Example
/// ```ignore
/// let matrix = DTensor::from_fn([10, 5], |idx| ...);
/// let fitter = RealMatrixFitter::new(matrix);
/// 
/// let coeffs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let values = fitter.evaluate(&coeffs);
/// let fitted_coeffs = fitter.fit(&values);
/// ```
pub struct RealMatrixFitter {
    matrix: DTensor<f64, 2>,  // (n_points, basis_size)
    svd: RefCell<Option<RealSVD>>,
}

impl RealMatrixFitter {
    /// Create a new fitter with the given real matrix
    pub fn new(matrix: DTensor<f64, 2>) -> Self {
        Self {
            matrix,
            svd: RefCell::new(None),
        }
    }
    
    /// Number of data points
    pub fn n_points(&self) -> usize {
        self.matrix.shape().0
    }
    
    /// Number of basis functions (coefficients)
    pub fn basis_size(&self) -> usize {
        self.matrix.shape().1
    }
    
    /// Evaluate: coeffs (real) → values (real)
    ///
    /// Computes: values = A * coeffs
    pub fn evaluate(&self, coeffs: &[f64]) -> Vec<f64> {
        assert_eq!(coeffs.len(), self.basis_size(), 
            "coeffs.len()={} must equal basis_size={}", coeffs.len(), self.basis_size());
        
        let n_points = self.n_points();
        let mut values = vec![0.0; n_points];
        
        for i in 0..n_points {
            let mut sum = 0.0;
            for j in 0..self.basis_size() {
                sum += self.matrix[[i, j]] * coeffs[j];
            }
            values[i] = sum;
        }
        
        values
    }
    
    /// Fit: values (real) → coeffs (real)
    ///
    /// Solves: min ||A * coeffs - values||^2 using SVD
    pub fn fit(&self, values: &[f64]) -> Vec<f64> {
        assert_eq!(values.len(), self.n_points(),
            "values.len()={} must equal n_points={}", values.len(), self.n_points());
        
        // Compute SVD lazily
        if self.svd.borrow().is_none() {
            let svd = compute_real_svd(&self.matrix);
            *self.svd.borrow_mut() = Some(svd);
        }
        
        let svd = self.svd.borrow();
        let svd = svd.as_ref().unwrap();
        
        // Solve: coeffs = V * S^{-1} * U^T * values
        solve_real_svd(svd, values)
    }
}

/// Fitter for complex matrix with real coefficients: A ∈ C^{n×m}
///
/// Solves: min ||A * coeffs - values||^2
/// where A is complex, coeffs are real, values are complex
///
/// Strategy: Flatten complex to real problem
///   A_real ∈ R^{2n×m}: [Re(A[0,:]); Im(A[0,:]); Re(A[1,:]); Im(A[1,:]); ...]
///   values_flat ∈ R^{2n}: [Re(v[0]); Im(v[0]); Re(v[1]); Im(v[1]); ...]
///
/// # Example
/// ```ignore
/// let matrix = DTensor::from_fn([10, 5], |idx| Complex::new(...));
/// let fitter = ComplexToRealFitter::new(&matrix);
/// 
/// let coeffs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let values = fitter.evaluate(&coeffs);  // → Vec<Complex<f64>>
/// let fitted_coeffs = fitter.fit(&values);  // ← Vec<Complex<f64>>, → Vec<f64>
/// ```
pub struct ComplexToRealFitter {
    // A_real ∈ R^{2n×m}: flattened complex matrix
    // A_real[2i,   :] = Re(A[i, :])
    // A_real[2i+1, :] = Im(A[i, :])
    matrix_real: DTensor<f64, 2>,  // (2*n_points, basis_size)
    pub matrix: DTensor<Complex<f64>, 2>,  // (n_points, basis_size) - original complex matrix
    svd: RefCell<Option<RealSVD>>,
    n_points: usize,      // Original complex point count
}

impl ComplexToRealFitter {
    /// Create from complex matrix A ∈ C^{n×m}
    pub fn new(matrix_complex: &DTensor<Complex<f64>, 2>) -> Self {
        let (n_points, basis_size) = *matrix_complex.shape();
        
        // Flatten to real: (2*n_points, basis_size)
        // A_real[2i,   j] = Re(A[i, j])
        // A_real[2i+1, j] = Im(A[i, j])
        let matrix_real = DTensor::<f64, 2>::from_fn(
            [2 * n_points, basis_size],
            |idx| {
                let i = idx[0] / 2;
                let j = idx[1];
                let val = matrix_complex[[i, j]];
                
                if idx[0] % 2 == 0 {
                    val.re  // Even row: real part
                } else {
                    val.im  // Odd row: imaginary part
                }
            }
        );
        
        Self {
            matrix_real,
            matrix: matrix_complex.clone(),
            svd: RefCell::new(None),
            n_points,
        }
    }
    
    /// Number of complex data points
    pub fn n_points(&self) -> usize {
        self.n_points
    }
    
    /// Number of basis functions (coefficients)
    pub fn basis_size(&self) -> usize {
        self.matrix_real.shape().1
    }
    
    /// Evaluate: coeffs (real) → values (complex)
    ///
    /// Computes: values = A * coeffs where A is complex
    pub fn evaluate(&self, coeffs: &[f64]) -> Vec<Complex<f64>> {
        assert_eq!(coeffs.len(), self.basis_size(),
            "coeffs.len()={} must equal basis_size={}", coeffs.len(), self.basis_size());
        
        // A_real * coeffs → [re_0, im_0, re_1, im_1, ...]
        let result_flat = {
            let mut result = vec![0.0; 2 * self.n_points];
            for i in 0..(2 * self.n_points) {
                let mut sum = 0.0;
                for j in 0..self.basis_size() {
                    sum += self.matrix_real[[i, j]] * coeffs[j];
                }
                result[i] = sum;
            }
            result
        };
        
        // Unflatten: [re_0, im_0, ...] → [z_0, z_1, ...]
        (0..self.n_points)
            .map(|i| Complex::new(result_flat[2*i], result_flat[2*i+1]))
            .collect()
    }
    
    /// Fit: values (complex) → coeffs (real)
    ///
    /// Solves: min ||A * coeffs - values||^2 using flattened real SVD
    pub fn fit(&self, values: &[Complex<f64>]) -> Vec<f64> {
        assert_eq!(values.len(), self.n_points(),
            "values.len()={} must equal n_points={}", values.len(), self.n_points());
        
        // Compute SVD lazily
        if self.svd.borrow().is_none() {
            let svd = compute_real_svd(&self.matrix_real);
            *self.svd.borrow_mut() = Some(svd);
        }
        
        // Flatten complex values → real: [z_0, z_1, ...] → [re_0, im_0, re_1, im_1, ...]
        let mut values_flat = vec![0.0; 2 * self.n_points];
        for (i, &z) in values.iter().enumerate() {
            values_flat[2*i] = z.re;
            values_flat[2*i+1] = z.im;
        }
        
        let svd = self.svd.borrow();
        let svd = svd.as_ref().unwrap();
        
        // Solve: coeffs = V * S^{-1} * U^T * values_flat
        solve_real_svd(svd, &values_flat)
    }
}

/// Fitter for complex matrix with complex coefficients: A ∈ C^{n×m}
///
/// Solves: min ||A * coeffs - values||^2
/// where A, coeffs, values are all complex
///
/// # Example
/// ```ignore
/// let matrix = DTensor::from_fn([10, 5], |idx| Complex::new(...));
/// let fitter = ComplexMatrixFitter::new(matrix);
/// 
/// let coeffs: Vec<Complex<f64>> = vec![...];
/// let values = fitter.evaluate(&coeffs);  // → Vec<Complex<f64>>
/// let fitted_coeffs = fitter.fit(&values);  // ← Vec<Complex<f64>>, → Vec<Complex<f64>>
/// ```
pub struct ComplexMatrixFitter {
    pub matrix: DTensor<Complex<f64>, 2>,  // (n_points, basis_size)
    svd: RefCell<Option<ComplexSVD>>,
}

impl ComplexMatrixFitter {
    /// Create a new fitter with the given complex matrix
    pub fn new(matrix: DTensor<Complex<f64>, 2>) -> Self {
        Self {
            matrix,
            svd: RefCell::new(None),
        }
    }
    
    /// Number of data points
    pub fn n_points(&self) -> usize {
        self.matrix.shape().0
    }
    
    /// Number of basis functions (coefficients)
    pub fn basis_size(&self) -> usize {
        self.matrix.shape().1
    }
    
    /// Evaluate: coeffs (complex) → values (complex)
    ///
    /// Computes: values = A * coeffs
    pub fn evaluate(&self, coeffs: &[Complex<f64>]) -> Vec<Complex<f64>> {
        assert_eq!(coeffs.len(), self.basis_size(),
            "coeffs.len()={} must equal basis_size={}", coeffs.len(), self.basis_size());
        
        let n_points = self.n_points();
        let mut values = vec![Complex::new(0.0, 0.0); n_points];
        
        for i in 0..n_points {
            let mut sum = Complex::new(0.0, 0.0);
            for j in 0..self.basis_size() {
                sum += self.matrix[[i, j]] * coeffs[j];
            }
            values[i] = sum;
        }
        
        values
    }
    
    /// Fit: values (complex) → coeffs (complex)
    ///
    /// Solves: min ||A * coeffs - values||^2 using complex SVD
    pub fn fit(&self, values: &[Complex<f64>]) -> Vec<Complex<f64>> {
        assert_eq!(values.len(), self.n_points(),
            "values.len()={} must equal n_points={}", values.len(), self.n_points());
        
        // Compute SVD lazily
        if self.svd.borrow().is_none() {
            let svd = compute_complex_svd(&self.matrix);
            *self.svd.borrow_mut() = Some(svd);
        }
        
        let svd = self.svd.borrow();
        let svd = svd.as_ref().unwrap();
        
        // Solve: coeffs = V * S^{-1} * U^H * values
        solve_complex_svd(svd, values)
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Compute SVD of a real matrix using mdarray-linalg
fn compute_real_svd(matrix: &DTensor<f64, 2>) -> RealSVD {
    use mdarray_linalg::{SVD, SVDDecomp};
    use mdarray_linalg_faer::Faer;
    
    let mut a = matrix.clone();
    let SVDDecomp { u, s, vt } = Faer.svd(&mut a)
        .expect("SVD computation failed");
    
    // Extract singular values from first row
    let min_dim = s.shape().0.min(s.shape().1);
    let s_vec: Vec<f64> = (0..min_dim).map(|i| s[[0, i]]).collect();
    
    RealSVD { u, s: s_vec, vt }
}

/// Solve linear system using precomputed SVD
fn solve_real_svd(svd: &RealSVD, values: &[f64]) -> Vec<f64> {
    use crate::gemm::matmul_par;
    
    let n = values.len();
    let basis_size = svd.vt.shape().1;
    
    // Convert values to column vector
    let values_col = DTensor::<f64, 2>::from_fn([n, 1], |idx| values[idx[0]]);
    
    // U^T * values
    let ut = svd.u.transpose().to_tensor();
    let ut_values = matmul_par(&ut, &values_col);
    
    // S^{-1} * (U^T * values)
    let min_dim = svd.s.len();
    let s_inv_ut_values = DTensor::<f64, 2>::from_fn([min_dim, 1], |idx| {
        let i = idx[0];
        ut_values[[i, 0]] / svd.s[i]
    });
    
    // coeffs = V * (S^{-1} * U^T * values) = V^T^T * (S^{-1} * U^T * values)
    let v = svd.vt.transpose().to_tensor();  // (basis_size, min_dim)
    let coeffs_col = matmul_par(&v, &s_inv_ut_values);  // (basis_size, 1)
    
    // Extract as Vec
    (0..basis_size).map(|i| coeffs_col[[i, 0]]).collect()
}

/// Compute SVD of a complex matrix directly
fn compute_complex_svd(matrix: &DTensor<Complex<f64>, 2>) -> ComplexSVD {
    use mdarray_linalg::{SVD, SVDDecomp};
    use mdarray_linalg_faer::Faer;
    
    // Convert Complex<f64> to faer's c64 for SVD
    let (n, m) = *matrix.shape();
    let mut matrix_c64 = mdarray::DTensor::<num_complex::Complex<f64>, 2>::from_fn([n, m], |idx| {
        matrix[idx]
    });
    
    // Compute complex SVD directly
    let SVDDecomp { u, s, vt } = Faer.svd(&mut matrix_c64)
        .expect("Complex SVD computation failed");
    
    // Extract singular values from first row (they are real even though stored as Complex)
    let min_dim = s.shape().0.min(s.shape().1);
    let s_vec: Vec<f64> = (0..min_dim).map(|i| s[[0, i]].re).collect();
    
    ComplexSVD {
        u,
        s: s_vec,
        vt,
    }
}

/// Solve linear system using precomputed complex SVD
fn solve_complex_svd(svd: &ComplexSVD, values: &[Complex<f64>]) -> Vec<Complex<f64>> {
    use crate::gemm::matmul_par;
    
    let n = values.len();
    let basis_size = svd.vt.shape().1;
    
    // Convert values to column vector
    let values_col = DTensor::<Complex<f64>, 2>::from_fn([n, 1], |idx| values[idx[0]]);
    
    // U^H * values (conjugate transpose)
    // U has shape (n, min_dim), so U^H has shape (min_dim, n)
    let (u_rows, u_cols) = *svd.u.shape();
    let uh = DTensor::<Complex<f64>, 2>::from_fn([u_cols, u_rows], |idx| {
        svd.u[[idx[1], idx[0]]].conj()  // transpose and conjugate
    });
    let uh_values = matmul_par(&uh, &values_col);
    
    // S^{-1} * (U^H * values)
    let min_dim = svd.s.len();
    let s_inv_uh_values = DTensor::<Complex<f64>, 2>::from_fn([min_dim, 1], |idx| {
        let i = idx[0];
        uh_values[[i, 0]] / svd.s[i]
    });
    
    // coeffs = V * (S^{-1} * U^H * values)
    // V = (V^T)^T, and since V^T is real → complex, no conjugate needed
    let v = svd.vt.transpose().to_tensor();  // (m, min_dim)
    let coeffs_col = matmul_par(&v, &s_inv_uh_values);
    
    // Extract as Vec
    (0..basis_size).map(|i| coeffs_col[[i, 0]]).collect()
}

