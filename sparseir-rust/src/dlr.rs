//! Discrete Lehmann Representation (DLR)
//!
//! This module provides the Discrete Lehmann Representation (DLR) basis,
//! which represents Green's functions as a linear combination of poles on the
//! real-frequency axis.

use crate::traits::{StatisticsType, Statistics};
use crate::freq::MatsubaraFreq;
use crate::fitter::RealMatrixFitter;
use mdarray::DTensor;
use num_complex::Complex;
use std::marker::PhantomData;

/// Generic single-pole Green's function at imaginary time τ
///
/// Computes G(τ) for either fermionic or bosonic statistics based on the type parameter S.
///
/// # Type Parameters
/// * `S` - Statistics type (Fermionic or Bosonic)
///
/// # Arguments
/// * `tau` - Imaginary time (can be outside [0, β))
/// * `omega` - Pole position (real frequency)
/// * `beta` - Inverse temperature
///
/// # Returns
/// Real-valued Green's function G(τ)
///
/// # Example
/// ```ignore
/// use sparseir_rust::traits::Fermionic;
/// let g_f = gtau_single_pole::<Fermionic>(0.5, 5.0, 1.0);
/// 
/// use sparseir_rust::traits::Bosonic;
/// let g_b = gtau_single_pole::<Bosonic>(0.5, 5.0, 1.0);
/// ```
pub fn gtau_single_pole<S: StatisticsType>(tau: f64, omega: f64, beta: f64) -> f64 {
    match S::STATISTICS {
        Statistics::Fermionic => fermionic_single_pole(tau, omega, beta),
        Statistics::Bosonic => bosonic_single_pole(tau, omega, beta),
    }
}

/// Compute fermionic single-pole Green's function at imaginary time τ
///
/// Evaluates G(τ) = -exp(-ω×τ) / (1 + exp(-β×ω)) for a single pole at frequency ω.
///
/// Supports extended τ ranges with anti-periodic boundary conditions:
/// - G(τ + β) = -G(τ) (fermionic anti-periodicity)
/// - Valid for τ ∈ (-β, 2β)
///
/// # Arguments
/// * `tau` - Imaginary time (can be outside [0, β))
/// * `omega` - Pole position (real frequency)
/// * `beta` - Inverse temperature
///
/// # Returns
/// Real-valued Green's function G(τ)
///
/// # Example
/// ```ignore
/// let beta = 1.0;
/// let omega = 5.0;
/// let tau = 0.5 * beta;
/// let g = fermionic_single_pole(tau, omega, beta);
/// ```
pub fn fermionic_single_pole(tau: f64, omega: f64, beta: f64) -> f64 {
    // Normalize τ to [0, β) and track sign from anti-periodicity
    // G(τ + β) = -G(τ) for fermions
    // Note: β is interpreted as β- (left limit), so tau > beta for extension
    let (tau_normalized, sign) = if tau < 0.0 {
        // -β < τ < 0: G(τ) = -G(τ + β)
        (tau + beta, -1.0)
    } else if tau > beta {
        // β < τ < 2β: G(τ) = -G(τ - β)  
        (tau - beta, -1.0)
    } else {
        // 0 ≤ τ ≤ β: normal range (β interpreted as β-)
        (tau, 1.0)
    };
    
    sign * (-(-omega * tau_normalized).exp() / (1.0 + (-beta * omega).exp()))
}

/// Compute bosonic single-pole Green's function at imaginary time τ
///
/// Evaluates G(τ) = exp(-ω×τ) / (1 - exp(-β×ω)) for a single pole at frequency ω.
///
/// Supports extended τ ranges with periodic boundary conditions:
/// - G(τ + β) = G(τ) (bosonic periodicity)
/// - Valid for τ ∈ (-β, 2β)
///
/// # Arguments
/// * `tau` - Imaginary time (can be outside [0, β))
/// * `omega` - Pole position (real frequency)
/// * `beta` - Inverse temperature
///
/// # Returns
/// Real-valued Green's function G(τ)
///
/// # Example
/// ```ignore
/// let beta = 1.0;
/// let omega = 5.0;
/// let tau = 0.5 * beta;
/// let g = bosonic_single_pole(tau, omega, beta);
/// ```
pub fn bosonic_single_pole(tau: f64, omega: f64, beta: f64) -> f64 {
    // Normalize τ to [0, β) using periodicity
    // G(τ + β) = G(τ) for bosons
    // Note: β is interpreted as β- (left limit), so tau > beta for extension
    let tau_normalized = if tau < 0.0 {
        // -β < τ < 0: G(τ) = G(τ + β)
        tau + beta
    } else if tau > beta {
        // β < τ < 2β: G(τ) = G(τ - β)
        tau - beta
    } else {
        // 0 ≤ τ ≤ β: normal range (β interpreted as β-)
        tau
    };
    
    (-omega * tau_normalized).exp() / (1.0 - (-beta * omega).exp())
}

/// Generic single-pole Green's function at Matsubara frequency
///
/// Computes G(iωn) = 1/(iωn - ω) for a single pole at frequency ω.
///
/// # Type Parameters
/// * `S` - Statistics type (Fermionic or Bosonic)
///
/// # Arguments
/// * `matsubara_freq` - Matsubara frequency
/// * `omega` - Pole position (real frequency)
/// * `beta` - Inverse temperature
///
/// # Returns
/// Complex-valued Green's function G(iωn)
pub fn giwn_single_pole<S: StatisticsType>(
    matsubara_freq: &MatsubaraFreq<S>,
    omega: f64,
    beta: f64,
) -> Complex<f64> {
    // G(iωn) = 1/(iωn - ω)
    let wn = matsubara_freq.value(beta);
    let denominator = Complex::new(0.0, 1.0) * wn - Complex::new(omega, 0.0);
    Complex::new(1.0, 0.0) / denominator
}

// ============================================================================
// Discrete Lehmann Representation
// ============================================================================

/// Discrete Lehmann Representation (DLR)
///
/// The DLR is a variant of the IR basis based on a "sketching" of the analytic
/// continuation kernel K. Instead of using singular value expansion, it represents
/// Green's functions as a linear combination of poles on the real-frequency axis:
///
/// ```text
/// G(iν) = Σ_i a[i] * reg[i] / (iν - ω[i])
/// ```
///
/// where:
/// - `ω[i]` are pole positions on the real axis
/// - `a[i]` are expansion coefficients  
/// - `reg[i]` are regularization factors (1 for fermions, tanh(βω/2) for bosons)
///
/// # Type Parameters
/// * `S` - Statistics type (Fermionic or Bosonic)
pub struct DiscreteLehmannRepresentation<S: StatisticsType> {
    /// Pole positions on the real-frequency axis ω ∈ [-ωmax, ωmax]
    pub poles: Vec<f64>,
    
    /// Inverse temperature β
    pub beta: f64,
    
    /// Maximum frequency ωmax
    pub wmax: f64,
    
    /// Accuracy of the representation
    pub accuracy: f64,
    
    /// Fitting matrix from IR: fitmat = -s · V(poles)
    /// Used for to_IR transformation
    fitmat: DTensor<f64, 2>,
    
    /// Fitter for from_IR transformation (uses SVD of fitmat)
    fitter: RealMatrixFitter,
    
    /// Marker for statistics type
    _phantom: PhantomData<S>,
}

impl<S: StatisticsType> DiscreteLehmannRepresentation<S> {
    /// Create DLR from IR basis with custom poles
    ///
    /// # Arguments
    /// * `basis` - The IR basis to construct DLR from
    /// * `poles` - Pole positions on the real-frequency axis
    ///
    /// # Returns
    /// A new DLR representation
    pub fn with_poles(
        basis: &impl crate::basis_trait::Basis<S>,
        poles: Vec<f64>,
    ) -> Self
    where
        S: 'static,
    {
        let beta = basis.beta();
        let wmax = basis.wmax();
        let accuracy = basis.accuracy();
        
        // Compute fitting matrix: fitmat = -s · V(poles)
        // This transforms DLR coefficients to IR coefficients
        let v_at_poles = basis.evaluate_omega(&poles);  // shape: [n_poles, basis_size]
        let s = basis.significance();  // Normalized by first singular value
        let first_s = s[0];
        
        let basis_size = basis.size();
        let n_poles = poles.len();
        
        // fitmat[l, i] = -s[l] * s[0] * V_l(pole[i])
        let fitmat = DTensor::<f64, 2>::from_fn([basis_size, n_poles], |idx| {
            let l = idx[0];
            let i = idx[1];
            -s[l] * first_s * v_at_poles[[i, l]]
        });
        
        // Create fitter for from_IR (inverse operation)
        let fitter = RealMatrixFitter::new(fitmat.clone());
        
        Self {
            poles,
            beta,
            wmax,
            accuracy,
            fitmat,
            fitter,
            _phantom: PhantomData,
        }
    }
    
    /// Create DLR from IR basis with default pole locations
    ///
    /// Uses the default omega sampling points from the basis.
    ///
    /// # Arguments
    /// * `basis` - The IR basis to construct DLR from
    ///
    /// # Returns
    /// A new DLR representation with default poles
    pub fn new(basis: &impl crate::basis_trait::Basis<S>) -> Self
    where
        S: 'static,
    {
        let poles = basis.default_omega_sampling_points();
        Self::with_poles(basis, poles)
    }
    
    /// Convert IR coefficients to DLR coefficients (generic over real/complex)
    ///
    /// # Type Parameters
    /// * `T` - Element type (f64 or Complex<f64>)
    ///
    /// # Arguments
    /// * `gl` - IR basis coefficients
    ///
    /// # Returns
    /// DLR coefficients (pole weights)
    pub fn from_IR<T>(&self, gl: &[T]) -> Vec<T>
    where
        T: num_complex::ComplexFloat + faer_traits::ComplexField + From<f64> + Copy + Default + 'static,
    {
        use std::any::TypeId;
        
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // Real case
            let gl_f64 = unsafe { &*(gl as *const [T] as *const [f64]) };
            let result = self.fitter.fit(gl_f64);
            unsafe { std::mem::transmute::<Vec<f64>, Vec<T>>(result) }
        } else if TypeId::of::<T>() == TypeId::of::<Complex<f64>>() {
            // Complex case
            let gl_complex = unsafe { &*(gl as *const [T] as *const [Complex<f64>]) };
            let result = self.fitter.fit_complex(gl_complex);
            unsafe { std::mem::transmute::<Vec<Complex<f64>>, Vec<T>>(result) }
        } else {
            panic!("Unsupported type for from_IR");
        }
    }
    
    /// Convert DLR coefficients to IR coefficients (generic over real/complex)
    ///
    /// # Type Parameters
    /// * `T` - Element type (f64 or Complex<f64>)
    ///
    /// # Arguments
    /// * `g_dlr` - DLR coefficients (pole weights)
    ///
    /// # Returns
    /// IR basis coefficients
    pub fn to_IR<T>(&self, g_dlr: &[T]) -> Vec<T>
    where
        T: num_complex::ComplexFloat + faer_traits::ComplexField + From<f64> + Copy + Default + 'static,
    {
        use std::any::TypeId;
        
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // Real case
            let g_dlr_f64 = unsafe { &*(g_dlr as *const [T] as *const [f64]) };
            let result = self.fitter.evaluate(g_dlr_f64);
            unsafe { std::mem::transmute::<Vec<f64>, Vec<T>>(result) }
        } else if TypeId::of::<T>() == TypeId::of::<Complex<f64>>() {
            // Complex case
            let g_dlr_complex = unsafe { &*(g_dlr as *const [T] as *const [Complex<f64>]) };
            let result = self.fitter.evaluate_complex(g_dlr_complex);
            unsafe { std::mem::transmute::<Vec<Complex<f64>>, Vec<T>>(result) }
        } else {
            panic!("Unsupported type for to_IR");
        }
    }
    
    /// Convert IR coefficients to DLR (N-dimensional, generic over real/complex)
    ///
    /// # Type Parameters
    /// * `T` - Element type (f64 or Complex<f64>)
    ///
    /// # Arguments
    /// * `gl` - IR coefficients as N-D tensor
    /// * `dim` - Dimension along which to transform
    ///
    /// # Returns
    /// DLR coefficients as N-D tensor
    pub fn from_IR_nd<T>(&self, gl: &mdarray::Tensor<T, mdarray::DynRank>, dim: usize) -> mdarray::Tensor<T, mdarray::DynRank>
    where
        T: num_complex::ComplexFloat + faer_traits::ComplexField + From<f64> + Copy + Default + 'static,
    {
        // Use movedim pattern from sampling.rs
        use mdarray::{Tensor, DynRank, DTensor, Shape};
        
        let mut gl_shape = vec![];
        gl.shape().with_dims(|dims| {
            gl_shape.extend_from_slice(dims);
        });
        
        let basis_size_original = gl_shape[dim];
        assert_eq!(
            basis_size_original,
            self.fitmat.shape().0,
            "IR basis size mismatch: expected {}, got {}",
            self.fitmat.shape().0,
            basis_size_original
        );
        
        // Move target dimension to position 0
        let gl_dim0 = crate::sampling::movedim(gl, dim, 0);
        
        // After movedim, the dimension is now at position 0
        let basis_size = gl_shape[dim];  // Size of the transformed dimension
        let extra_size: usize = gl_dim0.len() / basis_size;
        let gl_2d_dyn = gl_dim0.reshape(&[basis_size, extra_size][..]).to_tensor();
        
        // Convert to DTensor and fit (dispatch based on type)
        use std::any::TypeId;
        
        let g_dlr_2d: Tensor<T, DynRank> = if TypeId::of::<T>() == TypeId::of::<f64>() {
            // Real case
            let gl_2d = DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| {
                unsafe { *(&gl_2d_dyn[&[idx[0], idx[1]][..]] as *const T as *const f64) }
            });
            let result = self.fitter.fit_2d(&gl_2d);
            unsafe { std::mem::transmute::<Tensor<f64, DynRank>, Tensor<T, DynRank>>(result.into_dyn()) }
        } else if TypeId::of::<T>() == TypeId::of::<Complex<f64>>() {
            // Complex case
            let gl_2d = DTensor::<Complex<f64>, 2>::from_fn([basis_size, extra_size], |idx| {
                unsafe { *(&gl_2d_dyn[&[idx[0], idx[1]][..]] as *const T as *const Complex<f64>) }
            });
            let result = self.fitter.fit_complex_2d(&gl_2d);
            unsafe { std::mem::transmute::<Tensor<Complex<f64>, DynRank>, Tensor<T, DynRank>>(result.into_dyn()) }
        } else {
            panic!("Unsupported type for from_IR_nd");
        };
        
        // Reshape back
        let n_poles = self.poles.len();
        let mut g_dlr_shape = vec![n_poles];
        gl_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                g_dlr_shape.push(dims[i]);
            }
        });
        
        let g_dlr_dim0 = g_dlr_2d.reshape(&g_dlr_shape[..]).to_tensor();
        
        // Move dimension 0 back to original position
        crate::sampling::movedim(&g_dlr_dim0, 0, dim)
    }
    
    /// Convert DLR coefficients to IR (N-dimensional, generic over real/complex)
    ///
    /// # Type Parameters
    /// * `T` - Element type (f64 or Complex<f64>)
    ///
    /// # Arguments
    /// * `g_dlr` - DLR coefficients as N-D tensor
    /// * `dim` - Dimension along which to transform
    ///
    /// # Returns
    /// IR coefficients as N-D tensor
    pub fn to_IR_nd<T>(&self, g_dlr: &mdarray::Tensor<T, mdarray::DynRank>, dim: usize) -> mdarray::Tensor<T, mdarray::DynRank>
    where
        T: num_complex::ComplexFloat + faer_traits::ComplexField + From<f64> + Copy + Default + 'static,
    {
        use mdarray::{Tensor, DynRank, DTensor, Shape};
        
        let mut g_dlr_shape = vec![];
        g_dlr.shape().with_dims(|dims| {
            g_dlr_shape.extend_from_slice(dims);
        });
        
        let n_poles_original = g_dlr_shape[dim];
        assert_eq!(
            n_poles_original,
            self.poles.len(),
            "DLR size mismatch: expected {}, got {}",
            self.poles.len(),
            n_poles_original
        );
        
        // Move target dimension to position 0
        let g_dlr_dim0 = crate::sampling::movedim(g_dlr, dim, 0);
        
        // After movedim, the dimension is now at position 0
        let n_poles = g_dlr_shape[dim];  // Size of the transformed dimension
        let extra_size: usize = g_dlr_dim0.len() / n_poles;
        let g_dlr_2d_dyn = g_dlr_dim0.reshape(&[n_poles, extra_size][..]).to_tensor();
        
        // Convert to DTensor and evaluate (dispatch based on type)
        use std::any::TypeId;
        use crate::gemm::matmul_par;
        
        let gl_2d: Tensor<T, DynRank> = if TypeId::of::<T>() == TypeId::of::<f64>() {
            // Real case: gl = fitmat * g_dlr
            let g_dlr_2d = DTensor::<f64, 2>::from_fn([n_poles, extra_size], |idx| {
                unsafe { *(&g_dlr_2d_dyn[&[idx[0], idx[1]][..]] as *const T as *const f64) }
            });
            let result = matmul_par(&self.fitmat, &g_dlr_2d);
            unsafe { std::mem::transmute::<Tensor<f64, DynRank>, Tensor<T, DynRank>>(result.into_dyn()) }
        } else if TypeId::of::<T>() == TypeId::of::<Complex<f64>>() {
            // Complex case: gl = fitmat * g_dlr (fitmat is real, g_dlr is complex)
            let g_dlr_2d = DTensor::<Complex<f64>, 2>::from_fn([n_poles, extra_size], |idx| {
                unsafe { *(&g_dlr_2d_dyn[&[idx[0], idx[1]][..]] as *const T as *const Complex<f64>) }
            });
            
            // Split complex into real/imaginary and multiply separately
            let g_dlr_re = DTensor::<f64, 2>::from_fn([n_poles, extra_size], |idx| g_dlr_2d[idx].re);
            let g_dlr_im = DTensor::<f64, 2>::from_fn([n_poles, extra_size], |idx| g_dlr_2d[idx].im);
            
            let gl_re = matmul_par(&self.fitmat, &g_dlr_re);
            let gl_im = matmul_par(&self.fitmat, &g_dlr_im);
            
            // Combine to complex
            let gl_complex = DTensor::<Complex<f64>, 2>::from_fn(*gl_re.shape(), |idx| {
                Complex::new(gl_re[idx], gl_im[idx])
            });
            unsafe { std::mem::transmute::<Tensor<Complex<f64>, DynRank>, Tensor<T, DynRank>>(gl_complex.into_dyn()) }
        } else {
            panic!("Unsupported type for to_IR_nd");
        };
        
        // Reshape back
        let basis_size = self.fitmat.shape().0;
        let mut gl_shape = vec![basis_size];
        g_dlr_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                gl_shape.push(dims[i]);
            }
        });
        
        let gl_dim0 = gl_2d.reshape(&gl_shape[..]).to_tensor();
        
        // Move dimension 0 back to original position
        crate::sampling::movedim(&gl_dim0, 0, dim)
    }
}
