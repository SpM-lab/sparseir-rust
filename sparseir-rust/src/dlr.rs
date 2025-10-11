//! Discrete Lehmann Representation (DLR)
//!
//! This module provides the Discrete Lehmann Representation (DLR) basis,
//! which represents Green's functions as a linear combination of poles on the
//! real-frequency axis.

use crate::traits::{StatisticsType, Statistics};
use crate::freq::MatsubaraFreq;
use crate::fitter::RealMatrixFitter;
use crate::kernel::CentrosymmKernel;
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
    
    // ========================================================================
    // Public API (generic, user-friendly)
    // ========================================================================
    
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
        use mdarray::{Tensor, DynRank, DTensor, Shape};
        
        let mut gl_shape = vec![];
        gl.shape().with_dims(|dims| {
            gl_shape.extend_from_slice(dims);
        });
        
        let basis_size = gl_shape[dim];
        assert_eq!(basis_size, self.fitmat.shape().0,
                   "IR basis size mismatch: expected {}, got {}", self.fitmat.shape().0, basis_size);
        
        // Move target dimension to position 0
        let gl_dim0 = crate::sampling::movedim(gl, dim, 0);
        
        // Reshape to 2D
        let extra_size = gl_dim0.len() / basis_size;
        let gl_2d_dyn = gl_dim0.reshape(&[basis_size, extra_size][..]).to_tensor();
        
        let gl_2d = DTensor::<T, 2>::from_fn([basis_size, extra_size], |idx| {
            gl_2d_dyn[&[idx[0], idx[1]][..]]
        });
        
        // Fit using fitter's generic 2D method
        let g_dlr_2d = self.fitter.fit_2d_generic::<T>(&gl_2d);
        
        // Reshape back
        let n_poles = self.poles.len();
        let mut g_dlr_shape = vec![n_poles];
        gl_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                g_dlr_shape.push(dims[i]);
            }
        });
        
        let g_dlr_dim0 = g_dlr_2d.into_dyn().reshape(&g_dlr_shape[..]).to_tensor();
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
        
        let n_poles = g_dlr_shape[dim];
        assert_eq!(n_poles, self.poles.len(),
                   "DLR size mismatch: expected {}, got {}", self.poles.len(), n_poles);
        
        // Move target dimension to position 0
        let g_dlr_dim0 = crate::sampling::movedim(g_dlr, dim, 0);
        
        // Reshape to 2D
        let extra_size = g_dlr_dim0.len() / n_poles;
        let g_dlr_2d_dyn = g_dlr_dim0.reshape(&[n_poles, extra_size][..]).to_tensor();
        
        let g_dlr_2d = DTensor::<T, 2>::from_fn([n_poles, extra_size], |idx| {
            g_dlr_2d_dyn[&[idx[0], idx[1]][..]]
        });
        
        // Evaluate using fitter's generic 2D method
        let gl_2d = self.fitter.evaluate_2d_generic::<T>(&g_dlr_2d);
        
        // Reshape back
        let basis_size = self.fitmat.shape().0;
        let mut gl_shape = vec![basis_size];
        g_dlr_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                gl_shape.push(dims[i]);
            }
        });
        
        let gl_dim0 = gl_2d.into_dyn().reshape(&gl_shape[..]).to_tensor();
        crate::sampling::movedim(&gl_dim0, 0, dim)
    }
}

// ============================================================================
// Basis trait implementation for DLR
// ============================================================================

impl<S: StatisticsType> crate::basis_trait::Basis<S> for DiscreteLehmannRepresentation<S>
where
    S: 'static,
{
    fn beta(&self) -> f64 {
        self.beta
    }
    
    fn wmax(&self) -> f64 {
        self.wmax
    }
    
    fn lambda(&self) -> f64 {
        self.beta * self.wmax
    }
    
    fn size(&self) -> usize {
        self.poles.len()
    }
    
    fn accuracy(&self) -> f64 {
        self.accuracy
    }
    
    fn significance(&self) -> Vec<f64> {
        // All poles are equally significant in DLR
        vec![1.0; self.poles.len()]
    }
    
    fn default_tau_sampling_points(&self) -> Vec<f64> {
        // TODO: Could use the original IR basis sampling points
        // For now, return empty - not well-defined for DLR
        vec![]
    }
    
    fn default_matsubara_sampling_points(&self, _positive_only: bool) -> Vec<crate::freq::MatsubaraFreq<S>> {
        // TODO: Could use the original IR basis sampling points
        // For now, return empty - not well-defined for DLR
        vec![]
    }
    
    fn evaluate_tau(&self, tau: &[f64]) -> mdarray::DTensor<f64, 2> {
        use mdarray::DTensor;
        use crate::kernel::LogisticKernel;
        
        let n_points = tau.len();
        let n_poles = self.poles.len();
        
        // Evaluate TauPoles basis functions: u_i(τ) = -K(x, y_i)
        // where x = 2τ/β - 1, y_i = pole_i/ωmax
        let lambda = self.beta * self.wmax;
        let kernel = LogisticKernel::new(lambda);
        
        DTensor::<f64, 2>::from_fn([n_points, n_poles], |idx| {
            let tau_val = tau[idx[0]];
            let pole = self.poles[idx[1]];
            
            // Normalize tau to [0, β) with periodicity (for extended range support)
            let (tau_norm, sign) = match S::STATISTICS {
                Statistics::Fermionic => {
                    // Anti-periodic: G(τ + β) = -G(τ)
                    if tau_val < 0.0 {
                        (tau_val + self.beta, -1.0)
                    } else if tau_val > self.beta {
                        (tau_val - self.beta, -1.0)
                    } else {
                        (tau_val, 1.0)
                    }
                },
                Statistics::Bosonic => {
                    // Periodic: G(τ + β) = G(τ)
                    if tau_val < 0.0 {
                        (tau_val + self.beta, 1.0)
                    } else if tau_val > self.beta {
                        (tau_val - self.beta, 1.0)
                    } else {
                        (tau_val, 1.0)
                    }
                },
            };
            
            // Compute kernel value
            let x = 2.0 * tau_norm / self.beta - 1.0;
            let y = pole / self.wmax;
            
            sign * (-kernel.compute(x, y))
        })
    }
    
    fn evaluate_matsubara(&self, freqs: &[crate::freq::MatsubaraFreq<S>]) -> mdarray::DTensor<num_complex::Complex<f64>, 2> {
        use mdarray::DTensor;
        use num_complex::Complex;
        
        let n_points = freqs.len();
        let n_poles = self.poles.len();
        
        // Evaluate MatsubaraPoles basis functions
        DTensor::<Complex<f64>, 2>::from_fn([n_points, n_poles], |idx| {
            let freq = &freqs[idx[0]];
            let pole = self.poles[idx[1]];
            
            // iν = i * π * (2n + ζ) / β
            let iv = freq.value_imaginary(self.beta);
            
            match S::STATISTICS {
                Statistics::Fermionic => {
                    // u_i(iν) = 1 / (iν - pole_i)
                    Complex::new(1.0, 0.0) / (iv - Complex::new(pole, 0.0))
                },
                Statistics::Bosonic => {
                    // u_i(iν) = tanh(β·pole_i/2) / (iν - pole_i)
                    let reg = (self.beta * pole / 2.0).tanh();
                    Complex::new(reg, 0.0) / (iv - Complex::new(pole, 0.0))
                },
            }
        })
    }
    
    fn evaluate_omega(&self, omega: &[f64]) -> mdarray::DTensor<f64, 2> {
        use mdarray::DTensor;
        
        let n_points = omega.len();
        let n_poles = self.poles.len();
        
        // For DLR, this could return identity or delta function
        // For now, return zeros (not well-defined)
        DTensor::<f64, 2>::from_fn([n_points, n_poles], |_idx| 0.0)
    }
    
    fn default_omega_sampling_points(&self) -> Vec<f64> {
        // DLR poles ARE the omega sampling points
        self.poles.clone()
    }
}
