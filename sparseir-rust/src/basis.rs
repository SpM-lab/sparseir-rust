//! Finite temperature basis for SparseIR
//!
//! This module provides the `FiniteTempBasis` type which represents the
//! intermediate representation (IR) basis for a given temperature.

use ndarray::Array1;
use std::sync::Arc;

use crate::kernel::{KernelProperties, CentrosymmKernel, LogisticKernel};
use crate::poly::PiecewiseLegendrePolyVector;
use crate::polyfourier::PiecewiseLegendreFTVector;
use crate::sve::{SVEResult, compute_sve, TworkType};
use crate::traits::{StatisticsType, Fermionic, Bosonic};

/// Finite temperature basis for imaginary time/frequency Green's functions
///
/// For a continuation kernel `K` from real frequencies `ω ∈ [-ωmax, ωmax]` to
/// imaginary time `τ ∈ [0, β]`, this type stores the truncated singular
/// value expansion or IR basis:
///
/// ```text
/// K(τ, ω) ≈ sum(u[l](τ) * s[l] * v[l](ω) for l in 1:L)
/// ```
///
/// This basis is inferred from a reduced form by appropriate scaling of
/// the variables.
///
/// # Type Parameters
///
/// * `K` - Kernel type implementing `KernelProperties + CentrosymmKernel`
/// * `S` - Statistics type (`Fermionic` or `Bosonic`)
#[derive(Clone)]
pub struct FiniteTempBasis<K, S>
where
    K: KernelProperties + CentrosymmKernel + Clone + 'static,
    S: StatisticsType,
{
    /// The kernel used to construct this basis
    pub kernel: K,
    
    /// The SVE result (in scaled variables)
    pub sve_result: Arc<SVEResult>,
    
    /// Accuracy of the basis (relative error)
    pub accuracy: f64,
    
    /// Inverse temperature β
    pub beta: f64,
    
    /// Left singular functions on imaginary time axis τ ∈ [0, β]
    pub u: PiecewiseLegendrePolyVector,
    
    /// Right singular functions on real frequency axis ω ∈ [-ωmax, ωmax]
    pub v: PiecewiseLegendrePolyVector,
    
    /// Singular values
    pub s: Array1<f64>,
    
    /// Left singular functions on Matsubara frequency axis (Fourier transform of u)
    pub uhat: PiecewiseLegendreFTVector<S>,
    
    /// Full uhat (before truncation to basis size)
    pub uhat_full: PiecewiseLegendreFTVector<S>,
    
    _phantom: std::marker::PhantomData<S>,
}

impl<K, S> FiniteTempBasis<K, S>
where
    K: KernelProperties + CentrosymmKernel + Clone + 'static,
    S: StatisticsType,
{
    /// Create a new FiniteTempBasis
    ///
    /// # Arguments
    ///
    /// * `kernel` - Kernel implementing `KernelProperties + CentrosymmKernel`
    /// * `beta` - Inverse temperature (β > 0)
    /// * `epsilon` - Accuracy parameter (optional, defaults to NaN for auto)
    /// * `max_size` - Maximum number of basis functions (optional)
    ///
    /// # Returns
    ///
    /// A new FiniteTempBasis
    pub fn new(
        kernel: K,
        beta: f64,
        epsilon: Option<f64>,
        max_size: Option<usize>,
    ) -> Self {
        // Validate inputs
        if beta <= 0.0 {
            panic!("Inverse temperature beta must be positive, got {}", beta);
        }
        
        // Compute SVE
        let epsilon_value = epsilon.unwrap_or(f64::NAN);
        let sve_result = compute_sve(
            kernel.clone(),
            epsilon_value,
            None,  // cutoff
            max_size.map(|s| s as usize),
            TworkType::Auto,
        );
        
        Self::from_sve_result(kernel, beta, sve_result, epsilon, max_size)
    }
    
    /// Create basis from existing SVE result
    ///
    /// This is useful when you want to reuse the same SVE computation
    /// for both fermionic and bosonic bases.
    pub fn from_sve_result(
        kernel: K,
        beta: f64,
        sve_result: SVEResult,
        epsilon: Option<f64>,
        max_size: Option<usize>,
    ) -> Self {
        // Get truncated u, s, v from SVE result
        let (u_sve, s_sve, v_sve) = sve_result.part(epsilon, max_size);
        
        // Calculate accuracy
        let accuracy = if sve_result.s.len() > s_sve.len() {
            sve_result.s[s_sve.len()] / sve_result.s[0]
        } else {
            sve_result.s[sve_result.s.len() - 1] / sve_result.s[0]
        };
        
        // Get kernel parameters
        let lambda = kernel.lambda();
        let omega_max = lambda / beta;
        
        // Scale polynomials to new variables
        // tau = β/2 * (x + 1), w = ωmax * y
        
        // Transform u: x ∈ [-1, 1] → τ ∈ [0, β]
        let u_knots: Vec<f64> = u_sve.get_polys()[0].knots.iter()
            .map(|&x| beta / 2.0 * (x + 1.0))
            .collect();
        let u_delta_x: Vec<f64> = u_sve.get_polys()[0].delta_x.iter()
            .map(|&dx| beta / 2.0 * dx)
            .collect();
        let u_symm: Vec<i32> = u_sve.get_polys().iter()
            .map(|p| p.symm)
            .collect();
        
        let u = u_sve.rescale_domain(u_knots, Some(u_delta_x), Some(u_symm));
        
        // Transform v: y ∈ [-1, 1] → ω ∈ [-ωmax, ωmax]
        let v_knots: Vec<f64> = v_sve.get_polys()[0].knots.iter()
            .map(|&y| omega_max * y)
            .collect();
        let v_delta_x: Vec<f64> = v_sve.get_polys()[0].delta_x.iter()
            .map(|&dy| omega_max * dy)
            .collect();
        let v_symm: Vec<i32> = v_sve.get_polys().iter()
            .map(|p| p.symm)
            .collect();
        
        let v = v_sve.rescale_domain(v_knots, Some(v_delta_x), Some(v_symm));
        
        // Scale singular values
        // s_scaled = sqrt(β/2 * ωmax) * ωmax^(-ypower) * s_sve
        let ypower = kernel.ypower();
        let scale_factor = (beta / 2.0 * omega_max).sqrt() 
                         * omega_max.powi(-ypower);
        let s = s_sve.mapv(|x| scale_factor * x);
        
        // Construct uhat (Fourier transform of u)
        // HACK: Fourier transforms only work on unit interval, so we scale the data
        let uhat_base_full = sve_result.u.scale_data(beta.sqrt());
        let conv_rad = kernel.conv_radius();
        
        // Create statistics instance - we need a value of type S
        // For Fermionic: S = Fermionic, for Bosonic: S = Bosonic
        let stat_marker = unsafe { std::mem::zeroed::<S>() };
        
        let uhat_full = PiecewiseLegendreFTVector::<S>::from_poly_vector(
            &uhat_base_full,
            stat_marker,
            Some(conv_rad),
        );
        
        // Truncate uhat to basis size
        let uhat_polyvec: Vec<_> = uhat_full.polyvec.iter()
            .take(s.len())
            .cloned()
            .collect();
        let uhat = PiecewiseLegendreFTVector::from_vector(uhat_polyvec);
        
        Self {
            kernel,
            sve_result: Arc::new(sve_result),
            accuracy,
            beta,
            u,
            v,
            s,
            uhat,
            uhat_full,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Get the size of the basis (number of basis functions)
    pub fn size(&self) -> usize {
        self.s.len()
    }
    
    /// Get the cutoff parameter Λ = β * ωmax
    pub fn lambda(&self) -> f64 {
        self.kernel.lambda()
    }
    
    /// Get the frequency cutoff ωmax
    pub fn omega_max(&self) -> f64 {
        self.lambda() / self.beta
    }
    
    /// Get significance of each singular value (s[i] / s[0])
    pub fn significance(&self) -> Array1<f64> {
        let s0 = self.s[0];
        self.s.mapv(|s| s / s0)
    }
}

/// Type alias for fermionic basis with LogisticKernel
pub type FermionicBasis = FiniteTempBasis<LogisticKernel, Fermionic>;

/// Type alias for bosonic basis with LogisticKernel
pub type BosonicBasis = FiniteTempBasis<LogisticKernel, Bosonic>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basis_construction() {
        let beta = 10.0;
        let omega_max = 1.0;
        let epsilon = 1e-6;
        
        let kernel = LogisticKernel::new(beta * omega_max);
        let basis = FermionicBasis::new(kernel, beta, Some(epsilon), None);
        
        assert_eq!(basis.beta, beta);
        assert!((basis.omega_max() - omega_max).abs() < 1e-10);
        assert!(basis.size() > 0);
        assert!(basis.accuracy > 0.0);
        assert!(basis.accuracy < epsilon);
    }
    
    #[test]
    #[should_panic(expected = "beta must be positive")]
    fn test_negative_beta() {
        let kernel = LogisticKernel::new(1.0);
        let _ = FermionicBasis::new(kernel, -1.0, None, None);
    }
}

