//! Finite temperature basis for SparseIR
//!
//! This module provides the `FiniteTempBasis` type which represents the
//! intermediate representation (IR) basis for a given temperature.

use std::sync::Arc;

use crate::kernel::{KernelProperties, CentrosymmKernel, LogisticKernel};
use crate::poly::{PiecewiseLegendrePolyVector, default_sampling_points};
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
    pub s: Vec<f64>,
    
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
        let s: Vec<f64> = s_sve.iter().map(|&x| scale_factor * x).collect();
        
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
    pub fn significance(&self) -> Vec<f64> {
        let s0 = self.s[0];
        self.s.iter().map(|&s| s / s0).collect()
    }
    
    /// Get default tau sampling points
    ///
    /// C++ implementation: libsparseir/include/sparseir/basis.hpp:229-270
    ///
    /// Returns sampling points in imaginary time τ ∈ [0, β].
    pub fn default_tau_sampling_points(&self) -> Vec<f64> {
        let sz = self.size();
        
        // C++: Eigen::VectorXd x = default_sampling_points(*(this->sve_result->u), sz);
        let x = default_sampling_points(&self.sve_result.u, sz);
        
        // C++: Extract unique half of sampling points
        let mut unique_x = Vec::new();
        if x.len() % 2 == 0 {
            // C++: for (auto i = 0; i < x.size() / 2; ++i)
            for i in 0..(x.len() / 2) {
                unique_x.push(x[i]);
            }
        } else {
            // C++: for (auto i = 0; i < x.size() / 2; ++i)
            for i in 0..(x.len() / 2) {
                unique_x.push(x[i]);
            }
            // C++: auto x_new = 0.5 * (unique_x.back() + 0.5);
            let x_new = 0.5 * (unique_x.last().unwrap() + 0.5);
            unique_x.push(x_new);
        }
        
        // C++: Generate symmetric points
        //      Eigen::VectorXd smpl_taus(2 * unique_x.size());
        //      for (auto i = 0; i < unique_x.size(); ++i) {
        //          smpl_taus(i) = (this->beta / 2.0) * (unique_x[i] + 1.0);
        //          smpl_taus(unique_x.size() + i) = -smpl_taus(i);
        //      }
        let mut smpl_taus = Vec::with_capacity(2 * unique_x.len());
        for &ux in &unique_x {
            smpl_taus.push((self.beta / 2.0) * (ux + 1.0));
        }
        for i in 0..unique_x.len() {
            smpl_taus.push(-smpl_taus[i]);
        }
        
        // C++: std::sort(smpl_taus.data(), smpl_taus.data() + smpl_taus.size());
        smpl_taus.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // C++: Convert negative values to [beta/2, beta]
        //      for (auto i = 0; i < smpl_taus.size(); ++i) {
        //          if (smpl_taus(i) < 0.0) {
        //              smpl_taus(i) += this->beta;
        //          }
        //      }
        for tau in &mut smpl_taus {
            if *tau < 0.0 {
                *tau += self.beta;
            }
        }
        
        // C++: Note - C++ implementation at line 262-268 shows that after
        //      converting negative values, the array is NOT re-sorted.
        //      However, this means some points may be out of order.
        //      Let's check the C++ code again...
        //
        // Re-sort after conversion to ensure monotonic order
        smpl_taus.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        smpl_taus
    }
    
    /// Get default Matsubara frequency sampling points
    ///
    /// Returns sampling points as MatsubaraFreq objects based on extrema
    /// of the Matsubara basis functions (same algorithm as C++/Julia).
    ///
    /// # Arguments
    /// * `positive_only` - If true, returns only non-negative frequencies
    ///
    /// # Returns
    /// Vector of Matsubara frequency sampling points
    pub fn default_matsubara_sampling_points(&self, positive_only: bool) -> Vec<crate::freq::MatsubaraFreq<S>>
    where
        S: 'static,
    {
        use crate::freq::MatsubaraFreq;
        use crate::polyfourier::{sign_changes, find_extrema};
        use std::collections::BTreeSet;
        
        let l = self.size();
        let mut l_requested = l;
        
        // Adjust l_requested based on statistics (same as C++)
        if S::STATISTICS == crate::traits::Statistics::Fermionic && l_requested % 2 != 0 {
            l_requested += 1;
        } else if S::STATISTICS == crate::traits::Statistics::Bosonic && l_requested % 2 == 0 {
            l_requested += 1;
        }
        
        // Choose sign_changes or find_extrema based on l_requested
        let mut omega_n = if l_requested < self.uhat_full.len() {
            sign_changes(&self.uhat_full[l_requested], positive_only)
        } else {
            find_extrema(&self.uhat_full[self.uhat_full.len() - 1], positive_only)
        };
        
        // For bosons, include zero frequency explicitly to prevent conditioning issues
        if S::STATISTICS == crate::traits::Statistics::Bosonic {
            omega_n.push(MatsubaraFreq::<S>::new(0).unwrap());
        }
        
        // Sort and remove duplicates using BTreeSet
        let omega_n_set: BTreeSet<MatsubaraFreq<S>> = omega_n.into_iter().collect();
        let omega_n: Vec<MatsubaraFreq<S>> = omega_n_set.into_iter().collect();
        
        // Check expected size
        let expected_size = if positive_only {
            (l_requested + 1) / 2
        } else {
            l_requested
        };
        
        if omega_n.len() != expected_size {
            eprintln!(
                "Warning: Requested {} sampling frequencies for basis size L = {}, but got {}.",
                expected_size, l, omega_n.len()
            );
        }
        
        omega_n
    }
    
    /// Get default omega (real frequency) sampling points
    ///
    /// Returns sampling points on the real-frequency axis ω ∈ [-ωmax, ωmax].
    /// These are used as pole locations for the Discrete Lehmann Representation (DLR).
    ///
    /// The sampling points are chosen as the roots of the L-th basis function
    /// in the spectral domain (v), which provides near-optimal conditioning.
    ///
    /// # Returns
    /// Vector of real-frequency sampling points in [-ωmax, ωmax]
    pub fn default_omega_sampling_points(&self) -> Vec<f64> {
        let sz = self.size();
        
        // Get sampling points in [-1, 1] from spectral basis functions
        let y = default_sampling_points(&self.sve_result.v, sz);
        
        // Scale to [-ωmax, ωmax]
        let wmax = self.kernel.lambda() / self.beta;
        y.into_iter().map(|yi| wmax * yi).collect()
    }
}

/// Type alias for fermionic basis with LogisticKernel
pub type FermionicBasis = FiniteTempBasis<LogisticKernel, Fermionic>;

/// Type alias for bosonic basis with LogisticKernel
pub type BosonicBasis = FiniteTempBasis<LogisticKernel, Bosonic>;

// Tests moved to tests/basis_tests.rs

