//! Discrete Lehmann Representation (DLR) utilities
//!
//! This module provides utility functions for working with Green's functions
//! and spectral representations.

use crate::traits::{StatisticsType, Statistics};

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
