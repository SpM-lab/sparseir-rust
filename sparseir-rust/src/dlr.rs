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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Fermionic, Bosonic};

    /// Generic test for periodicity/anti-periodicity
    fn test_periodicity_generic<S: StatisticsType>(expected_sign: f64, stat_name: &str) {
        let beta = 1.0;
        let omega = 5.0;
        
        // Use interior points, avoiding boundaries
        for tau in [0.1, 0.3, 0.7] {
            let g_tau = gtau_single_pole::<S>(tau, omega, beta);
            let g_tau_plus_beta = gtau_single_pole::<S>(tau + beta, omega, beta);
            
            // For fermions: G(τ+β) = -G(τ) → sign = -1
            // For bosons: G(τ+β) = G(τ) → sign = 1
            let expected = expected_sign * g_tau;
            
            assert!(
                (expected - g_tau_plus_beta).abs() < 1e-14,
                "{} periodicity violated at τ={}: G(τ)={}, G(τ+β)={}, expected={}",
                stat_name, tau, g_tau, g_tau_plus_beta, expected
            );
        }
    }

    /// Generic test for boundary interpretation
    fn test_boundary_interpretation_generic<S: StatisticsType>(stat_name: &str) {
        let beta = 1.0;
        let omega = 5.0;
        let eps = 1e-10;
        
        // Test: β is interpreted as β- (left limit) and -β as -β+ (right limit)
        let g_beta_minus = gtau_single_pole::<S>(beta - eps, omega, beta);
        let g_beta = gtau_single_pole::<S>(beta, omega, beta);
        
        let g_minus_beta_plus = gtau_single_pole::<S>(-beta + eps, omega, beta);
        let g_minus_beta = gtau_single_pole::<S>(-beta, omega, beta);
        
        // β is treated as β- (stays in normal range, not wrapped)
        assert!(
            (g_beta - g_beta_minus).abs() < 1e-8,
            "{}: β should be β-: G(β)={}, G(β-)={}, diff={}",
            stat_name, g_beta, g_beta_minus, (g_beta - g_beta_minus).abs()
        );
        
        // -β is treated as -β+ (stays after wrapping)
        assert!(
            (g_minus_beta - g_minus_beta_plus).abs() < 1e-8,
            "{}: -β should be -β+: G(-β)={}, G(-β+)={}, diff={}",
            stat_name, g_minus_beta, g_minus_beta_plus, (g_minus_beta - g_minus_beta_plus).abs()
        );
    }

    #[test]
    fn test_fermionic_antiperiodicity() {
        // Fermions: G(τ+β) = -G(τ)
        test_periodicity_generic::<Fermionic>(-1.0, "Fermionic");
    }

    #[test]
    fn test_bosonic_periodicity() {
        // Bosons: G(τ+β) = G(τ)
        test_periodicity_generic::<Bosonic>(1.0, "Bosonic");
    }

    #[test]
    fn test_fermionic_boundary_interpretation() {
        test_boundary_interpretation_generic::<Fermionic>("Fermionic");
    }

    #[test]
    fn test_bosonic_boundary_interpretation() {
        test_boundary_interpretation_generic::<Bosonic>("Bosonic");
    }

}

