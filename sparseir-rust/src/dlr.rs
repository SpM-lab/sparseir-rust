//! Discrete Lehmann Representation (DLR) utilities
//!
//! This module provides utility functions for working with Green's functions
//! and spectral representations.

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
    let (tau_normalized, sign) = if tau < 0.0 {
        // -β < τ < 0: G(τ) = -G(τ + β)
        (tau + beta, -1.0)
    } else if tau >= beta {
        // β ≤ τ < 2β: G(τ) = -G(τ - β)  
        (tau - beta, -1.0)
    } else {
        // 0 ≤ τ < β: normal range
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
    let tau_normalized = if tau < 0.0 {
        // -β < τ < 0: G(τ) = G(τ + β)
        tau + beta
    } else if tau >= beta {
        // β ≤ τ < 2β: G(τ) = G(τ - β)
        tau - beta
    } else {
        // 0 ≤ τ < β: normal range
        tau
    };
    
    (-omega * tau_normalized).exp() / (1.0 - (-beta * omega).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fermionic_antiperiodicity() {
        let beta = 1.0;
        let omega = 5.0;
        
        // Test anti-periodicity: G(τ + β) = -G(τ)
        for tau in [0.0, 0.3, 0.7] {
            let g_tau = fermionic_single_pole(tau, omega, beta);
            let g_tau_plus_beta = fermionic_single_pole(tau + beta, omega, beta);
            
            assert!(
                (g_tau + g_tau_plus_beta).abs() < 1e-14,
                "Anti-periodicity violated at τ={}: G(τ)={}, G(τ+β)={}",
                tau, g_tau, g_tau_plus_beta
            );
        }
    }

    #[test]
    fn test_bosonic_periodicity() {
        let beta = 1.0;
        let omega = 5.0;
        
        // Test periodicity: G(τ + β) = G(τ)
        for tau in [0.0, 0.3, 0.7] {
            let g_tau = bosonic_single_pole(tau, omega, beta);
            let g_tau_plus_beta = bosonic_single_pole(tau + beta, omega, beta);
            
            assert!(
                (g_tau - g_tau_plus_beta).abs() < 1e-14,
                "Periodicity violated at τ={}: G(τ)={}, G(τ+β)={}",
                tau, g_tau, g_tau_plus_beta
            );
        }
    }

}

