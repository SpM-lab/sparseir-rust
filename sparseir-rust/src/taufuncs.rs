//! Imaginary time τ normalization utilities
//!
//! This module provides utility functions for normalizing imaginary time τ
//! with appropriate boundary conditions (periodic/anti-periodic).
//! 
//! These utilities are used internally (e.g., in DLR implementation) and
//! can also be used by external C-API or FFI layers.

use crate::traits::{StatisticsType, Statistics};

/// Check if τ is outside the normal range [0, β]
///
/// Returns whether τ is in an "odd" period (outside [0, β]).
/// This determines the sign for anti-periodic (fermionic) functions.
///
/// # Arguments
/// * `tau` - Imaginary time
/// * `beta` - Inverse temperature
///
/// # Returns
/// * `true` if τ < 0 or τ > β (odd period), `false` otherwise
#[inline]
fn is_odd_period(tau: f64, beta: f64) -> bool {
    tau < 0.0 || tau > beta
}

/// Normalize τ to the range [0, β] with statistics-dependent boundary conditions
///
/// Handles boundary conditions based on statistics:
/// - Fermions: Anti-periodic G(τ + β) = -G(τ)
/// - Bosons: Periodic G(τ + β) = G(τ)
///
/// # Type Parameters
/// * `S` - Statistics type (Fermionic or Bosonic)
///
/// # Arguments
/// * `tau` - Imaginary time (can be outside [0, β])
/// * `beta` - Inverse temperature
///
/// # Returns
/// * `(tau_normalized, sign)` - Normalized τ ∈ [0, β] and sign factor
///
/// # Boundary Interpretation
/// * `β` is interpreted as `β-` (left limit at β): `tau == beta` stays in normal range
/// * `-β` is interpreted as `-β + 0` (right limit at -β): `tau == -beta` wraps to normal range
///
/// # Examples
/// ```ignore
/// use sparseir_rust::taufuncs::normalize_tau;
/// use sparseir_rust::traits::Fermionic;
///
/// let (tau_norm, sign) = normalize_tau::<Fermionic>(-0.3, 1.0);
/// assert!((tau_norm - 0.7).abs() < 1e-14);
/// assert_eq!(sign, -1.0);
/// ```
pub fn normalize_tau<S: StatisticsType>(tau: f64, beta: f64) -> (f64, f64) {
    // Normalize τ to [0, β)
    let mut sign: f64;
    let mut tau_normalized: f64;
    if tau < 0.0 {
        tau_normalized = tau + beta;
        sign = -1.0;
    } else if tau > beta {
        tau_normalized = tau - beta;
        sign = -1.0;
    } else {
        tau_normalized = tau;
        sign = 1.0;
    };
    
    // Compute sign based on statistics and whether we're in an "odd" period
    match S::STATISTICS {
        Statistics::Fermionic => {
            // Anti-periodic: sign flips in odd periods
            return (tau_normalized, sign);
        }
        Statistics::Bosonic => {
            // Periodic: sign never flips
            return (tau_normalized, 1.0);
        }
    };
}

/// Normalize τ to the range [0, β] for fermionic functions
///
/// Convenience wrapper for `normalize_tau::<Fermionic>(tau, beta)`.
///
/// # Deprecated
/// Prefer using the generic `normalize_tau::<S>()` function.
#[inline]
pub fn normalize_tau_fermionic(tau: f64, beta: f64) -> (f64, f64) {
    use crate::traits::Fermionic;
    normalize_tau::<Fermionic>(tau, beta)
}

/// Normalize τ to the range [0, β] for bosonic functions
///
/// Convenience wrapper for `normalize_tau::<Bosonic>(tau, beta)`.
/// Returns only the normalized τ value (sign is always 1.0 for bosons).
///
/// # Deprecated
/// Prefer using the generic `normalize_tau::<S>()` function.
#[inline]
pub fn normalize_tau_bosonic(tau: f64, beta: f64) -> f64 {
    use crate::traits::Bosonic;
    normalize_tau::<Bosonic>(tau, beta).0
}

