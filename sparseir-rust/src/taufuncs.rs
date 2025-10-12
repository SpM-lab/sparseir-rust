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
/// use crate::taufuncs::normalize_tau;
/// use crate::traits::Fermionic;
///
/// let (tau_norm, sign) = normalize_tau::<Fermionic>(-0.3, 1.0);
/// assert!((tau_norm - 0.7).abs() < 1e-14);
/// assert_eq!(sign, -1.0);
/// ```
pub(crate) fn normalize_tau<S: StatisticsType>(tau: f64, beta: f64) -> (f64, f64) {
    // Normalize τ to [0, β)
    let sign: f64;
    let tau_normalized: f64;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Fermionic, Bosonic};

    #[test]
    fn test_is_odd_period() {
        let beta = 1.0;
        
        // Normal range
        assert!(!is_odd_period(0.0, beta));
        assert!(!is_odd_period(0.5, beta));
        assert!(!is_odd_period(beta, beta));  // β is β-
        
        // Odd periods
        assert!(is_odd_period(-0.1, beta));
        assert!(is_odd_period(1.1, beta));
    }

    #[test]
    fn test_normalize_tau_generic_fermionic() {
        let beta = 1.0;

        // Normal range
        let (tau_norm, sign) = normalize_tau::<Fermionic>(0.5, beta);
        assert!((tau_norm - 0.5).abs() < 1e-14);
        assert!((sign - 1.0).abs() < 1e-14);

        // At β (interpreted as β-)
        let (tau_norm, sign) = normalize_tau::<Fermionic>(beta, beta);
        assert!((tau_norm - beta).abs() < 1e-14);
        assert!((sign - 1.0).abs() < 1e-14);

        // Negative range
        let (tau_norm, sign) = normalize_tau::<Fermionic>(-0.3, beta);
        assert!((tau_norm - 0.7).abs() < 1e-14);
        assert!((sign - (-1.0)).abs() < 1e-14);

        // Extended range
        let (tau_norm, sign) = normalize_tau::<Fermionic>(1.2, beta);
        assert!((tau_norm - 0.2).abs() < 1e-14);
        assert!((sign - (-1.0)).abs() < 1e-14);
        
        // Test -β (interpreted as -β + 0, wraps to normal range)
        let (tau_norm, sign) = normalize_tau::<Fermionic>(-beta, beta);
        assert!(tau_norm.abs() < 1e-14);  // wraps to 0
        assert!((sign - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn test_normalize_tau_generic_bosonic() {
        let beta = 1.0;

        // Normal range
        let (tau_norm, sign) = normalize_tau::<Bosonic>(0.5, beta);
        assert!((tau_norm - 0.5).abs() < 1e-14);
        assert!((sign - 1.0).abs() < 1e-14);

        // At β (interpreted as β-)
        let (tau_norm, sign) = normalize_tau::<Bosonic>(beta, beta);
        assert!((tau_norm - beta).abs() < 1e-14);
        assert!((sign - 1.0).abs() < 1e-14);

        // Negative range
        let (tau_norm, sign) = normalize_tau::<Bosonic>(-0.3, beta);
        assert!((tau_norm - 0.7).abs() < 1e-14);
        assert!((sign - 1.0).abs() < 1e-14);

        // Extended range
        let (tau_norm, sign) = normalize_tau::<Bosonic>(1.2, beta);
        assert!((tau_norm - 0.2).abs() < 1e-14);
        assert!((sign - 1.0).abs() < 1e-14);
        
        // Test -β (interpreted as -β + 0, wraps to normal range)
        let (tau_norm, sign) = normalize_tau::<Bosonic>(-beta, beta);
        assert!(tau_norm.abs() < 1e-14);  // wraps to 0
        assert!((sign - 1.0).abs() < 1e-14);
    }
}
