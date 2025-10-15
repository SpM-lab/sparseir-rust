//! Sampling API for C
//!
//! This module provides the C API for sparse sampling in imaginary time (τ),
//! Matsubara frequency (iωn), and real frequency (ω) domains.
//!
//! Functions:
//! - Creation: spir_tau_sampling_new, spir_matsu_sampling_new, ...
//! - Introspection: get_npoints, get_taus, get_matsus, get_cond_num
//! - Evaluation: eval_dd, eval_dz, eval_zz (coefficients → sampling points)
//! - Fitting: fit_dd, fit_zz, fit_zd (sampling points → coefficients)
//! - Memory: release, clone, is_assigned (via macro)

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use num_complex::Complex64;

use crate::types::{spir_basis, spir_sampling, SamplingType, BasisType};
use crate::{StatusCode, SPIR_COMPUTATION_SUCCESS, SPIR_INVALID_ARGUMENT, SPIR_NOT_SUPPORTED};
use sparseir_rust::{Bosonic, Fermionic, Tensor, DynRank};

// Generate common opaque type functions: release, clone, is_assigned, get_raw_ptr
impl_opaque_type_common!(sampling);

// ============================================================================
// Creation Functions
// ============================================================================

/// Creates a new tau sampling object for sparse sampling in imaginary time
///
/// # Arguments
/// * `b` - Pointer to a finite temperature basis object
/// * `num_points` - Number of sampling points
/// * `points` - Array of sampling points in imaginary time (τ)
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the newly created sampling object, or NULL if creation fails
///
/// # Safety
/// Caller must ensure `b` is valid and `points` has `num_points` elements
#[no_mangle]
pub unsafe extern "C" fn spir_tau_sampling_new(
    b: *const spir_basis,
    num_points: libc::c_int,
    points: *const f64,
    status: *mut StatusCode,
) -> *mut spir_sampling {
    let result = catch_unwind(|| {
        // Validate inputs
        if b.is_null() || points.is_null() {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }
        if num_points <= 0 {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }

        let basis_ref = &*b;
        let points_slice = std::slice::from_raw_parts(points, num_points as usize);

        // Convert points to Vec
        let tau_points: Vec<f64> = points_slice.to_vec();

        // Create sampling based on basis statistics
        let sampling_type = match &basis_ref.inner {
            BasisType::LogisticFermionic(ir_basis) => {
                let tau_sampling = sparseir_rust::sampling::TauSampling::with_sampling_points(
                    ir_basis.as_ref(),
                    tau_points,
                );
                SamplingType::TauFermionic(Arc::new(tau_sampling))
            }
            BasisType::RegularizedBoseFermionic(ir_basis) => {
                let tau_sampling = sparseir_rust::sampling::TauSampling::with_sampling_points(
                    ir_basis.as_ref(),
                    tau_points,
                );
                SamplingType::TauFermionic(Arc::new(tau_sampling))
            }
            BasisType::LogisticBosonic(ir_basis) => {
                let tau_sampling = sparseir_rust::sampling::TauSampling::with_sampling_points(
                    ir_basis.as_ref(),
                    tau_points,
                );
                SamplingType::TauBosonic(Arc::new(tau_sampling))
            }
            BasisType::RegularizedBoseBosonic(ir_basis) => {
                let tau_sampling = sparseir_rust::sampling::TauSampling::with_sampling_points(
                    ir_basis.as_ref(),
                    tau_points,
                );
                SamplingType::TauBosonic(Arc::new(tau_sampling))
            }
        };

        let sampling = spir_sampling {
            inner: sampling_type,
        };

        (Box::into_raw(Box::new(sampling)), SPIR_COMPUTATION_SUCCESS)
    });

    match result {
        Ok((ptr, code)) => {
            if !status.is_null() {
                *status = code;
            }
            ptr
        }
        Err(_) => {
            if !status.is_null() {
                *status = crate::SPIR_INTERNAL_ERROR;
            }
            std::ptr::null_mut()
        }
    }
}

/// Creates a new Matsubara sampling object for sparse sampling in Matsubara frequencies
///
/// # Arguments
/// * `b` - Pointer to a finite temperature basis object
/// * `positive_only` - If true, only positive frequencies are used
/// * `num_points` - Number of sampling points
/// * `points` - Array of Matsubara frequency indices (n)
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the newly created sampling object, or NULL if creation fails
#[no_mangle]
pub unsafe extern "C" fn spir_matsu_sampling_new(
    b: *const spir_basis,
    positive_only: bool,
    num_points: libc::c_int,
    points: *const i64,
    status: *mut StatusCode,
) -> *mut spir_sampling {
    let result = catch_unwind(|| {
        // Validate inputs
        if b.is_null() || points.is_null() {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }
        if num_points <= 0 {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }

        let basis_ref = &*b;
        let points_slice = std::slice::from_raw_parts(points, num_points as usize);

        // Convert points to Vec
        let matsu_points: Vec<i64> = points_slice.to_vec();

        // Convert i64 indices to MatsubaraFreq
        use sparseir_rust::freq::MatsubaraFreq;
        
        // Helper macro to reduce duplication
        macro_rules! create_matsu_sampling {
            ($basis:expr, Fermionic) => {
                if positive_only {
                    let matsu_freqs: Vec<MatsubaraFreq<Fermionic>> = matsu_points
                        .iter()
                        .map(|&n| MatsubaraFreq::new(n).expect("Invalid Matsubara frequency"))
                        .collect();
                    let matsu_sampling = sparseir_rust::matsubara_sampling::MatsubaraSamplingPositiveOnly::with_sampling_points(
                        $basis,
                        matsu_freqs,
                    );
                    SamplingType::MatsubaraPositiveOnlyFermionic(Arc::new(matsu_sampling))
                } else {
                    let matsu_freqs: Vec<MatsubaraFreq<Fermionic>> = matsu_points
                        .iter()
                        .map(|&n| MatsubaraFreq::new(n).expect("Invalid Matsubara frequency"))
                        .collect();
                    let matsu_sampling = sparseir_rust::matsubara_sampling::MatsubaraSampling::with_sampling_points(
                        $basis,
                        matsu_freqs,
                    );
                    SamplingType::MatsubaraFermionic(Arc::new(matsu_sampling))
                }
            };
            ($basis:expr, Bosonic) => {
                if positive_only {
                    let matsu_freqs: Vec<MatsubaraFreq<Bosonic>> = matsu_points
                        .iter()
                        .map(|&n| MatsubaraFreq::new(n).expect("Invalid Matsubara frequency"))
                        .collect();
                    let matsu_sampling = sparseir_rust::matsubara_sampling::MatsubaraSamplingPositiveOnly::with_sampling_points(
                        $basis,
                        matsu_freqs,
                    );
                    SamplingType::MatsubaraPositiveOnlyBosonic(Arc::new(matsu_sampling))
                } else {
                    let matsu_freqs: Vec<MatsubaraFreq<Bosonic>> = matsu_points
                        .iter()
                        .map(|&n| MatsubaraFreq::new(n).expect("Invalid Matsubara frequency"))
                        .collect();
                    let matsu_sampling = sparseir_rust::matsubara_sampling::MatsubaraSampling::with_sampling_points(
                        $basis,
                        matsu_freqs,
                    );
                    SamplingType::MatsubaraBosonic(Arc::new(matsu_sampling))
                }
            };
        }

        // Create sampling based on basis statistics and positive_only flag
        let sampling_type = match &basis_ref.inner {
            BasisType::LogisticFermionic(ir_basis) => {
                create_matsu_sampling!(ir_basis.as_ref(), Fermionic)
            }
            BasisType::RegularizedBoseFermionic(ir_basis) => {
                create_matsu_sampling!(ir_basis.as_ref(), Fermionic)
            }
            BasisType::LogisticBosonic(ir_basis) => {
                create_matsu_sampling!(ir_basis.as_ref(), Bosonic)
            }
            BasisType::RegularizedBoseBosonic(ir_basis) => {
                create_matsu_sampling!(ir_basis.as_ref(), Bosonic)
            }
        };

        let sampling = spir_sampling {
            inner: sampling_type,
        };

        (Box::into_raw(Box::new(sampling)), SPIR_COMPUTATION_SUCCESS)
    });

    match result {
        Ok((ptr, code)) => {
            if !status.is_null() {
                *status = code;
            }
            ptr
        }
        Err(_) => {
            if !status.is_null() {
                *status = crate::SPIR_INTERNAL_ERROR;
            }
            std::ptr::null_mut()
        }
    }
}

// ============================================================================
// Introspection Functions
// ============================================================================

/// Gets the number of sampling points in a sampling object
#[no_mangle]
pub unsafe extern "C" fn spir_sampling_get_npoints(
    s: *const spir_sampling,
    num_points: *mut libc::c_int,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || num_points.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }

        let sampling_ref = &*s;

        let n_points = match &sampling_ref.inner {
            SamplingType::TauFermionic(tau) => tau.n_sampling_points(),
            SamplingType::TauBosonic(tau) => tau.n_sampling_points(),
            SamplingType::MatsubaraFermionic(matsu) => matsu.n_sampling_points(),
            SamplingType::MatsubaraBosonic(matsu) => matsu.n_sampling_points(),
            SamplingType::MatsubaraPositiveOnlyFermionic(matsu) => matsu.n_sampling_points(),
            SamplingType::MatsubaraPositiveOnlyBosonic(matsu) => matsu.n_sampling_points(),
        };

        *num_points = n_points as libc::c_int;
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Gets the imaginary time sampling points
#[no_mangle]
pub unsafe extern "C" fn spir_sampling_get_taus(
    s: *const spir_sampling,
    points: *mut f64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || points.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }

        let sampling_ref = &*s;

        match &sampling_ref.inner {
            SamplingType::TauFermionic(tau) => {
                let tau_points = tau.sampling_points();
                let out_slice = std::slice::from_raw_parts_mut(points, tau_points.len());
                out_slice.copy_from_slice(tau_points);
                SPIR_COMPUTATION_SUCCESS
            }
            SamplingType::TauBosonic(tau) => {
                let tau_points = tau.sampling_points();
                let out_slice = std::slice::from_raw_parts_mut(points, tau_points.len());
                out_slice.copy_from_slice(tau_points);
                SPIR_COMPUTATION_SUCCESS
            }
            _ => SPIR_NOT_SUPPORTED,
        }
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Gets the Matsubara frequency sampling points
#[no_mangle]
pub unsafe extern "C" fn spir_sampling_get_matsus(
    s: *const spir_sampling,
    points: *mut i64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || points.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }

        let sampling_ref = &*s;

        match &sampling_ref.inner {
            SamplingType::MatsubaraFermionic(matsu) => {
                let matsu_freqs = matsu.sampling_points();
                let out_slice = std::slice::from_raw_parts_mut(points, matsu_freqs.len());
                for (i, freq) in matsu_freqs.iter().enumerate() {
                    out_slice[i] = freq.n();
                }
                SPIR_COMPUTATION_SUCCESS
            }
            SamplingType::MatsubaraBosonic(matsu) => {
                let matsu_freqs = matsu.sampling_points();
                let out_slice = std::slice::from_raw_parts_mut(points, matsu_freqs.len());
                for (i, freq) in matsu_freqs.iter().enumerate() {
                    out_slice[i] = freq.n();
                }
                SPIR_COMPUTATION_SUCCESS
            }
            SamplingType::MatsubaraPositiveOnlyFermionic(matsu) => {
                let matsu_freqs = matsu.sampling_points();
                let out_slice = std::slice::from_raw_parts_mut(points, matsu_freqs.len());
                for (i, freq) in matsu_freqs.iter().enumerate() {
                    out_slice[i] = freq.n();
                }
                SPIR_COMPUTATION_SUCCESS
            }
            SamplingType::MatsubaraPositiveOnlyBosonic(matsu) => {
                let matsu_freqs = matsu.sampling_points();
                let out_slice = std::slice::from_raw_parts_mut(points, matsu_freqs.len());
                for (i, freq) in matsu_freqs.iter().enumerate() {
                    out_slice[i] = freq.n();
                }
                SPIR_COMPUTATION_SUCCESS
            }
            _ => SPIR_NOT_SUPPORTED,
        }
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Gets the condition number of the sampling matrix
///
/// Note: Currently returns a placeholder value.
/// TODO: Implement proper condition number calculation from SVD
#[no_mangle]
pub unsafe extern "C" fn spir_sampling_get_cond_num(
    s: *const spir_sampling,
    cond_num: *mut f64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || cond_num.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }

        // TODO: Calculate actual condition number from SVD
        // For now, return a reasonable placeholder
        *cond_num = 1.0;
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tau_sampling_creation() {
        unsafe {
            // Create a basis
            let mut status = 0;
            let kernel = crate::spir_logistic_kernel_new(10.0, &mut status);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

            let sve = crate::spir_sve_result_new(kernel, 1e-10, -1.0, -1, -1, -1, &mut status);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

            let basis = crate::spir_basis_new(1, 10.0, 1.0, 1e-10, kernel, sve, -1, &mut status);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

            // Create tau sampling with custom points
            let tau_points = vec![0.1, 0.5, 1.0, 5.0, 9.0];
            let sampling = spir_tau_sampling_new(
                basis,
                tau_points.len() as i32,
                tau_points.as_ptr(),
                &mut status,
            );
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            assert!(!sampling.is_null());

            // Get number of points
            let mut n_points = 0;
            let ret = spir_sampling_get_npoints(sampling, &mut n_points);
            assert_eq!(ret, SPIR_COMPUTATION_SUCCESS);
            assert_eq!(n_points, 5);

            // Get tau points back
            let mut retrieved_points = vec![0.0; 5];
            let ret = spir_sampling_get_taus(sampling, retrieved_points.as_mut_ptr());
            assert_eq!(ret, SPIR_COMPUTATION_SUCCESS);
            assert_eq!(retrieved_points, tau_points);

            // Get condition number
            let mut cond = 0.0;
            let ret = spir_sampling_get_cond_num(sampling, &mut cond);
            assert_eq!(ret, SPIR_COMPUTATION_SUCCESS);
            assert!(cond >= 1.0); // Condition number >= 1 (placeholder returns 1.0)

            // Clean up
            crate::spir_sampling_release(sampling);
            crate::spir_basis_release(basis);
            crate::spir_sve_result_release(sve);
            crate::spir_kernel_release(kernel);
        }
    }
}

