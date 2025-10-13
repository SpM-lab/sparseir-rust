//! Basis API
//!
//! Functions for creating and manipulating finite temperature basis objects.

use std::panic::catch_unwind;

use sparseir_rust::basis::FiniteTempBasis;

use crate::types::{spir_kernel, spir_sve_result, spir_basis};
use crate::{StatusCode, SPIR_SUCCESS, SPIR_INVALID_ARGUMENT, SPIR_INTERNAL_ERROR};

/// Create a finite temperature basis (libsparseir compatible)
///
/// # Arguments
/// * `statistics` - 0 for Bosonic, 1 for Fermionic
/// * `beta` - Inverse temperature (must be > 0)
/// * `omega_max` - Frequency cutoff (must be > 0)
/// * `epsilon` - Accuracy target (must be > 0)
/// * `k` - Kernel object (can be NULL if sve is provided)
/// * `sve` - Pre-computed SVE result (can be NULL, will compute if needed)
/// * `max_size` - Maximum basis size (-1 for no limit)
/// * `status` - Pointer to store status code
///
/// # Returns
/// * Pointer to basis object, or NULL on failure
///
/// # Safety
/// The caller must ensure `status` is a valid pointer.
#[no_mangle]
pub extern "C" fn spir_basis_new(
    statistics: libc::c_int,
    beta: f64,
    omega_max: f64,
    epsilon: f64,
    k: *const spir_kernel,
    sve: *const spir_sve_result,
    max_size: libc::c_int,
    status: *mut StatusCode,
) -> *mut spir_basis {
    if status.is_null() {
        return std::ptr::null_mut();
    }

    // Validate inputs
    if beta <= 0.0 || omega_max <= 0.0 || epsilon <= 0.0 {
        unsafe { *status = SPIR_INVALID_ARGUMENT; }
        return std::ptr::null_mut();
    }

    // Validate statistics
    if statistics != 0 && statistics != 1 {
        unsafe { *status = SPIR_INVALID_ARGUMENT; }
        return std::ptr::null_mut();
    }

    // Must have kernel (SVE can be provided for optimization but kernel is required for type info)
    if k.is_null() {
        unsafe { *status = SPIR_INVALID_ARGUMENT; }
        return std::ptr::null_mut();
    }

    // Convert max_size
    let max_size_opt = if max_size < 0 {
        None
    } else {
        Some(max_size as usize)
    };

    let result = catch_unwind(|| unsafe {
        let kernel_ref = &*k;
        
        // Check that kernel's lambda matches beta * omega_max
        let expected_lambda = beta * omega_max;
        let kernel_lambda = kernel_ref.lambda();
        if (kernel_lambda - expected_lambda).abs() > 1e-10 {
            return Err(format!(
                "Kernel lambda ({}) does not match beta * omega_max ({})",
                kernel_lambda, expected_lambda
            ));
        }

        // Dispatch based on kernel type and statistics
        if let Some(logistic) = kernel_ref.as_logistic() {
            if statistics == 1 {
                // Fermionic
                let basis: FiniteTempBasis<_, _> = if !sve.is_null() {
                    let sve_ref = &*sve;
                    FiniteTempBasis::from_sve_result(
                        (**logistic).clone(),
                        beta,
                        sve_ref.inner().as_ref().clone(),
                        Some(epsilon),
                        max_size_opt,
                    )
                } else {
                    FiniteTempBasis::new(
                        (**logistic).clone(),
                        beta,
                        Some(epsilon),
                        max_size_opt,
                    )
                };
                Ok(Box::into_raw(Box::new(spir_basis::new_logistic_fermionic(basis))))
            } else {
                // Bosonic
                let basis: FiniteTempBasis<_, _> = if !sve.is_null() {
                    let sve_ref = &*sve;
                    FiniteTempBasis::from_sve_result(
                        (**logistic).clone(),
                        beta,
                        sve_ref.inner().as_ref().clone(),
                        Some(epsilon),
                        max_size_opt,
                    )
                } else {
                    FiniteTempBasis::new(
                        (**logistic).clone(),
                        beta,
                        Some(epsilon),
                        max_size_opt,
                    )
                };
                Ok(Box::into_raw(Box::new(spir_basis::new_logistic_bosonic(basis))))
            }
        } else if let Some(reg_bose) = kernel_ref.as_regularized_bose() {
            if statistics == 1 {
                // Fermionic
                let basis: FiniteTempBasis<_, _> = if !sve.is_null() {
                    let sve_ref = &*sve;
                    FiniteTempBasis::from_sve_result(
                        (**reg_bose).clone(),
                        beta,
                        sve_ref.inner().as_ref().clone(),
                        Some(epsilon),
                        max_size_opt,
                    )
                } else {
                    FiniteTempBasis::new(
                        (**reg_bose).clone(),
                        beta,
                        Some(epsilon),
                        max_size_opt,
                    )
                };
                Ok(Box::into_raw(Box::new(spir_basis::new_regularized_bose_fermionic(basis))))
            } else {
                // Bosonic
                let basis: FiniteTempBasis<_, _> = if !sve.is_null() {
                    let sve_ref = &*sve;
                    FiniteTempBasis::from_sve_result(
                        (**reg_bose).clone(),
                        beta,
                        sve_ref.inner().as_ref().clone(),
                        Some(epsilon),
                        max_size_opt,
                    )
                } else {
                    FiniteTempBasis::new(
                        (**reg_bose).clone(),
                        beta,
                        Some(epsilon),
                        max_size_opt,
                    )
                };
                Ok(Box::into_raw(Box::new(spir_basis::new_regularized_bose_bosonic(basis))))
            }
        } else {
            Err("Unknown kernel type".to_string())
        }
    });

    match result {
        Ok(Ok(ptr)) => {
            unsafe { *status = SPIR_SUCCESS; }
            ptr
        }
        Ok(Err(_)) | Err(_) => {
            unsafe { *status = SPIR_INTERNAL_ERROR; }
            std::ptr::null_mut()
        }
    }
}

/// Release a basis object
///
/// # Arguments
/// * `b` - Basis to release (can be NULL)
///
/// # Safety
/// After calling this function, the basis pointer is invalid and must not be used.
#[no_mangle]
pub extern "C" fn spir_basis_release(b: *mut spir_basis) {
    if !b.is_null() {
        let _ = catch_unwind(|| unsafe {
            let _ = Box::from_raw(b);
        });
    }
}

/// Get the number of basis functions
///
/// # Arguments
/// * `b` - Basis object
/// * `size` - Pointer to store the size
///
/// # Returns
/// * `SPIR_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if b or size is null
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
#[no_mangle]
pub extern "C" fn spir_basis_get_size(
    b: *const spir_basis,
    size: *mut libc::c_int,
) -> StatusCode {
    if b.is_null() || size.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let basis = &*b;
        *size = basis.size() as libc::c_int;
        SPIR_SUCCESS
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get singular values from a basis
///
/// # Arguments
/// * `b` - Basis object
/// * `svals` - Pre-allocated array to store singular values (size must be >= basis size)
///
/// # Returns
/// * `SPIR_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if b or svals is null
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
#[no_mangle]
pub extern "C" fn spir_basis_get_svals(
    b: *const spir_basis,
    svals: *mut f64,
) -> StatusCode {
    if b.is_null() || svals.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let basis = &*b;
        let sval_vec = basis.svals();
        std::ptr::copy_nonoverlapping(sval_vec.as_ptr(), svals, sval_vec.len());
        SPIR_SUCCESS
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get statistics type (Fermionic or Bosonic) of a basis
///
/// # Arguments
/// * `b` - Basis object
/// * `statistics` - Pointer to store statistics (0 = Bosonic, 1 = Fermionic)
///
/// # Returns
/// * `SPIR_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if b or statistics is null
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
#[no_mangle]
pub extern "C" fn spir_basis_get_stats(
    b: *const spir_basis,
    statistics: *mut libc::c_int,
) -> StatusCode {
    if b.is_null() || statistics.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let basis = &*b;
        *statistics = basis.statistics();
        SPIR_SUCCESS
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get singular values (alias for spir_basis_get_svals for libsparseir compatibility)
#[no_mangle]
pub extern "C" fn spir_basis_get_singular_values(
    b: *const spir_basis,
    svals: *mut f64,
) -> StatusCode {
    spir_basis_get_svals(b, svals)
}

/// Get the number of default tau sampling points
///
/// # Arguments
/// * `b` - Basis object
/// * `num_points` - Pointer to store the number of points
///
/// # Returns
/// * `SPIR_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if b or num_points is null
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
#[no_mangle]
pub extern "C" fn spir_basis_get_n_default_taus(
    b: *const spir_basis,
    num_points: *mut libc::c_int,
) -> StatusCode {
    if b.is_null() || num_points.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let basis = &*b;
        let points = basis.default_tau_sampling_points();
        *num_points = points.len() as libc::c_int;
        SPIR_SUCCESS
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get default tau sampling points
///
/// # Arguments
/// * `b` - Basis object
/// * `points` - Pre-allocated array to store tau points
///
/// # Returns
/// * `SPIR_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if b or points is null
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
#[no_mangle]
pub extern "C" fn spir_basis_get_default_taus(
    b: *const spir_basis,
    points: *mut f64,
) -> StatusCode {
    if b.is_null() || points.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let basis = &*b;
        let tau_points = basis.default_tau_sampling_points();
        std::ptr::copy_nonoverlapping(tau_points.as_ptr(), points, tau_points.len());
        SPIR_SUCCESS
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get the number of default Matsubara sampling points
///
/// # Arguments
/// * `b` - Basis object
/// * `positive_only` - If true, return only positive frequencies
/// * `num_points` - Pointer to store the number of points
///
/// # Returns
/// * `SPIR_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if b or num_points is null
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
#[no_mangle]
pub extern "C" fn spir_basis_get_n_default_matsus(
    b: *const spir_basis,
    positive_only: bool,
    num_points: *mut libc::c_int,
) -> StatusCode {
    if b.is_null() || num_points.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let basis = &*b;
        let points = basis.default_matsubara_sampling_points(positive_only);
        *num_points = points.len() as libc::c_int;
        SPIR_SUCCESS
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Get default Matsubara sampling points
///
/// # Arguments
/// * `b` - Basis object
/// * `positive_only` - If true, return only positive frequencies
/// * `points` - Pre-allocated array to store Matsubara indices
///
/// # Returns
/// * `SPIR_SUCCESS` (0) on success
/// * `SPIR_INVALID_ARGUMENT` (-6) if b or points is null
/// * `SPIR_INTERNAL_ERROR` (-7) if internal panic occurs
#[no_mangle]
pub extern "C" fn spir_basis_get_default_matsus(
    b: *const spir_basis,
    positive_only: bool,
    points: *mut i64,
) -> StatusCode {
    if b.is_null() || points.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let basis = &*b;
        let matsu_points = basis.default_matsubara_sampling_points(positive_only);
        std::ptr::copy_nonoverlapping(matsu_points.as_ptr(), points, matsu_points.len());
        SPIR_SUCCESS
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::*;
    use crate::sve::*;
    use std::ptr;

    #[test]
    fn test_basis_from_sve() {
        // Create kernel
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
        assert_eq!(kernel_status, SPIR_SUCCESS);

        // Compute SVE
        let mut sve_status = SPIR_INTERNAL_ERROR;
        let sve = spir_sve_result_new(kernel, 1e-6, -1.0, -1, -1, -1, &mut sve_status);
        assert_eq!(sve_status, SPIR_SUCCESS);

        // Create basis from SVE (kernel is still required for type info)
        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            1,      // Fermionic
            10.0,   // beta
            1.0,    // omega_max
            1e-6,   // epsilon
            kernel, // kernel required (for type info)
            sve,    // SVE provided (optimization)
            -1,     // no max_size
            &mut basis_status,
        );
        assert_eq!(basis_status, SPIR_SUCCESS);
        assert!(!basis.is_null());

        // Get size
        let mut size = 0;
        let status = spir_basis_get_size(basis, &mut size);
        assert_eq!(status, SPIR_SUCCESS);
        assert!(size > 0);
        println!("Basis size: {}", size);

        // Get statistics
        let mut stats = -1;
        let status = spir_basis_get_stats(basis, &mut stats);
        assert_eq!(status, SPIR_SUCCESS);
        assert_eq!(stats, 1); // Fermionic

        // Cleanup
        spir_basis_release(basis);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }

    #[test]
    fn test_basis_from_kernel() {
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
        
        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            0,      // Bosonic
            10.0,   // beta
            1.0,    // omega_max
            1e-6,   // epsilon
            kernel,
            ptr::null(),  // no SVE (compute from kernel)
            -1,
            &mut basis_status,
        );
        assert_eq!(basis_status, SPIR_SUCCESS);
        assert!(!basis.is_null());

        let mut stats = -1;
        spir_basis_get_stats(basis, &mut stats);
        assert_eq!(stats, 0); // Bosonic

        spir_basis_release(basis);
        spir_kernel_release(kernel);
    }

    #[test]
    fn test_basis_tau_sampling() {
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
        
        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(0, 10.0, 1.0, 1e-6, kernel, ptr::null(), -1, &mut basis_status);
        
        // Get number of tau points
        let mut n_taus = 0;
        let status = spir_basis_get_n_default_taus(basis, &mut n_taus);
        assert_eq!(status, SPIR_SUCCESS);
        assert!(n_taus > 0);
        println!("Number of tau points: {}", n_taus);

        // Get tau points
        let mut taus = vec![0.0; n_taus as usize];
        let status = spir_basis_get_default_taus(basis, taus.as_mut_ptr());
        assert_eq!(status, SPIR_SUCCESS);

        println!("First 5 tau points:");
        for i in 0..std::cmp::min(5, taus.len()) {
            println!("  tau[{}] = {}", i, taus[i]);
        }

        spir_basis_release(basis);
        spir_kernel_release(kernel);
    }

    #[test]
    fn test_basis_matsubara_sampling() {
        let mut kernel_status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
        
        let mut basis_status = SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(1, 10.0, 1.0, 1e-6, kernel, ptr::null(), -1, &mut basis_status);
        
        // Get number of Matsubara points (positive only)
        let mut n_matsus = 0;
        let status = spir_basis_get_n_default_matsus(basis, true, &mut n_matsus);
        assert_eq!(status, SPIR_SUCCESS);
        assert!(n_matsus > 0);
        println!("Number of Matsubara points (positive): {}", n_matsus);

        // Get Matsubara points
        let mut matsus = vec![0i64; n_matsus as usize];
        let status = spir_basis_get_default_matsus(basis, true, matsus.as_mut_ptr());
        assert_eq!(status, SPIR_SUCCESS);

        println!("First 5 Matsubara indices:");
        for i in 0..std::cmp::min(5, matsus.len()) {
            println!("  n[{}] = {}", i, matsus[i]);
        }

        spir_basis_release(basis);
        spir_kernel_release(kernel);
    }
}

