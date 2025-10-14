//! Functions API for C
//!
//! This module provides C-compatible functions for working with basis functions.

use crate::{SPIR_SUCCESS, SPIR_INVALID_ARGUMENT};
use crate::types::spir_funcs;

/// Releases a funcs object
///
/// # Arguments
/// * `funcs` - Pointer to the funcs object to release
///
/// # Safety
/// The caller must ensure that `funcs` is a valid pointer returned from a previous
/// `spir_basis_get_u/v/uhat()` call, and that it is not used after this function returns.
#[no_mangle]
pub unsafe extern "C" fn spir_funcs_release(funcs: *mut spir_funcs) {
    if funcs.is_null() {
        return;
    }
    // Drop the boxed funcs object
    let _ = Box::from_raw(funcs);
}

/// Gets the number of basis functions
///
/// # Arguments
/// * `funcs` - Pointer to the funcs object
/// * `size` - Pointer to store the number of functions
///
/// # Returns
/// Status code (SPIR_SUCCESS on success)
#[no_mangle]
pub extern "C" fn spir_funcs_get_size(
    funcs: *const spir_funcs,
    size: *mut libc::c_int,
) -> crate::StatusCode {
    use std::panic::catch_unwind;
    use crate::{SPIR_SUCCESS, SPIR_INVALID_ARGUMENT, SPIR_INTERNAL_ERROR};

    if funcs.is_null() || size.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let f = &*funcs;
        *size = f.size() as libc::c_int;
        SPIR_SUCCESS
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Gets the number of knots for continuous functions
///
/// # Arguments
/// * `funcs` - Pointer to the funcs object
/// * `n_knots` - Pointer to store the number of knots
///
/// # Returns
/// Status code (SPIR_SUCCESS on success, SPIR_NOT_SUPPORTED if not continuous)
#[no_mangle]
pub extern "C" fn spir_funcs_get_n_knots(
    funcs: *const spir_funcs,
    n_knots: *mut libc::c_int,
) -> crate::StatusCode {
    use std::panic::catch_unwind;
    use crate::{SPIR_SUCCESS, SPIR_INVALID_ARGUMENT, SPIR_INTERNAL_ERROR, SPIR_NOT_SUPPORTED};

    if funcs.is_null() || n_knots.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let f = &*funcs;
        match f.knots() {
            Some(knots) => {
                *n_knots = knots.len() as libc::c_int;
                SPIR_SUCCESS
            },
            None => SPIR_NOT_SUPPORTED,
        }
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Gets the knot positions for continuous functions
///
/// # Arguments
/// * `funcs` - Pointer to the funcs object
/// * `knots` - Pre-allocated array to store knot positions
///
/// # Returns
/// Status code (SPIR_SUCCESS on success, SPIR_NOT_SUPPORTED if not continuous)
///
/// # Safety
/// The caller must ensure that `knots` has size >= `spir_funcs_get_n_knots(funcs)`
#[no_mangle]
pub extern "C" fn spir_funcs_get_knots(
    funcs: *const spir_funcs,
    knots: *mut f64,
) -> crate::StatusCode {
    use std::panic::catch_unwind;
    use crate::{SPIR_SUCCESS, SPIR_INVALID_ARGUMENT, SPIR_INTERNAL_ERROR, SPIR_NOT_SUPPORTED};

    if funcs.is_null() || knots.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let f = &*funcs;
        match f.knots() {
            Some(knot_vec) => {
                std::ptr::copy_nonoverlapping(knot_vec.as_ptr(), knots, knot_vec.len());
                SPIR_SUCCESS
            },
            None => SPIR_NOT_SUPPORTED,
        }
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Evaluate functions at a single point (continuous functions only)
///
/// # Arguments
/// * `funcs` - Pointer to the funcs object
/// * `x` - Point to evaluate at (tau coordinate in [-1, 1])
/// * `out` - Pre-allocated array to store function values
///
/// # Returns
/// Status code (SPIR_SUCCESS on success, SPIR_NOT_SUPPORTED if not continuous)
///
/// # Safety
/// The caller must ensure that `out` has size >= `spir_funcs_get_size(funcs)`
#[no_mangle]
pub extern "C" fn spir_funcs_eval(
    funcs: *const spir_funcs,
    x: f64,
    out: *mut f64,
) -> crate::StatusCode {
    use std::panic::catch_unwind;
    use crate::{SPIR_SUCCESS, SPIR_INVALID_ARGUMENT, SPIR_INTERNAL_ERROR, SPIR_NOT_SUPPORTED};

    if funcs.is_null() || out.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let f = &*funcs;
        match f.eval_continuous(x) {
            Some(values) => {
                std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
                SPIR_SUCCESS
            },
            None => SPIR_NOT_SUPPORTED,
        }
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Evaluate functions at a single Matsubara frequency
///
/// # Arguments
/// * `funcs` - Pointer to the funcs object
/// * `n` - Matsubara frequency index
/// * `out` - Pre-allocated array to store complex function values
///
/// # Returns
/// Status code (SPIR_SUCCESS on success, SPIR_NOT_SUPPORTED if not Matsubara type)
///
/// # Safety
/// The caller must ensure that `out` has size >= `spir_funcs_get_size(funcs)`
/// Complex numbers are laid out as [real, imag] pairs
#[no_mangle]
pub extern "C" fn spir_funcs_eval_matsu(
    funcs: *const spir_funcs,
    n: i64,
    out: *mut num_complex::Complex64,
) -> crate::StatusCode {
    use std::panic::catch_unwind;
    use crate::{SPIR_SUCCESS, SPIR_INVALID_ARGUMENT, SPIR_INTERNAL_ERROR, SPIR_NOT_SUPPORTED};

    if funcs.is_null() || out.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let f = &*funcs;
        match f.eval_matsubara(n) {
            Some(values) => {
                std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
                SPIR_SUCCESS
            },
            None => SPIR_NOT_SUPPORTED,
        }
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Batch evaluate functions at multiple points (continuous functions only)
///
/// # Arguments
/// * `funcs` - Pointer to the funcs object
/// * `order` - Memory layout: 0 for row-major, 1 for column-major
/// * `num_points` - Number of evaluation points
/// * `xs` - Array of points to evaluate at
/// * `out` - Pre-allocated array to store results
///
/// # Returns
/// Status code (SPIR_SUCCESS on success, SPIR_NOT_SUPPORTED if not continuous)
///
/// # Safety
/// - `xs` must have size >= `num_points`
/// - `out` must have size >= `num_points * spir_funcs_get_size(funcs)`
/// - Layout: row-major = out[point][func], column-major = out[func][point]
#[no_mangle]
pub extern "C" fn spir_funcs_batch_eval(
    funcs: *const spir_funcs,
    order: libc::c_int,
    num_points: libc::c_int,
    xs: *const f64,
    out: *mut f64,
) -> crate::StatusCode {
    use std::panic::catch_unwind;
    use crate::{SPIR_SUCCESS, SPIR_INVALID_ARGUMENT, SPIR_INTERNAL_ERROR, SPIR_NOT_SUPPORTED};

    if funcs.is_null() || xs.is_null() || out.is_null() || num_points <= 0 {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let f = &*funcs;
        let xs_slice = std::slice::from_raw_parts(xs, num_points as usize);
        
        match f.batch_eval_continuous(xs_slice) {
            Some(result_matrix) => {
                // result_matrix is Vec<Vec<f64>> where outer index is function, inner is point
                let n_funcs = result_matrix.len();
                let n_points = num_points as usize;
                
                if order == 0 {
                    // Row-major: out[point][func]
                    for i in 0..n_points {
                        for j in 0..n_funcs {
                            *out.add(i * n_funcs + j) = result_matrix[j][i];
                        }
                    }
                } else {
                    // Column-major: out[func][point]
                    for j in 0..n_funcs {
                        for i in 0..n_points {
                            *out.add(j * n_points + i) = result_matrix[j][i];
                        }
                    }
                }
                SPIR_SUCCESS
            },
            None => SPIR_NOT_SUPPORTED,
        }
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Batch evaluate functions at multiple Matsubara frequencies
///
/// # Arguments
/// * `funcs` - Pointer to the funcs object
/// * `order` - Memory layout: 0 for row-major, 1 for column-major
/// * `num_freqs` - Number of Matsubara frequencies
/// * `ns` - Array of Matsubara frequency indices
/// * `out` - Pre-allocated array to store complex results
///
/// # Returns
/// Status code (SPIR_SUCCESS on success, SPIR_NOT_SUPPORTED if not Matsubara type)
///
/// # Safety
/// - `ns` must have size >= `num_freqs`
/// - `out` must have size >= `num_freqs * spir_funcs_get_size(funcs)`
/// - Complex numbers are laid out as [real, imag] pairs
/// - Layout: row-major = out[freq][func], column-major = out[func][freq]
#[no_mangle]
pub extern "C" fn spir_funcs_batch_eval_matsu(
    funcs: *const spir_funcs,
    order: libc::c_int,
    num_freqs: libc::c_int,
    ns: *const i64,
    out: *mut num_complex::Complex64,
) -> crate::StatusCode {
    use std::panic::catch_unwind;
    use crate::{SPIR_SUCCESS, SPIR_INVALID_ARGUMENT, SPIR_INTERNAL_ERROR, SPIR_NOT_SUPPORTED};

    if funcs.is_null() || ns.is_null() || out.is_null() || num_freqs <= 0 {
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| unsafe {
        let f = &*funcs;
        let ns_slice = std::slice::from_raw_parts(ns, num_freqs as usize);
        
        match f.batch_eval_matsubara(ns_slice) {
            Some(result_matrix) => {
                // result_matrix is Vec<Vec<Complex64>> where outer index is function, inner is freq
                let n_funcs = result_matrix.len();
                let n_freqs = num_freqs as usize;
                
                if order == 0 {
                    // Row-major: out[freq][func]
                    for i in 0..n_freqs {
                        for j in 0..n_funcs {
                            *out.add(i * n_funcs + j) = result_matrix[j][i];
                        }
                    }
                } else {
                    // Column-major: out[func][freq]
                    for j in 0..n_funcs {
                        for i in 0..n_freqs {
                            *out.add(j * n_freqs + i) = result_matrix[j][i];
                        }
                    }
                }
                SPIR_SUCCESS
            },
            None => SPIR_NOT_SUPPORTED,
        }
    });

    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::*;
    use crate::basis::*;
    use std::ptr;

    #[test]
    fn test_funcs_basic_lifecycle() {
        use crate::kernel::*;
        use crate::basis::*;

        // Create a kernel
        let mut kernel_status = crate::SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
        assert_eq!(kernel_status, SPIR_SUCCESS);
        assert!(!kernel.is_null());

        // Create a basis
        let mut basis_status = crate::SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            1,      // Fermionic
            10.0,   // beta
            1.0,    // omega_max
            1e-6,   // epsilon
            kernel,
            ptr::null(),  // no SVE
            -1,     // no max_size
            &mut basis_status,
        );
        assert_eq!(basis_status, SPIR_SUCCESS);
        assert!(!basis.is_null());

        // Get u funcs
        let mut u_status = crate::SPIR_INTERNAL_ERROR;
        let u_funcs = unsafe { spir_basis_get_u(basis, &mut u_status) };
        assert_eq!(u_status, SPIR_SUCCESS);
        assert!(!u_funcs.is_null());
        println!("✓ Created u funcs");

        // Get v funcs
        let mut v_status = crate::SPIR_INTERNAL_ERROR;
        let v_funcs = unsafe { spir_basis_get_v(basis, &mut v_status) };
        assert_eq!(v_status, SPIR_SUCCESS);
        assert!(!v_funcs.is_null());
        println!("✓ Created v funcs");

        // Get uhat funcs
        let mut uhat_status = crate::SPIR_INTERNAL_ERROR;
        let uhat_funcs = unsafe { spir_basis_get_uhat(basis, &mut uhat_status) };
        assert_eq!(uhat_status, SPIR_SUCCESS);
        assert!(!uhat_funcs.is_null());
        println!("✓ Created uhat funcs");

        // Clean up
        unsafe {
            spir_funcs_release(u_funcs);
            spir_funcs_release(v_funcs);
            spir_funcs_release(uhat_funcs);
            spir_basis_release(basis);
            spir_kernel_release(kernel);
        }

        println!("✓ All funcs released successfully");
    }

    #[test]
    fn test_funcs_introspection() {
        use crate::kernel::*;
        use crate::basis::*;

        // Create a kernel and basis
        let mut kernel_status = crate::SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
        assert_eq!(kernel_status, SPIR_SUCCESS);

        let mut basis_status = crate::SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            1, 10.0, 1.0, 1e-6, kernel, ptr::null(), -1, &mut basis_status,
        );
        assert_eq!(basis_status, SPIR_SUCCESS);

        // Get u funcs
        let mut u_status = crate::SPIR_INTERNAL_ERROR;
        let u_funcs = unsafe { spir_basis_get_u(basis, &mut u_status) };
        assert_eq!(u_status, SPIR_SUCCESS);

        // Test get_size
        let mut size = 0;
        let status = spir_funcs_get_size(u_funcs, &mut size);
        assert_eq!(status, SPIR_SUCCESS);
        assert!(size > 0);
        println!("✓ Funcs size: {}", size);

        // Test get_n_knots
        let mut n_knots = 0;
        let status = spir_funcs_get_n_knots(u_funcs, &mut n_knots);
        assert_eq!(status, SPIR_SUCCESS);
        assert!(n_knots > 0);
        println!("✓ Number of knots: {}", n_knots);

        // Test get_knots
        let mut knots = vec![0.0; n_knots as usize];
        let status = spir_funcs_get_knots(u_funcs, knots.as_mut_ptr());
        assert_eq!(status, SPIR_SUCCESS);
        println!("✓ First 5 knots:");
        for i in 0..std::cmp::min(5, knots.len()) {
            println!("  knot[{}] = {}", i, knots[i]);
        }

        // Test with uhat funcs (should return NOT_SUPPORTED for knots)
        let mut uhat_status = crate::SPIR_INTERNAL_ERROR;
        let uhat_funcs = unsafe { spir_basis_get_uhat(basis, &mut uhat_status) };
        assert_eq!(uhat_status, SPIR_SUCCESS);

        let mut n_knots_uhat = 0;
        let status = spir_funcs_get_n_knots(uhat_funcs, &mut n_knots_uhat);
        assert_eq!(status, crate::SPIR_NOT_SUPPORTED);
        println!("✓ uhat funcs correctly returns NOT_SUPPORTED for knots");

        // Cleanup
        unsafe {
            spir_funcs_release(u_funcs);
            spir_funcs_release(uhat_funcs);
            spir_basis_release(basis);
            spir_kernel_release(kernel);
        }
    }

    #[test]
    fn test_funcs_evaluation() {
        use crate::kernel::*;
        use crate::basis::*;
        use num_complex::Complex64;

        // Create a kernel and basis
        let mut kernel_status = crate::SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut kernel_status);
        assert_eq!(kernel_status, SPIR_SUCCESS);

        let mut basis_status = crate::SPIR_INTERNAL_ERROR;
        let basis = spir_basis_new(
            1, 10.0, 1.0, 1e-6, kernel, ptr::null(), -1, &mut basis_status,
        );
        assert_eq!(basis_status, SPIR_SUCCESS);

        // Get u funcs
        let mut u_status = crate::SPIR_INTERNAL_ERROR;
        let u_funcs = unsafe { spir_basis_get_u(basis, &mut u_status) };
        assert_eq!(u_status, SPIR_SUCCESS);

        // Test single point eval
        let mut size = 0;
        let status = spir_funcs_get_size(u_funcs, &mut size);
        assert_eq!(status, SPIR_SUCCESS);

        let mut values = vec![0.0; size as usize];
        let status = spir_funcs_eval(u_funcs, 0.0, values.as_mut_ptr());
        assert_eq!(status, SPIR_SUCCESS);
        println!("✓ Evaluated u at x=0: {} functions", size);
        println!("  u[0](0) = {}", values[0]);

        // Test batch eval
        let xs = vec![-0.5, 0.0, 0.5];
        let mut batch_out = vec![0.0; (size as usize) * xs.len()];
        let status = spir_funcs_batch_eval(
            u_funcs,
            1, // column-major
            xs.len() as i32,
            xs.as_ptr(),
            batch_out.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_SUCCESS);
        println!("✓ Batch evaluated u at 3 points (column-major)");

        // Get uhat funcs
        let mut uhat_status = crate::SPIR_INTERNAL_ERROR;
        let uhat_funcs = unsafe { spir_basis_get_uhat(basis, &mut uhat_status) };
        assert_eq!(uhat_status, SPIR_SUCCESS);

        // Test Matsubara eval
        let mut matsu_values = vec![Complex64::new(0.0, 0.0); size as usize];
        let status = spir_funcs_eval_matsu(uhat_funcs, 1, matsu_values.as_mut_ptr());
        assert_eq!(status, SPIR_SUCCESS);
        println!("✓ Evaluated uhat at n=1");
        println!("  uhat[0](iω_1) = {}", matsu_values[0]);

        // Test batch Matsubara eval
        let matsu_ns = vec![1i64, 3, 5];
        let mut batch_matsu_out = vec![Complex64::new(0.0, 0.0); (size as usize) * matsu_ns.len()];
        let status = spir_funcs_batch_eval_matsu(
            uhat_funcs,
            1, // column-major
            matsu_ns.len() as i32,
            matsu_ns.as_ptr(),
            batch_matsu_out.as_mut_ptr(),
        );
        assert_eq!(status, SPIR_SUCCESS);
        println!("✓ Batch evaluated uhat at 3 Matsubara frequencies");

        // Cleanup
        unsafe {
            spir_funcs_release(u_funcs);
            spir_funcs_release(uhat_funcs);
            spir_basis_release(basis);
            spir_kernel_release(kernel);
        }
    }
}

