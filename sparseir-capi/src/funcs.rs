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
}

