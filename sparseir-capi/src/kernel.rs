//! Kernel API for C
//!
//! Functions for creating and manipulating kernel objects.

use std::panic::catch_unwind;

use crate::types::SparseIRKernel;
use crate::{StatusCode, SPIR_SUCCESS, SPIR_ERROR_NULL_POINTER, SPIR_ERROR_INVALID_ARGUMENT, SPIR_ERROR_PANIC};

/// Create a new Logistic kernel
///
/// # Arguments
/// * `lambda` - The kernel parameter Λ = β * ωmax (must be > 0)
/// * `out` - Pointer to store the created kernel
///
/// # Returns
/// * `SPIR_SUCCESS` on success
/// * `SPIR_ERROR_NULL_POINTER` if out is null
/// * `SPIR_ERROR_INVALID_ARGUMENT` if lambda <= 0
/// * `SPIR_ERROR_PANIC` if internal panic occurs
///
/// # Safety
/// The caller must ensure `out` is a valid pointer.
///
/// # Example (C)
/// ```c
/// SparseIRKernel* kernel = NULL;
/// int status = spir_kernel_logistic_new(10.0, &kernel);
/// if (status == SPIR_SUCCESS) {
///     // Use kernel...
///     spir_kernel_release(kernel);
/// }
/// ```
#[no_mangle]
pub extern "C" fn spir_kernel_logistic_new(
    lambda: f64,
    out: *mut *mut SparseIRKernel,
) -> StatusCode {
    // Input validation
    if out.is_null() {
        return SPIR_ERROR_NULL_POINTER;
    }

    if lambda <= 0.0 {
        return SPIR_ERROR_INVALID_ARGUMENT;
    }

    // Catch panics to prevent unwinding across FFI boundary
    let result = catch_unwind(|| {
        let kernel = SparseIRKernel::new_logistic(lambda);
        unsafe {
            *out = Box::into_raw(Box::new(kernel));
        }
        SPIR_SUCCESS
    });

    result.unwrap_or(SPIR_ERROR_PANIC)
}

/// Create a new RegularizedBose kernel
///
/// # Arguments
/// * `lambda` - The kernel parameter Λ = β * ωmax (must be > 0)
/// * `out` - Pointer to store the created kernel
///
/// # Returns
/// * `SPIR_SUCCESS` on success
/// * `SPIR_ERROR_NULL_POINTER` if out is null
/// * `SPIR_ERROR_INVALID_ARGUMENT` if lambda <= 0
/// * `SPIR_ERROR_PANIC` if internal panic occurs
#[no_mangle]
pub extern "C" fn spir_kernel_regularized_bose_new(
    lambda: f64,
    out: *mut *mut SparseIRKernel,
) -> StatusCode {
    if out.is_null() {
        return SPIR_ERROR_NULL_POINTER;
    }

    if lambda <= 0.0 {
        return SPIR_ERROR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| {
        let kernel = SparseIRKernel::new_regularized_bose(lambda);
        unsafe {
            *out = Box::into_raw(Box::new(kernel));
        }
        SPIR_SUCCESS
    });

    result.unwrap_or(SPIR_ERROR_PANIC)
}

/// Release a kernel object
///
/// # Arguments
/// * `kernel` - Kernel to release (can be NULL)
///
/// # Safety
/// After calling this function, the kernel pointer is invalid and must not be used.
#[no_mangle]
pub extern "C" fn spir_kernel_release(kernel: *mut SparseIRKernel) {
    if !kernel.is_null() {
        let _ = catch_unwind(|| unsafe {
            // Convert back to Box and drop
            let _ = Box::from_raw(kernel);
        });
    }
}

/// Get the lambda parameter of a kernel
///
/// # Arguments
/// * `kernel` - Kernel object
/// * `out` - Pointer to store the lambda value
///
/// # Returns
/// * `SPIR_SUCCESS` on success
/// * `SPIR_ERROR_NULL_POINTER` if kernel or out is null
/// * `SPIR_ERROR_PANIC` if internal panic occurs
#[no_mangle]
pub extern "C" fn spir_kernel_lambda(
    kernel: *const SparseIRKernel,
    out: *mut f64,
) -> StatusCode {
    if kernel.is_null() || out.is_null() {
        return SPIR_ERROR_NULL_POINTER;
    }

    let result = catch_unwind(|| unsafe {
        let k = &*kernel;
        *out = k.lambda();
        SPIR_SUCCESS
    });

    result.unwrap_or(SPIR_ERROR_PANIC)
}

/// Compute kernel value K(x, y)
///
/// # Arguments
/// * `kernel` - Kernel object
/// * `x` - First argument (typically in [-1, 1])
/// * `y` - Second argument (typically in [-1, 1])
/// * `out` - Pointer to store the result
///
/// # Returns
/// * `SPIR_SUCCESS` on success
/// * `SPIR_ERROR_NULL_POINTER` if kernel or out is null
/// * `SPIR_ERROR_PANIC` if internal panic occurs
#[no_mangle]
pub extern "C" fn spir_kernel_compute(
    kernel: *const SparseIRKernel,
    x: f64,
    y: f64,
    out: *mut f64,
) -> StatusCode {
    if kernel.is_null() || out.is_null() {
        return SPIR_ERROR_NULL_POINTER;
    }

    let result = catch_unwind(|| unsafe {
        let k = &*kernel;
        *out = k.compute(x, y);
        SPIR_SUCCESS
    });

    result.unwrap_or(SPIR_ERROR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn test_logistic_kernel_creation() {
        let mut kernel: *mut SparseIRKernel = ptr::null_mut();
        let status = spir_kernel_logistic_new(10.0, &mut kernel);
        
        assert_eq!(status, SPIR_SUCCESS);
        assert!(!kernel.is_null());
        
        spir_kernel_release(kernel);
    }

    #[test]
    fn test_regularized_bose_kernel_creation() {
        let mut kernel: *mut SparseIRKernel = ptr::null_mut();
        let status = spir_kernel_regularized_bose_new(10.0, &mut kernel);
        
        assert_eq!(status, SPIR_SUCCESS);
        assert!(!kernel.is_null());
        
        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_lambda() {
        let mut kernel: *mut SparseIRKernel = ptr::null_mut();
        spir_kernel_logistic_new(10.0, &mut kernel);
        
        let mut lambda = 0.0;
        let status = spir_kernel_lambda(kernel, &mut lambda);
        
        assert_eq!(status, SPIR_SUCCESS);
        assert_eq!(lambda, 10.0);
        
        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_compute() {
        let mut kernel: *mut SparseIRKernel = ptr::null_mut();
        spir_kernel_logistic_new(10.0, &mut kernel);
        
        let mut result = 0.0;
        let status = spir_kernel_compute(kernel, 0.5, 0.5, &mut result);
        
        assert_eq!(status, SPIR_SUCCESS);
        assert!(result > 0.0);  // Kernel should be positive
        
        spir_kernel_release(kernel);
    }

    #[test]
    fn test_null_pointer_errors() {
        // Null out pointer
        let status = spir_kernel_logistic_new(10.0, ptr::null_mut());
        assert_eq!(status, SPIR_ERROR_NULL_POINTER);
        
        // Null kernel pointer
        let mut lambda = 0.0;
        let status = spir_kernel_lambda(ptr::null(), &mut lambda);
        assert_eq!(status, SPIR_ERROR_NULL_POINTER);
    }

    #[test]
    fn test_invalid_lambda() {
        let mut kernel: *mut SparseIRKernel = ptr::null_mut();
        
        // Zero lambda
        let status = spir_kernel_logistic_new(0.0, &mut kernel);
        assert_eq!(status, SPIR_ERROR_INVALID_ARGUMENT);
        
        // Negative lambda
        let status = spir_kernel_logistic_new(-1.0, &mut kernel);
        assert_eq!(status, SPIR_ERROR_INVALID_ARGUMENT);
    }
}

