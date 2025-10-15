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

// ============================================================================
// Evaluation Functions (coefficients → sampling points)
// ============================================================================

/// Evaluate basis coefficients at sampling points (double → double)
///
/// Transforms IR basis coefficients to values at sampling points.
///
/// # Note
/// Currently only supports column-major order (SPIR_ORDER_COLUMN_MAJOR = 1).
/// Row-major support will be added in a future update.
#[no_mangle]
pub unsafe extern "C" fn spir_sampling_eval_dd(
    s: *const spir_sampling,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    input: *const f64,
    out: *mut f64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Validate inputs
        if s.is_null() || input_dims.is_null() || input.is_null() || out.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }
        if ndim <= 0 || target_dim < 0 || target_dim >= ndim {
            return SPIR_INVALID_ARGUMENT;
        }
        
        // Only support column-major for now
        if order != crate::SPIR_ORDER_COLUMN_MAJOR {
            return SPIR_NOT_SUPPORTED;
        }

        let sampling_ref = &*s;
        let dims_slice = std::slice::from_raw_parts(input_dims, ndim as usize);
        let dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();
        
        // Calculate total input size
        let total_input: usize = dims.iter().product();
        let input_slice = std::slice::from_raw_parts(input, total_input);
        
        // Create input tensor from slice (column-major layout)
        // mdarray uses column-major by default, matching our input
        let input_vec: Vec<f64> = input_slice.to_vec();
        let flat_tensor = Tensor::<f64, (usize,)>::from(input_vec);
        let input_tensor = flat_tensor.into_dyn().reshape(&dims[..]).to_tensor();
        
        // Validate that input dimension matches basis size
        let expected_basis_size = match &sampling_ref.inner {
            SamplingType::TauFermionic(tau) => tau.basis_size(),
            SamplingType::TauBosonic(tau) => tau.basis_size(),
            _ => return SPIR_NOT_SUPPORTED,
        };
        
        if dims[target_dim as usize] != expected_basis_size {
            eprintln!("eval_dd: dimension mismatch! dims[{}] = {}, expected basis_size = {}",
                target_dim, dims[target_dim as usize], expected_basis_size);
            return crate::SPIR_INPUT_DIMENSION_MISMATCH;
        }
        
        // Evaluate based on sampling type (only tau sampling supports dd)
        let (result_tensor, n_points) = match &sampling_ref.inner {
            SamplingType::TauFermionic(tau) => {
                let n = tau.n_sampling_points();
                (tau.evaluate_nd(&input_tensor, target_dim as usize), n)
            }
            SamplingType::TauBosonic(tau) => {
                let n = tau.n_sampling_points();
                (tau.evaluate_nd(&input_tensor, target_dim as usize), n)
            }
            _ => return SPIR_NOT_SUPPORTED,
        };
        
        // Flatten result and copy to output (column-major)
        let total_output = result_tensor.len();
        let result_flat = result_tensor.into_dyn().reshape(&[total_output]).to_tensor();
        
        for i in 0..total_output {
            *out.add(i) = result_flat[&[i][..]];
        }
        
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Evaluate basis coefficients at sampling points (double → complex)
///
/// For Matsubara sampling: transforms real IR coefficients to complex values.
#[no_mangle]
pub unsafe extern "C" fn spir_sampling_eval_dz(
    s: *const spir_sampling,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    input: *const f64,
    out: *mut Complex64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Validate inputs
        if s.is_null() || input_dims.is_null() || input.is_null() || out.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }
        if ndim <= 0 || target_dim < 0 || target_dim >= ndim {
            return SPIR_INVALID_ARGUMENT;
        }
        
        // Only support column-major for now
        if order != crate::SPIR_ORDER_COLUMN_MAJOR {
            return SPIR_NOT_SUPPORTED;
        }

        let sampling_ref = &*s;
        let dims_slice = std::slice::from_raw_parts(input_dims, ndim as usize);
        let dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();
        
        let total_input: usize = dims.iter().product();
        let input_slice = std::slice::from_raw_parts(input, total_input);
        
        let input_vec: Vec<f64> = input_slice.to_vec();
        let flat_tensor = Tensor::<f64, (usize,)>::from(input_vec);
        let input_tensor = flat_tensor.into_dyn().reshape(&dims[..]).to_tensor();
        
        // Evaluate based on sampling type (Matsubara positive-only)
        let (result_tensor, n_points) = match &sampling_ref.inner {
            SamplingType::MatsubaraPositiveOnlyFermionic(matsu) => {
                let n = matsu.n_sampling_points();
                (matsu.evaluate_nd(&input_tensor, target_dim as usize), n)
            }
            SamplingType::MatsubaraPositiveOnlyBosonic(matsu) => {
                let n = matsu.n_sampling_points();
                (matsu.evaluate_nd(&input_tensor, target_dim as usize), n)
            }
            _ => return SPIR_NOT_SUPPORTED,
        };
        
        // Flatten result and copy to output (column-major)
        let total_output = result_tensor.len();
        let result_flat = result_tensor.into_dyn().reshape(&[total_output]).to_tensor();
        
        for i in 0..total_output {
            *out.add(i) = result_flat[&[i][..]];
        }
        
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Evaluate basis coefficients at sampling points (complex → complex)
///
/// For Matsubara sampling: transforms complex coefficients to complex values.
#[no_mangle]
pub unsafe extern "C" fn spir_sampling_eval_zz(
    s: *const spir_sampling,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    input: *const Complex64,
    out: *mut Complex64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || input_dims.is_null() || input.is_null() || out.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }
        if ndim <= 0 || target_dim < 0 || target_dim >= ndim {
            return SPIR_INVALID_ARGUMENT;
        }
        
        if order != crate::SPIR_ORDER_COLUMN_MAJOR {
            return SPIR_NOT_SUPPORTED;
        }

        let sampling_ref = &*s;
        let dims_slice = std::slice::from_raw_parts(input_dims, ndim as usize);
        let dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();
        
        let total_input: usize = dims.iter().product();
        let input_slice = std::slice::from_raw_parts(input, total_input);
        
        let input_vec: Vec<Complex64> = input_slice.to_vec();
        let flat_tensor = Tensor::<Complex64, (usize,)>::from(input_vec);
        let input_tensor = flat_tensor.into_dyn().reshape(&dims[..]).to_tensor();
        
        // Evaluate (full Matsubara sampling)
        let (result_tensor, n_points) = match &sampling_ref.inner {
            SamplingType::MatsubaraFermionic(matsu) => {
                let n = matsu.n_sampling_points();
                (matsu.evaluate_nd(&input_tensor, target_dim as usize), n)
            }
            SamplingType::MatsubaraBosonic(matsu) => {
                let n = matsu.n_sampling_points();
                (matsu.evaluate_nd(&input_tensor, target_dim as usize), n)
            }
            _ => return SPIR_NOT_SUPPORTED,
        };
        
        // Calculate output dimensions
        let mut output_dims = dims.clone();
        output_dims[target_dim as usize] = n_points;
        
        let total_output = result_tensor.len();
        
        for i in 0..total_output {
            let mut indices = vec![0; result_tensor.rank()];
            let mut remaining = i;
            for j in 0..result_tensor.rank() {
                indices[j] = remaining % output_dims[j];
                remaining /= output_dims[j];
            }
            *out.add(i) = result_tensor[&indices[..]];
        }
        
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

// ============================================================================
// Fitting Functions (sampling points → coefficients)
// ============================================================================

/// Fit basis coefficients from sampling point values (double → double)
#[no_mangle]
pub unsafe extern "C" fn spir_sampling_fit_dd(
    s: *const spir_sampling,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    input: *const f64,
    out: *mut f64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || input_dims.is_null() || input.is_null() || out.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }
        if ndim <= 0 || target_dim < 0 || target_dim >= ndim {
            return SPIR_INVALID_ARGUMENT;
        }
        
        if order != crate::SPIR_ORDER_COLUMN_MAJOR {
            return SPIR_NOT_SUPPORTED;
        }

        let sampling_ref = &*s;
        let dims_slice = std::slice::from_raw_parts(input_dims, ndim as usize);
        let dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();
        
        let total_input: usize = dims.iter().product();
        let input_slice = std::slice::from_raw_parts(input, total_input);
        
        let input_vec: Vec<f64> = input_slice.to_vec();
        let flat_tensor = Tensor::<f64, (usize,)>::from(input_vec);
        let input_tensor = flat_tensor.into_dyn().reshape(&dims[..]).to_tensor();
        
        // Fit (tau sampling only)
        let (result_tensor, basis_size) = match &sampling_ref.inner {
            SamplingType::TauFermionic(tau) => {
                let bs = tau.basis_size();
                (tau.fit_nd(&input_tensor, target_dim as usize), bs)
            }
            SamplingType::TauBosonic(tau) => {
                let bs = tau.basis_size();
                (tau.fit_nd(&input_tensor, target_dim as usize), bs)
            }
            _ => return SPIR_NOT_SUPPORTED,
        };
        
        // Flatten result and copy to output (column-major)
        let total_output = result_tensor.len();
        let result_flat = result_tensor.into_dyn().reshape(&[total_output]).to_tensor();
        
        for i in 0..total_output {
            *out.add(i) = result_flat[&[i][..]];
        }
        
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Fit basis coefficients from sampling point values (complex → complex)
#[no_mangle]
pub unsafe extern "C" fn spir_sampling_fit_zz(
    s: *const spir_sampling,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    input: *const Complex64,
    out: *mut Complex64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || input_dims.is_null() || input.is_null() || out.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }
        if ndim <= 0 || target_dim < 0 || target_dim >= ndim {
            return SPIR_INVALID_ARGUMENT;
        }
        
        if order != crate::SPIR_ORDER_COLUMN_MAJOR {
            return SPIR_NOT_SUPPORTED;
        }

        let sampling_ref = &*s;
        let dims_slice = std::slice::from_raw_parts(input_dims, ndim as usize);
        let dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();
        
        let total_input: usize = dims.iter().product();
        let input_slice = std::slice::from_raw_parts(input, total_input);
        
        let input_vec: Vec<Complex64> = input_slice.to_vec();
        let flat_tensor = Tensor::<Complex64, (usize,)>::from(input_vec);
        let input_tensor = flat_tensor.into_dyn().reshape(&dims[..]).to_tensor();
        
        // Fit (full Matsubara sampling)
        let (result_tensor, basis_size) = match &sampling_ref.inner {
            SamplingType::MatsubaraFermionic(matsu) => {
                let bs = matsu.basis_size();
                (matsu.fit_nd(&input_tensor, target_dim as usize), bs)
            }
            SamplingType::MatsubaraBosonic(matsu) => {
                let bs = matsu.basis_size();
                (matsu.fit_nd(&input_tensor, target_dim as usize), bs)
            }
            _ => return SPIR_NOT_SUPPORTED,
        };
        
        // Calculate output dimensions
        let mut output_dims = dims.clone();
        output_dims[target_dim as usize] = basis_size;
        
        let total_output = result_tensor.len();
        
        for i in 0..total_output {
            let mut indices = vec![0; result_tensor.rank()];
            let mut remaining = i;
            for j in 0..result_tensor.rank() {
                indices[j] = remaining % output_dims[j];
                remaining /= output_dims[j];
            }
            *out.add(i) = result_tensor[&indices[..]];
        }
        
        SPIR_COMPUTATION_SUCCESS
    }));

    result.unwrap_or(crate::SPIR_INTERNAL_ERROR)
}

/// Fit basis coefficients from Matsubara sampling (complex → double, positive only)
#[no_mangle]
pub unsafe extern "C" fn spir_sampling_fit_zd(
    s: *const spir_sampling,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    input: *const Complex64,
    out: *mut f64,
) -> StatusCode {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if s.is_null() || input_dims.is_null() || input.is_null() || out.is_null() {
            return SPIR_INVALID_ARGUMENT;
        }
        if ndim <= 0 || target_dim < 0 || target_dim >= ndim {
            return SPIR_INVALID_ARGUMENT;
        }
        
        if order != crate::SPIR_ORDER_COLUMN_MAJOR {
            return SPIR_NOT_SUPPORTED;
        }

        let sampling_ref = &*s;
        let dims_slice = std::slice::from_raw_parts(input_dims, ndim as usize);
        let dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();
        
        let total_input: usize = dims.iter().product();
        let input_slice = std::slice::from_raw_parts(input, total_input);
        
        let input_vec: Vec<Complex64> = input_slice.to_vec();
        let flat_tensor = Tensor::<Complex64, (usize,)>::from(input_vec);
        let input_tensor = flat_tensor.into_dyn().reshape(&dims[..]).to_tensor();
        
        // Fit (positive-only Matsubara → real coefficients)
        let (result_tensor, basis_size) = match &sampling_ref.inner {
            SamplingType::MatsubaraPositiveOnlyFermionic(matsu) => {
                let bs = matsu.basis_size();
                (matsu.fit_nd(&input_tensor, target_dim as usize), bs)
            }
            SamplingType::MatsubaraPositiveOnlyBosonic(matsu) => {
                let bs = matsu.basis_size();
                (matsu.fit_nd(&input_tensor, target_dim as usize), bs)
            }
            _ => return SPIR_NOT_SUPPORTED,
        };
        
        // Calculate output dimensions
        let mut output_dims = dims.clone();
        output_dims[target_dim as usize] = basis_size;
        
        let total_output = result_tensor.len();
        
        for i in 0..total_output {
            let mut indices = vec![0; result_tensor.rank()];
            let mut remaining = i;
            for j in 0..result_tensor.rank() {
                indices[j] = remaining % output_dims[j];
                remaining /= output_dims[j];
            }
            *out.add(i) = result_tensor[&indices[..]];
        }
        
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

            let sve = crate::spir_sve_result_new(kernel,1e-6, -1.0, -1, -1, -1, &mut status);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

            // Limit basis size to 5
            let basis = crate::spir_basis_new(1, 10.0, 1.0,1e-6, kernel, sve, 5, &mut status);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

            // Get actual basis size
            let mut actual_basis_size = 0;
            let ret = crate::spir_basis_get_size(basis, &mut actual_basis_size);
            assert_eq!(ret, SPIR_COMPUTATION_SUCCESS);

            // Create tau sampling with enough points (at least basis_size)
            let tau_points: Vec<f64> = (0..actual_basis_size)
                .map(|i| (i as f64 + 1.0) * 10.0 / (actual_basis_size as f64 + 1.0))
                .collect();
            
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
            assert_eq!(n_points, actual_basis_size);

            // Get tau points back
            let mut retrieved_points = vec![0.0; actual_basis_size as usize];
            let ret = spir_sampling_get_taus(sampling, retrieved_points.as_mut_ptr());
            assert_eq!(ret, SPIR_COMPUTATION_SUCCESS);
            
            // Check that retrieved points match
            for (i, (&retrieved, &original)) in retrieved_points.iter().zip(tau_points.iter()).enumerate() {
                assert!((retrieved - original).abs() < 1e-10, 
                    "Point {} mismatch: {} vs {}", i, retrieved, original);
            }

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

