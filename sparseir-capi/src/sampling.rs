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
use crate::utils::{copy_tensor_to_c_array, convert_dims_for_row_major, MemoryOrder};
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
    let result = catch_unwind(AssertUnwindSafe(|| {
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
            // DLR: not supported for tau sampling (DLR is discrete, not continuous)
            BasisType::DLRLogisticFermionic(_) |
            BasisType::DLRLogisticBosonic(_) |
            BasisType::DLRRegularizedBoseFermionic(_) |
            BasisType::DLRRegularizedBoseBosonic(_) => {
                return (std::ptr::null_mut(), SPIR_NOT_SUPPORTED);
            }
        };

        let sampling = spir_sampling {
            inner: sampling_type,
        };

        (Box::into_raw(Box::new(sampling)), SPIR_COMPUTATION_SUCCESS)
    }));

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
    let result = catch_unwind(AssertUnwindSafe(|| {
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
            // DLR: not supported for Matsubara sampling
            BasisType::DLRLogisticFermionic(_) |
            BasisType::DLRLogisticBosonic(_) |
            BasisType::DLRRegularizedBoseFermionic(_) |
            BasisType::DLRRegularizedBoseBosonic(_) => {
                return (std::ptr::null_mut(), SPIR_NOT_SUPPORTED);
            }
        };

        let sampling = spir_sampling {
            inner: sampling_type,
        };

        (Box::into_raw(Box::new(sampling)), SPIR_COMPUTATION_SUCCESS)
    }));

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

/// Creates a new tau sampling object with custom sampling points and pre-computed matrix
///
/// # Arguments
/// * `order` - Memory layout order (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
/// * `statistics` - Statistics type (SPIR_STATISTICS_FERMIONIC or SPIR_STATISTICS_BOSONIC)
/// * `basis_size` - Basis size
/// * `num_points` - Number of sampling points
/// * `points` - Array of sampling points in imaginary time (τ)
/// * `matrix` - Pre-computed matrix for the sampling points (num_points x basis_size)
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the newly created sampling object, or NULL if creation fails
///
/// # Safety
/// Caller must ensure `points` and `matrix` have correct sizes
#[no_mangle]
pub unsafe extern "C" fn spir_tau_sampling_new_with_matrix(
    order: libc::c_int,
    statistics: libc::c_int,
    basis_size: libc::c_int,
    num_points: libc::c_int,
    points: *const f64,
    matrix: *const f64,
    status: *mut StatusCode,
) -> *mut spir_sampling {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Validate inputs
        if points.is_null() || matrix.is_null() {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }
        if num_points <= 0 || basis_size <= 0 {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }

        // Parse order
        let mem_order = match MemoryOrder::from_c_int(order) {
            Ok(o) => o,
            Err(_) => return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT),
        };

        // Convert points to Vec
        let points_slice = std::slice::from_raw_parts(points, num_points as usize);
        let tau_points: Vec<f64> = points_slice.to_vec();

        // Convert matrix to Tensor
        let matrix_size = (num_points as usize) * (basis_size as usize);
        let matrix_slice = std::slice::from_raw_parts(matrix, matrix_size);
        let matrix_vec: Vec<f64> = matrix_slice.to_vec();
        
        // Convert dims based on order
        let orig_dims = vec![num_points as usize, basis_size as usize];
        let (dims, _) = convert_dims_for_row_major(&orig_dims, 0, mem_order);
        
        // Create tensor (mdarray is row-major)
        let flat_tensor = Tensor::<f64, (usize,)>::from(matrix_vec);
        let dyn_tensor = flat_tensor.into_dyn().reshape(&dims[..]).to_tensor();
        
        // Convert DynRank to fixed 2D shape
        let matrix_tensor: sparseir_rust::DTensor<f64, 2> = unsafe {
            std::mem::transmute(dyn_tensor)
        };

        // Create sampling based on statistics
        let sampling_type = match statistics {
            0 => {  // SPIR_STATISTICS_FERMIONIC
                let tau_sampling = sparseir_rust::sampling::TauSampling::<Fermionic>::from_matrix(
                    tau_points,
                    matrix_tensor,
                );
                SamplingType::TauFermionic(Arc::new(tau_sampling))
            }
            1 => {  // SPIR_STATISTICS_BOSONIC
                let tau_sampling = sparseir_rust::sampling::TauSampling::<Bosonic>::from_matrix(
                    tau_points,
                    matrix_tensor,
                );
                SamplingType::TauBosonic(Arc::new(tau_sampling))
            }
            _ => return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT),
        };

        let sampling = spir_sampling {
            inner: sampling_type,
        };

        (Box::into_raw(Box::new(sampling)), SPIR_COMPUTATION_SUCCESS)
    }));

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

/// Creates a new Matsubara sampling object with custom sampling points and pre-computed matrix
///
/// # Arguments
/// * `order` - Memory layout order (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
/// * `statistics` - Statistics type (SPIR_STATISTICS_FERMIONIC or SPIR_STATISTICS_BOSONIC)
/// * `basis_size` - Basis size
/// * `positive_only` - If true, only positive frequencies are used
/// * `num_points` - Number of sampling points
/// * `points` - Array of Matsubara frequency indices (n)
/// * `matrix` - Pre-computed complex matrix (num_points x basis_size)
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the newly created sampling object, or NULL if creation fails
///
/// # Safety
/// Caller must ensure `points` and `matrix` have correct sizes
#[no_mangle]
pub unsafe extern "C" fn spir_matsu_sampling_new_with_matrix(
    order: libc::c_int,
    statistics: libc::c_int,
    basis_size: libc::c_int,
    positive_only: bool,
    num_points: libc::c_int,
    points: *const i64,
    matrix: *const Complex64,
    status: *mut StatusCode,
) -> *mut spir_sampling {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Validate inputs
        if points.is_null() || matrix.is_null() {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }
        if num_points <= 0 || basis_size <= 0 {
            return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT);
        }

        // Parse order
        let mem_order = match MemoryOrder::from_c_int(order) {
            Ok(o) => o,
            Err(_) => return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT),
        };

        // Convert points to Vec<MatsubaraFreq>
        let points_slice = std::slice::from_raw_parts(points, num_points as usize);
        let matsu_points: Vec<i64> = points_slice.to_vec();
        
        use sparseir_rust::freq::MatsubaraFreq;

        // Convert matrix to Tensor
        let matrix_size = (num_points as usize) * (basis_size as usize);
        let matrix_slice = std::slice::from_raw_parts(matrix, matrix_size);
        let matrix_vec: Vec<Complex64> = matrix_slice.to_vec();
        
        // Convert dims based on order
        let orig_dims = vec![num_points as usize, basis_size as usize];
        let (dims, _) = convert_dims_for_row_major(&orig_dims, 0, mem_order);
        
        // Create tensor (mdarray is row-major)
        let flat_tensor = Tensor::<Complex64, (usize,)>::from(matrix_vec);
        let dyn_tensor = flat_tensor.into_dyn().reshape(&dims[..]).to_tensor();
        
        // Convert DynRank to fixed 2D shape
        let matrix_tensor: sparseir_rust::DTensor<Complex64, 2> = unsafe {
            std::mem::transmute(dyn_tensor)
        };

        // Create sampling based on statistics and positive_only
        let sampling_type = match (statistics, positive_only) {
            (0, true) => {  // Fermionic, positive-only
                let matsu_freqs: Vec<MatsubaraFreq<Fermionic>> = matsu_points
                    .iter()
                    .map(|&n| MatsubaraFreq::new(n).expect("Invalid Matsubara frequency"))
                    .collect();
                let matsu_sampling = sparseir_rust::matsubara_sampling::MatsubaraSamplingPositiveOnly::from_matrix(
                    matsu_freqs,
                    matrix_tensor.clone(),
                );
                SamplingType::MatsubaraPositiveOnlyFermionic(Arc::new(matsu_sampling))
            }
            (0, false) => {  // Fermionic, full range
                let matsu_freqs: Vec<MatsubaraFreq<Fermionic>> = matsu_points
                    .iter()
                    .map(|&n| MatsubaraFreq::new(n).expect("Invalid Matsubara frequency"))
                    .collect();
                let matsu_sampling = sparseir_rust::matsubara_sampling::MatsubaraSampling::from_matrix(
                    matsu_freqs,
                    matrix_tensor.clone(),
                );
                SamplingType::MatsubaraFermionic(Arc::new(matsu_sampling))
            }
            (1, true) => {  // Bosonic, positive-only
                let matsu_freqs: Vec<MatsubaraFreq<Bosonic>> = matsu_points
                    .iter()
                    .map(|&n| MatsubaraFreq::new(n).expect("Invalid Matsubara frequency"))
                    .collect();
                let matsu_sampling = sparseir_rust::matsubara_sampling::MatsubaraSamplingPositiveOnly::from_matrix(
                    matsu_freqs,
                    matrix_tensor.clone(),
                );
                SamplingType::MatsubaraPositiveOnlyBosonic(Arc::new(matsu_sampling))
            }
            (1, false) => {  // Bosonic, full range
                let matsu_freqs: Vec<MatsubaraFreq<Bosonic>> = matsu_points
                    .iter()
                    .map(|&n| MatsubaraFreq::new(n).expect("Invalid Matsubara frequency"))
                    .collect();
                let matsu_sampling = sparseir_rust::matsubara_sampling::MatsubaraSampling::from_matrix(
                    matsu_freqs,
                    matrix_tensor.clone(),
                );
                SamplingType::MatsubaraBosonic(Arc::new(matsu_sampling))
            }
            _ => return (std::ptr::null_mut(), SPIR_INVALID_ARGUMENT),
        };

        let sampling = spir_sampling {
            inner: sampling_type,
        };

        (Box::into_raw(Box::new(sampling)), SPIR_COMPUTATION_SUCCESS)
    }));

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
        
        // Parse order
        let mem_order = match MemoryOrder::from_c_int(order) {
            Ok(o) => o,
            Err(_) => return SPIR_INVALID_ARGUMENT,
        };

        let sampling_ref = &*s;
        let dims_slice = std::slice::from_raw_parts(input_dims, ndim as usize);
        let orig_dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();
        
        // Convert dims and target_dim for row-major mdarray
        let (dims, mdarray_target_dim) = convert_dims_for_row_major(
            &orig_dims,
            target_dim as usize,
            mem_order,
        );
        
        // Calculate total input size
        let total_input: usize = dims.iter().product();
        let input_slice = std::slice::from_raw_parts(input, total_input);
        
        // Create input tensor (mdarray is row-major)
        let input_vec: Vec<f64> = input_slice.to_vec();
        let flat_tensor = Tensor::<f64, (usize,)>::from(input_vec);
        let input_tensor = flat_tensor.into_dyn().reshape(&dims[..]).to_tensor();
        
        // Validate that input dimension matches basis size
        let expected_basis_size = match &sampling_ref.inner {
            SamplingType::TauFermionic(tau) => tau.basis_size(),
            SamplingType::TauBosonic(tau) => tau.basis_size(),
            _ => return SPIR_NOT_SUPPORTED,
        };
        
        if dims[mdarray_target_dim] != expected_basis_size {
            return crate::SPIR_INPUT_DIMENSION_MISMATCH;
        }
        
        // Evaluate based on sampling type (only tau sampling supports dd)
        let result_tensor = match &sampling_ref.inner {
            SamplingType::TauFermionic(tau) => {
                tau.evaluate_nd(&input_tensor, mdarray_target_dim)
            }
            SamplingType::TauBosonic(tau) => {
                tau.evaluate_nd(&input_tensor, mdarray_target_dim)
            }
            _ => return SPIR_NOT_SUPPORTED,
        };
        
        // Copy result to output (order-independent: flat copy)
        copy_tensor_to_c_array(result_tensor, out);
        
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
        
        // Parse order
        let mem_order = match MemoryOrder::from_c_int(order) {
            Ok(o) => o,
            Err(_) => return SPIR_INVALID_ARGUMENT,
        };

        let sampling_ref = &*s;
        let dims_slice = std::slice::from_raw_parts(input_dims, ndim as usize);
        let orig_dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();
        
        // Convert dims and target_dim for row-major mdarray
        let (dims, mdarray_target_dim) = convert_dims_for_row_major(
            &orig_dims,
            target_dim as usize,
            mem_order,
        );
        
        let total_input: usize = dims.iter().product();
        let input_slice = std::slice::from_raw_parts(input, total_input);
        
        let input_vec: Vec<f64> = input_slice.to_vec();
        let flat_tensor = Tensor::<f64, (usize,)>::from(input_vec);
        let input_tensor = flat_tensor.into_dyn().reshape(&dims[..]).to_tensor();
        
        // Evaluate based on sampling type (Matsubara positive-only)
        let result_tensor = match &sampling_ref.inner {
            SamplingType::MatsubaraPositiveOnlyFermionic(matsu) => {
                matsu.evaluate_nd(&input_tensor, mdarray_target_dim)
            }
            SamplingType::MatsubaraPositiveOnlyBosonic(matsu) => {
                matsu.evaluate_nd(&input_tensor, mdarray_target_dim)
            }
            _ => return SPIR_NOT_SUPPORTED,
        };
        
        // Copy result to output (order-independent: flat copy)
        copy_tensor_to_c_array(result_tensor, out);
        
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
        
        // Parse order
        let mem_order = match MemoryOrder::from_c_int(order) {
            Ok(o) => o,
            Err(_) => return SPIR_INVALID_ARGUMENT,
        };

        let sampling_ref = &*s;
        let dims_slice = std::slice::from_raw_parts(input_dims, ndim as usize);
        let orig_dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();
        
        // Convert dims and target_dim for row-major mdarray
        let (dims, mdarray_target_dim) = convert_dims_for_row_major(
            &orig_dims,
            target_dim as usize,
            mem_order,
        );
        
        let total_input: usize = dims.iter().product();
        let input_slice = std::slice::from_raw_parts(input, total_input);
        
        let input_vec: Vec<Complex64> = input_slice.to_vec();
        let flat_tensor = Tensor::<Complex64, (usize,)>::from(input_vec);
        let input_tensor = flat_tensor.into_dyn().reshape(&dims[..]).to_tensor();
        
        // Evaluate (full Matsubara sampling)
        let result_tensor = match &sampling_ref.inner {
            SamplingType::MatsubaraFermionic(matsu) => {
                matsu.evaluate_nd(&input_tensor, mdarray_target_dim)
            }
            SamplingType::MatsubaraBosonic(matsu) => {
                matsu.evaluate_nd(&input_tensor, mdarray_target_dim)
            }
            _ => return SPIR_NOT_SUPPORTED,
        };
        
        // Copy result to output (order-independent: flat copy)
        copy_tensor_to_c_array(result_tensor, out);
        
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
        
        // Parse order
        let mem_order = match MemoryOrder::from_c_int(order) {
            Ok(o) => o,
            Err(_) => return SPIR_INVALID_ARGUMENT,
        };

        let sampling_ref = &*s;
        let dims_slice = std::slice::from_raw_parts(input_dims, ndim as usize);
        let orig_dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();
        
        // Convert dims and target_dim for row-major mdarray
        let (dims, mdarray_target_dim) = convert_dims_for_row_major(
            &orig_dims,
            target_dim as usize,
            mem_order,
        );
        
        let total_input: usize = dims.iter().product();
        let input_slice = std::slice::from_raw_parts(input, total_input);
        
        let input_vec: Vec<f64> = input_slice.to_vec();
        let flat_tensor = Tensor::<f64, (usize,)>::from(input_vec);
        let input_tensor = flat_tensor.into_dyn().reshape(&dims[..]).to_tensor();
        
        // Fit (tau sampling only)
        let result_tensor = match &sampling_ref.inner {
            SamplingType::TauFermionic(tau) => {
                tau.fit_nd(&input_tensor, mdarray_target_dim)
            }
            SamplingType::TauBosonic(tau) => {
                tau.fit_nd(&input_tensor, mdarray_target_dim)
            }
            _ => return SPIR_NOT_SUPPORTED,
        };
        
        // Copy result to output (order-independent: flat copy)
        copy_tensor_to_c_array(result_tensor, out);
        
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
        
        // Parse order
        let mem_order = match MemoryOrder::from_c_int(order) {
            Ok(o) => o,
            Err(_) => return SPIR_INVALID_ARGUMENT,
        };

        let sampling_ref = &*s;
        let dims_slice = std::slice::from_raw_parts(input_dims, ndim as usize);
        let orig_dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();
        
        // Convert dims and target_dim for row-major mdarray
        let (dims, mdarray_target_dim) = convert_dims_for_row_major(
            &orig_dims,
            target_dim as usize,
            mem_order,
        );
        
        let total_input: usize = dims.iter().product();
        let input_slice = std::slice::from_raw_parts(input, total_input);
        
        let input_vec: Vec<Complex64> = input_slice.to_vec();
        let flat_tensor = Tensor::<Complex64, (usize,)>::from(input_vec);
        let input_tensor = flat_tensor.into_dyn().reshape(&dims[..]).to_tensor();
        
        // Fit (full Matsubara sampling)
        let result_tensor = match &sampling_ref.inner {
            SamplingType::MatsubaraFermionic(matsu) => {
                matsu.fit_nd(&input_tensor, mdarray_target_dim)
            }
            SamplingType::MatsubaraBosonic(matsu) => {
                matsu.fit_nd(&input_tensor, mdarray_target_dim)
            }
            _ => return SPIR_NOT_SUPPORTED,
        };
        
        // Copy result to output (order-independent: flat copy)
        copy_tensor_to_c_array(result_tensor, out);
        
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
        
        // Parse order
        let mem_order = match MemoryOrder::from_c_int(order) {
            Ok(o) => o,
            Err(_) => return SPIR_INVALID_ARGUMENT,
        };

        let sampling_ref = &*s;
        let dims_slice = std::slice::from_raw_parts(input_dims, ndim as usize);
        let orig_dims: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();
        
        // Convert dims and target_dim for row-major mdarray
        let (dims, mdarray_target_dim) = convert_dims_for_row_major(
            &orig_dims,
            target_dim as usize,
            mem_order,
        );
        
        let total_input: usize = dims.iter().product();
        let input_slice = std::slice::from_raw_parts(input, total_input);
        
        let input_vec: Vec<Complex64> = input_slice.to_vec();
        let flat_tensor = Tensor::<Complex64, (usize,)>::from(input_vec);
        let input_tensor = flat_tensor.into_dyn().reshape(&dims[..]).to_tensor();
        
        // Fit (positive-only Matsubara → real coefficients)
        let result_tensor = match &sampling_ref.inner {
            SamplingType::MatsubaraPositiveOnlyFermionic(matsu) => {
                matsu.fit_nd(&input_tensor, mdarray_target_dim)
            }
            SamplingType::MatsubaraPositiveOnlyBosonic(matsu) => {
                matsu.fit_nd(&input_tensor, mdarray_target_dim)
            }
            _ => return SPIR_NOT_SUPPORTED,
        };
        
        // Copy result to output (order-independent: flat copy)
        copy_tensor_to_c_array(result_tensor, out);
        
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

