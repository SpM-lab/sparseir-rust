//! Main SVE computation functions

use mdarray::DTensor;
use crate::numeric::CustomNumeric;
use crate::kernel::{CentrosymmKernel, KernelProperties, SVEHints};

use super::result::SVEResult;
use super::strategy::{SVEStrategy, CentrosymmSVE};
use super::types::{TworkType, SVDStrategy, safe_epsilon};

/// Main SVE computation function
/// 
/// Automatically chooses the appropriate SVE strategy based on kernel properties
/// and working precision based on epsilon.
/// 
/// # Arguments
/// 
/// * `kernel` - The kernel to expand
/// * `epsilon` - Required accuracy
/// * `cutoff` - Relative tolerance for singular value truncation
/// * `max_num_svals` - Maximum number of singular values to keep
/// * `twork` - Working precision type (Auto for automatic selection)
/// 
/// # Returns
/// 
/// SVEResult containing singular functions and values
pub fn compute_sve<K>(
    kernel: K,
    epsilon: f64,
    cutoff: Option<f64>,
    max_num_svals: Option<usize>,
    twork: TworkType,
) -> SVEResult
where
    K: CentrosymmKernel + KernelProperties + Clone + 'static,
{
    // Determine safe epsilon and working precision
    let (safe_epsilon, twork_actual, _svd_strategy) = 
        safe_epsilon(epsilon, twork, SVDStrategy::Auto);
    
    // Dispatch based on working precision
    match twork_actual {
        TworkType::Float64 => {
            compute_sve_with_precision::<f64, K>(
                kernel, 
                safe_epsilon, 
                cutoff, 
                max_num_svals
            )
        }
        TworkType::Float64X2 => {
            compute_sve_with_precision::<twofloat::TwoFloat, K>(
                kernel, 
                safe_epsilon, 
                cutoff, 
                max_num_svals
            )
        }
        _ => panic!("Invalid TworkType: {:?}", twork_actual),
    }
}

/// Compute SVE with specific precision type
fn compute_sve_with_precision<T, K>(
    kernel: K,
    epsilon: f64,
    cutoff: Option<f64>,
    max_num_svals: Option<usize>,
) -> SVEResult
where
    T: CustomNumeric + Send + Sync + Clone + 'static,
    K: CentrosymmKernel + KernelProperties + Clone + 'static,
    K::SVEHintsType<T>: SVEHints<T> + Clone,
{
    // 1. Determine SVE strategy (automatically chooses CentrosymmSVE for centrosymmetric kernels)
    let sve = determine_sve::<T, K>(kernel, epsilon);
    
    // 2. Compute matrices
    let matrices = sve.matrices();
    
    // 3. Compute SVD for each matrix
    let mut u_list = Vec::new();
    let mut s_list = Vec::new();
    let mut v_list = Vec::new();
    
    for matrix in matrices.iter() {
        let (u, s, v) = compute_svd(matrix);
        
        u_list.push(u);
        s_list.push(s);
        v_list.push(v);
    }
    
    // 4. Truncate based on cutoff
    let rtol = cutoff.unwrap_or(2.0 * f64::EPSILON);
    let rtol_t = T::from_f64(rtol);
    let (u_trunc, s_trunc, v_trunc) = truncate(
        u_list, 
        s_list, 
        v_list, 
        rtol_t, 
        max_num_svals
    );
    
    // 5. Post-process to create SVEResult
    sve.postprocess(u_trunc, s_trunc, v_trunc)
}

/// Determine the appropriate SVE strategy
/// 
/// For centrosymmetric kernels, uses CentrosymmSVE for efficient computation
/// by exploiting even/odd symmetry.
fn determine_sve<T, K>(
    kernel: K,
    epsilon: f64,
) -> Box<dyn SVEStrategy<T>>
where
    T: CustomNumeric + Send + Sync + Clone + 'static,
    K: CentrosymmKernel + KernelProperties + Clone + 'static,
    K::SVEHintsType<T>: SVEHints<T> + Clone,
{
    // CentrosymmKernel trait implies centrosymmetric
    Box::new(CentrosymmSVE::new(kernel, epsilon))
}

/// Compute SVD of a matrix
/// 
/// Uses xprec-svd for high-precision SVD computation.
/// 
/// # Returns
/// 
/// Tuple of (U, singular_values, V) where A = U * S * V^T
pub fn compute_svd<T: CustomNumeric + 'static>(
    matrix: &DTensor<T, 2>
) -> (DTensor<T, 2>, Vec<T>, DTensor<T, 2>) {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        compute_svd_f64_xprec(matrix)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<twofloat::TwoFloat>() {
        compute_svd_twofloat_xprec(matrix)
    } else {
        compute_svd_f64_xprec(matrix)
    }
}

/// Compute SVD for f64 using mdarray-linalg (Faer backend)
fn compute_svd_f64_xprec<T: CustomNumeric>(
    matrix: &DTensor<T, 2>
) -> (DTensor<T, 2>, Vec<T>, DTensor<T, 2>) {
    use mdarray_linalg::SVD;
    use mdarray_linalg_faer::Faer;
    
    let matrix_f64 = DTensor::<f64, 2>::from_fn(*matrix.shape(), |idx| matrix[idx].to_f64());
    
    // Clone for SVD (mdarray-linalg requires &mut)
    let mut matrix_copy = matrix_f64.clone();
    let result = Faer.svd(&mut matrix_copy)
        .expect("SVD computation failed");
    
    // Extract singular values from first row (mdarray-linalg stores in s[[0, i]])
    let min_dim = result.s.shape().0.min(result.s.shape().1);
    let s_vec: Vec<T> = (0..min_dim)
        .map(|i| T::from_f64(result.s[[0, i]]))  // First row, not diagonal!
        .collect();
    
    // Convert back to T
    let u = DTensor::<T, 2>::from_fn(*result.u.shape(), |idx| T::from_f64(result.u[idx]));
    
    // mdarray-linalg returns vt (V^T), but we need v (V)
    // First convert vt to type T, then transpose
    let vt = DTensor::<T, 2>::from_fn(*result.vt.shape(), |idx| T::from_f64(result.vt[idx]));
    let v = vt.transpose().to_tensor();  // (V^T)^T = V
    
    (u, s_vec, v)
}

/// Compute SVD for TwoFloat using xprec-svd
fn compute_svd_twofloat_xprec<T: CustomNumeric>(
    matrix: &DTensor<T, 2>
) -> (DTensor<T, 2>, Vec<T>, DTensor<T, 2>) {
    let matrix_twofloat = DTensor::<xprec_svd::TwoFloatPrecision, 2>::from_fn(*matrix.shape(), |idx| {
        xprec_svd::TwoFloatPrecision::from_f64(matrix[idx].to_f64())
    });
    
    let result = xprec_svd::jacobi_svd(&matrix_twofloat);
    
    // Convert back to T (via f64 since TwoFloatPrecision needs to_f64())
    let u = DTensor::<T, 2>::from_fn(*result.u.shape(), |idx| {
        T::from_f64(result.u[idx].to_f64())
    });
    let s: Vec<T> = result.s.iter().map(|x| {
        T::from_f64(x.to_f64())
    }).collect();
    let v = DTensor::<T, 2>::from_fn(*result.v.shape(), |idx| {
        T::from_f64(result.v[idx].to_f64())
    });
    
    (u, s, v)
}

/// Truncate SVD results based on cutoff and maximum size
/// 
/// # Arguments
/// 
/// * `u_list` - List of U matrices
/// * `s_list` - List of singular value vectors
/// * `v_list` - List of V matrices
/// * `rtol` - Relative tolerance for truncation
/// * `max_num_svals` - Maximum number of singular values to keep
/// 
/// # Returns
/// 
/// Tuple of (truncated_u_list, truncated_s_list, truncated_v_list)
pub fn truncate<T: CustomNumeric>(
    u_list: Vec<DTensor<T, 2>>,
    s_list: Vec<Vec<T>>,
    v_list: Vec<DTensor<T, 2>>,
    rtol: T,
    max_num_svals: Option<usize>,
) -> (Vec<DTensor<T, 2>>, Vec<Vec<T>>, Vec<DTensor<T, 2>>) {
    let zero = T::zero();
    
    // Validate
    if let Some(max) = max_num_svals {
        if (max as isize) < 0 {
            panic!("max_num_svals must be non-negative");
        }
    }
    if rtol < zero || rtol > T::from_f64(1.0) {
        panic!("rtol must be in [0, 1]");
    }
    
    // Find global maximum singular value
    let mut all_svals = Vec::new();
    for s in &s_list {
        all_svals.extend(s.iter().copied());
    }
    
    let max_sval = all_svals.iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .unwrap_or(zero);
    
    // Determine cutoff
    let cutoff = if let Some(max_count) = max_num_svals {
        if max_count < all_svals.len() {
            let mut sorted = all_svals.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let nth = sorted[max_count - 1];
            if rtol * max_sval > nth {
                rtol * max_sval
            } else {
                nth
            }
        } else {
            rtol * max_sval
        }
    } else {
        rtol * max_sval
    };
    
    // Truncate each result
    let mut u_trunc = Vec::new();
    let mut s_trunc = Vec::new();
    let mut v_trunc = Vec::new();
    
    for i in 0..s_list.len() {
        let s = &s_list[i];
        let u = &u_list[i];
        let v = &v_list[i];
        
        // Count singular values above cutoff
        let mut n_keep = 0;
        for &val in s.iter() {
            if val >= cutoff {
                n_keep += 1;
            }
        }
        
        if n_keep > 0 {
            // Slice U: keep first n_keep columns
            let u_shape = *u.shape();
            let u_sliced = DTensor::<T, 2>::from_fn([u_shape.0, n_keep], |idx| {
                u[[idx[0], idx[1]]]
            });
            u_trunc.push(u_sliced);
            
            s_trunc.push(s[..n_keep].to_vec());
            
            // Slice V: keep first n_keep columns
            let v_shape = *v.shape();
            let v_sliced = DTensor::<T, 2>::from_fn([v_shape.0, n_keep], |idx| {
                v[[idx[0], idx[1]]]
            });
            v_trunc.push(v_sliced);
        }
    }
    
    (u_trunc, s_trunc, v_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_by_rtol() {
        let u = vec![DTensor::<f64, 2>::from_elem([3, 3], 1.0)];
        let s = vec![vec![10.0, 5.0, 0.1]];
        let v = vec![DTensor::<f64, 2>::from_elem([3, 3], 1.0)];
        
        // rtol = 0.1, max_sval = 10.0, cutoff = 1.0
        // Keep values >= 1.0: [10.0, 5.0]
        let (_, s_trunc, _) = truncate(u, s, v, 0.1, None);
        
        assert_eq!(s_trunc[0].len(), 2);
        assert_eq!(s_trunc[0][0], 10.0);
        assert_eq!(s_trunc[0][1], 5.0);
    }

    #[test]
    fn test_truncate_by_max_size() {
        let u = vec![DTensor::<f64, 2>::from_elem([3, 3], 1.0)];
        let s = vec![vec![10.0, 5.0, 2.0]];
        let v = vec![DTensor::<f64, 2>::from_elem([3, 3], 1.0)];
        
        // max_num_svals = 2
        let (_, s_trunc, _) = truncate(u, s, v, 0.0, Some(2));
        
        assert_eq!(s_trunc[0].len(), 2);
    }

    #[test]
    #[should_panic(expected = "rtol must be in [0, 1]")]
    fn test_truncate_invalid_rtol() {
        let u = vec![DTensor::<f64, 2>::from_elem([1, 1], 1.0)];
        let s = vec![vec![1.0]];
        let v = vec![DTensor::<f64, 2>::from_elem([1, 1], 1.0)];
        
        truncate(u, s, v, 1.5, None);
    }
}

