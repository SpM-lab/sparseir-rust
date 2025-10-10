use sparseir_rust::basis::FiniteTempBasis;
use sparseir_rust::kernel::LogisticKernel;
use sparseir_rust::sampling::TauSampling;
use sparseir_rust::traits::{Fermionic, Bosonic, StatisticsType};
use sparseir_rust::kernel::{KernelProperties, CentrosymmKernel};
use num_complex::Complex;
use mdarray::{Tensor, DynRank};

mod common;
use common::{SimpleRng, RandomGenerate, movedim, ErrorNorm};

/// Generic test for evaluate_nd/fit_nd roundtrip (generic over element type and statistics)
fn test_evaluate_nd_roundtrip<T, S>()
where
    T: RandomGenerate + num_complex::ComplexFloat + faer_traits::ComplexField + From<f64> + Copy + Default + ErrorNorm + 'static,
    S: StatisticsType,
    LogisticKernel: KernelProperties + CentrosymmKernel + Clone + 'static,
{
    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);
    
    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, epsilon, None);
    let sampling = TauSampling::new(&basis);
    
    let n_k = 5;
    let n_omega = 7;
    
    // Test for all dimensions (dim = 0, 1, 2)
    for dim in 0..3 {
        // Generate random test data (dim=0 format: [basis_size, n_k, n_omega])
        let (coeffs_0, _values_0) = generate_random_test_data::<T, _, _>(
            sampling.sampling_points(), &basis, 42 + dim as u64, &[n_k, n_omega]
        );
        
        // Move to target dimension
        // coeffs: [basis_size, n_k, n_omega] → move axis 0 to position dim
        let coeffs_dim = movedim(&coeffs_0, 0, dim);
        
        // Evaluate along target dimension (using generic API)
        let evaluated_values = sampling.evaluate_nd::<T>(&coeffs_dim, dim);
        
        // Fit back along target dimension (using generic API)
        let fitted_coeffs_dim = sampling.fit_nd::<T>(&evaluated_values, dim);
        
        // Move back to dim=0 for comparison
        let fitted_coeffs_0 = movedim(&fitted_coeffs_dim, dim, 0);
        
        let basis_size = basis.size();
        
        // Check roundtrip (compare in dim=0 format)
        for k in 0..n_k {
            for omega in 0..n_omega {
                for l in 0..basis_size {
                    let orig = coeffs_0[&[l, k, omega][..]];
                    let fitted = fitted_coeffs_0[&[l, k, omega][..]];
                    // ErrorNorm returns f64 for both f64 and Complex<f64>
                    let abs_error = (orig - fitted).error_norm();
                    
                    assert!(
                        abs_error < 1e-10,
                        "ND roundtrip (dim={}) error at ({},{},{}): error={}",
                        dim, l, k, omega, abs_error
                    );
                }
            }
        }
    }
}

#[test]
fn test_evaluate_nd_fermionic_real() {
    test_evaluate_nd_roundtrip::<f64, Fermionic>();
}

#[test]
fn test_evaluate_nd_fermionic_complex() {
    test_evaluate_nd_roundtrip::<Complex<f64>, Fermionic>();
}

#[test]
fn test_evaluate_nd_bosonic_real() {
    test_evaluate_nd_roundtrip::<f64, Bosonic>();
}

#[test]
fn test_evaluate_nd_bosonic_complex() {
    test_evaluate_nd_roundtrip::<Complex<f64>, Bosonic>();
}

// ============================================================================
// Test data generation helpers
// ============================================================================

/// Generate random test data with N-dimensional structure (generic over element type)
///
/// Creates coefficients and corresponding values at sampling points:
/// - Coefficients: shape [basis_size, ...extra_dims], randomly generated
/// - Values: shape [n_points, ...extra_dims], evaluated at sampling points
///
/// Coefficients are scaled by singular values for physical relevance.
///
/// # Type Parameters
/// * `T` - Element type (f64 or Complex<f64>)
///
/// # Arguments
/// * `sampling_points` - Sampling points in τ ∈ [0, β]
/// * `basis` - Finite temperature basis (needed for evaluation)
/// * `seed` - Random seed for reproducible coefficient generation
/// * `extra_dims` - Additional dimensions beyond basis/sampling (e.g., &[n_k, n_omega])
///
/// # Returns
/// (coeffs_tensor, values_tensor) where values are evaluated at sampling_points
#[allow(dead_code)]
fn generate_random_test_data<T, K, S>(
    sampling_points: &[f64],
    basis: &FiniteTempBasis<K, S>,
    seed: u64,
    extra_dims: &[usize],
) -> (Tensor<T, DynRank>, Tensor<T, DynRank>)
where
    T: RandomGenerate + num_complex::ComplexFloat + faer_traits::ComplexField + From<f64> + Copy + Default + 'static,
    K: KernelProperties + CentrosymmKernel + Clone + 'static,
    S: StatisticsType,
{
    let mut rng = SimpleRng::new(seed);
    
    let basis_size = basis.size();
    let n_points = sampling_points.len();
    
    let total_extra: usize = extra_dims.iter().product();
    let total_extra = if total_extra == 0 { 1 } else { total_extra };
    
    // Create coeffs shape: [basis_size, ...extra_dims]
    let mut coeffs_shape = vec![basis_size];
    coeffs_shape.extend_from_slice(extra_dims);
    
    let mut coeffs: Tensor<T, DynRank> = Tensor::zeros(&coeffs_shape[..]);
    
    // Generate random coefficients scaled by singular values
    for flat_idx in 0..total_extra {
        // Convert flat index to multi-index for extra dimensions
        let mut extra_idx = Vec::new();
        let mut remainder = flat_idx;
        for &dim_size in extra_dims.iter().rev() {
            extra_idx.push(remainder % dim_size);
            remainder /= dim_size;
        }
        extra_idx.reverse();
        
        // Generate coefficients for each basis function
        for l in 0..basis_size {
            // Random value in [0, 1) (or complex with re, im in [0, 1))
            let random_base: T = rng.next();
            
            // Map to [-1, 1): x * 2 - 1
            let random_centered = random_base * 2.0.into() - 1.0.into();
            
            // Scale by singular value for physical relevance
            let scaled_coeff = random_centered * basis.s[l].into();
            
            // Set coefficient at [l, ...extra_idx]
            let mut full_idx = vec![l];
            full_idx.extend_from_slice(&extra_idx);
            coeffs[&full_idx[..]] = scaled_coeff;
        }
    }
    
    // Evaluate coefficients at sampling points
    // values[i, ...extra_idx] = Σ_l coeffs[l, ...extra_idx] * u_l(τ_i)
    let mut values_shape = vec![n_points];
    values_shape.extend_from_slice(extra_dims);
    let mut values: Tensor<T, DynRank> = Tensor::zeros(&values_shape[..]);
    
    for flat_idx in 0..total_extra {
        // Convert flat index to multi-index for extra dimensions
        let mut extra_idx = Vec::new();
        let mut remainder = flat_idx;
        for &dim_size in extra_dims.iter().rev() {
            extra_idx.push(remainder % dim_size);
            remainder /= dim_size;
        }
        extra_idx.reverse();
        
        // Evaluate at each sampling point
        for (i, &tau) in sampling_points.iter().enumerate() {
            let mut value = T::zero();
            
            // Sum over basis functions: Σ_l coeffs[l, ...extra_idx] * u_l(τ)
            for l in 0..basis_size {
                let mut coeff_idx = vec![l];
                coeff_idx.extend_from_slice(&extra_idx);
                let coeff = coeffs[&coeff_idx[..]];
                
                // u_l(τ) is real, convert to type T
                let u_l_tau = basis.u[l].evaluate(tau);
                value = value + coeff * u_l_tau.into();
            }
            
            // Set value at [i, ...extra_idx]
            let mut value_idx = vec![i];
            value_idx.extend_from_slice(&extra_idx);
            values[&value_idx[..]] = value;
        }
    }
    
    (coeffs, values)
}

