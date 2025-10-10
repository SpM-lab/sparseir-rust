mod common;

use sparseir_rust::{LogisticKernel, FiniteTempBasis};
use sparseir_rust::matsubara_sampling::{MatsubaraSampling, MatsubaraSamplingPositiveOnly};
use sparseir_rust::freq::MatsubaraFreq;
use sparseir_rust::traits::{Fermionic, Bosonic, StatisticsType};
use num_complex::Complex;
use common::{generate_test_data_tau_and_matsubara, ErrorNorm};

/// Test MatsubaraSampling (symmetric mode, complex coefficients) roundtrip
#[test]
fn test_matsubara_sampling_roundtrip_fermionic() {
    test_matsubara_sampling_roundtrip_generic::<Fermionic>();
}

#[test]
fn test_matsubara_sampling_roundtrip_bosonic() {
    test_matsubara_sampling_roundtrip_generic::<Bosonic>();
}

fn test_matsubara_sampling_roundtrip_generic<S: StatisticsType + 'static>() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;
    
    // Create basis
    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, Some(epsilon), None);
    
    // Create symmetric Matsubara sampling points (positive and negative)
    let sampling_points = basis.default_matsubara_sampling_points(false);
    
    // Create sampling
    let sampling = MatsubaraSampling::with_sampling_points(&basis, sampling_points.clone());
    
    // Generate test data (we only need Matsubara values)
    let (_coeffs_random, _gtau_values, giwn_values) = 
        generate_test_data_tau_and_matsubara::<Complex<f64>, S, _>(
            &basis,
            &[0.5 * beta], // dummy tau point
            &sampling_points,
            12345,
        );
    
    // Fit to get coefficients
    let coeffs_fitted = sampling.fit(&giwn_values);
    
    // Evaluate back
    let giwn_reconstructed = sampling.evaluate(&coeffs_fitted);
    
    // Check roundtrip accuracy
    let max_error = giwn_values.iter()
        .zip(giwn_reconstructed.iter())
        .map(|(a, b)| (*a - *b).error_norm())
        .fold(0.0f64, f64::max);
    
    println!("MatsubaraSampling {:?} roundtrip max error: {}", S::STATISTICS, max_error);
    assert!(max_error < 1e-7, "Roundtrip error too large: {}", max_error);
}

/// Test MatsubaraSamplingPositiveOnly (positive frequencies only, real coefficients) roundtrip
#[test]
fn test_matsubara_sampling_positive_only_roundtrip_fermionic() {
    test_matsubara_sampling_positive_only_roundtrip_generic::<Fermionic>();
}

#[test]
fn test_matsubara_sampling_positive_only_roundtrip_bosonic() {
    test_matsubara_sampling_positive_only_roundtrip_generic::<Bosonic>();
}

fn test_matsubara_sampling_positive_only_roundtrip_generic<S: StatisticsType + 'static>() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;
    
    // Create basis
    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, Some(epsilon), None);
    
    // Use positive-only sampling points
    let sampling_points = basis.default_matsubara_sampling_points(true);
    let n_matsubara = sampling_points.len();
    
    // Create sampling
    let sampling = MatsubaraSamplingPositiveOnly::with_sampling_points(&basis, sampling_points.clone());
    
    // Generate test data (we only need Matsubara values)
    let (_coeffs_random, _gtau_values, giwn_values) = 
        generate_test_data_tau_and_matsubara::<f64, S, _>(
            &basis,
            &[0.5 * beta], // dummy tau point
            &sampling_points,
            12345,
        );
    
    // Fit to get real coefficients
    let coeffs_fitted = sampling.fit(&giwn_values);
    
    // Evaluate back
    let giwn_reconstructed = sampling.evaluate(&coeffs_fitted);
    
    // Check roundtrip accuracy
    let max_error = giwn_values.iter()
        .zip(giwn_reconstructed.iter())
        .map(|(a, b)| (*a - *b).error_norm())
        .fold(0.0f64, f64::max);
    
    println!("MatsubaraSamplingPositiveOnly {:?} roundtrip max error: {}", S::STATISTICS, max_error);
    assert!(max_error < 1e-7, "Roundtrip error too large: {}", max_error);
}

/// Test that basis sizes are consistent
#[test]
fn test_matsubara_sampling_dimensions() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;
    
    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, Some(epsilon), None);
    
    let sampling_points = basis.default_matsubara_sampling_points(true);
    
    let sampling = MatsubaraSamplingPositiveOnly::with_sampling_points(&basis, sampling_points.clone());
    
    assert_eq!(sampling.basis_size(), basis.size());
    assert_eq!(sampling.n_sampling_points(), sampling_points.len());
}

/// Test MatsubaraSampling evaluate_nd/fit_nd roundtrip
#[test]
fn test_matsubara_sampling_nd_roundtrip_fermionic() {
    test_matsubara_sampling_nd_roundtrip_generic::<Fermionic>();
}

#[test]
fn test_matsubara_sampling_nd_roundtrip_bosonic() {
    test_matsubara_sampling_nd_roundtrip_generic::<Bosonic>();
}

fn test_matsubara_sampling_nd_roundtrip_generic<S: StatisticsType + 'static>() {
    use mdarray::Tensor;
    use num_complex::Complex;
    use common::generate_nd_test_data;
    
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;
    
    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, Some(epsilon), None);
    
    let sampling_points = basis.default_matsubara_sampling_points(false);  // Symmetric (positive and negative)
    
    let sampling = MatsubaraSampling::with_sampling_points(&basis, sampling_points.clone());
    
    let n_k = 4;
    let n_omega = 5;
    
    // Test for all dimensions (dim = 0, 1, 2)
    for dim in 0..3 {
        // Generate test data (dim=0 format: [basis_size, n_k, n_omega])
        let (coeffs_0, _gtau_0, _giwn_0) = generate_nd_test_data::<Complex<f64>, _, _>(
            &basis, &[], &sampling_points, 42 + dim as u64, &[n_k, n_omega]
        );
        
        // Move to target dimension
        let coeffs_dim = common::movedim(&coeffs_0, 0, dim);
        
        // Evaluate and fit along target dimension
        let values_dim = sampling.evaluate_nd(&coeffs_dim, dim);
        let coeffs_fitted_dim = sampling.fit_nd(&values_dim, dim);
        
        // Move back to dim=0 for comparison
        let coeffs_fitted_0 = common::movedim(&coeffs_fitted_dim, dim, 0);
        
        // Check roundtrip
        let max_error = coeffs_0.iter().zip(coeffs_fitted_0.iter())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0, f64::max);
        
        println!("MatsubaraSampling {:?} dim={} roundtrip error: {}", S::STATISTICS, dim, max_error);
        assert!(max_error < 1e-10, "ND roundtrip (dim={}) error too large: {}", dim, max_error);
    }
}

/// Test MatsubaraSamplingPositiveOnly evaluate_nd/fit_nd roundtrip
#[test]
fn test_matsubara_sampling_positive_only_nd_roundtrip_fermionic() {
    test_matsubara_sampling_positive_only_nd_roundtrip_generic::<Fermionic>();
}

#[test]
fn test_matsubara_sampling_positive_only_nd_roundtrip_bosonic() {
    test_matsubara_sampling_positive_only_nd_roundtrip_generic::<Bosonic>();
}

fn test_matsubara_sampling_positive_only_nd_roundtrip_generic<S: StatisticsType + 'static>() {
    use mdarray::Tensor;
    use common::generate_nd_test_data;
    
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;
    
    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, Some(epsilon), None);
    
    // Use positive-only sampling points
    let sampling_points = basis.default_matsubara_sampling_points(true);
    
    let sampling = MatsubaraSamplingPositiveOnly::with_sampling_points(&basis, sampling_points.clone());
    
    let n_k = 4;
    let n_omega = 5;
    
    // Test for all dimensions (dim = 0, 1, 2)
    for dim in 0..3 {
        // Generate test data (dim=0 format: [basis_size, n_k, n_omega])
        let (coeffs_0, _gtau_0, _giwn_0) = generate_nd_test_data::<f64, _, _>(
            &basis, &[], &sampling_points, 42 + dim as u64, &[n_k, n_omega]
        );
        
        // Move to target dimension
        let coeffs_dim = common::movedim(&coeffs_0, 0, dim);
        
        // Evaluate and fit along target dimension
        let values_dim = sampling.evaluate_nd(&coeffs_dim, dim);
        let coeffs_fitted_dim = sampling.fit_nd(&values_dim, dim);
        
        // Move back to dim=0 for comparison
        let coeffs_fitted_0 = common::movedim(&coeffs_fitted_dim, dim, 0);
        
        // Check roundtrip
        let max_error = coeffs_0.iter().zip(coeffs_fitted_0.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0, f64::max);
        
        println!("MatsubaraSamplingPositiveOnly {:?} dim={} roundtrip error: {}", S::STATISTICS, dim, max_error);
        assert!(max_error < 1e-7, "ND roundtrip (dim={}) error too large: {}", dim, max_error);
    }
}

