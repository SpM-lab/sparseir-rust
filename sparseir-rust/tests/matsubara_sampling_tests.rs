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

fn test_matsubara_sampling_roundtrip_generic<S: StatisticsType>() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;
    
    // Create basis
    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, Some(epsilon), None);
    
    // Create Matsubara sampling points (symmetric: negative and positive)
    // For Fermions: use odd n (... -3, -1, 1, 3, ...)
    // For Bosons: use even n (... -4, -2, 0, 2, 4, ...)
    let n_matsubara = basis.size();
    let sampling_points: Vec<MatsubaraFreq<S>> = (-(n_matsubara as i64)..(n_matsubara as i64))
        .filter_map(|k| {
            // k -> n: for Fermions n=2k+1, for Bosons n=2k
            let n = match S::STATISTICS {
                sparseir_rust::traits::Statistics::Fermionic => 2 * k + 1,
                sparseir_rust::traits::Statistics::Bosonic => 2 * k,
            };
            MatsubaraFreq::<S>::new(n).ok()
        })
        .take(n_matsubara)
        .collect();
    
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
    assert!(max_error < 1e-10, "Roundtrip error too large: {}", max_error);
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

fn test_matsubara_sampling_positive_only_roundtrip_generic<S: StatisticsType>() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;
    
    // Create basis
    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, Some(epsilon), None);
    
    // Create Matsubara sampling points (positive only)
    // For Fermions: use odd positive n (1, 3, 5, ...)
    // For Bosons: use even non-negative n (0, 2, 4, ...)
    let n_matsubara = basis.size();
    let sampling_points: Vec<MatsubaraFreq<S>> = (0..2 * n_matsubara as i64)
        .filter_map(|k| {
            // k -> n: for Fermions n=2k+1, for Bosons n=2k
            let n = match S::STATISTICS {
                sparseir_rust::traits::Statistics::Fermionic => 2 * k + 1,
                sparseir_rust::traits::Statistics::Bosonic => 2 * k,
            };
            MatsubaraFreq::<S>::new(n).ok()
        })
        .take(n_matsubara)
        .collect();
    
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
    
    let n_matsubara = basis.size();
    let sampling_points: Vec<MatsubaraFreq<Fermionic>> = (0..n_matsubara as i64)
        .map(|k| MatsubaraFreq::new(2 * k + 1).unwrap()) // Fermions: n = 2k+1 (odd)
        .collect();
    
    let sampling = MatsubaraSamplingPositiveOnly::with_sampling_points(&basis, sampling_points);
    
    assert_eq!(sampling.basis_size(), basis.size());
    assert_eq!(sampling.n_sampling_points(), n_matsubara);
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

fn test_matsubara_sampling_nd_roundtrip_generic<S: StatisticsType>() {
    use mdarray::Tensor;
    use num_complex::Complex;
    
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;
    
    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, Some(epsilon), None);
    
    let n_matsubara = basis.size();
    let sampling_points: Vec<MatsubaraFreq<S>> = (-(n_matsubara as i64)..(n_matsubara as i64))
        .filter_map(|k| {
            let n = match S::STATISTICS {
                sparseir_rust::traits::Statistics::Fermionic => 2 * k + 1,
                sparseir_rust::traits::Statistics::Bosonic => 2 * k,
            };
            MatsubaraFreq::<S>::new(n).ok()
        })
        .take(n_matsubara)
        .collect();
    
    let sampling = MatsubaraSampling::with_sampling_points(&basis, sampling_points);
    
    // Test that evaluate_nd with 1D tensor matches evaluate
    let basis_size = basis.size();
    let coeffs_1d_vec: Vec<Complex<f64>> = (0..basis_size)
        .map(|i| Complex::new(i as f64 * 0.1, (i % 3) as f64 * 0.05))
        .collect();
    
    // Use evaluate (1D)
    let values_1d = sampling.evaluate(&coeffs_1d_vec);
    
    // Use evaluate_nd with 1D tensor
    let mut coeffs_1d_tensor = Tensor::<Complex<f64>, _>::zeros(vec![basis_size]);
    for i in 0..basis_size {
        coeffs_1d_tensor[&[i][..]] = coeffs_1d_vec[i];
    }
    let values_1d_nd = sampling.evaluate_nd(&coeffs_1d_tensor, 0);
    
    // Compare
    let max_eval_diff = values_1d.iter().zip(values_1d_nd.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0, f64::max);
    
    println!("MatsubaraSampling {:?} ND eval vs 1D eval max diff: {}", S::STATISTICS, max_eval_diff);
    assert!(max_eval_diff < 1e-12, "ND eval differs from 1D eval: {}", max_eval_diff);
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

fn test_matsubara_sampling_positive_only_nd_roundtrip_generic<S: StatisticsType>() {
    use mdarray::Tensor;
    
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;
    
    let kernel = LogisticKernel::new(wmax * beta);
    let basis = FiniteTempBasis::<_, S>::new(kernel, beta, Some(epsilon), None);
    
    let n_matsubara = basis.size();
    let sampling_points: Vec<MatsubaraFreq<S>> = (0..2 * n_matsubara as i64)
        .filter_map(|k| {
            let n = match S::STATISTICS {
                sparseir_rust::traits::Statistics::Fermionic => 2 * k + 1,
                sparseir_rust::traits::Statistics::Bosonic => 2 * k,
            };
            MatsubaraFreq::<S>::new(n).ok()
        })
        .take(n_matsubara)
        .collect();
    
    let sampling = MatsubaraSamplingPositiveOnly::with_sampling_points(&basis, sampling_points);
    
    // Test that evaluate_nd with 1D tensor matches evaluate
    let basis_size = basis.size();
    let coeffs_1d_vec: Vec<f64> = (0..basis_size)
        .map(|i| i as f64 * 0.1)
        .collect();
    
    // Use evaluate (1D)
    let values_1d = sampling.evaluate(&coeffs_1d_vec);
    
    // Use evaluate_nd with 1D tensor
    let mut coeffs_1d_tensor = Tensor::<f64, _>::zeros(vec![basis_size]);
    for i in 0..basis_size {
        coeffs_1d_tensor[&[i][..]] = coeffs_1d_vec[i];
    }
    let values_1d_nd = sampling.evaluate_nd(&coeffs_1d_tensor, 0);
    
    // Compare
    let max_eval_diff = values_1d.iter().zip(values_1d_nd.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0, f64::max);
    
    println!("MatsubaraSamplingPositiveOnly {:?} ND eval vs 1D eval max diff: {}", S::STATISTICS, max_eval_diff);
    assert!(max_eval_diff < 1e-12, "ND eval differs from 1D eval: {}", max_eval_diff);
}

