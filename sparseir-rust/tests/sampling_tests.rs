use sparseir_rust::basis::FiniteTempBasis;
use sparseir_rust::kernel::LogisticKernel;
use sparseir_rust::sampling::TauSampling;
use sparseir_rust::traits::{Fermionic, Bosonic};

fn create_test_basis_fermionic() -> FiniteTempBasis<LogisticKernel, Fermionic> {
    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);
    let kernel = LogisticKernel::new(beta * wmax);
    FiniteTempBasis::new(kernel, beta, epsilon, None)
}

fn create_test_basis_bosonic() -> FiniteTempBasis<LogisticKernel, Bosonic> {
    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);
    let kernel = LogisticKernel::new(beta * wmax);
    FiniteTempBasis::new(kernel, beta, epsilon, None)
}

#[test]
fn test_tau_sampling_construction_fermionic() {
    let basis = create_test_basis_fermionic();
    
    let sampling: TauSampling<Fermionic> = TauSampling::new(&basis);
    
    assert_eq!(sampling.basis_size(), basis.size());
    assert!(sampling.n_sampling_points() > 0);
    assert!(sampling.n_sampling_points() <= 2 * basis.size());
    
    // Check sampling points are within [0, β]
    let beta = basis.beta;
    for &tau in sampling.sampling_points() {
        assert!(tau >= 0.0 && tau <= beta, "τ={} outside [0, β={}]", tau, beta);
    }
}

#[test]
fn test_tau_sampling_construction_bosonic() {
    let basis = create_test_basis_bosonic();
    
    let sampling: TauSampling<Bosonic> = TauSampling::new(&basis);
    
    assert_eq!(sampling.basis_size(), basis.size());
    assert!(sampling.n_sampling_points() > 0);
}

#[test]
fn test_evaluate_fit_roundtrip_fermionic() {
    let basis = create_test_basis_fermionic();
    
    let sampling: TauSampling<Fermionic> = TauSampling::new(&basis);
    
    // Create test coefficients (use singular values as test data)
    let coeffs: Vec<f64> = basis.s.iter().copied().collect();
    
    // Evaluate: coeffs → values at sampling points
    let values = sampling.evaluate(&coeffs);
    assert_eq!(values.len(), sampling.n_sampling_points());
    
    // Fit back: values → coeffs
    let fitted_coeffs = sampling.fit(&values);
    assert_eq!(fitted_coeffs.len(), sampling.basis_size());
    
    // Check roundtrip accuracy
    for (orig, fitted) in coeffs.iter().zip(fitted_coeffs.iter()) {
        let rel_error = (orig - fitted).abs() / orig.abs().max(1e-10);
        assert!(
            rel_error < 1e-10,
            "Roundtrip error too large: orig={}, fitted={}, rel_error={}",
            orig,
            fitted,
            rel_error
        );
    }
}

#[test]
fn test_evaluate_fit_roundtrip_bosonic() {
    let basis = create_test_basis_bosonic();
    
    let sampling: TauSampling<Bosonic> = TauSampling::new(&basis);
    
    // Create test coefficients
    let coeffs: Vec<f64> = basis.s.iter().copied().collect();
    
    // Evaluate and fit
    let values = sampling.evaluate(&coeffs);
    let fitted_coeffs = sampling.fit(&values);
    
    // Check roundtrip accuracy
    for (orig, fitted) in coeffs.iter().zip(fitted_coeffs.iter()) {
        let rel_error = (orig - fitted).abs() / orig.abs().max(1e-10);
        assert!(rel_error < 1e-10);
    }
}

#[test]
fn test_custom_sampling_points() {
    let basis = create_test_basis_fermionic();
    
    // Create custom sampling points (uniform grid)
    let beta = basis.beta;
    let n_points = 20;
    let custom_points: Vec<f64> = (0..n_points)
        .map(|i| (i as f64 + 0.5) * beta / (n_points as f64))
        .collect();
    
    let sampling: TauSampling<Fermionic> = TauSampling::with_sampling_points(
        &basis,
        custom_points.clone(),
        true,
    );
    
    assert_eq!(sampling.n_sampling_points(), n_points);
    assert_eq!(sampling.sampling_points(), &custom_points[..]);
    
    // Test evaluate still works
    let coeffs: Vec<f64> = basis.s.iter().copied().collect();
    let values = sampling.evaluate(&coeffs);
    assert_eq!(values.len(), n_points);
}

#[test]
fn test_sampling_matrix_shape() {
    let basis = create_test_basis_fermionic();
    
    let sampling: TauSampling<Fermionic> = TauSampling::new(&basis);
    
    let matrix = sampling.matrix();
    let shape = *matrix.shape();
    
    // Matrix should be (n_sampling_points, basis_size)
    assert_eq!(shape.0, sampling.n_sampling_points());
    assert_eq!(shape.1, sampling.basis_size());
}

#[test]
fn test_evaluate_zero_coefficients() {
    let basis = create_test_basis_fermionic();
    
    let sampling: TauSampling<Fermionic> = TauSampling::new(&basis);
    
    // All zero coefficients
    let coeffs = vec![0.0; sampling.basis_size()];
    let values = sampling.evaluate(&coeffs);
    
    // All values should be zero
    for &val in &values {
        assert!(val.abs() < 1e-14, "Expected ~0, got {}", val);
    }
}

#[test]
fn test_evaluate_single_basis_function() {
    let basis = create_test_basis_fermionic();
    
    let sampling: TauSampling<Fermionic> = TauSampling::new(&basis);
    
    // Only first coefficient is 1, rest are 0
    let mut coeffs = vec![0.0; sampling.basis_size()];
    coeffs[0] = 1.0;
    
    let values = sampling.evaluate(&coeffs);
    
    // Values should match u_0(τ_i)
    for (i, &val) in values.iter().enumerate() {
        let tau = sampling.sampling_points()[i];
        let expected = basis.u[0].evaluate(tau);
        let diff = (val - expected).abs();
        assert!(diff < 1e-12, "Mismatch at i={}: got {}, expected {}", i, val, expected);
    }
}

#[test]
#[should_panic(expected = "No sampling points given")]
fn test_empty_sampling_points() {
    let basis = create_test_basis_fermionic();
    
    // Empty sampling points should panic
    let _sampling: TauSampling<Fermionic> = TauSampling::with_sampling_points(
        &basis,
        vec![],
        true,
    );
}

#[test]
#[should_panic(expected = "outside")]
fn test_sampling_point_outside_range() {
    let basis = create_test_basis_fermionic();
    
    // Sampling point outside [0, β] should panic
    let invalid_points = vec![basis.beta + 1.0];
    let _sampling: TauSampling<Fermionic> = TauSampling::with_sampling_points(
        &basis,
        invalid_points,
        true,
    );
}

#[test]
#[should_panic(expected = "must match basis size")]
fn test_evaluate_wrong_size() {
    let basis = create_test_basis_fermionic();
    
    let sampling: TauSampling<Fermionic> = TauSampling::new(&basis);
    
    // Wrong number of coefficients should panic
    let wrong_coeffs = vec![1.0; sampling.basis_size() + 5];
    let _values = sampling.evaluate(&wrong_coeffs);
}

#[test]
#[should_panic(expected = "must match number of sampling points")]
fn test_fit_wrong_size() {
    let basis = create_test_basis_fermionic();
    
    let sampling: TauSampling<Fermionic> = TauSampling::new(&basis);
    
    // Wrong number of values should panic
    let wrong_values = vec![1.0; sampling.n_sampling_points() + 5];
    let _coeffs = sampling.fit(&wrong_values);
}

