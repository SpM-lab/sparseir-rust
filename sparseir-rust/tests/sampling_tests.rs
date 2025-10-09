use sparseir_rust::basis::FiniteTempBasis;
use sparseir_rust::kernel::LogisticKernel;
use sparseir_rust::sampling::TauSampling;
use sparseir_rust::traits::{Fermionic, Bosonic};
use mdarray::{Tensor, DynRank, Shape};

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

/// Single-pole Green's function for fermions: G(τ) = -e^(-ω*τ) / (1 + e^(-β*ω))
/// 
/// For ω > 0 (particle pole), this gives the retarded Green's function in imaginary time.
/// 
/// Supports extended τ range using anti-periodic boundary condition: G(τ + β) = -G(τ)
/// - For -β < τ < 0: use G(τ) = -G(τ + β)
/// - For β < τ < 2β: use G(τ) = -G(τ - β)
fn fermionic_single_pole(tau: f64, omega: f64, beta: f64) -> f64 {
    // Normalize τ to [0, β) and track sign from anti-periodicity
    // G(τ + β) = -G(τ) for fermions
    let (tau_normalized, sign) = if tau < 0.0 {
        // -β < τ < 0: G(τ) = -G(τ + β)
        (tau + beta, -1.0)
    } else if tau >= beta {
        // β ≤ τ < 2β: G(τ) = -G(τ - β)  
        (tau - beta, -1.0)
    } else {
        // 0 ≤ τ < β: normal range
        (tau, 1.0)
    };
    
    sign * (-(-omega * tau_normalized).exp() / (1.0 + (-beta * omega).exp()))
}

/// Single-pole Green's function for bosons: G(τ) = e^(-ω*τ) / (1 - e^(-β*ω))
/// 
/// For ω > 0, this gives the bosonic Green's function in imaginary time.
/// 
/// Supports extended τ range using periodic boundary condition: G(τ + β) = G(τ)
/// - For -β < τ < 0: use G(τ) = G(τ + β)
/// - For β < τ < 2β: use G(τ) = G(τ - β)
fn bosonic_single_pole(tau: f64, omega: f64, beta: f64) -> f64 {
    // Normalize τ to [0, β) using periodicity
    // G(τ + β) = G(τ) for bosons
    let tau_normalized = if tau < 0.0 {
        // -β < τ < 0: G(τ) = G(τ + β)
        tau + beta
    } else if tau >= beta {
        // β ≤ τ < 2β: G(τ) = G(τ - β)
        tau - beta
    } else {
        // 0 ≤ τ < β: normal range
        tau
    };
    
    (-omega * tau_normalized).exp() / (1.0 - (-beta * omega).exp())
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
    
    // Physical test: Single pole at ω = 0.5 * wmax
    let beta = basis.beta;
    let omega = 0.5 * basis.omega_max();
    
    // Compute Green's function values at sampling points
    let g_values: Vec<f64> = sampling.sampling_points()
        .iter()
        .map(|&tau| fermionic_single_pole(tau, omega, beta))
        .collect();
    
    // Fit to IR basis coefficients
    let coeffs = sampling.fit(&g_values);
    assert_eq!(coeffs.len(), sampling.basis_size());
    
    // Evaluate back: coeffs → values
    let fitted_values = sampling.evaluate(&coeffs);
    
    // Check roundtrip accuracy
    for (orig, fitted) in g_values.iter().zip(fitted_values.iter()) {
        let abs_error = (orig - fitted).abs();
        assert!(
            abs_error < 1e-10,
            "Roundtrip error too large: orig={}, fitted={}, error={}",
            orig,
            fitted,
            abs_error
        );
    }
}

#[test]
fn test_evaluate_fit_roundtrip_bosonic() {
    let basis = create_test_basis_bosonic();
    let sampling: TauSampling<Bosonic> = TauSampling::new(&basis);
    
    // Physical test: Single pole at ω = 0.5 * wmax
    let beta = basis.beta;
    let omega = 0.5 * basis.omega_max();
    
    // Compute Green's function values at sampling points
    let g_values: Vec<f64> = sampling.sampling_points()
        .iter()
        .map(|&tau| bosonic_single_pole(tau, omega, beta))
        .collect();
    
    // Fit to IR basis coefficients
    let coeffs = sampling.fit(&g_values);
    
    // Evaluate back: coeffs → values
    let fitted_values = sampling.evaluate(&coeffs);
    
    // Check roundtrip accuracy
    for (orig, fitted) in g_values.iter().zip(fitted_values.iter()) {
        let abs_error = (orig - fitted).abs();
        assert!(abs_error < 1e-10, "Bosonic roundtrip error: {}", abs_error);
    }
}

#[test]
fn test_fermionic_antiperiodicity() {
    let beta = 1.0;
    let omega = 5.0;
    
    // Test anti-periodicity: G(τ + β) = -G(τ)
    for i in 0..10 {
        let tau = i as f64 * 0.1 * beta;
        let g_tau = fermionic_single_pole(tau, omega, beta);
        let g_tau_plus_beta = fermionic_single_pole(tau + beta, omega, beta);
        
        let diff = (g_tau + g_tau_plus_beta).abs();
        assert!(diff < 1e-14, 
                "Anti-periodicity violated at τ={}: G(τ)={}, G(τ+β)={}", 
                tau, g_tau, g_tau_plus_beta);
    }
    
    // Test for negative τ: G(τ) = -G(τ + β)
    for i in 1..10 {
        let tau = -i as f64 * 0.1 * beta;
        let g_tau = fermionic_single_pole(tau, omega, beta);
        let g_tau_plus_beta = fermionic_single_pole(tau + beta, omega, beta);
        
        let diff = (g_tau + g_tau_plus_beta).abs();
        assert!(diff < 1e-14, 
                "Anti-periodicity violated at τ={}: G(τ)={}, G(τ+β)={}", 
                tau, g_tau, g_tau_plus_beta);
    }
}

#[test]
fn test_bosonic_periodicity() {
    let beta = 1.0;
    let omega = 5.0;
    
    // Test periodicity: G(τ + β) = G(τ)
    for i in 0..10 {
        let tau = i as f64 * 0.1 * beta;
        let g_tau = bosonic_single_pole(tau, omega, beta);
        let g_tau_plus_beta = bosonic_single_pole(tau + beta, omega, beta);
        
        let diff = (g_tau - g_tau_plus_beta).abs();
        assert!(diff < 1e-14, 
                "Periodicity violated at τ={}: G(τ)={}, G(τ+β)={}", 
                tau, g_tau, g_tau_plus_beta);
    }
    
    // Test for negative τ: G(τ) = G(τ + β)
    for i in 1..10 {
        let tau = -i as f64 * 0.1 * beta;
        let g_tau = bosonic_single_pole(tau, omega, beta);
        let g_tau_plus_beta = bosonic_single_pole(tau + beta, omega, beta);
        
        let diff = (g_tau - g_tau_plus_beta).abs();
        assert!(diff < 1e-14, 
                "Periodicity violated at τ={}: G(τ)={}, G(τ+β)={}", 
                tau, g_tau, g_tau_plus_beta);
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

#[test]
fn test_evaluate_nd_2d() {
    let basis = create_test_basis_fermionic();
    let sampling: TauSampling<Fermionic> = TauSampling::new(&basis);
    
    // Create 2D coefficients: (basis_size, n_extra)
    let basis_size = sampling.basis_size();
    let n_extra = 5;
    
    // Create using DTensor<f64, 2> then convert to DynRank
    let coeffs_2d_typed = mdarray::DTensor::<f64, 2>::from_fn([basis_size, n_extra], |idx| {
        (idx[0] as f64 + 1.0) * (idx[1] as f64 + 1.0)
    });
    
    // Convert to DynRank
    let shape_vec = vec![basis_size, n_extra];
    let mut coeffs_2d: Tensor<f64, DynRank> = Tensor::zeros(shape_vec.as_slice());
    for i in 0..basis_size {
        for j in 0..n_extra {
            coeffs_2d[&[i, j][..]] = coeffs_2d_typed[[i, j]];
        }
    }
    
    // Evaluate along dim=0
    let values_2d = sampling.evaluate_nd(&coeffs_2d, 0);
    
    // Check shape
    assert_eq!(values_2d.rank(), 2);
    assert_eq!(values_2d.shape().dim(0), sampling.n_sampling_points());
    assert_eq!(values_2d.shape().dim(1), n_extra);
    
    // Verify: each column should be independent
    for j in 0..n_extra {
        let coeffs_col: Vec<f64> = (0..basis_size).map(|i| coeffs_2d[&[i, j][..]]).collect();
        let values_col_expected = sampling.evaluate(&coeffs_col);
        
        for i in 0..sampling.n_sampling_points() {
            let expected = values_col_expected[i];
            let actual = values_2d[&[i, j][..]];
            let diff = (actual - expected).abs();
            assert!(diff < 1e-12, "Mismatch at ({}, {}): expected {}, got {}", i, j, expected, actual);
        }
    }
}

#[test]
fn test_fit_nd_2d() {
    let basis = create_test_basis_fermionic();
    let sampling: TauSampling<Fermionic> = TauSampling::new(&basis);
    
    // Create 2D values: (n_sampling_points, n_extra)
    let n_points = sampling.n_sampling_points();
    let n_extra = 5;
    
    let values_2d_typed = mdarray::DTensor::<f64, 2>::from_fn([n_points, n_extra], |idx| {
        (idx[0] as f64 + 1.0) * (idx[1] as f64 + 1.0)
    });
    
    let mut values_2d: Tensor<f64, DynRank> = Tensor::zeros(&vec![n_points, n_extra][..]);
    for i in 0..n_points {
        for j in 0..n_extra {
            values_2d[&[i, j][..]] = values_2d_typed[[i, j]];
        }
    }
    
    // Fit along dim=0
    let coeffs_2d = sampling.fit_nd(&values_2d, 0);
    
    // Check shape
    assert_eq!(coeffs_2d.rank(), 2);
    assert_eq!(coeffs_2d.shape().dim(0), sampling.basis_size());
    assert_eq!(coeffs_2d.shape().dim(1), n_extra);
    
    // Verify roundtrip: fit then evaluate
    let values_roundtrip = sampling.evaluate_nd(&coeffs_2d, 0);
    
    for j in 0..n_extra {
        for i in 0..n_points {
            let original = values_2d[&[i, j][..]];
            let roundtrip = values_roundtrip[&[i, j][..]];
            let diff = (original - roundtrip).abs();
            assert!(diff < 1e-10, "Roundtrip error at ({}, {}): {}", i, j, diff);
        }
    }
}

#[test]
fn test_evaluate_nd_3d() {
    let basis = create_test_basis_fermionic();
    let sampling: TauSampling<Fermionic> = TauSampling::new(&basis);
    
    // Create 3D coefficients: (basis_size, n_k, n_omega)
    let basis_size = sampling.basis_size();
    let n_k = 3;
    let n_omega = 4;
    
    let coeffs_3d_typed = mdarray::DTensor::<f64, 3>::from_fn([basis_size, n_k, n_omega], |idx| {
        (idx[0] as f64 + 1.0) + (idx[1] as f64) * 0.1 + (idx[2] as f64) * 0.01
    });
    
    let mut coeffs_3d: Tensor<f64, DynRank> = Tensor::zeros(&vec![basis_size, n_k, n_omega][..]);
    for l in 0..basis_size {
        for k in 0..n_k {
            for omega in 0..n_omega {
                coeffs_3d[&[l, k, omega][..]] = coeffs_3d_typed[[l, k, omega]];
            }
        }
    }
    
    // Evaluate along dim=0
    let values_3d = sampling.evaluate_nd(&coeffs_3d, 0);
    
    // Check shape
    assert_eq!(values_3d.rank(), 3);
    assert_eq!(values_3d.shape().dim(0), sampling.n_sampling_points());
    assert_eq!(values_3d.shape().dim(1), n_k);
    assert_eq!(values_3d.shape().dim(2), n_omega);
}

#[test]
fn test_fit_nd_3d_roundtrip() {
    let basis = create_test_basis_fermionic();
    let sampling: TauSampling<Fermionic> = TauSampling::new(&basis);
    
    // Physical test: Multiple poles at different ω for different k-points
    // Simulating G(τ, k, ω) where each (k, ω) has a pole at ω_pole(k, ω)
    let beta = basis.beta;
    let wmax = basis.omega_max();
    let n_k = 3;
    let n_omega = 4;
    
    // Create 3D Green's function: G(τ, k, ω_index)
    let mut values_3d: Tensor<f64, DynRank> = Tensor::zeros(&vec![sampling.n_sampling_points(), n_k, n_omega][..]);
    
    for (i, &tau) in sampling.sampling_points().iter().enumerate() {
        for k in 0..n_k {
            for omega_idx in 0..n_omega {
                // Different pole position for each (k, omega_idx)
                let omega_pole = wmax * (0.2 + 0.15 * k as f64 + 0.1 * omega_idx as f64);
                let g_val = fermionic_single_pole(tau, omega_pole, beta);
                values_3d[&[i, k, omega_idx][..]] = g_val;
            }
        }
    }
    
    // Fit: values → coeffs
    let coeffs_3d = sampling.fit_nd(&values_3d, 0);
    
    // Evaluate back: coeffs → values
    let fitted_values_3d = sampling.evaluate_nd(&coeffs_3d, 0);
    
    // Check roundtrip accuracy
    for k in 0..n_k {
        for omega_idx in 0..n_omega {
            for i in 0..sampling.n_sampling_points() {
                let original = values_3d[&[i, k, omega_idx][..]];
                let fitted = fitted_values_3d[&[i, k, omega_idx][..]];
                let abs_error = (original - fitted).abs();
                assert!(abs_error < 1e-10, 
                        "Error at (tau_idx={}, k={}, ω_idx={}): error={}", 
                        i, k, omega_idx, abs_error);
            }
        }
    }
}

