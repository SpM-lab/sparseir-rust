//! Tests for FiniteTempBasis functionality

use crate::basis::{FermionicBasis, FiniteTempBasis};
use crate::kernel::{LogisticKernel, RegularizedBoseKernel};
use crate::traits::Bosonic;

#[test]
fn test_basis_construction() {
    let beta = 10.0;
    let omega_max = 1.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * omega_max);
    let basis = FermionicBasis::new(kernel, beta, Some(epsilon), None);

    assert_eq!(basis.beta, beta);
    assert!((basis.omega_max() - omega_max).abs() < 1e-10);
    assert!(basis.size() > 0);
    assert!(basis.accuracy > 0.0);
    assert!(basis.accuracy < epsilon);
}

#[test]
#[should_panic(expected = "beta must be positive")]
fn test_negative_beta() {
    let kernel = LogisticKernel::new(1.0);
    let _ = FermionicBasis::new(kernel, -1.0, None, None);
}

#[test]
fn test_default_tau_sampling_points_conditioning() {
    // Test parameters: beta=1.0, lambda=10.0
    let beta = 1.0;
    let lambda = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(lambda);
    let basis = FermionicBasis::new(kernel, beta, Some(epsilon), None);

    println!("\n=== Default Tau Sampling Points Test ===");
    println!("Beta: {}, Lambda: {}, Epsilon: {}", beta, lambda, epsilon);
    println!("Basis size: {}", basis.size());

    // Get default sampling points
    let tau_points = basis.default_tau_sampling_points();
    println!("Number of sampling points: {}", tau_points.len());

    // Verify range: [-beta/2, beta/2] (matches C++ implementation)
    let beta_half = beta / 2.0;
    for &tau in &tau_points {
        assert!(
            tau >= -beta_half && tau <= beta_half,
            "tau={} out of range [{}, {}]",
            tau,
            -beta_half,
            beta_half
        );
    }

    // Verify sorted (monotonically increasing)
    for i in 1..tau_points.len() {
        assert!(
            tau_points[i] >= tau_points[i - 1],
            "Points not sorted: tau[{}]={} < tau[{}]={}",
            i,
            tau_points[i],
            i - 1,
            tau_points[i - 1]
        );
    }

    // Verify symmetry around 0 (for range [-beta/2, beta/2])
    // Points should come in pairs: tau and -tau
    let tol = 1e-10;

    for &tau in &tau_points {
        let tau_reflected = -tau;
        let has_pair = tau_points.iter().any(|&t| (t - tau_reflected).abs() < tol);
        assert!(
            has_pair || tau.abs() < tol,
            "tau={} lacks symmetric pair around 0",
            tau
        );
    }
    println!("✅ Sampling points are symmetric around 0");

    // Evaluate sampling matrix: matrix[i,l] = u_l(tau_i)
    // Use the Basis trait method which handles tau normalization
    use crate::basis_trait::Basis;
    let matrix = basis.evaluate_tau(&tau_points);

    let num_points = tau_points.len();
    let basis_size = basis.size();
    println!("Sampling matrix shape: {}x{}", num_points, basis_size);

    // Compute SVD using mdarray-linalg (Faer backend)
    use mdarray_linalg::prelude::SVD;
    use mdarray_linalg_faer::Faer;
    let mut matrix_copy = matrix.clone();
    let svd = Faer.svd(&mut *matrix_copy).expect("SVD computation failed");

    println!("\nSampling matrix SVD:");
    let min_dim = svd.s.shape().0.min(svd.s.shape().1);
    println!("  Rank: {}", min_dim);
    // mdarray-linalg stores singular values in first row: s[[0, i]]
    println!("  First singular value: {:.6e}", svd.s[[0, 0]]);
    println!("  Last singular value: {:.6e}", svd.s[[0, min_dim - 1]]);

    let condition_number = svd.s[[0, 0]] / svd.s[[0, min_dim - 1]];
    println!("  Condition number: {:.6e}", condition_number);

    // Reference condition number (from Julia/C++ for beta=1, lambda=10)
    // This is approximate - actual value depends on implementation details
    let reference_cond = 25.0; // ~24.4 observed

    // Check: condition number should not be significantly worse
    // (within factor of 2 of reference)
    assert!(
        condition_number < reference_cond * 2.0,
        "Condition number too large: {:.2e} (reference: {:.2e})",
        condition_number,
        reference_cond
    );

    // Julia check: cond > 1e8 triggers warning
    assert!(
        condition_number < 1e8,
        "Sampling matrix is poorly conditioned: cond = {:.6e}",
        condition_number
    );

    println!(
        "✅ Condition number: {:.2e} (reference: {:.2e}, threshold: {:.2e})",
        condition_number,
        reference_cond,
        reference_cond * 2.0
    );
}

#[test]
fn test_regularized_bose_basis_construction() {
    let beta = 10.0;
    let omega_max = 1.0;
    let epsilon = 1e-6;

    let kernel = RegularizedBoseKernel::new(beta * omega_max);
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(epsilon), None);

    assert_eq!(basis.beta, beta);
    assert!((basis.omega_max() - omega_max).abs() < 1e-10);
    assert!(basis.size() > 0);
    assert!(basis.accuracy > 0.0);
    assert!(basis.accuracy < epsilon);

    println!("\n=== RegularizedBoseKernel Basis Test ===");
    println!(
        "Beta: {}, Omega_max: {}, Epsilon: {}",
        beta, omega_max, epsilon
    );
    println!("Basis size: {}", basis.size());
    println!("Accuracy: {:.6e}", basis.accuracy);
}

#[test]
fn test_regularized_bose_basis_different_parameters() {
    // Test with different beta and omega_max values
    let test_cases = vec![(1.0, 1.0, 1e-6), (10.0, 10.0, 1e-6), (100.0, 1.0, 1e-6)];

    for (beta, omega_max, epsilon) in test_cases {
        let kernel = RegularizedBoseKernel::new(beta * omega_max);
        let basis = FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(
            kernel,
            beta,
            Some(epsilon),
            None,
        );

        assert_eq!(basis.beta, beta);
        assert!((basis.omega_max() - omega_max).abs() < 1e-10);
        assert!(basis.size() > 0);
        assert!(basis.accuracy > 0.0);
        assert!(basis.accuracy < epsilon);

        println!(
            "Beta={}, Omega_max={}: size={}, accuracy={:.6e}",
            beta,
            omega_max,
            basis.size(),
            basis.accuracy
        );
    }
}
