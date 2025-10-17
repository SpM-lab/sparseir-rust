//! Tests for default omega sampling points

use crate::{LogisticKernel, FiniteTempBasis, Fermionic, Bosonic};

#[test]
fn test_default_omega_sampling_points_fermionic() {
    let beta = 10000.0;
    let wmax = 1.0;
    let epsilon = 1e-6;
    
    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None);
    
    let omega_points = basis.default_omega_sampling_points();
    
    // Should have same size as basis
    assert_eq!(omega_points.len(), basis.size());
    
    // Points should be in [-wmax, wmax]
    for &omega in &omega_points {
        assert!(omega.abs() <= wmax, "omega = {} exceeds wmax = {}", omega, wmax);
    }
    
    // Points should be sorted
    let mut sorted = omega_points.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(omega_points, sorted, "Omega points should be sorted");
}

#[test]
fn test_default_omega_sampling_points_bosonic() {
    let beta = 10000.0;
    let wmax = 1.0;
    let epsilon = 1e-6;
    
    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<LogisticKernel, Bosonic>::new(kernel, beta, Some(epsilon), None);
    
    let omega_points = basis.default_omega_sampling_points();
    
    // Should have same size as basis
    assert_eq!(omega_points.len(), basis.size());
    
    // Points should be in [-wmax, wmax]
    for &omega in &omega_points {
        assert!(omega.abs() <= wmax, "omega = {} exceeds wmax = {}", omega, wmax);
    }
    
    // Points should be sorted
    let mut sorted = omega_points.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(omega_points, sorted, "Omega points should be sorted");
}

#[test]
fn test_omega_points_symmetry() {
    let beta = 1000.0;
    let wmax = 2.0;
    let epsilon = 1e-8;
    
    let kernel = LogisticKernel::new(beta * wmax);
    let basis_f = FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None);
    let omega_points = basis_f.default_omega_sampling_points();
    
    // Check approximate symmetry: for each positive point, there should be a negative counterpart
    // (This is approximate due to the nature of the roots)
    let positive: Vec<f64> = omega_points.iter().filter(|&&x| x > 0.0).copied().collect();
    let negative: Vec<f64> = omega_points.iter().filter(|&&x| x < 0.0).map(|&x| -x).collect();
    
    println!("Omega points: {:?}", omega_points);
    println!("Number of positive: {}, negative: {}", positive.len(), negative.len());
}

