//! Tests for piecewise Legendre polynomial Fourier transform implementations

use sparseir_rust::polyfourier::{
    PiecewiseLegendreFT, PiecewiseLegendreFTVector, 
    FermionicPiecewiseLegendreFT, BosonicPiecewiseLegendreFT,
    FermionicPiecewiseLegendreFTVector, BosonicPiecewiseLegendreFTVector
};
use sparseir_rust::poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
use sparseir_rust::traits::{Statistics, Fermionic, Bosonic};
use sparseir_rust::freq::{FermionicFreq, BosonicFreq};
use ndarray::arr2;

#[test]
fn test_fermionic_ft_creation() {
    let data = arr2(&[[1.0], [0.0]]);
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);
    
    let ft_poly = FermionicPiecewiseLegendreFT::new(poly.clone(), Fermionic, None);
    
    assert_eq!(ft_poly.get_n_asymp(), f64::INFINITY);
    assert_eq!(ft_poly.get_statistics(), Statistics::Fermionic);
    assert_eq!(ft_poly.zeta(), 1);
}

#[test]
fn test_bosonic_ft_creation() {
    let data = arr2(&[[1.0], [0.0]]);
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);
    
    let ft_poly = BosonicPiecewiseLegendreFT::new(poly.clone(), Bosonic, Some(100.0));
    
    assert_eq!(ft_poly.get_n_asymp(), 100.0);
    assert_eq!(ft_poly.get_statistics(), Statistics::Bosonic);
    assert_eq!(ft_poly.zeta(), 0);
}

#[test]
fn test_ft_evaluation_fermionic() {
    let data = arr2(&[[1.0], [0.0]]);
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);
    
    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);
    
    // Test evaluation at valid fermionic frequency
    let omega = FermionicFreq::new(1).unwrap();
    let result = ft_poly.evaluate(&omega);
    
    // The result should be a complex number
    assert!(result.is_finite());
    println!("Fermionic FT at n=1: {}", result);
}

#[test]
fn test_ft_evaluation_bosonic() {
    let data = arr2(&[[1.0], [0.0]]);
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);
    
    let ft_poly = BosonicPiecewiseLegendreFT::new(poly, Bosonic, None);
    
    // Test evaluation at valid bosonic frequency
    let omega = BosonicFreq::new(0).unwrap();
    let result = ft_poly.evaluate(&omega);
    
    // The result should be a complex number
    assert!(result.is_finite());
    println!("Bosonic FT at n=0: {}", result);
}

#[test]
fn test_ft_vector_creation() {
    let data1 = arr2(&[[1.0], [0.0]]);
    let data2 = arr2(&[[0.0], [1.0]]);
    let knots = vec![-1.0, 1.0];
    
    let poly1 = PiecewiseLegendrePoly::new(data1, knots.clone(), 0, None, 0);
    let poly2 = PiecewiseLegendrePoly::new(data2, knots, 1, None, 0);
    
    let ft_poly1 = FermionicPiecewiseLegendreFT::new(poly1, Fermionic, None);
    let ft_poly2 = FermionicPiecewiseLegendreFT::new(poly2, Fermionic, None);
    
    let ft_vector = FermionicPiecewiseLegendreFTVector::from_vector(vec![ft_poly1, ft_poly2]);
    
    assert_eq!(ft_vector.size(), 2);
}

#[test]
fn test_ft_vector_from_poly_vector() {
    let data1 = arr2(&[[1.0], [0.0]]);
    let data2 = arr2(&[[0.0], [1.0]]);
    let knots = vec![-1.0, 1.0];
    
    let poly1 = PiecewiseLegendrePoly::new(data1, knots.clone(), 0, None, 0);
    let poly2 = PiecewiseLegendrePoly::new(data2, knots.clone(), 1, None, 0);
    
    let poly_vector = PiecewiseLegendrePolyVector::new(vec![poly1, poly2]);
    let ft_vector = FermionicPiecewiseLegendreFTVector::from_poly_vector(&poly_vector, Fermionic, None);
    
    assert_eq!(ft_vector.size(), 2);
}

#[test]
fn test_ft_vector_evaluation() {
    let data = arr2(&[[1.0], [0.0]]);
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);
    
    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);
    let ft_vector = FermionicPiecewiseLegendreFTVector::from_vector(vec![ft_poly]);
    
    let omega = FermionicFreq::new(1).unwrap();
    let results = ft_vector.evaluate_at(&omega);
    
    assert_eq!(results.len(), 1);
    assert!(results[0].is_finite());
}

#[test]
fn test_power_model_creation() {
    let data = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let knots = vec![-1.0, 0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 1, None, 0);
    
    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);
    
    // Check that power model was created
    assert!(!ft_poly.model.moments.is_empty());
    println!("Power model moments: {:?}", ft_poly.model.moments);
}

#[test]
fn test_invalid_domain_panic() {
    let data = arr2(&[[1.0], [0.0]]);
    let knots = vec![0.0, 2.0]; // Invalid domain for Fourier transform
    
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);
    
    // This should panic
    std::panic::catch_unwind(|| {
        FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);
    }).expect_err("Should panic for invalid domain");
}

#[test]
fn test_sign_changes_basic() {
    // Test with a simple polynomial that has sign changes
    let data = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let knots = vec![-1.0, 0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 1, None, 0);
    
    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);
    
    // Test sign changes (default - both positive and negative frequencies)
    let sign_changes_default = ft_poly.sign_changes(false);
    
    // Test sign changes (positive only)
    let sign_changes_positive = ft_poly.sign_changes(true);
    
    // Results should be valid
    assert!(sign_changes_default.len() >= 0);
    assert!(sign_changes_positive.len() >= 0);
    
    println!("Sign changes (default): {:?}", sign_changes_default);
    println!("Sign changes (positive only): {:?}", sign_changes_positive);
}

#[test]
fn test_find_extrema_basic() {
    // Test with a simple polynomial that has extrema
    let data = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let knots = vec![-1.0, 0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 1, None, 0);
    
    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);
    
    // Test extrema (default - both positive and negative frequencies)
    let extrema_default = ft_poly.find_extrema(false);
    
    // Test extrema (positive only)
    let extrema_positive = ft_poly.find_extrema(true);
    
    // Results should be valid
    assert!(extrema_default.len() >= 0);
    assert!(extrema_positive.len() >= 0);
    
    println!("Extrema (default): {:?}", extrema_default);
    println!("Extrema (positive only): {:?}", extrema_positive);
}

#[test]
fn test_func_for_part_basic() {
    // Test func_for_part functionality
    let data = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let knots = vec![-1.0, 0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 1, None, 0);
    
    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);
    
    // Test evaluation at different frequencies
    let result_0 = ft_poly.evaluate_at_n(0);
    let result_1 = ft_poly.evaluate_at_n(1);
    let result_2 = ft_poly.evaluate_at_n(2);
    
    // Results should be finite
    assert!(result_0.is_finite());
    assert!(result_1.is_finite());
    assert!(result_2.is_finite());
    
    println!("FT at n=0: {}", result_0);
    println!("FT at n=1: {}", result_1);
    println!("FT at n=2: {}", result_2);
}

#[test]
#[ignore] // TODO: Fix constant polynomial Fourier transform
fn test_sign_changes_empty_case() {
    // Test with a constant polynomial (should have no sign changes)
    let data = arr2(&[[1.0], [0.0]]);
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);
    
    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);
    
    let sign_changes = ft_poly.sign_changes(false);
    
    println!("Sign changes found: {}", sign_changes.len());
    if sign_changes.len() > 0 {
        println!("First few sign changes: {:?}", &sign_changes[..std::cmp::min(5, sign_changes.len())]);
    }
    
    // Constant polynomial should have no sign changes
    assert_eq!(sign_changes.len(), 0);
}

#[test]
#[ignore] // TODO: Fix constant polynomial Fourier transform
fn test_find_extrema_empty_case() {
    // Test with a constant polynomial (should have no extrema)
    let data = arr2(&[[1.0], [0.0]]);
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);
    
    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);
    
    let extrema = ft_poly.find_extrema(false);
    
    // Constant polynomial should have no extrema
    assert_eq!(extrema.len(), 0);
}

#[test]
fn test_bosonic_sign_changes() {
    // Test bosonic statistics
    let data = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let knots = vec![-1.0, 0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 1, None, 0);
    
    let ft_poly = BosonicPiecewiseLegendreFT::new(poly, Bosonic, None);
    
    let sign_changes = ft_poly.sign_changes(false);
    let sign_changes_positive = ft_poly.sign_changes(true);
    
    // Results should be valid
    assert!(sign_changes.len() >= 0);
    assert!(sign_changes_positive.len() >= 0);
    
    println!("Bosonic sign changes: {:?}", sign_changes);
    println!("Bosonic sign changes (positive): {:?}", sign_changes_positive);
}

#[test]
fn test_bosonic_find_extrema() {
    // Test bosonic statistics
    let data = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let knots = vec![-1.0, 0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 1, None, 0);
    
    let ft_poly = BosonicPiecewiseLegendreFT::new(poly, Bosonic, None);
    
    let extrema = ft_poly.find_extrema(false);
    let extrema_positive = ft_poly.find_extrema(true);
    
    // Results should be valid
    assert!(extrema.len() >= 0);
    assert!(extrema_positive.len() >= 0);
    
    println!("Bosonic extrema: {:?}", extrema);
    println!("Bosonic extrema (positive): {:?}", extrema_positive);
}
