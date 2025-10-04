//! Tests for SVE module

use sparseir_rust::*;
use std::sync::Arc;

#[test]
fn test_twork_type_values() {
    // Test that enum values match C-API constants
    assert_eq!(sve::TworkType::Float64 as i32, 0);
    assert_eq!(sve::TworkType::Float64X2 as i32, 1);
    assert_eq!(sve::TworkType::Auto as i32, -1);
}

#[test]
fn test_svd_strategy_values() {
    // Test that enum values match C-API constants
    assert_eq!(sve::SVDStrategy::Fast as i32, 0);
    assert_eq!(sve::SVDStrategy::Accurate as i32, 1);
    assert_eq!(sve::SVDStrategy::Auto as i32, -1);
}

#[test]
fn test_compute_sve_placeholder() {
    // Test the placeholder compute_sve function
    let kernel = Arc::new(kernel::LogisticKernel::new(1.0));
    let epsilon = 1e-12;
    
    let result = sve::compute_sve::<f64>(
        kernel,
        epsilon,
        Some(1e-10),
        Some(10),
        Some(20),
        sve::TworkType::Float64,
    );
    
    // Basic checks on placeholder result
    assert_eq!(result.epsilon, epsilon);
    assert_eq!(result.u.get_polys().len(), 1);
    assert_eq!(result.v.get_polys().len(), 1);
    assert_eq!(result.s.len(), 1);
}

#[test]
fn test_sveresult_creation() {
    // Create dummy polynomials for testing
    let dummy_poly = poly::PiecewiseLegendrePoly::new(
        ndarray::Array2::zeros((1, 1)),
        vec![0.0, 1.0],
        0,
        None,
        0
    );
    
    let u = poly::PiecewiseLegendrePolyVector::new(vec![dummy_poly.clone()]);
    let s = ndarray::Array1::from_vec(vec![1.0]);
    let v = poly::PiecewiseLegendrePolyVector::new(vec![dummy_poly]);
    
    // Create SVEResult
    let sve_result = sve::SVEResult::new(u.clone(), s.clone(), v.clone(), 1e-12);
    
    // Test basic properties
    assert_eq!(sve_result.epsilon, 1e-12);
    assert_eq!(sve_result.s.len(), 1);
    assert_eq!(sve_result.u.get_polys().len(), 1);
    assert_eq!(sve_result.v.get_polys().len(), 1);
    
    // Test part method with no truncation
    let (u_part, s_part, v_part) = sve_result.part(Some(1e-15), None);
    assert_eq!(s_part.len(), 1);
    assert_eq!(u_part.get_polys().len(), 1);
    assert_eq!(v_part.get_polys().len(), 1);
}

#[test]
fn test_sampling_sve_creation() {
    // Test SamplingSVE creation
    let kernel = Arc::new(kernel::LogisticKernel::new(1.0));
    let sve = sve::SamplingSVE::new(kernel, 1e-12, Some(10));
    
    // Basic check that it was created successfully
    // (No public fields to test, but creation should not panic)
    assert!(true);
}

#[test]
fn test_centrosymm_sve_creation() {
    // Test CentrosymmSVE creation
    let kernel = Arc::new(kernel::LogisticKernel::new(1.0));
    let sve = sve::CentrosymmSVE::new(kernel, 1e-12, Some(10));
    
    // Basic check that it was created successfully
    // (No public fields to test, but creation should not panic)
    assert!(true);
}