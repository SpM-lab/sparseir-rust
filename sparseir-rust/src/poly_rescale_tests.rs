//! Tests for polynomial domain rescaling and data scaling

use crate::poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
use mdarray::DTensor;

#[test]
fn test_rescale_domain_single_poly() {
    // Create a simple polynomial on [-1, 1]
    let data_vec = vec![
        1.0, 2.0,  // const term
        0.5, 1.0,  // linear term
        0.2, 0.3,  // quadratic term
    ];
    let data = DTensor::<f64, 2>::from_fn([3, 2], |idx| data_vec[idx[0] * 2 + idx[1]]);
    
    let knots = vec![-1.0, 0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);
    
    // Test evaluation on original domain
    let val_original = poly.evaluate(0.5);
    
    // Rescale to [0, 2]: x_new = x_old + 1
    let new_knots = vec![0.0, 1.0, 2.0];
    let poly_rescaled = poly.rescale_domain(new_knots.clone(), None, None);
    
    // Check domain changed
    assert_eq!(poly_rescaled.xmin, 0.0);
    assert_eq!(poly_rescaled.xmax, 2.0);
    assert_eq!(poly_rescaled.knots, new_knots);
    
    // Evaluate at corresponding point: x=0.5 on [-1,1] → x=1.5 on [0,2]
    let val_rescaled = poly_rescaled.evaluate(1.5);
    
    // Values should be equal (same data, just different domain mapping)
    assert!((val_original - val_rescaled).abs() < 1e-14,
            "Rescaled value differs: {} vs {}", val_original, val_rescaled);
}

#[test]
fn test_scale_data_single_poly() {
    // Create a polynomial
    let data_vec = vec![1.0, 2.0, 0.5, 1.0];
    let data = DTensor::<f64, 2>::from_fn([2, 2], |idx| data_vec[idx[0] * 2 + idx[1]]);
    
    let knots = vec![-1.0, 0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);
    
    let val_original = poly.evaluate(0.5);
    
    // Scale data by 2.0
    let poly_scaled = poly.scale_data(2.0);
    let val_scaled = poly_scaled.evaluate(0.5);
    
    // Scaled value should be exactly 2x original
    assert!((val_scaled - 2.0 * val_original).abs() < 1e-14,
            "Scaled value incorrect: {} vs {}", val_scaled, 2.0 * val_original);
    
    // Domain should be unchanged
    assert_eq!(poly_scaled.xmin, poly.xmin);
    assert_eq!(poly_scaled.xmax, poly.xmax);
}

#[test]
fn test_rescale_domain_vector() {
    // Create vector with 2 polynomials
    let data1 = DTensor::<f64, 2>::from_fn([2, 1], |idx| if idx[0] == 0 { 1.0 } else { 0.5 });
    let data2 = DTensor::<f64, 2>::from_fn([2, 1], |idx| if idx[0] == 0 { 2.0 } else { 1.0 });
    
    let knots = vec![-1.0, 1.0];
    let poly1 = PiecewiseLegendrePoly::new(data1, knots.clone(), 0, None, 1);  // symm=1
    let poly2 = PiecewiseLegendrePoly::new(data2, knots.clone(), 1, None, -1); // symm=-1
    
    let polyvec = PiecewiseLegendrePolyVector::new(vec![poly1, poly2]);
    
    // Rescale to [0, β] where β=2.0
    let beta = 2.0;
    let new_knots = vec![0.0, beta];
    let new_delta_x = vec![beta];
    
    let polyvec_rescaled = polyvec.rescale_domain(
        new_knots.clone(),
        Some(new_delta_x),
        None,  // Keep original symmetry
    );
    
    // Check all polynomials have new domain
    for poly in polyvec_rescaled.get_polys() {
        assert_eq!(poly.xmin, 0.0);
        assert_eq!(poly.xmax, beta);
    }
    
    // Check symmetry preserved
    assert_eq!(polyvec_rescaled.get_polys()[0].symm, 1);
    assert_eq!(polyvec_rescaled.get_polys()[1].symm, -1);
}

#[test]
fn test_scale_data_vector() {
    // Create vector with 2 polynomials
    let data1 = DTensor::<f64, 2>::from_fn([2, 1], |idx| if idx[0] == 0 { 1.0 } else { 0.5 });
    let data2 = DTensor::<f64, 2>::from_fn([2, 1], |idx| if idx[0] == 0 { 2.0 } else { 1.0 });
    
    let knots = vec![-1.0, 1.0];
    let poly1 = PiecewiseLegendrePoly::new(data1, knots.clone(), 0, None, 0);
    let poly2 = PiecewiseLegendrePoly::new(data2, knots, 1, None, 0);
    
    let polyvec = PiecewiseLegendrePolyVector::new(vec![poly1, poly2]);
    
    // Evaluate before scaling
    let val1_before = polyvec.get_polys()[0].evaluate(0.0);
    let val2_before = polyvec.get_polys()[1].evaluate(0.0);
    
    // Scale by sqrt(2.0)
    let factor = 2.0_f64.sqrt();
    let polyvec_scaled = polyvec.scale_data(factor);
    
    // Evaluate after scaling
    let val1_after = polyvec_scaled.get_polys()[0].evaluate(0.0);
    let val2_after = polyvec_scaled.get_polys()[1].evaluate(0.0);
    
    // Check scaling
    assert!((val1_after - factor * val1_before).abs() < 1e-14);
    assert!((val2_after - factor * val2_before).abs() < 1e-14);
}

#[test]
fn test_rescale_with_symmetry_change() {
    // Test rescaling with symmetry parameter change
    let data = DTensor::<f64, 2>::from_fn([2, 1], |idx| if idx[0] == 0 { 1.0 } else { 0.5 });
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots.clone(), 0, None, 1);
    
    assert_eq!(poly.symm, 1);
    
    // Rescale to [0, 2] with symm=-1
    let new_knots = vec![0.0, 2.0];
    let poly_rescaled = poly.rescale_domain(new_knots, None, Some(-1));
    
    assert_eq!(poly_rescaled.symm, -1);
    assert_eq!(poly_rescaled.xmin, 0.0);
    assert_eq!(poly_rescaled.xmax, 2.0);
}

#[test]
fn test_basis_transformation_example() {
    // Example: Transform from SVE domain x ∈ [-1, 1] to τ ∈ [0, β]
    // Transformation: τ = β/2 * (x + 1)
    // So: x = -1 → τ = 0, x = 1 → τ = β
    
    let beta = 10.0;
    
    // Original polynomial on [-1, 1]
    let data_vec = vec![1.0, 1.5, 0.2, 0.3, 0.1, 0.1];
    let data = DTensor::<f64, 2>::from_fn([3, 2], |idx| data_vec[idx[0] * 2 + idx[1]]);
    let sve_knots = vec![-1.0, 0.0, 1.0];
    let poly_sve = PiecewiseLegendrePoly::new(data, sve_knots, 0, None, 0);
    
    // Transform to [0, β]
    let tau_knots: Vec<f64> = poly_sve.knots.iter()
        .map(|&x| beta / 2.0 * (x + 1.0))
        .collect();
    let tau_delta_x: Vec<f64> = poly_sve.delta_x.iter()
        .map(|&dx| beta / 2.0 * dx)
        .collect();
    
    let poly_tau = poly_sve.rescale_domain(tau_knots.clone(), Some(tau_delta_x), None);
    
    // Check transformation
    assert!((poly_tau.xmin - 0.0).abs() < 1e-14);
    assert!((poly_tau.xmax - beta).abs() < 1e-14);
    assert_eq!(poly_tau.knots.len(), 3);
    assert!((poly_tau.knots[0] - 0.0).abs() < 1e-14);
    assert!((poly_tau.knots[1] - beta / 2.0).abs() < 1e-14);
    assert!((poly_tau.knots[2] - beta).abs() < 1e-14);
}

