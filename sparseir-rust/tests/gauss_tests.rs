use ndarray::{Array1, Array2};
use sparseir_rust::{legendre, legendre_custom, legendre_twofloat, CustomNumeric, Rule, TwoFloat};
use sparseir_rust::gauss::{legendre_vandermonde, legendre_generic};
use sparseir_rust::interpolation1d::{legendre_collocation_matrix, interpolate_1d_legendre, evaluate_interpolated_polynomial};

#[test]
fn test_rule_constructor() {
    let x = Array1::from(vec![0.0, 1.0]);
    let w = Array1::from(vec![0.5, 0.5]);

    let rule = Rule::new(x.clone(), w.clone(), -1.0, 1.0);
    assert_eq!(rule.x, x);
    assert_eq!(rule.w, w);
    assert_eq!(rule.a, -1.0);
    assert_eq!(rule.b, 1.0);
}

#[test]
fn test_rule_from_vectors() {
    let x = vec![0.0, 1.0];
    let w = vec![0.5, 0.5];

    let rule = Rule::from_vectors(x.clone(), w.clone(), -1.0, 1.0);
    assert_eq!(rule.x.to_vec(), x);
    assert_eq!(rule.w.to_vec(), w);
}

#[test]
fn test_rule_empty() {
    let rule = Rule::<f64>::empty();
    assert_eq!(rule.x.len(), 0);
    assert_eq!(rule.w.len(), 0);
    assert_eq!(rule.a, -1.0);
    assert_eq!(rule.b, 1.0);
}

#[test]
fn test_rule_validation() {
    let x = Array1::from(vec![0.0, 1.0]);
    let w = Array1::from(vec![0.5, 0.5]);

    let rule = Rule::new(x, w, -1.0, 1.0);
    assert!(rule.validate());
}

#[test]
fn test_rule_join() {
    let rule1 = legendre::<f64>(4).reseat(-4.0, -1.0);
    let rule2 = legendre::<f64>(4).reseat(-1.0, 1.0);
    let rule3 = legendre::<f64>(4).reseat(1.0, 3.0);

    let joined = Rule::join(&[rule1, rule2, rule3]);

    assert!(joined.validate());
    assert_eq!(joined.a, -4.0);
    assert_eq!(joined.b, 3.0);
}

#[test]
fn test_rule_reseat() {
    let original_rule = legendre::<f64>(4);
    let reseated = original_rule.reseat(-2.0, 2.0);

    assert!(reseated.validate());
    assert_eq!(reseated.a, -2.0);
    assert_eq!(reseated.b, 2.0);
}

#[test]
fn test_rule_scale() {
    let x = Array1::from(vec![0.0, 1.0]);
    let w = Array1::from(vec![1.0, 1.0]);

    let rule = Rule::new(x, w, -1.0, 1.0);
    let scaled = rule.scale(2.0);

    assert_eq!(scaled.w[0], 2.0);
    assert_eq!(scaled.w[1], 2.0);
}

#[test]
fn test_rule_piecewise() {
    let edges = vec![-4.0, -1.0, 1.0, 3.0];
    let rule = legendre::<f64>(20).piecewise(&edges);

    assert!(rule.validate());
    assert_eq!(rule.a, -4.0);
    assert_eq!(rule.b, 3.0);
}

#[test]
fn test_gauss_validation_like_cpp() {
    // Test similar to C++ gaussValidate function
    let rule = legendre::<f64>(20);

    // Check interval validity: a <= b
    assert!(rule.a <= rule.b);

    // Check that all points are within [a, b]
    for &xi in rule.x.iter() {
        assert!(xi >= rule.a && xi <= rule.b);
    }

    // Check that points are sorted
    for i in 1..rule.x.len() {
        assert!(rule.x[i] >= rule.x[i - 1]);
    }

    // Check that x and w have same length
    assert_eq!(rule.x.len(), rule.w.len());

    // Check x_forward and x_backward consistency
    for i in 0..rule.x.len() {
        let expected_forward = rule.x[i] - rule.a;
        let expected_backward = rule.b - rule.x[i];

        assert!((rule.x_forward[i] - expected_forward).abs() < 1e-14);
        assert!((rule.x_backward[i] - expected_backward).abs() < 1e-14);
    }
}

#[test]
fn test_rule_constructor_with_defaults() {
    // Test like C++ Rule constructor with default a, b
    let x = Array1::from(vec![0.0, 1.0]);
    let w = Array1::from(vec![0.5, 0.5]);

    let rule1 = Rule::new(x.clone(), w.clone(), -1.0, 1.0);
    let rule2 = Rule::new(x, w, -1.0, 1.0);

    assert_eq!(rule1.a, rule2.a);
    assert_eq!(rule1.b, rule2.b);
    assert_eq!(rule1.x, rule2.x);
    assert_eq!(rule1.w, rule2.w);
}

#[test]
fn test_reseat_functionality() {
    // Test reseat functionality first
    let original_rule = legendre::<f64>(4);
    let reseated = original_rule.reseat(-4.0, -1.0);

    assert!(reseated.validate());
    assert_eq!(reseated.a, -4.0);
    assert_eq!(reseated.b, -1.0);
}

#[test]
fn test_join_functionality() {
    // Test join functionality
    let rule1 = legendre::<f64>(4).reseat(-4.0, -1.0);
    let rule2 = legendre::<f64>(4).reseat(-1.0, 1.0);
    let rule3 = legendre::<f64>(4).reseat(1.0, 3.0);

    let joined = Rule::join(&[rule1, rule2, rule3]);

    assert!(joined.validate());
    assert_eq!(joined.a, -4.0);
    assert_eq!(joined.b, 3.0);
}

#[test]
fn test_piecewise_like_cpp() {
    // Test piecewise functionality like C++ test
    let edges = vec![-4.0, -1.0, 1.0, 3.0];
    let rule = legendre::<f64>(20).piecewise(&edges);

    assert!(rule.validate());
    assert_eq!(rule.a, -4.0);
    assert_eq!(rule.b, 3.0);
}

#[test]
fn test_large_legendre_rule() {
    // Test large rule like C++ test with n=200
    let rule = legendre::<f64>(200);

    assert!(rule.validate());
    assert_eq!(rule.a, -1.0);
    assert_eq!(rule.b, 1.0);
    assert_eq!(rule.x.len(), 200);
    assert_eq!(rule.w.len(), 200);
}

#[test]
fn test_legendre_function() {
    // Test legendre function with different orders
    for n in 1..=5 {
        let rule = legendre::<f64>(n);
        assert_eq!(rule.x.len(), n);
        assert_eq!(rule.w.len(), n);
        assert!(rule.validate());
    }

    // Test n=0 case
    let rule = legendre::<f64>(0);
    assert_eq!(rule.x.len(), 0);
    assert_eq!(rule.w.len(), 0);
}

// CustomNumeric tests
#[test]
fn test_legendre_custom_f64() {
    // Test legendre_custom function with f64
    for n in 1..=5 {
        let rule = legendre_custom::<f64>(n);
        assert_eq!(rule.x.len(), n);
        assert_eq!(rule.w.len(), n);
        assert!(rule.validate_custom());
    }

    // Test n=0 case
    let rule = legendre_custom::<f64>(0);
    assert_eq!(rule.x.len(), 0);
    assert_eq!(rule.w.len(), 0);
}

#[test]
fn test_legendre_twofloat() {
    // Test legendre_twofloat function with TwoFloat
    for n in 1..=3 {
        // Smaller range for TwoFloat due to complexity
        let rule = legendre_twofloat(n);
        assert_eq!(rule.x.len(), n);
        assert_eq!(rule.w.len(), n);
        assert!(rule.validate_twofloat());
    }

    // Test n=0 case
    let rule = legendre_twofloat(0);
    assert_eq!(rule.x.len(), 0);
    assert_eq!(rule.w.len(), 0);
}

#[test]
fn test_rule_custom_methods() {
    // Test Rule custom methods with f64
    let x = Array1::from(vec![0.0, 1.0]);
    let w = Array1::from(vec![0.5, 0.5]);

    let rule = Rule::new_custom(x.clone(), w.clone(), -1.0, 1.0);
    assert!(rule.validate_custom());

    let reseated = rule.reseat_custom(-2.0, 0.0);
    assert!(reseated.validate_custom());
    assert_eq!(reseated.a, -2.0);
    assert_eq!(reseated.b, 0.0);

    let scaled = rule.scale_custom(2.0);
    assert!(scaled.validate_custom());
    assert_eq!(scaled.w[0], 1.0);
    assert_eq!(scaled.w[1], 1.0);
}

#[test]
fn test_rule_twofloat_methods() {
    // Test with TwoFloat
    let x_tf = Array1::from(vec![TwoFloat::from(0.0), TwoFloat::from(1.0)]);
    let w_tf = Array1::from(vec![TwoFloat::from(0.5), TwoFloat::from(0.5)]);

    let rule_tf = Rule::new_twofloat(x_tf, w_tf, TwoFloat::from(-1.0), TwoFloat::from(1.0));
    assert!(rule_tf.validate_twofloat());
}

// ===== TwoFloat Gauss Integration Precision Tests =====

/// Test function: f(x) = {cos((π/2) * x)}²
/// Integral over [-1, 1] should be exactly 1.0
fn test_function(x: TwoFloat) -> TwoFloat {
    let pi = TwoFloat::from_f64(std::f64::consts::PI);
    let cos_val = (pi / TwoFloat::from_f64(2.0) * x).cos();
    cos_val * cos_val
}

/// Analytical integral of f(x) = {cos((π/2) * x)}² over [-1, 1]
/// ∫_{-1}^{1} cos²((π/2) * x) dx = 1.0
fn analytical_integral() -> TwoFloat {
    TwoFloat::from_f64(1.0)
}

#[test]
fn test_twofloat_gauss_rule_validation() {
    println!("TwoFloat Gauss Rule Validation Test");
    println!("===================================");

    let test_points = vec![5, 10, 20, 50];

    for n in test_points {
        let rule = legendre_twofloat(n);

        println!("Testing rule with {} points:", n);
        println!("  Interval: [{}, {}]", rule.a.to_f64(), rule.b.to_f64());
        println!("  Points: {}", rule.x.len());
        println!("  Weights: {}", rule.w.len());

        // Validate the rule
        let is_valid = rule.validate_twofloat();
        println!(
            "  Validation: {}",
            if is_valid { "✅ PASS" } else { "❌ FAIL" }
        );

        // Check weight sum (should be 2.0 for [-1, 1])
        let mut weight_sum = TwoFloat::from_f64(0.0);
        for &w in rule.w.iter() {
            weight_sum = weight_sum + w;
        }
        let expected_sum = TwoFloat::from_f64(2.0);
        let weight_error = (weight_sum - expected_sum).abs();

        println!(
            "  Weight sum: {} (expected: 2.0, error: {:.2e})",
            weight_sum.to_f64(),
            weight_error.to_f64()
        );

        // Check symmetry (for even n, should be symmetric)
        if n % 2 == 0 {
            let mid = n / 2;
            let sym_check = (rule.x[mid - 1] + rule.x[mid]).abs() < TwoFloat::epsilon();
            println!(
                "  Symmetry check: {}",
                if sym_check { "✅ PASS" } else { "❌ FAIL" }
            );
        }

        println!();
    }
}

#[test]
fn test_twofloat_integration_convergence_analysis() {
    println!("TwoFloat Integration Convergence Analysis");
    println!("========================================");

    let analytical = analytical_integral();

    // Test convergence with specific number of points
    let test_points = vec![100, 150, 200];

    for n in test_points {
        let rule = legendre_twofloat(n);
        let mut integral = TwoFloat::from_f64(0.0);

        for i in 0..rule.x.len() {
            let f_val = test_function(rule.x[i]);
            integral = integral + f_val * rule.w[i];
        }

        let error = (integral - analytical).abs().to_f64();
        let rel_error = error / analytical.to_f64().abs();

        println!(
            "n={:3}: error={:.2e}, rel_error={:.2e}",
            n, error, rel_error
        );
        // This target is too loose for TwoFloat.
        // The numerical precision of math functions in twofloat is not that good.
        assert!(rel_error < 1e-15);
    }
}

/// Evaluate Legendre polynomial P_n(x) at point x
fn evaluate_legendre_polynomial<T: CustomNumeric>(x: T, n: usize) -> T {
    if n == 0 {
        T::from_f64(1.0)
    } else if n == 1 {
        x
    } else {
        let mut p_prev2 = T::from_f64(1.0);
        let mut p_prev1 = x;
        
        for i in 2..=n {
            let i_f64 = i as f64;
            let p_curr = ((T::from_f64(2.0 * i_f64 - 1.0) * x * p_prev1) - (T::from_f64(i_f64 - 1.0) * p_prev2)) / T::from_f64(i_f64);
            p_prev2 = p_prev1;
            p_prev1 = p_curr;
        }
        
        p_prev1
    }
}


#[test]
fn test_legendre_vandermonde_basic() {
    // Test with simple 3-point grid
    let x = vec![-1.0, 0.0, 1.0];
    let v = legendre_vandermonde(&x, 2);
    
    // Check dimensions
    assert_eq!(v.nrows(), 3);
    assert_eq!(v.ncols(), 3);
    
    // Check first column (P_0 = 1)
    for i in 0..3 {
        assert!((v[[i, 0]] - 1.0).abs() < 1e-12);
    }
    
    // Check second column (P_1 = x)
    for i in 0..3 {
        assert!((v[[i, 1]] - x[i]).abs() < 1e-12);
    }
    
    // Check third column (P_2 = (3x^2 - 1)/2)
    for i in 0..3 {
        let expected = (3.0 * x[i] * x[i] - 1.0) / 2.0;
        assert!((v[[i, 2]] - expected).abs() < 1e-12);
    }
}

/// Generic test function for 1D Legendre interpolation of sin(x) - MOVED TO interpolation1d_tests.rs
fn test_interpolate_1d_legendre_sin_generic<T: CustomNumeric + 'static>(
    n_points: usize,
    tolerance: T,
    test_points: Vec<T>,
) where
    T: std::fmt::Display,
{
    // Create Gauss rule using generic function
    let gauss_rule = legendre_generic::<T>(n_points).reseat(
        T::from_f64(-1.0),
        T::from_f64(1.0),
    );
    
    // Sample sin(x) at Gauss points
    let values: Vec<T> = gauss_rule.x.iter().map(|&x| x.sin()).collect();
    
    // Get interpolation coefficients
    let coeffs = interpolate_1d_legendre(&values, &gauss_rule);
    
    // Test interpolation at grid points (should be exact)
    for &x_grid in &gauss_rule.x {
        let expected = x_grid.sin();
        let interpolated = evaluate_interpolated_polynomial(x_grid, &coeffs);
        assert!((interpolated - expected).abs() < T::from_f64(1e-12),
            "Interpolation failed at grid point {}: expected {}, got {}", 
            x_grid, expected, interpolated);
    }
    
    // Test interpolation at interior points
    for &x_test in &test_points {
        let expected = x_test.sin();
        let interpolated = evaluate_interpolated_polynomial(x_test, &coeffs);
        let error = (interpolated - expected).abs();
        assert!(error < tolerance,
            "High-precision interpolation failed at point {}: expected {}, got {}, error={} > tolerance={}", 
            x_test, expected, interpolated, error, tolerance);
    }
}


#[test]
#[ignore] // MOVED TO interpolation1d_tests.rs
fn _test_interpolate_1d_legendre_sin_f64_high_precision() {
    // Test high-precision interpolation of sin(x) with f64
    test_interpolate_1d_legendre_sin_generic::<f64>(
        100,  // n_points
        f64::EPSILON * 100.0,  // tolerance: EPSILON * 100
        vec![-0.8, -0.5, -0.2, 0.1, 0.4, 0.7, 0.9],  // test_points
    );
}

#[test]
#[ignore] // MOVED TO interpolation1d_tests.rs
fn _test_interpolate_1d_legendre_sin_twofloat_ultra_high_precision() {
    // Test ultra high-precision interpolation of sin(x) with TwoFloat
    test_interpolate_1d_legendre_sin_generic::<TwoFloat>(
        200,  // n_points (higher for better precision)
        TwoFloat::from_f64(1e-19),  // tolerance: 1e-19 (achieved maximum precision)
        vec![
            TwoFloat::from_f64(-0.8),
            TwoFloat::from_f64(-0.5),
            TwoFloat::from_f64(-0.2),
            TwoFloat::from_f64(0.1),
            TwoFloat::from_f64(0.4),
            TwoFloat::from_f64(0.7),
            TwoFloat::from_f64(0.9),
        ],  // test_points
    );
}


/// Test that the collocation matrix is approximately the inverse of the Vandermonde matrix - MOVED TO interpolation1d_tests.rs
#[test]
#[ignore] // MOVED TO interpolation1d_tests.rs
fn _test_legendre_collocation_matrix_inverse() {
    // Test with different sizes
    for n in [2, 3, 5, 10] {
        let gauss_rule = legendre_generic::<f64>(n).reseat(-1.0, 1.0);
        
        // Create Vandermonde matrix
        let vandermonde = legendre_vandermonde(&gauss_rule.x.to_vec(), n - 1);
        
        // Create collocation matrix
        let collocation = legendre_collocation_matrix(&gauss_rule);
        
        // Compute V * C and check if it's approximately the identity matrix
        let mut product = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    product[[i, j]] += vandermonde[[i, k]] * collocation[[k, j]];
                }
            }
        }
        
        // Check that V * C ≈ I
        let identity: Array2<f64> = Array2::eye(n);
        let error = (&product - &identity).mapv(|x: f64| x.abs()).sum() / (n * n) as f64;
        
        println!("n={}, error={}", n, error);
        assert!(error < 1e-10, 
            "Collocation matrix is not inverse of Vandermonde matrix for n={}: error={}", n, error);
    }
}

/// Test the new fast interpolation method - MOVED TO interpolation1d_tests.rs
#[test]
#[ignore] // MOVED TO interpolation1d_tests.rs
fn _test_interpolate_1d_legendre_fast() {
    // Test with different sizes and functions
    for n in [2, 3, 5] {
        let gauss_rule = legendre_generic::<f64>(n).reseat(-1.0, 1.0);
        
        // Test different functions
        let test_functions = vec![
            |x: f64| x,                    // Linear
            |x: f64| x * x,                // Quadratic
            |x: f64| x * x * x,            // Cubic
            |x: f64| x.sin(),              // Sine
        ];
        
        for (func_idx, func) in test_functions.iter().enumerate() {
            // Sample function at Gauss points
            let values: Vec<f64> = gauss_rule.x.iter().map(|&x| func(x)).collect();
            
            // Get coefficients using the fast method
            let coeffs = interpolate_1d_legendre(&values, &gauss_rule);
            
            // Test interpolation at grid points (should be exact)
            for (i, &x_grid) in gauss_rule.x.iter().enumerate() {
                let expected = func(x_grid);
                let interpolated = evaluate_interpolated_polynomial(x_grid, &coeffs);
                let error = (interpolated - expected).abs();
                
                assert!(error < 1e-12,
                    "Interpolation failed for n={}, func={}, point={}: expected {}, got {}, error={}", 
                    n, func_idx, x_grid, expected, interpolated, error);
            }
            
            println!("n={}, func={}: interpolation successful", n, func_idx);
        }
    }
}

