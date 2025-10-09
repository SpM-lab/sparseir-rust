use sparseir_rust::{legendre_custom, legendre_twofloat, TwoFloat};

/// Helper function to check if two vectors are approximately equal within tolerance
fn vecs_approx_equal<T>(a: &[T], b: &[T], tolerance: T) -> bool
where
    T: Copy + std::ops::Sub<Output = T> + PartialOrd,
    T: std::fmt::Display,
{
    if a.len() != b.len() {
        return false;
    }

    for i in 0..a.len() {
        let diff = if a[i] > b[i] {
            a[i] - b[i]
        } else {
            b[i] - a[i]
        };
        if diff > tolerance {
            return false;
        }
    }
    true
}

/// Helper function to compute Legendre polynomial P_n(x)
fn legendre_polynomial(n: usize, x: f64) -> f64 {
    match n {
        0 => 1.0,
        1 => x,
        _ => {
            let mut p0 = 1.0;
            let mut p1 = x;

            for k in 2..=n {
                let k_f = k as f64;
                let k1_f = (k - 1) as f64;

                let p2 = ((2.0 * k1_f + 1.0) * x * p1 - k1_f * p0) / k_f;
                p0 = p1;
                p1 = p2;
            }
            p1
        }
    }
}

/// Helper function to compute Legendre polynomial with TwoFloat
fn legendre_polynomial_twofloat(n: usize, x: TwoFloat) -> TwoFloat {
    match n {
        0 => TwoFloat::from(1.0),
        1 => x,
        _ => {
            let mut p0 = TwoFloat::from(1.0);
            let mut p1 = x;

            for k in 2..=n {
                let k_f = TwoFloat::from(k as f64);
                let k1_f = TwoFloat::from((k - 1) as f64);

                let p2 =
                    ((TwoFloat::from(2.0) * k1_f + TwoFloat::from(1.0)) * x * p1 - k1_f * p0) / k_f;
                p0 = p1;
                p1 = p2;
            }
            p1
        }
    }
}

/// Test high-precision Gauss-Legendre rule with f64
/// Similar to C++ test but using f64 with 1e-13 tolerance
#[test]
fn test_high_precision_legendre_f64() {
    let n = 16;
    let rule = legendre_custom::<f64>(n);

    // Expected values computed with high precision (similar to C++ DDouble test)
    let x_expected = vec![
        -0.9894009349916499,
        -0.9445750230732325,
        -0.8656312023878318,
        -0.755404408355003,
        -0.6178762444026438,
        -0.45801677765722737,
        -0.2816035507792589,
        -0.09501250983763743,
        0.09501250983763743,
        0.2816035507792589,
        0.45801677765722737,
        0.6178762444026438,
        0.755404408355003,
        0.8656312023878318,
        0.9445750230732325,
        0.9894009349916499,
    ];

    let w_expected = vec![
        0.027152459411754124,
        0.06225352393864806,
        0.0951585116824928,
        0.12462897125553389,
        0.14959598881657682,
        0.16915651939500254,
        0.18260341504492367,
        0.18945061045506834,
        0.18945061045506834,
        0.18260341504492367,
        0.16915651939500254,
        0.14959598881657682,
        0.12462897125553389,
        0.0951585116824928,
        0.06225352393864806,
        0.027152459411754124,
    ];

    // Check with high precision tolerance (1e-13)
    let tolerance = 1e-13;

    // Check x values
    for i in 0..n {
        assert!(
            (rule.x[i] - x_expected[i]).abs() < tolerance,
            "x[{}] mismatch: expected {}, got {}",
            i,
            x_expected[i],
            rule.x[i]
        );
    }

    // Check w values
    for i in 0..n {
        assert!(
            (rule.w[i] - w_expected[i]).abs() < tolerance,
            "w[{}] mismatch: expected {}, got {}",
            i,
            w_expected[i],
            rule.w[i]
        );
    }

    // Check interval
    assert_eq!(rule.a, -1.0);
    assert_eq!(rule.b, 1.0);

    // Check x_forward and x_backward consistency with high precision
    for i in 0..rule.x.len() {
        let expected_forward = rule.x[i] - rule.a;
        let expected_backward = rule.b - rule.x[i];

        assert!(
            (rule.x_forward[i] - expected_forward).abs() < tolerance,
            "x_forward[{}] inconsistent",
            i
        );
        assert!(
            (rule.x_backward[i] - expected_backward).abs() < tolerance,
            "x_backward[{}] inconsistent",
            i
        );
    }
}

/// Test high-precision Gauss-Legendre rule with TwoFloat
/// Similar to C++ DDouble test but using TwoFloat
#[test]
fn test_high_precision_legendre_twofloat() {
    let n = 6; // Smaller n for TwoFloat due to complexity
    let rule = legendre_twofloat(n);

    // Check that the rule is valid
    assert!(rule.validate_twofloat());

    // Check that all points are within [-1, 1]
    for &xi in rule.x.iter() {
        assert!(xi >= TwoFloat::from(-1.0) && xi <= TwoFloat::from(1.0));
    }

    // Check that points are sorted
    for i in 1..rule.x.len() {
        assert!(rule.x[i] >= rule.x[i - 1]);
    }

    // Check x_forward and x_backward consistency with TwoFloat precision
    let tolerance = TwoFloat::from(1e-15); // Higher precision for TwoFloat
    for i in 0..rule.x.len() {
        let expected_forward = rule.x[i] - rule.a;
        let expected_backward = rule.b - rule.x[i];

        assert!(
            (rule.x_forward[i] - expected_forward).abs() < tolerance,
            "x_forward[{}] inconsistent",
            i
        );
        assert!(
            (rule.x_backward[i] - expected_backward).abs() < tolerance,
            "x_backward[{}] inconsistent",
            i
        );
    }

    // Test orthogonality property: sum of weights should be 2.0
    let weight_sum: TwoFloat = rule.w.iter().fold(TwoFloat::from(0.0), |acc, &w| acc + w);
    assert!(
        (weight_sum - TwoFloat::from(2.0)).abs() < TwoFloat::from(1e-14),
        "Sum of weights should be 2.0, got {}",
        weight_sum
    );
}

/// Test Legendre polynomial evaluation at Gauss-Legendre nodes
/// This tests the orthogonality property
#[test]
fn test_legendre_polynomial_at_nodes() {
    let n = 8;
    let rule = legendre_custom::<f64>(n);

    // Test that P_0(x) = 1 at all nodes
    for i in 0..n {
        let p0 = legendre_polynomial(0, rule.x[i]);
        assert!((p0 - 1.0).abs() < 1e-14, "P_0(x[{}]) should be 1.0", i);
    }

    // Test that P_1(x) = x at all nodes
    for i in 0..n {
        let p1 = legendre_polynomial(1, rule.x[i]);
        assert!(
            (p1 - rule.x[i]).abs() < 1e-14,
            "P_1(x[{}]) should equal x[{}]",
            i,
            i
        );
    }

    // Test that P_n(x) = 0 at all nodes (where n is the order of the rule)
    // This is the defining property of Gauss-Legendre nodes
    for i in 0..n {
        let pn = legendre_polynomial(n, rule.x[i]);
        assert!(
            pn.abs() < 1e-12,
            "P_{}(x[{}]) should be approximately 0, got {}",
            n,
            i,
            pn
        );
    }
}

/// Test Legendre polynomial evaluation with TwoFloat
#[test]
fn test_legendre_polynomial_twofloat_at_nodes() {
    let n = 4; // Smaller n for TwoFloat
    let rule = legendre_twofloat(n);

    // Test that P_0(x) = 1 at all nodes
    for i in 0..n {
        let p0 = legendre_polynomial_twofloat(0, rule.x[i]);
        assert!(
            (p0 - TwoFloat::from(1.0)).abs() < TwoFloat::from(1e-15),
            "P_0(x[{}]) should be 1.0",
            i
        );
    }

    // Test that P_1(x) = x at all nodes
    for i in 0..n {
        let p1 = legendre_polynomial_twofloat(1, rule.x[i]);
        assert!(
            (p1 - rule.x[i]).abs() < TwoFloat::from(1e-15),
            "P_1(x[{}]) should equal x[{}]",
            i,
            i
        );
    }

    // Test that P_n(x) = 0 at all nodes
    for i in 0..n {
        let pn = legendre_polynomial_twofloat(n, rule.x[i]);
        assert!(
            pn.abs() < TwoFloat::from(1e-14),
            "P_{}(x[{}]) should be approximately 0, got {}",
            n,
            i,
            pn
        );
    }
}

/// Test large Gauss-Legendre rule like C++ test with n=200
#[test]
fn test_large_legendre_rule_high_precision() {
    let n = 200;
    let rule = legendre_custom::<f64>(n);

    // Check basic properties
    assert!(rule.validate_custom());
    assert_eq!(rule.a, -1.0);
    assert_eq!(rule.b, 1.0);
    assert_eq!(rule.x.len(), n);
    assert_eq!(rule.w.len(), n);

    // Check that all points are within [-1, 1]
    for &xi in rule.x.iter() {
        assert!(xi >= -1.0 && xi <= 1.0);
    }

    // Check that points are sorted
    for i in 1..rule.x.len() {
        assert!(rule.x[i] >= rule.x[i - 1]);
    }

    // Check sum of weights should be 2.0
    let weight_sum: f64 = rule.w.iter().sum();
    assert!(
        (weight_sum - 2.0).abs() < 1e-14,
        "Sum of weights should be 2.0, got {}",
        weight_sum
    );

    // Check x_forward and x_backward consistency with high precision
    let tolerance = 1e-14;
    for i in 0..rule.x.len() {
        let expected_forward = rule.x[i] - rule.a;
        let expected_backward = rule.b - rule.x[i];

        assert!(
            (rule.x_forward[i] - expected_forward).abs() < tolerance,
            "x_forward[{}] inconsistent",
            i
        );
        assert!(
            (rule.x_backward[i] - expected_backward).abs() < tolerance,
            "x_backward[{}] inconsistent",
            i
        );
    }
}

/// Test piecewise functionality with high precision
#[test]
fn test_piecewise_high_precision() {
    let edges = vec![-4.0, -1.0, 1.0, 3.0];
    let rule = legendre_custom::<f64>(20).piecewise(&edges);

    assert!(rule.validate_custom());
    assert_eq!(rule.a, -4.0);
    assert_eq!(rule.b, 3.0);

    // Check that all points are within the overall interval
    for &xi in rule.x.iter() {
        assert!(xi >= -4.0 && xi <= 3.0);
    }

    // Check that points are sorted
    for i in 1..rule.x.len() {
        assert!(rule.x[i] >= rule.x[i - 1]);
    }

    // Check sum of weights should be 7.0 (length of interval)
    let weight_sum: f64 = rule.w.iter().sum();
    assert!(
        (weight_sum - 7.0).abs() < 1e-13,
        "Sum of weights should be 7.0, got {}",
        weight_sum
    );
}
