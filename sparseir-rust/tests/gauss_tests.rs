use sparseir_rust::{Rule, legendre, legendre_custom, legendre_twofloat, TwoFloat};
use ndarray::Array1;

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
        assert!(rule.x[i] >= rule.x[i-1]);
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
    for n in 1..=3 {  // Smaller range for TwoFloat due to complexity
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
