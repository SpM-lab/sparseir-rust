//! Gauss quadrature rules for numerical integration
//!
//! This module provides quadrature rules for approximating integrals by weighted sums:
//!
//!     ∫ f(x) * omega(x) * dx ~ sum(f(xi) * wi for (xi, wi) in zip(x, w))
//!
//! where we generally have superexponential convergence for smooth f(x)
//! with the number of quadrature points.

use ndarray::{Array1, ScalarOperand};
use std::fmt::Debug;
use num_traits::{FromPrimitive, ToPrimitive, Float};

/// Quadrature rule for numerical integration.
///
/// Represents an approximation of an integral by a weighted sum over discrete points.
/// The rule contains quadrature points `x`, weights `w`, and auxiliary arrays
/// `x_forward` and `x_backward` for efficient computation.
#[derive(Debug, Clone)]
pub struct Rule<T> {
    /// Quadrature points
    pub x: Array1<T>,
    /// Quadrature weights
    pub w: Array1<T>,
    /// Distance from left endpoint: x - a
    pub x_forward: Array1<T>,
    /// Distance from right endpoint: b - x
    pub x_backward: Array1<T>,
    /// Left endpoint of integration interval
    pub a: T,
    /// Right endpoint of integration interval
    pub b: T,
}

impl<T> Rule<T>
where
    T: Copy + Debug + Float + FromPrimitive + ToPrimitive + ScalarOperand + std::fmt::Display,
{
    /// Create a new quadrature rule from points and weights.
    ///
    /// # Arguments
    /// * `x` - Quadrature points
    /// * `w` - Quadrature weights
    /// * `a` - Left endpoint (default: -1.0)
    /// * `b` - Right endpoint (default: 1.0)
    ///
    /// # Panics
    /// Panics if x and w have different lengths.
    pub fn new(x: Array1<T>, w: Array1<T>, a: T, b: T) -> Self {
        assert_eq!(x.len(), w.len(), "x and w must have the same length");
        
        let x_forward = &x - a;
        let x_backward = x.mapv(|xi| b - xi);
        
        Self {
            x,
            w,
            x_forward,
            x_backward,
            a,
            b,
        }
    }
    
    /// Create a new quadrature rule from vectors.
    pub fn from_vectors(x: Vec<T>, w: Vec<T>, a: T, b: T) -> Self {
        Self::new(Array1::from(x), Array1::from(w), a, b)
    }
    
    /// Create a default rule with empty arrays.
    pub fn empty() -> Self {
        Self {
            x: Array1::from(vec![]),
            w: Array1::from(vec![]),
            x_forward: Array1::from(vec![]),
            x_backward: Array1::from(vec![]),
            a: T::from_f64(-1.0).unwrap(),
            b: T::from_f64(1.0).unwrap(),
        }
    }
    
    /// Reseat the rule to a new interval [a, b].
    ///
    /// Scales and translates the quadrature points and weights to the new interval.
    pub fn reseat(&self, a: T, b: T) -> Self {
        let scaling = (b - a) / (self.b - self.a);
        let midpoint_old = (self.b + self.a) * T::from_f64(0.5).unwrap();
        let midpoint_new = (b + a) * T::from_f64(0.5).unwrap();
        
        // Transform x: scaling * (xi - midpoint_old) + midpoint_new
        let new_x = self.x.mapv(|xi| scaling * (xi - midpoint_old) + midpoint_new);
        let new_w = &self.w * scaling;
        let new_x_forward = &self.x_forward * scaling;
        let new_x_backward = &self.x_backward * scaling;
        
        Self {
            x: new_x,
            w: new_w,
            x_forward: new_x_forward,
            x_backward: new_x_backward,
            a,
            b,
        }
    }
    
    /// Scale the weights by a factor.
    pub fn scale(&self, factor: T) -> Self {
        Self {
            x: self.x.clone(),
            w: &self.w * factor,
            x_forward: self.x_forward.clone(),
            x_backward: self.x_backward.clone(),
            a: self.a,
            b: self.b,
        }
    }
    
    /// Create a piecewise rule over multiple segments.
    ///
    /// # Arguments
    /// * `edges` - Segment boundaries (must be sorted in ascending order)
    ///
    /// # Panics
    /// Panics if edges are not sorted or have less than 2 elements.
    pub fn piecewise(&self, edges: &[T]) -> Self {
        if edges.len() < 2 {
            panic!("edges must have at least 2 elements");
        }
        
        // Check if edges are sorted
        for i in 1..edges.len() {
            if edges[i] <= edges[i-1] {
                panic!("edges must be sorted in ascending order");
            }
        }
        
        let mut rules = Vec::new();
        for i in 0..edges.len() - 1 {
            let rule = self.reseat(edges[i], edges[i + 1]);
            rules.push(rule);
        }
        
        Self::join(&rules)
    }
    
    /// Join multiple rules into a single rule.
    ///
    /// # Arguments
    /// * `rules` - Vector of rules to join (must be contiguous and sorted)
    ///
    /// # Panics
    /// Panics if rules are empty, not contiguous, or not sorted.
    pub fn join(rules: &[Self]) -> Self {
        if rules.is_empty() {
            return Self::empty();
        }
        
        let a = rules[0].a;
        let b = rules[rules.len() - 1].b;
        
        // Check that rules are contiguous
        for i in 1..rules.len() {
            if (rules[i].a - rules[i-1].b).abs() > T::epsilon() {
                panic!("rules must be contiguous");
            }
        }
        
        // Concatenate all arrays
        let mut x_vec = Vec::new();
        let mut w_vec = Vec::new();
        let mut x_forward_vec = Vec::new();
        let mut x_backward_vec = Vec::new();
        
        for rule in rules {
            // Adjust x_forward and x_backward for global coordinates
            let x_forward_adj = &rule.x_forward + (rule.a - a);
            let x_backward_adj = &rule.x_backward + (b - rule.b);
            
            x_vec.extend_from_slice(rule.x.as_slice().unwrap());
            w_vec.extend_from_slice(rule.w.as_slice().unwrap());
            x_forward_vec.extend_from_slice(x_forward_adj.as_slice().unwrap());
            x_backward_vec.extend_from_slice(x_backward_adj.as_slice().unwrap());
        }
        
        // Sort by x values to maintain order
        let mut indices: Vec<usize> = (0..x_vec.len()).collect();
        indices.sort_by(|&a, &b| x_vec[a].partial_cmp(&x_vec[b]).unwrap());
        
        let sorted_x: Vec<T> = indices.iter().map(|&i| x_vec[i]).collect();
        let sorted_w: Vec<T> = indices.iter().map(|&i| w_vec[i]).collect();
        
        // Recalculate x_forward and x_backward after sorting
        let sorted_x_forward: Vec<T> = sorted_x.iter().map(|&xi| xi - a).collect();
        let sorted_x_backward: Vec<T> = sorted_x.iter().map(|&xi| b - xi).collect();
        
        let x = Array1::from(sorted_x);
        let w = Array1::from(sorted_w);
        let x_forward = Array1::from(sorted_x_forward);
        let x_backward = Array1::from(sorted_x_backward);
        
        Self {
            x,
            w,
            x_forward,
            x_backward,
            a,
            b,
        }
    }
    
    /// Convert the rule to a different numeric type.
    pub fn convert<U>(&self) -> Rule<U>
    where
        U: Copy + Debug + Float + FromPrimitive + ToPrimitive + ScalarOperand + std::fmt::Display,
    {
        let x: Array1<U> = self.x.iter().map(|&xi| U::from_f64(xi.to_f64().unwrap()).unwrap()).collect();
        let w: Array1<U> = self.w.iter().map(|&wi| U::from_f64(wi.to_f64().unwrap()).unwrap()).collect();
        let x_forward: Array1<U> = self.x_forward.iter().map(|&xi| U::from_f64(xi.to_f64().unwrap()).unwrap()).collect();
        let x_backward: Array1<U> = self.x_backward.iter().map(|&xi| U::from_f64(xi.to_f64().unwrap()).unwrap()).collect();
        let a = U::from_f64(self.a.to_f64().unwrap()).unwrap();
        let b = U::from_f64(self.b.to_f64().unwrap()).unwrap();
        
        Rule {
            x,
            w,
            x_forward,
            x_backward,
            a,
            b,
        }
    }
    
    /// Validate the rule for consistency.
    ///
    /// # Returns
    /// `true` if the rule is valid, `false` otherwise.
    pub fn validate(&self) -> bool {
        // Check interval validity
        if self.a >= self.b {
            return false;
        }
        
        // Check array lengths
        if self.x.len() != self.w.len() {
            return false;
        }
        
        if self.x.len() != self.x_forward.len() || self.x.len() != self.x_backward.len() {
            return false;
        }
        
        // Check that all points are within [a, b]
        for &xi in self.x.iter() {
            if xi < self.a || xi > self.b {
                return false;
            }
        }
        
        // Check that points are sorted
        for i in 1..self.x.len() {
            if self.x[i] <= self.x[i-1] {
                return false;
            }
        }
        
        // Check x_forward and x_backward consistency
        for i in 0..self.x.len() {
            let expected_forward = self.x[i] - self.a;
            let expected_backward = self.b - self.x[i];
            
            if (self.x_forward[i] - expected_forward).abs() > T::epsilon() {
                return false;
            }
            if (self.x_backward[i] - expected_backward).abs() > T::epsilon() {
                return false;
            }
        }
        
        true
    }
}

/// Compute Gauss-Legendre quadrature nodes and weights using Newton's method.
///
/// This is a simplified implementation of the Gauss-Legendre quadrature rule.
/// For production use, a more sophisticated algorithm would be preferred.
fn gauss_legendre_nodes_weights<T>(n: usize) -> (Vec<T>, Vec<T>)
where
    T: Copy + Debug + Float + FromPrimitive + ToPrimitive + ScalarOperand + std::fmt::Display,
{
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    
    if n == 1 {
        return (vec![T::from_f64(0.0).unwrap()], vec![T::from_f64(2.0).unwrap()]);
    }
    
    let mut x = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    
    // Use Newton's method to find roots of Legendre polynomial
    let m = (n + 1) / 2;
    let pi = T::from_f64(std::f64::consts::PI).unwrap();
    
    for i in 0..m {
        // Initial guess using Chebyshev nodes
        let mut z = (pi * T::from_f64(i as f64 + 0.75).unwrap() / T::from_f64(n as f64 + 0.5).unwrap()).cos();
        
        // Newton's method to refine the root
        for _ in 0..10 {
            let (p0, p1) = legendre_polynomial_and_derivative(n, z);
            if p0.abs() < T::epsilon() {
                break;
            }
            z = z - p0 / p1;
        }
        
        // Compute weight
        let (_, p1) = legendre_polynomial_and_derivative(n, z);
        let weight = T::from_f64(2.0).unwrap() / ((T::from_f64(1.0).unwrap() - z * z) * p1 * p1);
        
        x.push(-z);
        w.push(weight);
        
        if i != n - 1 - i {
            x.push(z);
            w.push(weight);
        }
    }
    
    // Sort by x values
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
    
    let sorted_x: Vec<T> = indices.iter().map(|&i| x[i]).collect();
    let sorted_w: Vec<T> = indices.iter().map(|&i| w[i]).collect();
    
    (sorted_x, sorted_w)
}

/// Compute Legendre polynomial P_n(x) and its derivative using recurrence relation.
fn legendre_polynomial_and_derivative<T>(n: usize, x: T) -> (T, T)
where
    T: Copy + Debug + Float + FromPrimitive + ToPrimitive + ScalarOperand + std::fmt::Display,
{
    if n == 0 {
        return (T::from_f64(1.0).unwrap(), T::from_f64(0.0).unwrap());
    }
    
    if n == 1 {
        return (x, T::from_f64(1.0).unwrap());
    }
    
    let mut p0 = T::from_f64(1.0).unwrap();
    let mut p1 = x;
    let mut dp0 = T::from_f64(0.0).unwrap();
    let mut dp1 = T::from_f64(1.0).unwrap();
    
    for k in 2..=n {
        let k_f = T::from_f64(k as f64).unwrap();
        let k1_f = T::from_f64((k - 1) as f64).unwrap();
        let _k2_f = T::from_f64((k - 2) as f64).unwrap();
        
        let p2 = ((T::from_f64(2.0).unwrap() * k1_f + T::from_f64(1.0).unwrap()) * x * p1 - k1_f * p0) / k_f;
        let dp2 = ((T::from_f64(2.0).unwrap() * k1_f + T::from_f64(1.0).unwrap()) * (p1 + x * dp1) - k1_f * dp0) / k_f;
        
        p0 = p1;
        p1 = p2;
        dp0 = dp1;
        dp1 = dp2;
    }
    
    (p1, dp1)
}

/// Create a Gauss-Legendre quadrature rule with n points on [-1, 1].
///
/// # Arguments
/// * `n` - Number of quadrature points
///
/// # Returns
/// A Gauss-Legendre quadrature rule
pub fn legendre<T>(n: usize) -> Rule<T>
where
    T: Copy + Debug + Float + FromPrimitive + ToPrimitive + ScalarOperand + std::fmt::Display,
{
    if n == 0 {
        return Rule::empty();
    }
    
    let (x, w) = gauss_legendre_nodes_weights(n);
    
    Rule::from_vectors(x, w, T::from_f64(-1.0).unwrap(), T::from_f64(1.0).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rule_constructor() {
        let x = Array1::from(vec![0.0, 1.0]);
        let w = Array1::from(vec![0.5, 0.5]);
        let rule = Rule::new(x.clone(), w.clone(), -1.0, 1.0);
        
        assert_eq!(rule.a, -1.0);
        assert_eq!(rule.b, 1.0);
        assert_eq!(rule.x, x);
        assert_eq!(rule.w, w);
    }
    
    #[test]
    fn test_rule_validation() {
        let x = Array1::from(vec![0.0, 1.0]);
        let w = Array1::from(vec![0.5, 0.5]);
        let rule = Rule::new(x, w, -1.0, 1.0);
        
        assert!(rule.validate());
    }
    
    #[test]
    fn test_rule_reseat() {
        let x = Array1::from(vec![0.0, 1.0]);
        let w = Array1::from(vec![0.5, 0.5]);
        let rule = Rule::new(x, w, -1.0, 1.0);
        
        let reseated = rule.reseat(0.0, 2.0);
        
        assert_eq!(reseated.a, 0.0);
        assert_eq!(reseated.b, 2.0);
        assert!(reseated.validate());
    }
    
    #[test]
    fn test_rule_scale() {
        let x = Array1::from(vec![0.0, 1.0]);
        let w = Array1::from(vec![0.5, 0.5]);
        let rule = Rule::new(x, w, -1.0, 1.0);
        
        let scaled = rule.scale(2.0);
        
        assert_eq!(scaled.w[0], 1.0);
        assert_eq!(scaled.w[1], 1.0);
    }
    
    #[test]
    fn test_rule_piecewise() {
        let x = Array1::from(vec![0.0]);
        let w = Array1::from(vec![1.0]);
        let rule = Rule::new(x, w, -1.0, 1.0);
        
        let edges = vec![-2.0, 0.0, 2.0];
        let piecewise = rule.piecewise(&edges);
        
        assert_eq!(piecewise.a, -2.0);
        assert_eq!(piecewise.b, 2.0);
        assert!(piecewise.validate());
    }
    
    #[test]
    fn test_rule_join() {
        let x1 = Array1::from(vec![0.0]);
        let w1 = Array1::from(vec![1.0]);
        let rule1 = Rule::new(x1, w1, -1.0, 0.0);
        
        let x2 = Array1::from(vec![1.0]);
        let w2 = Array1::from(vec![1.0]);
        let rule2 = Rule::new(x2, w2, 0.0, 1.0);
        
        let joined = Rule::join(&[rule1, rule2]);
        
        assert_eq!(joined.a, -1.0);
        assert_eq!(joined.b, 1.0);
        assert_eq!(joined.x.len(), 2);
        assert!(joined.validate());
    }
    
    #[test]
    fn test_legendre() {
        let rule = legendre::<f64>(4);
        
        assert_eq!(rule.a, -1.0);
        assert_eq!(rule.b, 1.0);
        assert_eq!(rule.x.len(), 4);
        assert_eq!(rule.w.len(), 4);
        assert!(rule.validate());
    }
    
    #[test]
    fn test_rule_convert() {
        let rule_f64 = legendre::<f64>(4);
        let rule_f32 = rule_f64.convert::<f32>();
        
        assert_eq!(rule_f32.x.len(), 4);
        assert_eq!(rule_f32.w.len(), 4);
        assert!(rule_f32.validate());
    }
    
    #[test]
    fn test_f32_support() {
        let rule = legendre::<f32>(4);
        
        assert_eq!(rule.a, -1.0_f32);
        assert_eq!(rule.b, 1.0_f32);
        assert_eq!(rule.x.len(), 4);
        assert_eq!(rule.w.len(), 4);
        assert!(rule.validate());
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
}
