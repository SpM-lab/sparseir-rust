//! Gauss quadrature rules for numerical integration (mdarray version)
//!
//! This module provides quadrature rules for approximating integrals by weighted sums.
//!
//! The integral of f(x) * omega(x) is approximated by a weighted sum:
//!
//! sum(f(xi) * wi for (xi, wi) in zip(x, w))
//!
//! where we generally have superexponential convergence for smooth f(x)
//! with the number of quadrature points.

use crate::numeric::CustomNumeric;
use mdarray::Tensor;
use std::fmt::Debug;

/// Quadrature rule for numerical integration (mdarray version).
///
/// Represents an approximation of an integral by a weighted sum over discrete points.
/// The rule contains quadrature points `x`, weights `w`, and auxiliary arrays
/// `x_forward` and `x_backward` for efficient computation.
#[derive(Debug, Clone)]
pub struct Rule<T> {
    /// Quadrature points
    pub x: Tensor<T, (usize,)>,
    /// Quadrature weights
    pub w: Tensor<T, (usize,)>,
    /// Distance from left endpoint: x - a
    pub x_forward: Tensor<T, (usize,)>,
    /// Distance from right endpoint: b - x
    pub x_backward: Tensor<T, (usize,)>,
    /// Left endpoint of integration interval
    pub a: T,
    /// Right endpoint of integration interval
    pub b: T,
}

impl<T> Rule<T>
where
    T: CustomNumeric,
{
    /// Create a new quadrature rule from points and weights.
    ///
    /// # Arguments
    /// * `x` - Quadrature points (Tensor)
    /// * `w` - Quadrature weights (Tensor)
    /// * `a` - Left endpoint (default: -1.0)
    /// * `b` - Right endpoint (default: 1.0)
    ///
    /// # Panics
    /// Panics if x and w have different lengths.
    pub fn new(x: Tensor<T, (usize,)>, w: Tensor<T, (usize,)>, a: T, b: T) -> Self {
        assert_eq!(x.len(), w.len(), "x and w must have the same length");

        // x_forward = x - a
        let x_forward = Tensor::from_fn((x.len(),), |idx| x[[idx[0]]] - a);
        // x_backward = b - x
        let x_backward = Tensor::from_fn((x.len(),), |idx| b - x[[idx[0]]]);

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
        let n = x.len();
        let x_tensor = Tensor::from_fn((n,), |idx| x[idx[0]]);
        let w_tensor = Tensor::from_fn((n,), |idx| w[idx[0]]);
        Self::new(x_tensor, w_tensor, a, b)
    }

    /// Create a default rule with empty arrays.
    pub fn empty() -> Self {
        let empty_tensor: Tensor<T, (usize,)> = Tensor::from_fn((0,), |_| unreachable!());
        Self {
            x: empty_tensor.clone(),
            w: empty_tensor.clone(),
            x_forward: empty_tensor.clone(),
            x_backward: empty_tensor,
            a: <T as CustomNumeric>::from_f64(-1.0),
            b: <T as CustomNumeric>::from_f64(1.0),
        }
    }

    /// Reseat the rule to a new interval [a, b].
    ///
    /// Scales and translates the quadrature points and weights to the new interval.
    pub fn reseat(&self, a: T, b: T) -> Self {
        let scaling = (b - a) / (self.b - self.a);
        let midpoint_old = (self.b + self.a) * <T as CustomNumeric>::from_f64(0.5);
        let midpoint_new = (b + a) * <T as CustomNumeric>::from_f64(0.5);

        let n = self.x.len();
        
        // Transform x: scaling * (xi - midpoint_old) + midpoint_new
        let new_x = Tensor::from_fn((n,), |idx| {
            scaling * (self.x[[idx[0]]] - midpoint_old) + midpoint_new
        });
        let new_w = Tensor::from_fn((n,), |idx| self.w[[idx[0]]] * scaling);
        let new_x_forward = Tensor::from_fn((n,), |idx| self.x_forward[[idx[0]]] * scaling);
        let new_x_backward = Tensor::from_fn((n,), |idx| self.x_backward[[idx[0]]] * scaling);

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
        let n = self.w.len();
        let new_w = Tensor::from_fn((n,), |idx| self.w[[idx[0]]] * factor);
        
        Self {
            x: self.x.clone(),
            w: new_w,
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
            if edges[i] <= edges[i - 1] {
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
    /// Concatenates the quadrature points and weights of multiple rules.
    pub fn join(rules: &[Self]) -> Self {
        if rules.is_empty() {
            return Self::empty();
        }

        let mut all_x = Vec::new();
        let mut all_w = Vec::new();
        let mut all_x_forward = Vec::new();
        let mut all_x_backward = Vec::new();

        for rule in rules {
            for i in 0..rule.x.len() {
                all_x.push(rule.x[[i]]);
                all_w.push(rule.w[[i]]);
                all_x_forward.push(rule.x_forward[[i]]);
                all_x_backward.push(rule.x_backward[[i]]);
            }
        }

        let n = all_x.len();
        let x = Tensor::from_fn((n,), |idx| all_x[idx[0]]);
        let w = Tensor::from_fn((n,), |idx| all_w[idx[0]]);
        let x_forward = Tensor::from_fn((n,), |idx| all_x_forward[idx[0]]);
        let x_backward = Tensor::from_fn((n,), |idx| all_x_backward[idx[0]]);

        Self {
            x,
            w,
            x_forward,
            x_backward,
            a: rules.first().unwrap().a,
            b: rules.last().unwrap().b,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rule_creation() {
        let x = vec![0.0_f64, 0.5, 1.0];
        let w = vec![0.2, 0.6, 0.2];
        let rule = Rule::from_vectors(x, w, 0.0, 1.0);
        
        assert_eq!(rule.x.len(), 3);
        assert_eq!(rule.w.len(), 3);
        assert_eq!(rule.x[[0]], 0.0);
        assert_eq!(rule.x[[1]], 0.5);
        assert_eq!(rule.x[[2]], 1.0);
    }
    
    #[test]
    fn test_rule_reseat() {
        let x = vec![-1.0_f64, 0.0, 1.0];
        let w = vec![1.0/3.0, 4.0/3.0, 1.0/3.0];
        let rule = Rule::from_vectors(x, w, -1.0, 1.0);
        
        let reseated = rule.reseat(0.0, 2.0);
        assert_eq!(reseated.a, 0.0);
        assert_eq!(reseated.b, 2.0);
        assert!((reseated.x[[0]] - 0.0).abs() < 1e-10);
        assert!((reseated.x[[1]] - 1.0).abs() < 1e-10);
        assert!((reseated.x[[2]] - 2.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_rule_scale() {
        let x = vec![0.0_f64, 1.0];
        let w = vec![0.5, 0.5];
        let rule = Rule::from_vectors(x, w, 0.0, 1.0);
        
        let scaled = rule.scale(2.0);
        assert_eq!(scaled.w[[0]], 1.0);
        assert_eq!(scaled.w[[1]], 1.0);
        assert_eq!(scaled.x[[0]], 0.0);  // x unchanged
    }
}

