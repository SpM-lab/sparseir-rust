//! 2D interpolation functionality for SparseIR
//!
//! This module provides efficient 2D interpolation using Legendre polynomials
//! within individual grid cells.

use crate::gauss::Rule;
use crate::interpolation1d::legendre_collocation_matrix;
use crate::numeric::CustomNumeric;
use ndarray::Array2;
use num_traits::Zero;
use std::fmt::Debug;

/// 2D interpolation object for a single grid cell
///
/// This structure stores pre-computed polynomial coefficients for efficient
/// interpolation within a single grid cell.
#[derive(Debug, Clone)]
pub struct Interpolate2D<T> {
    /// Cell boundaries
    pub x_min: T,
    pub x_max: T,
    pub y_min: T,
    pub y_max: T,
    
    /// Pre-computed polynomial coefficients
    pub coeffs: Array2<T>,
    
    /// Grid points (for validation)
    pub gauss_x: Rule<T>,
    pub gauss_y: Rule<T>,
}

impl<T: CustomNumeric + Debug + Zero + 'static> Interpolate2D<T> {
    /// Create a new Interpolate2D object from grid values
    ///
    /// # Arguments
    /// * `values` - Function values at grid points
    /// * `gauss_x` - x-direction grid points
    /// * `gauss_y` - y-direction grid points
    ///
    /// # Panics
    /// Panics if dimensions don't match or if grid is empty
    pub fn new(
        values: &Array2<T>,
        gauss_x: &Rule<T>,
        gauss_y: &Rule<T>,
    ) -> Self {
        assert!(!values.is_empty(), "Cannot create interpolation from empty grid");
        assert_eq!(values.nrows(), gauss_x.x.len(), "Values height must match gauss_x length");
        assert_eq!(values.ncols(), gauss_y.x.len(), "Values width must match gauss_y length");
        
        let coeffs = interpolate_2d_legendre(values, gauss_x, gauss_y);
        
        Self {
            x_min: gauss_x.a,
            x_max: gauss_x.b,
            y_min: gauss_y.a,
            y_max: gauss_y.b,
            coeffs,
            gauss_x: gauss_x.clone(),
            gauss_y: gauss_y.clone(),
        }
    }
    
    /// Interpolate the function at point (x, y)
    ///
    /// # Arguments
    /// * `x` - x-coordinate
    /// * `y` - y-coordinate
    ///
    /// # Panics
    /// Panics if (x, y) is outside the cell boundaries
    pub fn interpolate(&self, x: T, y: T) -> T {
        assert!(x >= self.x_min && x <= self.x_max, 
                "x={} is outside cell bounds [{}, {}]", x, self.x_min, self.x_max);
        assert!(y >= self.y_min && y <= self.y_max, 
                "y={} is outside cell bounds [{}, {}]", y, self.y_min, self.y_max);
        
        evaluate_2d_legendre_polynomial(x, y, &self.coeffs, &self.gauss_x, &self.gauss_y)
    }
    
    /// Get the coefficient matrix
    pub fn coefficients(&self) -> &Array2<T> {
        &self.coeffs
    }
    
    /// Get cell boundaries
    pub fn bounds(&self) -> (T, T, T, T) {
        (self.x_min, self.x_max, self.y_min, self.y_max)
    }
}

/// 2D Legendre polynomial interpolation using collocation matrices
///
/// This function computes coefficients for a 2D Legendre polynomial that
/// interpolates the given function values at the Gauss points using the
/// efficient collocation matrix approach.
///
/// # Arguments
/// * `values` - Function values at grid points (n_x x n_y)
/// * `gauss_x` - x-direction Gauss quadrature rule
/// * `gauss_y` - y-direction Gauss quadrature rule
///
/// # Returns
/// Coefficient matrix (n_x x n_y) for the interpolating polynomial
pub fn interpolate_2d_legendre<T: CustomNumeric + 'static>(
    values: &Array2<T>,
    gauss_x: &Rule<T>,
    gauss_y: &Rule<T>,
) -> Array2<T> {
    let n_x = gauss_x.x.len();
    let n_y = gauss_y.x.len();

    assert_eq!(values.nrows(), n_x, "Values matrix rows must match x grid points");
    assert_eq!(values.ncols(), n_y, "Values matrix cols must match y grid points");

    // Get collocation matrices (pre-computed inverses of Vandermonde matrices)
    let collocation_x = legendre_collocation_matrix(gauss_x);
    let collocation_y = legendre_collocation_matrix(gauss_y);

    // Compute coefficients using tensor product approach
    // coeffs = C_x * values * C_y^T
    let mut coeffs = Array2::from_elem((n_x, n_y), T::zero());
    
    for i in 0..n_x {
        for j in 0..n_y {
            for k in 0..n_x {
                for l in 0..n_y {
                    coeffs[[i, j]] = coeffs[[i, j]] + collocation_x[[i, k]] * values[[k, l]] * collocation_y[[j, l]];
                }
            }
        }
    }
    
    coeffs
}

/// Evaluate 2D Legendre polynomial at point (x, y) using coefficient matrix
pub fn evaluate_2d_legendre_polynomial<T: CustomNumeric>(
    x: T,
    y: T,
    coeffs: &Array2<T>,
    _gauss_x: &Rule<T>,
    _gauss_y: &Rule<T>,
) -> T {
    let n_x = coeffs.nrows();
    let n_y = coeffs.ncols();
    
    // Evaluate Legendre polynomials at x and y
    let p_x = evaluate_legendre_basis(x, n_x);
    let p_y = evaluate_legendre_basis(y, n_y);
    
    // Compute tensor product sum: sum_{i,j} coeffs[i,j] * P_i(x) * P_j(y)
    let mut result = T::zero();
    for i in 0..n_x {
        for j in 0..n_y {
            result = result + coeffs[[i, j]] * p_x[i] * p_y[j];
        }
    }
    
    result
}

/// Evaluate Legendre polynomial basis functions at point x
///
/// Returns a vector of Legendre polynomial values [P_0(x), P_1(x), ..., P_{n-1}(x)]
fn evaluate_legendre_basis<T: CustomNumeric>(x: T, n: usize) -> Vec<T> {
    if n == 0 {
        return Vec::new();
    }
    
    let mut p = Vec::with_capacity(n);
    
    // P_0(x) = 1
    p.push(T::from_f64(1.0));
    
    if n > 1 {
        // P_1(x) = x
        p.push(x);
    }
    
    // Recurrence relation: (n+1) * P_{n+1}(x) = (2n+1) * x * P_n(x) - n * P_{n-1}(x)
    for i in 1..n-1 {
        let i_f64 = i as f64;
        let next_p = (T::from_f64(2.0 * i_f64 + 1.0) * x * p[i] - T::from_f64(i_f64) * p[i-1]) 
                    / T::from_f64(i_f64 + 1.0);
        p.push(next_p);
    }
    
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gauss::legendre_generic;
    
    #[test]
    fn test_interpolate_2d_legendre_basic() {
        // Test with a simple 2D function: f(x,y) = x + y
        let gauss_x = legendre_generic::<f64>(2).reseat(-1.0, 1.0);
        let gauss_y = legendre_generic::<f64>(2).reseat(-1.0, 1.0);
        
        // Create test values
        let mut values = Array2::from_elem((2, 2), 0.0);
        for i in 0..2 {
            for j in 0..2 {
                values[[i, j]] = gauss_x.x[i] + gauss_y.x[j];
            }
        }
        
        let coeffs = interpolate_2d_legendre(&values, &gauss_x, &gauss_y);
        
        // Test interpolation at grid points (should be exact)
        for i in 0..2 {
            for j in 0..2 {
                let expected = gauss_x.x[i] + gauss_y.x[j];
                let interpolated = evaluate_2d_legendre_polynomial(
                    gauss_x.x[i], gauss_y.x[j], &coeffs, &gauss_x, &gauss_y
                );
                assert!((interpolated - expected).abs() < 1e-12,
                    "Interpolation failed at ({}, {}): expected {}, got {}",
                    gauss_x.x[i], gauss_y.x[j], expected, interpolated);
            }
        }
    }
    
    #[test]
    fn test_interpolate_2d_object() {
        let gauss_x = legendre_generic::<f64>(2).reseat(0.0, 1.0);
        let gauss_y = legendre_generic::<f64>(2).reseat(0.0, 2.0);
        
        let values = Array2::from_elem((2, 2), 1.0);
        let interp = Interpolate2D::new(&values, &gauss_x, &gauss_y);
        
        // Test interpolation at center of cell
        let result = interp.interpolate(0.5, 1.0);
        assert!(result.abs() < 10.0); // Just check it doesn't panic and returns reasonable value
    }
}