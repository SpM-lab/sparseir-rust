//! Kernel matrix discretization for SparseIR
//!
//! This module provides functionality to discretize kernels using Gauss quadrature
//! rules and store them as matrices for numerical computation.

use crate::gauss::Rule;
use crate::kernel::{CentrosymmKernel, KernelProperties, SymmetryType};
use crate::numeric::CustomNumeric;
use ndarray::Array2;
use num_traits::ToPrimitive;

/// This structure stores a discrete kernel matrix along with the corresponding
/// Gauss quadrature rules for x and y coordinates. This enables easy application
/// of weights for SVE computation and maintains the relationship between matrix
/// elements and their corresponding quadrature points.
#[derive(Debug, Clone)]
pub struct DiscretizedKernel<T> {
    /// Discrete kernel matrix
    pub matrix: Array2<T>,
    /// Gauss quadrature rule for x coordinates
    pub gauss_x: Rule<T>,
    /// Gauss quadrature rule for y coordinates  
    pub gauss_y: Rule<T>,
}

impl<T: CustomNumeric + Clone> DiscretizedKernel<T> {
    /// Create a new DiscretizedKernel
    pub fn new(matrix: Array2<T>, gauss_x: Rule<T>, gauss_y: Rule<T>) -> Self {
        Self { matrix, gauss_x, gauss_y }
    }
    
    /// Delegate to matrix methods
    pub fn is_empty(&self) -> bool {
        self.matrix.is_empty()
    }
    
    pub fn nrows(&self) -> usize {
        self.matrix.nrows()
    }
    
    pub fn ncols(&self) -> usize {
        self.matrix.ncols()
    }
    
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.matrix.iter()
    }
    
    /// Apply weights for SVE computation
    /// 
    /// This applies the square root of Gauss weights to the matrix,
    /// which is required before performing SVD for SVE computation.
    /// The original matrix remains unchanged.
    pub fn apply_weights_for_sve(&self) -> Array2<T> {
        let mut weighted_matrix = self.matrix.clone();
        
        // Apply square root of x-direction weights to rows
        for i in 0..self.gauss_x.x.len() {
            let weight_sqrt = self.gauss_x.w[i].sqrt();
            weighted_matrix.row_mut(i).mapv_inplace(|x| x * weight_sqrt);
        }
        
        // Apply square root of y-direction weights to columns
        for j in 0..self.gauss_y.x.len() {
            let weight_sqrt = self.gauss_y.w[j].sqrt();
            weighted_matrix.column_mut(j).mapv_inplace(|x| x * weight_sqrt);
        }
        
        weighted_matrix
    }
    
    /// Remove weights from matrix (inverse of apply_weights_for_sve)
    pub fn remove_weights_from_sve(&mut self) {
        // Remove weights from U matrix (x-direction)
        for i in 0..self.gauss_x.x.len() {
            let weight_sqrt = self.gauss_x.w[i].sqrt();
            self.matrix.row_mut(i).mapv_inplace(|x| x / weight_sqrt);
        }
        
        // Remove weights from V matrix (y-direction) 
        for j in 0..self.gauss_y.x.len() {
            let weight_sqrt = self.gauss_y.w[j].sqrt();
            self.matrix.column_mut(j).mapv_inplace(|x| x / weight_sqrt);
        }
    }
    
    /// Get the number of Gauss points in x direction
    pub fn n_gauss_x(&self) -> usize {
        self.gauss_x.x.len()
    }
    
    /// Get the number of Gauss points in y direction
    pub fn n_gauss_y(&self) -> usize {
        self.gauss_y.x.len()
    }
}



/// Compute matrix from Gauss quadrature rules
/// 
/// This function evaluates the kernel at all combinations of Gauss points
/// and returns a DiscretizedKernel containing the matrix and quadrature rules.
pub fn matrix_from_gauss<T: CustomNumeric + ToPrimitive + num_traits::Zero + Clone, K: CentrosymmKernel + KernelProperties>(
    kernel: &K,
    gauss_x: &Rule<T>,
    gauss_y: &Rule<T>,
    symmetry: SymmetryType,
) -> DiscretizedKernel<T> {
    // Check that Gauss points are within [0, xmax] and [0, ymax]
    let kernel_xmax = kernel.xmax();
    let kernel_ymax = kernel.ymax();
    let tolerance = 1e-12;
    
    // Check x points are in [0, xmax]
    for &x in &gauss_x.x {
        let x_f64 = x.to_f64();
        assert!(
            x_f64 >= -tolerance && x_f64 <= kernel_xmax + tolerance,
            "Gauss x point {} is outside [0, {}]", x_f64, kernel_xmax
        );
    }
    
    // Check y points are in [0, ymax]
    for &y in &gauss_y.x {
        let y_f64 = y.to_f64();
        assert!(
            y_f64 >= -tolerance && y_f64 <= kernel_ymax + tolerance,
            "Gauss y point {} is outside [0, {}]", y_f64, kernel_ymax
        );
    }
    
    let n = gauss_x.x.len();
    let m = gauss_y.x.len();
    let mut result = Array2::zeros((n, m));
    
    // Evaluate kernel at all combinations of Gauss points
    for i in 0..n {
        for j in 0..m {
            let x = gauss_x.x[i];
            let y = gauss_y.x[j];
            
            // Use T type directly for kernel computation
            // Note: gauss_x and gauss_y should already be scaled to [0, 1] interval
            result[[i, j]] = kernel.compute_reduced(x, y, symmetry);
        }
    }
    
    DiscretizedKernel::new(result, gauss_x.clone(), gauss_y.clone())
}
