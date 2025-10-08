//! Kernel matrix discretization for SparseIR (mdarray version)
//!
//! This module provides functionality to discretize kernels using Gauss quadrature
//! rules and store them as matrices for numerical computation.

use crate::gauss_mdarray::Rule;
use crate::kernel::{CentrosymmKernel, KernelProperties, SymmetryType};
use crate::numeric::CustomNumeric;
use mdarray::Tensor;
use std::fmt::Debug;

/// This structure stores a discrete kernel matrix along with the corresponding
/// Gauss quadrature rules for x and y coordinates. This enables easy application
/// of weights for SVE computation and maintains the relationship between matrix
/// elements and their corresponding quadrature points.
#[derive(Debug, Clone)]
pub struct DiscretizedKernel<T> {
    /// Discrete kernel matrix
    pub matrix: Tensor<T, (usize, usize)>,
    /// Gauss quadrature rule for x coordinates
    pub gauss_x: Rule<T>,
    /// Gauss quadrature rule for y coordinates  
    pub gauss_y: Rule<T>,
    /// X-axis segment boundaries (from SVEHints)
    pub segments_x: Vec<T>,
    /// Y-axis segment boundaries (from SVEHints)
    pub segments_y: Vec<T>,
}

impl<T: CustomNumeric + Clone> DiscretizedKernel<T> {
    /// Create a new DiscretizedKernel
    pub fn new(
        matrix: Tensor<T, (usize, usize)>,
        gauss_x: Rule<T>,
        gauss_y: Rule<T>,
        segments_x: Vec<T>,
        segments_y: Vec<T>,
    ) -> Self {
        Self {
            matrix,
            gauss_x,
            gauss_y,
            segments_x,
            segments_y,
        }
    }

    /// Create a new DiscretizedKernel without segments (legacy)
    pub fn new_legacy(
        matrix: Tensor<T, (usize, usize)>,
        gauss_x: Rule<T>,
        gauss_y: Rule<T>,
    ) -> Self {
        Self {
            matrix,
            gauss_x: gauss_x.clone(),
            gauss_y: gauss_y.clone(),
            segments_x: vec![gauss_x.a, gauss_x.b],
            segments_y: vec![gauss_y.a, gauss_y.b],
        }
    }

    /// Delegate to matrix methods
    pub fn is_empty(&self) -> bool {
        self.matrix.is_empty()
    }

    pub fn nrows(&self) -> usize {
        self.matrix.shape().0
    }

    pub fn ncols(&self) -> usize {
        self.matrix.shape().1
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.matrix.iter()
    }

    /// Apply weights for SVE computation
    ///
    /// This applies the square root of Gauss weights to the matrix,
    /// which is required before performing SVD for SVE computation.
    /// The original matrix remains unchanged.
    pub fn apply_weights_for_sve(&self) -> Tensor<T, (usize, usize)> {
        let nrows = self.nrows();
        let ncols = self.ncols();

        // Apply sqrt(wx) * sqrt(wy) to each element
        let weighted_matrix = Tensor::from_fn((nrows, ncols), |idx| {
            let i = idx[0];
            let j = idx[1];
            let weight_x_sqrt = self.gauss_x.w[[i]].sqrt();
            let weight_y_sqrt = self.gauss_y.w[[j]].sqrt();
            self.matrix[[i, j]] * weight_x_sqrt * weight_y_sqrt
        });

        weighted_matrix
    }

    /// Remove weights from matrix (inverse of apply_weights_for_sve)
    pub fn remove_weights(&self) -> Tensor<T, (usize, usize)> {
        let nrows = self.nrows();
        let ncols = self.ncols();

        // Remove sqrt(wx) * sqrt(wy) from each element
        let unweighted_matrix = Tensor::from_fn((nrows, ncols), |idx| {
            let i = idx[0];
            let j = idx[1];
            let weight_x_sqrt = self.gauss_x.w[[i]].sqrt();
            let weight_y_sqrt = self.gauss_y.w[[j]].sqrt();
            self.matrix[[i, j]] / (weight_x_sqrt * weight_y_sqrt)
        });

        unweighted_matrix
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

/// Compute matrix from Gauss quadrature rules with segments from SVEHints
///
/// This function evaluates the kernel at all combinations of Gauss points
/// and returns a DiscretizedKernel containing the matrix, quadrature rules, and segments.
pub fn matrix_from_gauss_with_segments<
    T: CustomNumeric + Clone + Send + Sync,
    K: CentrosymmKernel + KernelProperties,
    H: crate::kernel::SVEHints<T>,
>(
    kernel: &K,
    gauss_x: &Rule<T>,
    gauss_y: &Rule<T>,
    symmetry: SymmetryType,
    hints: &H,
) -> DiscretizedKernel<T> {
    let segments_x = hints.segments_x();
    let segments_y = hints.segments_y();

    let nx = gauss_x.x.len();
    let ny = gauss_y.x.len();

    // Compute matrix: K[i,j] = kernel_symmetrized(x[i], y[j])
    let matrix = Tensor::from_fn((nx, ny), |idx| {
        let i = idx[0];
        let j = idx[1];
        let x = gauss_x.x[[i]];
        let y = gauss_y.x[[j]];
        kernel.compute_reduced(x, y, symmetry)
    });

    DiscretizedKernel::new(matrix, gauss_x.clone(), gauss_y.clone(), segments_x, segments_y)
}

/// Legacy function: Compute matrix from Gauss quadrature rules
pub fn matrix_from_gauss<T: CustomNumeric + Clone + Send + Sync, K: CentrosymmKernel + KernelProperties>(
    kernel: &K,
    gauss_x: &Rule<T>,
    gauss_y: &Rule<T>,
    symmetry: SymmetryType,
) -> DiscretizedKernel<T> {
    let nx = gauss_x.x.len();
    let ny = gauss_y.x.len();

    let matrix = Tensor::from_fn((nx, ny), |idx| {
        let i = idx[0];
        let j = idx[1];
        let x = gauss_x.x[[i]];
        let y = gauss_y.x[[j]];
        kernel.compute_reduced(x, y, symmetry)
    });

    DiscretizedKernel::new_legacy(matrix, gauss_x.clone(), gauss_y.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::LogisticKernel;

    #[test]
    fn test_discretized_kernel_basic() {
        let matrix = Tensor::from_fn((3, 3), |idx| (idx[0] * 3 + idx[1]) as f64);
        let x = vec![0.0, 0.5, 1.0];
        let w = vec![0.3, 0.4, 0.3];
        let gauss_x = Rule::from_vectors(x.clone(), w.clone(), 0.0, 1.0);
        let gauss_y = Rule::from_vectors(x, w, 0.0, 1.0);

        let dk = DiscretizedKernel::new(
            matrix,
            gauss_x,
            gauss_y,
            vec![0.0, 1.0],
            vec![0.0, 1.0],
        );

        assert_eq!(dk.nrows(), 3);
        assert_eq!(dk.ncols(), 3);
        assert_eq!(dk.n_gauss_x(), 3);
        assert_eq!(dk.n_gauss_y(), 3);
    }

    #[test]
    fn test_apply_weights() {
        let matrix = Tensor::from_elem((2, 2), 1.0_f64);
        let x = vec![-1.0, 1.0];
        let w = vec![1.0, 1.0];
        let gauss_x = Rule::from_vectors(x.clone(), w.clone(), -1.0, 1.0);
        let gauss_y = Rule::from_vectors(x, w, -1.0, 1.0);

        let dk = DiscretizedKernel::new_legacy(matrix, gauss_x, gauss_y);
        let weighted = dk.apply_weights_for_sve();

        // sqrt(1.0) * sqrt(1.0) = 1.0, so matrix should be unchanged
        assert_eq!(weighted[[0, 0]], 1.0);
        assert_eq!(weighted[[1, 1]], 1.0);
    }

    #[test]
    fn test_matrix_from_gauss() {
        let kernel = LogisticKernel::new(10.0);
        let x = vec![0.0, 0.5];
        let w = vec![0.5, 0.5];
        let gauss_x = Rule::from_vectors(x.clone(), w.clone(), 0.0, 1.0);
        let gauss_y = Rule::from_vectors(x, w, 0.0, 1.0);

        let dk = matrix_from_gauss(&kernel, &gauss_x, &gauss_y, SymmetryType::Even);

        assert_eq!(dk.nrows(), 2);
        assert_eq!(dk.ncols(), 2);
    }
}

