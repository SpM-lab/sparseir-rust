//! Utility functions for C API
//!
//! This module provides helper functions for order conversion and dimension handling.

use crate::{SPIR_ORDER_ROW_MAJOR, SPIR_ORDER_COLUMN_MAJOR};

/// Memory layout order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    RowMajor,    // Rightmost dimension varies fastest (C, Python)
    ColumnMajor, // Leftmost dimension varies fastest (Fortran, Julia, MATLAB)
}

impl MemoryOrder {
    /// Convert from C int to MemoryOrder
    pub fn from_c_int(order: libc::c_int) -> Result<Self, ()> {
        match order {
            SPIR_ORDER_ROW_MAJOR => Ok(Self::RowMajor),
            SPIR_ORDER_COLUMN_MAJOR => Ok(Self::ColumnMajor),
            _ => Err(()),
        }
    }
}

/// Calculate collapsed 3D shape from N-dimensional array
///
/// This follows libsparseir's collapse_to_3d function.
///
/// # Arguments
/// * `dims` - Original dimensions
/// * `target_dim` - The dimension to preserve
///
/// # Returns
/// (before_size, target_size, after_size) where:
/// - before_size = product of dims[0..target_dim]
/// - target_size = dims[target_dim]
/// - after_size = product of dims[target_dim+1..]
pub fn collapse_to_3d(dims: &[usize], target_dim: usize) -> (usize, usize, usize) {
    assert!(target_dim < dims.len(), "target_dim must be < ndim");
    
    let before: usize = if target_dim == 0 { 1 } else { dims[..target_dim].iter().product() };
    let target = dims[target_dim];
    let after: usize = if target_dim == dims.len() - 1 { 1 } else { dims[target_dim+1..].iter().product() };
    
    (before, target, after)
}

/// Copy N-dimensional tensor to C array (column-major layout)
///
/// Flattens the tensor and copies all elements to the output pointer.
/// This is a zero-copy operation for the reshape (metadata-only),
/// followed by a simple linear copy.
///
/// # Arguments
/// * `tensor` - Source tensor (any rank)
/// * `out` - Destination C array pointer
///
/// # Safety
/// Caller must ensure `out` has space for `tensor.len()` elements
pub unsafe fn copy_tensor_to_c_array<T: Copy>(
    tensor: sparseir_rust::Tensor<T, sparseir_rust::DynRank>,
    out: *mut T,
) {
    let total = tensor.len();
    let flat = tensor.into_dyn().reshape(&[total]).to_tensor();
    
    for i in 0..total {
        *out.add(i) = flat[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_collapse_to_3d() {
        // Shape: [2, 3, 4, 5], target_dim=2
        let dims = vec![2, 3, 4, 5];
        let (before, target, after) = collapse_to_3d(&dims, 2);
        
        assert_eq!(before, 2 * 3);  // 6
        assert_eq!(target, 4);
        assert_eq!(after, 5);
        
        // Edge cases
        let dims2 = vec![10, 20];
        assert_eq!(collapse_to_3d(&dims2, 0), (1, 10, 20));
        assert_eq!(collapse_to_3d(&dims2, 1), (10, 20, 1));
    }
    
    #[test]
    fn test_memory_order_conversion() {
        assert_eq!(MemoryOrder::from_c_int(SPIR_ORDER_ROW_MAJOR), Ok(MemoryOrder::RowMajor));
        assert_eq!(MemoryOrder::from_c_int(SPIR_ORDER_COLUMN_MAJOR), Ok(MemoryOrder::ColumnMajor));
        assert_eq!(MemoryOrder::from_c_int(99), Err(()));
    }
}

