//! Utility functions for C API
//!
//! This module provides helper functions for order conversion and dimension handling.

use crate::{SPIR_ORDER_COLUMN_MAJOR, SPIR_ORDER_ROW_MAJOR};

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

/// Convert dimensions and target_dim for row-major mdarray
///
/// mdarray uses row-major (C order) by default. When the C-API caller
/// specifies column-major (Fortran/Julia order), we need to reverse
/// dimensions and adjust target_dim to match mdarray's row-major layout.
///
/// This follows libsparseir's pattern for order handling.
///
/// # Arguments
/// * `dims` - Original dimensions from C-API
/// * `target_dim` - Original target dimension from C-API
/// * `order` - Memory order specified by caller
///
/// # Returns
/// (mdarray_dims, mdarray_target_dim) - Dimensions and target_dim for row-major mdarray
///
/// # Example
/// ```ignore
/// // Julia: dims=[5, 3], target_dim=0, order=COLUMN_MAJOR
/// convert_dims_for_row_major(&[5, 3], 0, MemoryOrder::ColumnMajor)
/// → ([3, 5], 1)  // For row-major mdarray
/// ```
pub fn convert_dims_for_row_major(
    dims: &[usize],
    target_dim: usize,
    order: MemoryOrder,
) -> (Vec<usize>, usize) {
    match order {
        MemoryOrder::RowMajor => {
            // Already row-major, use as-is
            (dims.to_vec(), target_dim)
        }
        MemoryOrder::ColumnMajor => {
            // Convert column-major to row-major:
            // Reverse dims and flip target_dim
            let mut rev_dims = dims.to_vec();
            rev_dims.reverse();
            let rev_target_dim = dims.len() - 1 - target_dim;
            (rev_dims, rev_target_dim)
        }
    }
}

/// Copy N-dimensional tensor to C array
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
    fn test_memory_order_conversion() {
        assert_eq!(
            MemoryOrder::from_c_int(SPIR_ORDER_ROW_MAJOR),
            Ok(MemoryOrder::RowMajor)
        );
        assert_eq!(
            MemoryOrder::from_c_int(SPIR_ORDER_COLUMN_MAJOR),
            Ok(MemoryOrder::ColumnMajor)
        );
        assert_eq!(MemoryOrder::from_c_int(99), Err(()));
    }
}
