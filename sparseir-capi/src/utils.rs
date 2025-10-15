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
    fn test_memory_order_conversion() {
        assert_eq!(MemoryOrder::from_c_int(SPIR_ORDER_ROW_MAJOR), Ok(MemoryOrder::RowMajor));
        assert_eq!(MemoryOrder::from_c_int(SPIR_ORDER_COLUMN_MAJOR), Ok(MemoryOrder::ColumnMajor));
        assert_eq!(MemoryOrder::from_c_int(99), Err(()));
    }
}

