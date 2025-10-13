# C-API Implementation Strategy

**Created**: October 13, 2025  
**Status**: Planning Phase  
**Goal**: Implement Rust-based C-API compatible with libsparseir C interface

---

## Overview

The C-API provides FFI (Foreign Function Interface) bindings for Python, Fortran, and other languages. All Rust core functionality is complete. This document outlines the strategy for wrapping it with a C-compatible interface.

---

## Opaque Types (5 types)

All opaque types use the `DECLARE_OPAQUE_TYPE` macro which generates:
- `_release()`: Destructor
- `_clone()`: Copy constructor
- `_is_assigned()`: Validity check
- `_get_raw_ptr()`: Debug accessor

### Required Opaque Types
1. **`spir_kernel`** - Kernel objects (LogisticKernel, RegularizedBoseKernel)
2. **`spir_sve_result`** - SVE computation results
3. **`spir_basis`** - IR basis objects (FiniteTempBasis)
4. **`spir_funcs`** - Basis function objects (PiecewiseLegendrePolyVector)
5. **`spir_sampling`** - Sampling objects (TauSampling, MatsubaraSampling)

### Additional Type (not in DECLARE_OPAQUE_TYPE)
6. **`spir_dlr`** - DLR representation (DiscreteLehmannRepresentation)

---

## Implementation Approach

### Phase 1: Project Setup (Priority 1) 🔴
1. **Create `sparseir-capi` crate**
   ```
   sparseir-rust/
   └── sparseir-capi/
       ├── Cargo.toml
       ├── build.rs (optional - for generating bindings)
       └── src/
           ├── lib.rs       # Main entry point
           ├── types.rs     # Opaque type definitions
           ├── kernel.rs    # Kernel API (~8 functions)
           ├── sve.rs       # SVE API (~10 functions)
           ├── basis.rs     # Basis API (~15 functions)
           ├── sampling.rs  # Sampling API (~15 functions)
           ├── dlr.rs       # DLR API (~8 functions)
           └── utils.rs     # Common utilities
   ```

2. **Cargo.toml dependencies**
   ```toml
   [dependencies]
   sparseir-rust = { path = "../sparseir-rust" }
   libc = "0.2"
   num-complex = "0.4"
   
   [lib]
   crate-type = ["cdylib", "staticlib"]
   ```

### Phase 2: Opaque Type Infrastructure (Priority 2)

**Strategy**: Use `Box` for ownership, `Arc` for shared ownership

```rust
// types.rs

use std::sync::Arc;
use sparseir_rust::kernel::{LogisticKernel, RegularizedBoseKernel};

/// Opaque kernel type - can hold different kernel types
pub enum KernelType {
    Logistic(LogisticKernel),
    RegularizedBose(RegularizedBoseKernel),
}

#[repr(C)]
pub struct spir_kernel {
    inner: Arc<KernelType>,
}

// Similar patterns for other opaque types
```

### Phase 3: Function Categories (Priority by dependency)

#### Category 1: Kernel API (Foundation) 🔴
**Priority**: Highest (no dependencies)
- `spir_logistic_kernel_new(lambda, *status)` → `LogisticKernel::new()`
- `spir_reg_bose_kernel_new(lambda, *status)` → `RegularizedBoseKernel::new()`
- `spir_kernel_domain(*k, *xmin, *xmax, *ymin, *ymax)` → `kernel.xmax()`, etc.
- `spir_kernel_release(*k)` → Drop
- `spir_kernel_clone(*k)` → Clone
- `spir_kernel_is_assigned(*k)` → null check

**Estimated**: 8 functions, ~200 lines

#### Category 2: SVE API (Depends on Kernel) 🟠
**Priority**: High
- `spir_sve_result_new(*k, epsilon, cutoff, lmax, n_gauss, Twork, *status)`
- `spir_sve_result_get_size(*sve, *size)`
- `spir_sve_result_get_svals(*sve, *s)`
- `spir_sve_result_get_u(*sve)` → Returns `spir_funcs`
- `spir_sve_result_get_v(*sve)` → Returns `spir_funcs`

**Estimated**: 10 functions, ~300 lines

#### Category 3: Basis API (Depends on SVE) 🟠
**Priority**: High
- `spir_basis_new(*k, beta, epsilon, *status)`
- `spir_basis_get_beta(*b, *beta)`
- `spir_basis_get_wmax(*b, *wmax)`
- `spir_basis_get_size(*b, *size)`
- `spir_basis_get_u(*b)` → Returns `spir_funcs`
- `spir_basis_get_v(*b)` → Returns `spir_funcs`
- `spir_basis_get_uhat(*b)` → Returns `spir_funcs`
- `spir_basis_get_default_taus(*b, n_points, *points)`
- `spir_basis_get_default_matsus(*b, positive_only, *n_freqs, *freqs)`

**Estimated**: 15 functions, ~400 lines

#### Category 4: Funcs API (For evaluating polynomials) 🟡
**Priority**: Medium
- `spir_funcs_eval(*f, *x, n_x, *result)` → Batch evaluation
- `spir_funcs_eval_matsu(*f, *freqs, n_freqs, *result)` → Matsubara eval
- `spir_funcs_get_size(*f, *size)`
- `spir_funcs_get_roots(*f, *roots)`

**Estimated**: 8 functions, ~250 lines

#### Category 5: Sampling API (Depends on Basis) 🟡
**Priority**: Medium
- `spir_tau_sampling_new(*b, *status)`
- `spir_matsu_sampling_new(*b, positive_only, *status)`
- `spir_sampling_eval_dd(*s, *coeffs, shape, ndim, order, *result, *status)`
- `spir_sampling_fit_dd(*s, *values, shape, ndim, order, *result, *status)`
- `spir_sampling_get_n_points(*s, *n)`
- `spir_sampling_get_cond_num(*s, *cond)`

**Estimated**: 15 functions, ~500 lines

#### Category 6: DLR API (Depends on Basis) 🟢
**Priority**: Low (but complete in Rust)
- `spir_dlr_new(*b, *status)`
- `spir_dlr_from_ir_dd(*dlr, *gl, shape, ndim, dim, order, *result, *status)`
- `spir_dlr_to_ir_dd(*dlr, *g_dlr, shape, ndim, dim, order, *result, *status)`
- `spir_dlr_get_poles(*dlr, *poles)`

**Estimated**: 8 functions, ~300 lines

---

## Implementation Strategy

### Incremental Approach (Recommended)

**Week 1**: Kernel + SVE API
1. Setup `sparseir-capi` crate structure
2. Implement opaque type infrastructure
3. Implement Kernel API (8 functions)
4. Implement SVE API (10 functions)
5. Basic testing with simple C program

**Week 2**: Basis + Funcs API
1. Implement Basis API (15 functions)
2. Implement Funcs API (8 functions)
3. Test with Python ctypes

**Week 3**: Sampling + DLR API
1. Implement Sampling API (15 functions)
2. Implement DLR API (8 functions)
3. Integration testing

**Week 4**: Polish and Validation
1. Performance benchmarking
2. Memory leak testing (valgrind)
3. Python wrapper integration
4. Fortran wrapper integration

---

## Technical Considerations

### 1. Memory Management Pattern

```rust
// Ownership: Box for single ownership
#[no_mangle]
pub extern "C" fn spir_kernel_release(ptr: *mut spir_kernel) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}

// Sharing: Arc for shared ownership
#[no_mangle]
pub extern "C" fn spir_kernel_clone(ptr: *const spir_kernel) -> *mut spir_kernel {
    if ptr.is_null() { return std::ptr::null_mut(); }
    unsafe {
        let kernel = &*ptr;
        Box::into_raw(Box::new(spir_kernel {
            inner: Arc::clone(&kernel.inner),
        }))
    }
}
```

### 2. Error Handling Pattern

```rust
fn set_status(status_ptr: *mut i32, code: i32) {
    if !status_ptr.is_null() {
        unsafe { *status_ptr = code; }
    }
}

#[no_mangle]
pub extern "C" fn spir_logistic_kernel_new(
    lambda: f64,
    status: *mut i32,
) -> *mut spir_kernel {
    if lambda < 0.0 {
        set_status(status, SPIR_INVALID_ARGUMENT);
        return std::ptr::null_mut();
    }
    
    let kernel = LogisticKernel::new(lambda);
    set_status(status, SPIR_COMPUTATION_SUCCESS);
    Box::into_raw(Box::new(spir_kernel {
        inner: Arc::new(KernelType::Logistic(kernel)),
    }))
}
```

### 3. Array Passing Pattern

```rust
// Multi-dimensional arrays: use shape + stride info
#[no_mangle]
pub extern "C" fn spir_sampling_eval_dd(
    sampling: *const spir_sampling,
    coeffs: *const f64,
    shape: *const usize,
    ndim: usize,
    order: i32,  // SPIR_ORDER_COLUMN_MAJOR or ROW_MAJOR
    result: *mut f64,
    status: *mut i32,
) -> i32 {
    // 1. Null checks
    // 2. Convert C array to Rust Tensor
    // 3. Call Rust implementation
    // 4. Copy result back to C array
}
```

### 4. Statistics Type Mapping

```rust
const SPIR_STATISTICS_FERMIONIC: i32 = 1;
const SPIR_STATISTICS_BOSONIC: i32 = 0;

fn statistics_from_c(stats: i32) -> Result<Statistics, ()> {
    match stats {
        SPIR_STATISTICS_FERMIONIC => Ok(Statistics::Fermionic),
        SPIR_STATISTICS_BOSONIC => Ok(Statistics::Bosonic),
        _ => Err(()),
    }
}
```

---

## Testing Strategy

### Unit Tests
- Test each FFI function in isolation
- Verify memory management (no leaks)
- Test error handling

### Integration Tests
- Create C test programs
- Python ctypes tests
- Fortran tests (if available)

### Compatibility Tests
- Compare results with C++ libsparseir
- Numerical precision validation
- Performance benchmarking

---

## Success Criteria

1. ✅ **API Compatibility**: All C-API functions implemented
2. ✅ **Memory Safety**: Zero memory leaks (valgrind clean)
3. ✅ **Numerical Accuracy**: Results match C++ implementation
4. ✅ **Performance**: Within 10% of C++ performance
5. ✅ **Integration**: Python/Fortran wrappers work without modification

---

## Estimated Effort

| Category | Functions | Lines of Code | Time Estimate |
|----------|-----------|---------------|---------------|
| Setup | - | 100 | 0.5 day |
| Kernel API | 8 | 200 | 0.5 day |
| SVE API | 10 | 300 | 1 day |
| Basis API | 15 | 400 | 1.5 days |
| Funcs API | 8 | 250 | 1 day |
| Sampling API | 15 | 500 | 2 days |
| DLR API | 8 | 300 | 1 day |
| Testing | - | 500 | 2 days |
| **Total** | **64** | **~2,550** | **~10 days** |

---

## Next Immediate Steps

1. ✅ Create `sparseir-capi/` directory
2. ✅ Set up `Cargo.toml` with FFI configuration
3. ✅ Implement opaque type infrastructure in `types.rs`
4. ✅ Implement Kernel API as proof of concept
5. ✅ Write simple C test program to verify
6. → Iterate through remaining categories

---

## Reference

- **C-API Header**: `libsparseir/include/sparseir/sparseir.h`
- **C++ Implementation**: `libsparseir/src/cinterface.cpp`
- **Python Wrapper**: `libsparseir/python/pylibsparseir/core.py`
- **Fortran Wrapper**: `libsparseir/fortran/sparseir.f90`

