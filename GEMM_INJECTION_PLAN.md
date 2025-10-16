# GEMM Function Pointer Injection Implementation Plan

**⚠️ IMPORTANT NOTE**: This is an **optional enhancement feature**, NOT part of the public libsparseir C-API (`sparseir.h`). The GEMM registration functions (`spir_register_dgemm`, etc.) exist only in libsparseir's internal implementation (`src/gemm.cpp`) and are not exposed in the public header.

For **100% C-API compatibility**, these functions are not required. See `CAPI_SPECIFICATION.md` for the complete public API.

## 概要

C-API経由で外部のBLAS/GEMM実装を注入できる機能を実装する。これにより、ユーザーは自分の好きなBLAS実装（ILP64、MKL、OpenBLAS、Accelerateなど）をランタイムに切り替えられる。

**Status**: Optional enhancement (not required for C-API compatibility)

## 設計原則

1. **ビルド時の依存関係を最小化**: メインライブラリはBLAS非依存（Pure Rust Faer）
2. **ランタイム柔軟性**: ユーザーがBLAS実装を自由に切り替え可能
3. **パフォーマンス**: デフォルトのFaerでも十分な性能、BLASはオプション
4. **安全性**: 関数ポインタの型安全性を保証
5. **互換性**: libsparseirのC-APIと完全互換

## アーキテクチャ

### 1. レイヤー構造

```
┌─────────────────────────────────────────────────┐
│ C-API (sparseir-capi)                          │
│ - spir_register_blas_functions()               │
│ - spir_register_ilp64_functions()              │
│ - spir_clear_blas_functions()                  │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│ BLAS Dispatcher (sparseir-rust/gemm.rs)        │
│ - GemmBackend trait                            │
│ - BLAS_DISPATCHER (global state)               │
│ - matmul_par() → dispatch to backend           │
└────────────────┬────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼─────┐          ┌────────▼────────┐
│ Faer    │          │ External BLAS   │
│ Backend │          │ Function Ptrs   │
│(default)│          │ (user-injected) │
└─────────┘          └─────────────────┘
```

### 2. 関数ポインタ型定義

```rust
// sparseir-rust/src/gemm.rs

/// BLAS GEMM function pointer type (LP64: 32-bit int)
pub type DgemmFnPtr = unsafe extern "C" fn(
    order: libc::c_int,       // CblasRowMajor / CblasColMajor
    transa: libc::c_int,      // CblasNoTrans / CblasTrans / CblasConjTrans
    transb: libc::c_int,
    m: libc::c_int,
    n: libc::c_int,
    k: libc::c_int,
    alpha: libc::c_double,
    a: *const libc::c_double,
    lda: libc::c_int,
    b: *const libc::c_double,
    ldb: libc::c_int,
    beta: libc::c_double,
    c: *mut libc::c_double,
    ldc: libc::c_int,
);

/// BLAS ZGEMM function pointer type (LP64: 32-bit int)
pub type ZgemmFnPtr = unsafe extern "C" fn(
    order: libc::c_int,
    transa: libc::c_int,
    transb: libc::c_int,
    m: libc::c_int,
    n: libc::c_int,
    k: libc::c_int,
    alpha: *const num_complex::Complex<f64>,
    a: *const num_complex::Complex<f64>,
    lda: libc::c_int,
    b: *const num_complex::Complex<f64>,
    ldb: libc::c_int,
    beta: *const num_complex::Complex<f64>,
    c: *mut num_complex::Complex<f64>,
    ldc: libc::c_int,
);

/// ILP64 BLAS GEMM function pointer type (ILP64: 64-bit int)
pub type Dgemm64FnPtr = unsafe extern "C" fn(
    order: libc::c_int,
    transa: libc::c_int,
    transb: libc::c_int,
    m: i64,
    n: i64,
    k: i64,
    alpha: libc::c_double,
    a: *const libc::c_double,
    lda: i64,
    b: *const libc::c_double,
    ldb: i64,
    beta: libc::c_double,
    c: *mut libc::c_double,
    ldc: i64,
);

/// ILP64 BLAS ZGEMM function pointer type (ILP64: 64-bit int)
pub type Zgemm64FnPtr = unsafe extern "C" fn(
    order: libc::c_int,
    transa: libc::c_int,
    transb: libc::c_int,
    m: i64,
    n: i64,
    k: i64,
    alpha: *const num_complex::Complex<f64>,
    a: *const num_complex::Complex<f64>,
    lda: i64,
    b: *const num_complex::Complex<f64>,
    ldb: i64,
    beta: *const num_complex::Complex<f64>,
    c: *mut num_complex::Complex<f64>,
    ldc: i64,
);
```

### 3. Backend Trait

```rust
// sparseir-rust/src/gemm.rs

/// GEMM backend trait for dispatch
pub trait GemmBackend: Send + Sync {
    /// Matrix multiplication: C = A * B (f64)
    fn dgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f64],
        b: &[f64],
        c: &mut [f64],
    );

    /// Matrix multiplication: C = A * B (Complex<f64>)
    fn zgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[num_complex::Complex<f64>],
        b: &[num_complex::Complex<f64>],
        c: &mut [num_complex::Complex<f64>],
    );

    /// Returns true if this backend uses 64-bit integers (ILP64)
    fn is_ilp64(&self) -> bool {
        false
    }
}

/// Default Faer backend (Pure Rust, no BLAS dependency)
pub struct FaerBackend;

impl GemmBackend for FaerBackend {
    fn dgemm(&self, m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        // Use mdarray_linalg_faer::Faer
        // Current implementation in matmul_par
    }

    fn zgemm(&self, m: usize, n: usize, k: usize, 
             a: &[num_complex::Complex<f64>], 
             b: &[num_complex::Complex<f64>], 
             c: &mut [num_complex::Complex<f64>]) {
        // Use mdarray_linalg_faer::Faer for complex
    }
}

/// External BLAS backend (user-injected function pointers)
pub struct ExternalBlasBackend {
    dgemm: DgemmFnPtr,
    zgemm: ZgemmFnPtr,
    ilp64: bool,
}

impl GemmBackend for ExternalBlasBackend {
    fn dgemm(&self, m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        // Call user-provided dgemm function pointer
        unsafe {
            (self.dgemm)(
                101, // CblasColMajor
                111, // CblasNoTrans
                111, // CblasNoTrans
                m as i32, n as i32, k as i32,
                1.0,
                a.as_ptr(), m as i32,
                b.as_ptr(), k as i32,
                0.0,
                c.as_mut_ptr(), m as i32,
            );
        }
    }

    fn zgemm(&self, m: usize, n: usize, k: usize, 
             a: &[num_complex::Complex<f64>], 
             b: &[num_complex::Complex<f64>], 
             c: &mut [num_complex::Complex<f64>]) {
        // Call user-provided zgemm function pointer
        unsafe {
            let alpha = num_complex::Complex::new(1.0, 0.0);
            let beta = num_complex::Complex::new(0.0, 0.0);
            (self.zgemm)(
                101, // CblasColMajor
                111, // CblasNoTrans
                111, // CblasNoTrans
                m as i32, n as i32, k as i32,
                &alpha,
                a.as_ptr(), m as i32,
                b.as_ptr(), k as i32,
                &beta,
                c.as_mut_ptr(), m as i32,
            );
        }
    }

    fn is_ilp64(&self) -> bool {
        self.ilp64
    }
}

/// ILP64 BLAS backend (64-bit integer variant)
pub struct ExternalBlas64Backend {
    dgemm64: Dgemm64FnPtr,
    zgemm64: Zgemm64FnPtr,
}

impl GemmBackend for ExternalBlas64Backend {
    fn dgemm(&self, m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        // Call ILP64 dgemm function pointer
        unsafe {
            (self.dgemm64)(
                101, // CblasColMajor
                111, // CblasNoTrans
                111, // CblasNoTrans
                m as i64, n as i64, k as i64,
                1.0,
                a.as_ptr(), m as i64,
                b.as_ptr(), k as i64,
                0.0,
                c.as_mut_ptr(), m as i64,
            );
        }
    }

    fn zgemm(&self, m: usize, n: usize, k: usize, 
             a: &[num_complex::Complex<f64>], 
             b: &[num_complex::Complex<f64>], 
             c: &mut [num_complex::Complex<f64>]) {
        // Call ILP64 zgemm function pointer
        unsafe {
            let alpha = num_complex::Complex::new(1.0, 0.0);
            let beta = num_complex::Complex::new(0.0, 0.0);
            (self.zgemm64)(
                101, // CblasColMajor
                111, // CblasNoTrans
                111, // CblasNoTrans
                m as i64, n as i64, k as i64,
                &alpha,
                a.as_ptr(), m as i64,
                b.as_ptr(), k as i64,
                &beta,
                c.as_mut_ptr(), m as i64,
            );
        }
    }

    fn is_ilp64(&self) -> bool {
        true
    }
}
```

### 4. Global Dispatcher

```rust
// sparseir-rust/src/gemm.rs

use std::sync::RwLock;
use once_cell::sync::Lazy;

/// Global BLAS dispatcher (thread-safe)
static BLAS_DISPATCHER: Lazy<RwLock<Box<dyn GemmBackend>>> = Lazy::new(|| {
    RwLock::new(Box::new(FaerBackend))
});

/// Set BLAS backend (LP64)
pub fn set_blas_backend(dgemm: DgemmFnPtr, zgemm: ZgemmFnPtr) {
    let backend = ExternalBlasBackend {
        dgemm,
        zgemm,
        ilp64: false,
    };
    let mut dispatcher = BLAS_DISPATCHER.write().unwrap();
    *dispatcher = Box::new(backend);
}

/// Set ILP64 BLAS backend
pub fn set_ilp64_backend(dgemm64: Dgemm64FnPtr, zgemm64: Zgemm64FnPtr) {
    let backend = ExternalBlas64Backend {
        dgemm64,
        zgemm64,
    };
    let mut dispatcher = BLAS_DISPATCHER.write().unwrap();
    *dispatcher = Box::new(backend);
}

/// Clear BLAS backend (reset to Faer)
pub fn clear_blas_backend() {
    let mut dispatcher = BLAS_DISPATCHER.write().unwrap();
    *dispatcher = Box::new(FaerBackend);
}

/// Get current BLAS backend information
pub fn get_backend_info() -> (bool, bool) {
    let dispatcher = BLAS_DISPATCHER.read().unwrap();
    let is_external = !matches!(&**dispatcher as &dyn std::any::Any, Some(&FaerBackend));
    let is_ilp64 = dispatcher.is_ilp64();
    (is_external, is_ilp64)
}
```

### 5. Updated matmul_par

```rust
// sparseir-rust/src/gemm.rs

/// Parallel matrix multiplication: C = A * B
///
/// Dispatches to registered BLAS backend (external or Faer).
pub fn matmul_par<T>(a: &DTensor<T, 2>, b: &DTensor<T, 2>) -> DTensor<T, 2>
where
    T: num_complex::ComplexFloat + faer_traits::ComplexField + num_traits::One + Copy + 'static,
{
    let dispatcher = BLAS_DISPATCHER.read().unwrap();
    
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    
    // Validate dimensions
    assert_eq!(k, b.shape()[0], "Matrix dimension mismatch: A.cols ({}) != B.rows ({})", k, b.shape()[0]);
    
    // Type dispatch: f64 or Complex<f64>
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        // f64 case
        let a_slice = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f64, m * k) };
        let b_slice = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f64, k * n) };
        
        let mut c_vec = vec![0.0f64; m * n];
        dispatcher.dgemm(m, n, k, a_slice, b_slice, &mut c_vec);
        
        // Convert back to DTensor
        unsafe {
            let c_ptr = c_vec.as_ptr() as *const T;
            let c_slice = std::slice::from_raw_parts(c_ptr, m * n);
            DTensor::from_shape_vec([m, n], c_slice.to_vec()).unwrap()
        }
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<num_complex::Complex<f64>>() {
        // Complex<f64> case
        let a_slice = unsafe { 
            std::slice::from_raw_parts(a.as_ptr() as *const num_complex::Complex<f64>, m * k) 
        };
        let b_slice = unsafe { 
            std::slice::from_raw_parts(b.as_ptr() as *const num_complex::Complex<f64>, k * n) 
        };
        
        let mut c_vec = vec![num_complex::Complex::new(0.0, 0.0); m * n];
        dispatcher.zgemm(m, n, k, a_slice, b_slice, &mut c_vec);
        
        // Convert back to DTensor
        unsafe {
            let c_ptr = c_vec.as_ptr() as *const T;
            let c_slice = std::slice::from_raw_parts(c_ptr, m * n);
            DTensor::from_shape_vec([m, n], c_slice.to_vec()).unwrap()
        }
    } else {
        panic!("Unsupported type for matmul_par");
    }
}
```

### 6. C-API Functions

```rust
// sparseir-capi/src/gemm.rs (新規ファイル)

use crate::StatusCode;
use crate::{SPIR_SUCCESS, SPIR_INVALID_ARGUMENT};

/// Register custom BLAS functions (LP64: 32-bit integers)
///
/// # Arguments
/// * `cblas_dgemm` - Function pointer to CBLAS dgemm
/// * `cblas_zgemm` - Function pointer to CBLAS zgemm
///
/// # Returns
/// * `SPIR_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if function pointers are null
///
/// # Safety
/// The provided function pointers must be valid and thread-safe.
/// They will be called from multiple threads.
///
/// # Example (from C)
/// ```c
/// #include <cblas.h>
/// spir_register_blas_functions(
///     (void*)cblas_dgemm,
///     (void*)cblas_zgemm
/// );
/// ```
#[no_mangle]
pub unsafe extern "C" fn spir_register_blas_functions(
    cblas_dgemm: *const libc::c_void,
    cblas_zgemm: *const libc::c_void,
) -> StatusCode {
    if cblas_dgemm.is_null() || cblas_zgemm.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let dgemm_fn: sparseir_rust::gemm::DgemmFnPtr = std::mem::transmute(cblas_dgemm);
    let zgemm_fn: sparseir_rust::gemm::ZgemmFnPtr = std::mem::transmute(cblas_zgemm);

    sparseir_rust::gemm::set_blas_backend(dgemm_fn, zgemm_fn);
    SPIR_SUCCESS
}

/// Register ILP64 BLAS functions (64-bit integers)
///
/// # Arguments
/// * `cblas_dgemm64` - Function pointer to ILP64 CBLAS dgemm
/// * `cblas_zgemm64` - Function pointer to ILP64 CBLAS zgemm
///
/// # Returns
/// * `SPIR_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if function pointers are null
///
/// # Safety
/// The provided function pointers must be valid, use 64-bit integers,
/// and be thread-safe.
///
/// # Example (from C)
/// ```c
/// // MKL ILP64 example
/// #include <mkl.h>
/// #define MKL_INT long long
/// spir_register_ilp64_functions(
///     (void*)cblas_dgemm,  // MKL's ILP64 version
///     (void*)cblas_zgemm   // MKL's ILP64 version
/// );
/// ```
#[no_mangle]
pub unsafe extern "C" fn spir_register_ilp64_functions(
    cblas_dgemm64: *const libc::c_void,
    cblas_zgemm64: *const libc::c_void,
) -> StatusCode {
    if cblas_dgemm64.is_null() || cblas_zgemm64.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let dgemm64_fn: sparseir_rust::gemm::Dgemm64FnPtr = std::mem::transmute(cblas_dgemm64);
    let zgemm64_fn: sparseir_rust::gemm::Zgemm64FnPtr = std::mem::transmute(cblas_zgemm64);

    sparseir_rust::gemm::set_ilp64_backend(dgemm64_fn, zgemm64_fn);
    SPIR_SUCCESS
}

/// Clear registered BLAS functions (reset to default Faer backend)
///
/// # Returns
/// * `SPIR_SUCCESS` always
///
/// # Example (from C)
/// ```c
/// spir_clear_blas_functions();
/// ```
#[no_mangle]
pub unsafe extern "C" fn spir_clear_blas_functions() -> StatusCode {
    sparseir_rust::gemm::clear_blas_backend();
    SPIR_SUCCESS
}

/// Get current BLAS backend information
///
/// # Arguments
/// * `is_external` - Output: 1 if external BLAS is registered, 0 if using default Faer
/// * `is_ilp64` - Output: 1 if ILP64 backend, 0 if LP64 or Faer
///
/// # Returns
/// * `SPIR_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if output pointers are null
///
/// # Example (from C)
/// ```c
/// int is_external, is_ilp64;
/// spir_get_blas_backend_info(&is_external, &is_ilp64);
/// printf("External: %d, ILP64: %d\n", is_external, is_ilp64);
/// ```
#[no_mangle]
pub unsafe extern "C" fn spir_get_blas_backend_info(
    is_external: *mut libc::c_int,
    is_ilp64: *mut libc::c_int,
) -> StatusCode {
    if is_external.is_null() || is_ilp64.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    let (ext, ilp) = sparseir_rust::gemm::get_backend_info();
    *is_external = if ext { 1 } else { 0 };
    *is_ilp64 = if ilp { 1 } else { 0 };

    SPIR_SUCCESS
}
```

### 7. C Header Additions

```c
// libsparseir/include/sparseir/gemm.h (新規ファイル)

#ifndef SPARSEIR_GEMM_H
#define SPARSEIR_GEMM_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Register custom BLAS functions (LP64: 32-bit integers)
 * 
 * @param cblas_dgemm Function pointer to CBLAS dgemm
 * @param cblas_zgemm Function pointer to CBLAS zgemm
 * @return SPIR_SUCCESS on success, SPIR_INVALID_ARGUMENT if pointers are null
 * 
 * @note The provided function pointers must be thread-safe
 * 
 * Example:
 * @code
 * #include <cblas.h>
 * spir_register_blas_functions((void*)cblas_dgemm, (void*)cblas_zgemm);
 * @endcode
 */
int spir_register_blas_functions(const void* cblas_dgemm, const void* cblas_zgemm);

/**
 * @brief Register ILP64 BLAS functions (64-bit integers)
 * 
 * @param cblas_dgemm64 Function pointer to ILP64 CBLAS dgemm
 * @param cblas_zgemm64 Function pointer to ILP64 CBLAS zgemm
 * @return SPIR_SUCCESS on success, SPIR_INVALID_ARGUMENT if pointers are null
 * 
 * @note The provided function pointers must use 64-bit integers and be thread-safe
 * 
 * Example (MKL ILP64):
 * @code
 * #include <mkl.h>
 * spir_register_ilp64_functions((void*)cblas_dgemm, (void*)cblas_zgemm);
 * @endcode
 */
int spir_register_ilp64_functions(const void* cblas_dgemm64, const void* cblas_zgemm64);

/**
 * @brief Clear registered BLAS functions (reset to default Faer backend)
 * 
 * @return SPIR_SUCCESS always
 * 
 * Example:
 * @code
 * spir_clear_blas_functions();
 * @endcode
 */
int spir_clear_blas_functions(void);

/**
 * @brief Get current BLAS backend information
 * 
 * @param is_external Output: 1 if external BLAS is registered, 0 if using default Faer
 * @param is_ilp64 Output: 1 if ILP64 backend, 0 if LP64 or Faer
 * @return SPIR_SUCCESS on success, SPIR_INVALID_ARGUMENT if output pointers are null
 * 
 * Example:
 * @code
 * int is_external, is_ilp64;
 * spir_get_blas_backend_info(&is_external, &is_ilp64);
 * printf("External: %d, ILP64: %d\n", is_external, is_ilp64);
 * @endcode
 */
int spir_get_blas_backend_info(int* is_external, int* is_ilp64);

#ifdef __cplusplus
}
#endif

#endif // SPARSEIR_GEMM_H
```

## 実装ステップ

### Phase 1: Core Infrastructure (sparseir-rust)
1. ✅ `gemm.rs`に関数ポインタ型定義を追加
2. ✅ `GemmBackend` traitを定義
3. ✅ `FaerBackend`実装（デフォルト）
4. ✅ `ExternalBlasBackend`実装（LP64）
5. ✅ `ExternalBlas64Backend`実装（ILP64）
6. ✅ グローバルディスパッチャ (`BLAS_DISPATCHER`)
7. ✅ `set_blas_backend`, `set_ilp64_backend`, `clear_blas_backend`関数
8. ✅ `matmul_par`を更新してディスパッチャ経由で呼ぶ
9. ✅ Cargo.tomlに`once_cell`依存を追加

### Phase 2: C-API Layer (sparseir-capi)
1. ✅ `src/gemm.rs`を新規作成
2. ✅ `spir_register_blas_functions`実装
3. ✅ `spir_register_ilp64_functions`実装
4. ✅ `spir_clear_blas_functions`実装
5. ✅ `spir_get_blas_backend_info`実装
6. ✅ `lib.rs`に`mod gemm;`と`pub use gemm::*;`を追加

### Phase 3: C Header (libsparseir)
1. ✅ `include/sparseir/gemm.h`を新規作成
2. ✅ `include/sparseir/sparseir.h`に`#include "gemm.h"`を追加
3. ✅ ドキュメント追加

### Phase 4: Testing
1. ⏳ Rustユニットテスト（`sparseir-rust/src/gemm.rs`内）
   - デフォルトFaerバックエンドのテスト
   - 外部BLAS登録とディスパッチのテスト
   - ILP64バックエンドのテスト
   - スレッド安全性のテスト
2. ⏳ C-API統合テスト（`sparseir-capi/tests/gemm_tests.rs`）
   - Mock BLAS関数での登録テスト
   - クリア機能のテスト
   - バックエンド情報取得のテスト
3. ⏳ C++統合テスト（`libsparseir/test/cpp/gemm_injection_test.cxx`）
   - OpenBLAS登録テスト
   - MKL ILP64登録テスト（オプション）
   - Accelerate登録テスト（macOSのみ）

### Phase 5: Documentation
1. ⏳ API docstrings（Rust側）
2. ⏳ C headerコメント
3. ⏳ 使用例（C, C++, Julia, Python）
4. ⏳ GEMM_INJECTION_USAGE.mdガイド

## 使用例

### C++からの使用

```cpp
#include <sparseir/sparseir.h>
#include <cblas.h>

int main() {
    // Register OpenBLAS
    spir_register_blas_functions(
        (void*)cblas_dgemm,
        (void*)cblas_zgemm
    );
    
    // Use SparseIR (now using OpenBLAS for GEMM)
    spir_kernel* kernel = spir_logistic_kernel_new(1000.0);
    spir_basis* basis = spir_basis_new(kernel, 1.0, 1e-10);
    
    // ... perform computations ...
    
    // Check backend info
    int is_external, is_ilp64;
    spir_get_blas_backend_info(&is_external, &is_ilp64);
    printf("External BLAS: %d, ILP64: %d\n", is_external, is_ilp64);
    
    // Clear and return to Faer
    spir_clear_blas_functions();
    
    spir_basis_free(basis);
    spir_kernel_free(kernel);
    
    return 0;
}
```

### Juliaからの使用

```julia
using SparseIR_jll  # Assuming Julia bindings

# Register system BLAS (OpenBLAS or MKL)
import LinearAlgebra.BLAS
dgemm_ptr = @cfunction(BLAS.dgemm!, Nothing, ...)
zgemm_ptr = @cfunction(BLAS.zgemm!, Nothing, ...)

spir_register_blas_functions(dgemm_ptr, zgemm_ptr)

# Use SparseIR
basis = FiniteTempBasis(β=1.0, ωmax=1000.0, ε=1e-10)

# ... computations ...

# Clear when done
spir_clear_blas_functions()
```

## パフォーマンス考慮事項

### メモリレイアウト
- mdarray/ndarrayは**Row-major**（C-order）
- BLASは**Column-major**（Fortran-order）
- **注意**: トランスポーズフラグや順序変換が必要

### オーバーヘッド
- 関数ポインタディスパッチ: ~1ns（negligible）
- RwLockオーバーヘッド: ~10ns（read lock、ほぼ無視可能）
- 型チェック（TypeId）: compile-timeで最適化される

### ベンチマーク目標
- Faer vs OpenBLAS vs MKL比較
- 小行列（<100x100）: Faerが有利な可能性
- 大行列（>1000x1000）: 最適化されたBLASが有利

## 安全性保証

### メモリ安全性
- 関数ポインタは`unsafe`ブロック内でのみ呼び出し
- バッファサイズを事前検証
- Null pointer checks

### スレッド安全性
- `BLAS_DISPATCHER`は`RwLock`で保護
- 複数スレッドからの読み取り並行可能
- 書き込み（登録/クリア）は排他ロック

### 型安全性
- 関数ポインタの型を厳密に定義
- `transmute`は`unsafe`関数内でのみ使用
- 型チェックでパニック回避

## 今後の拡張可能性

### SGEMM/CGEMM対応（f32/Complex<f32>）
現在の設計はf64に特化しているが、同様の手法でf32対応も可能。

### 他のLAPACK関数
GEMM以外のLAPACK関数（GESV, GESVD等）も同様の仕組みで注入可能。

### 動的ライブラリローディング
`libloading` crateを使えば、dlopen/dlsym経由でBLASを動的ロード可能。

### Accelerate Framework（macOS）特化
macOSのAccelerateは最適化が進んでいるため、専用バックエンドを作る価値がある。

## チェックリスト

### Phase 1: Core (sparseir-rust)
- [ ] 関数ポインタ型定義
- [ ] `GemmBackend` trait
- [ ] `FaerBackend`実装
- [ ] `ExternalBlasBackend`実装
- [ ] `ExternalBlas64Backend`実装
- [ ] グローバルディスパッチャ
- [ ] 登録/クリア関数
- [ ] `matmul_par`更新
- [ ] 依存関係追加（`once_cell`）

### Phase 2: C-API (sparseir-capi)
- [ ] `src/gemm.rs`作成
- [ ] `spir_register_blas_functions`
- [ ] `spir_register_ilp64_functions`
- [ ] `spir_clear_blas_functions`
- [ ] `spir_get_blas_backend_info`
- [ ] `lib.rs`に統合

### Phase 3: C Header (libsparseir)
- [ ] `include/sparseir/gemm.h`作成
- [ ] `sparseir.h`に追加
- [ ] ドキュメント

### Phase 4: Testing
- [ ] Rustユニットテスト
- [ ] C-API統合テスト
- [ ] C++統合テスト
- [ ] ベンチマーク

### Phase 5: Documentation
- [ ] Rustdoc
- [ ] C header docs
- [ ] 使用例
- [ ] ユーザーガイド

## まとめ

この設計により：
1. ✅ **ビルド時BLAS不要**: デフォルトはPure Rust（Faer）
2. ✅ **ランタイム柔軟性**: ユーザーが好きなBLASを注入可能
3. ✅ **ILP64対応**: 大規模行列計算に対応
4. ✅ **型安全性**: Rust型システムで保護
5. ✅ **スレッド安全性**: RwLockで並行アクセス可能
6. ✅ **パフォーマンス**: ディスパッチオーバーヘッドは無視可能
7. ✅ **互換性**: libsparseirのC-APIと完全互換

実装優先度: **Phase 1 → Phase 2 → Phase 4（基本テスト）**

