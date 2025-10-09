# GEMM モジュール最終設計

## 基本方針

1. **MatMulのみ使用** - matvecは不要（全てmat-matに統一）
2. **sparseir-rust側でラッパー** - 将来のBLAS委譲に備える
3. **Faer並列化を活用** - `.parallelize()`メソッド

## アーキテクチャ

```
┌─────────────────────────────────────┐
│  sampling.rs (TauSampling)          │
│  - evaluate() → gemm::matvec()      │
│  - fit() → gemm::solve_via_svd()    │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  gemm.rs (薄いラッパー層)            │
│  - matmul() → Faer.matmul()         │
│  - matvec() → reshape + matmul      │
│  - solve_via_svd()                  │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  mdarray-linalg-faer                │
│  - Faer.matmul()                    │
│  - .parallelize() オプション         │
└─────────────────────────────────────┘
```

## gemm.rs 実装

```rust
// src/gemm.rs

//! Matrix multiplication utilities
//!
//! Thin wrapper over mdarray-linalg-faer to enable future
//! delegation to BLAS Fortran pointers if needed.

use mdarray::Tensor;
use mdarray_linalg::{MatMul, MatMulBuilder};
use mdarray_linalg_faer::Faer;

// ===== Core Operations =====

/// Matrix-Matrix multiplication: C = A @ B
/// 
/// Uses Faer backend by default. Can be replaced with BLAS
/// Fortran pointer calls in the future.
pub fn matmul(
    a: &Tensor<f64, (usize, usize)>,
    b: &Tensor<f64, (usize, usize)>,
) -> Tensor<f64, (usize, usize)> {
    Faer.matmul(a, b).eval()
}

/// Matrix-Matrix multiplication with parallelization
pub fn matmul_par(
    a: &Tensor<f64, (usize, usize)>,
    b: &Tensor<f64, (usize, usize)>,
) -> Tensor<f64, (usize, usize)> {
    Faer.matmul(a, b).parallelize().eval()
}

/// Matrix transpose
pub fn transpose(
    a: &Tensor<f64, (usize, usize)>
) -> Tensor<f64, (usize, usize)> {
    let (m, n) = *a.shape();
    Tensor::from_fn((n, m), |idx| a[[idx[1], idx[0]]])
}

// ===== Vector Operations (via reshape) =====

/// Matrix-Vector multiplication: y = A @ x
/// 
/// Implemented as reshape + matmul for simplicity.
/// A: (m, k), x: (k,) → y: (m,)
pub fn matvec(
    a: &Tensor<f64, (usize, usize)>,
    x: &Tensor<f64, (usize,)>,
) -> Tensor<f64, (usize,)> {
    let (m, k) = *a.shape();
    assert_eq!(k, x.len(), "Dimension mismatch: A cols={}, x len={}", k, x.len());
    
    // Reshape x: (k,) → (k, 1)
    let x_2d = Tensor::from_fn((k, 1), |idx| x[[idx[0]]]);
    
    // C = A @ x_2d: (m, k) @ (k, 1) → (m, 1)
    let result_2d = matmul(a, &x_2d);
    
    // Reshape back: (m, 1) → (m,)
    Tensor::from_fn((m,), |idx| result_2d[[idx[0], 0]])
}

/// Matrix^T @ Vector: y = A^T @ x
/// A: (m, k), x: (m,) → y: (k,)
pub fn matmul_t_vec(
    a: &Tensor<f64, (usize, usize)>,
    x: &Tensor<f64, (usize,)>,
) -> Tensor<f64, (usize,)> {
    let a_t = transpose(a);
    matvec(&a_t, x)
}

// ===== SVD-based Linear Solver =====

/// Solve A @ x = b using SVD pseudoinverse
/// 
/// Given SVD: A = U @ diag(s) @ V^T
/// Solution: x = V @ diag(1/s) @ U^T @ b
pub fn solve_via_svd(
    u: &Tensor<f64, (usize, usize)>,
    s: &Tensor<f64, (usize,)>,
    v: &Tensor<f64, (usize, usize)>,
    b: &Tensor<f64, (usize,)>,
) -> Tensor<f64, (usize,)> {
    // Step 1: temp1 = U^T @ b
    let temp1 = matmul_t_vec(u, b);
    
    // Step 2: temp2 = diag(1/s) @ temp1
    let temp2 = Tensor::from_fn((s.len(),), |idx| {
        temp1[[idx[0]]] / s[[idx[0]]]
    });
    
    // Step 3: x = V @ temp2
    matvec(v, &temp2)
}
```

## Faerの並列化制御

### MatMulBuilderインターフェース
```rust
Faer.matmul(a, b)
    .parallelize()    // 並列化を有効化
    .scale(2.0)       // 結果を2倍（オプション）
    .eval()           // 実行
```

### 並列数の制御

#### Option 1: Rayonのグローバル設定
```rust
// main関数またはテスト開始時
rayon::ThreadPoolBuilder::new()
    .num_threads(4)
    .build_global()
    .unwrap();
```

#### Option 2: 環境変数
```bash
# 実行時に指定
RAYON_NUM_THREADS=4 cargo test
RAYON_NUM_THREADS=8 cargo run --release
```

#### Option 3: Faer独自設定（あれば）
```rust
// Faerのドキュメント要確認
// pulp crateを使っているかも
```

### sparseir-rustでの使用例

```rust
// デフォルト（自動並列化）
let c = gemm::matmul(&a, &b);

// 明示的な並列化（大きい行列用）
let c = gemm::matmul_par(&a, &b);
```

## 実装サイズ見積もり

```rust
// src/gemm.rs (~80行)
- matmul()         5行
- matmul_par()     5行
- transpose()      5行
- matvec()         10行
- matmul_t_vec()   5行
- solve_via_svd()  15行
- tests            35行
```

## sampling.rsでの使用

```rust
use crate::gemm;

impl TauSampling {
    pub fn evaluate(&self, coeffs: &Tensor<f64, (usize,)>) 
        -> Tensor<f64, (usize,)> 
    {
        // Automatically use Faer backend
        gemm::matvec(&self.matrix, coeffs)
    }
    
    pub fn fit(&self, values: &Tensor<f64, (usize,)>) 
        -> Result<Tensor<f64, (usize,)>, Error> 
    {
        let svd = self.matrix_svd.as_ref().ok_or(Error::NoSVD)?;
        Ok(gemm::solve_via_svd(&svd.u, &svd.s, &svd.v, values))
    }
}
```

## 将来のBLAS委譲パターン

```rust
// 将来の拡張例
pub fn matmul(
    a: &Tensor<f64, (usize, usize)>,
    b: &Tensor<f64, (usize, usize)>,
) -> Tensor<f64, (usize, usize)> {
    #[cfg(feature = "blas-fortran")]
    {
        // BLAS Fortran pointer call
        unsafe { call_dgemm_fortran(a, b) }
    }
    
    #[cfg(not(feature = "blas-fortran"))]
    {
        // Default: Faer
        Faer.matmul(a, b).eval()
    }
}
```

## 次のアクション

1. ✅ **Faer MatVec確認テスト**を作成・実行
2. ⏩ **gemm.rs実装**（1時間）
3. ⏩ **sampling.rs実装**（1-2時間）

---

Generated: 2025-10-09
Status: Final design - ready for implementation
