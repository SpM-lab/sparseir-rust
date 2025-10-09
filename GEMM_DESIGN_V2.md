# GEMM モジュール設計 v2 (Faer前提)

## 基本方針

**全ての演算をmat-mat（行列-行列積）に統一**  
→ Faerの高速MatMulを活用

## コア実装: matmul_2d のみ

```rust
// src/gemm.rs

use mdarray::Tensor;
use mdarray_linalg::{MatMul, MatMulBuilder};
use mdarray_linalg_faer::Faer;

/// Matrix-Matrix multiplication: C = A @ B
/// A: (m, k), B: (k, n) → C: (m, n)
pub fn matmul(
    a: &Tensor<f64, (usize, usize)>,
    b: &Tensor<f64, (usize, usize)>,
) -> Tensor<f64, (usize, usize)> {
    let (m, k1) = *a.shape();
    let (k2, n) = *b.shape();
    assert_eq!(k1, k2, "Inner dimensions must match: {} vs {}", k1, k2);
    
    Faer.matmul(a, b).eval()
}
```

## ラッパー関数（便利機能）

### 1. Matrix-Vector multiplication
```rust
/// Matrix-Vector: y = A @ x
/// A: (m, k), x: (k,) → y: (m,)
pub fn matvec(
    a: &Tensor<f64, (usize, usize)>,
    x: &Tensor<f64, (usize,)>,
) -> Tensor<f64, (usize,)> {
    let (m, k) = *a.shape();
    assert_eq!(k, x.len());
    
    // Reshape x: (k,) → (k, 1)
    let x_2d = Tensor::from_fn((k, 1), |idx| x[[idx[0]]]);
    
    // result_2d = A @ x_2d: (m, 1)
    let result_2d = matmul(a, &x_2d);
    
    // Reshape back: (m, 1) → (m,)
    Tensor::from_fn((m,), |idx| result_2d[[idx[0], 0]])
}
```

### 2. Vector-Matrix multiplication
```rust
/// Vector-Matrix: y = x @ A
/// x: (k,), A: (k, n) → y: (n,)
pub fn vecmat(
    x: &Tensor<f64, (usize,)>,
    a: &Tensor<f64, (usize, usize)>,
) -> Tensor<f64, (usize,)> {
    let (k, n) = *a.shape();
    assert_eq!(k, x.len());
    
    // Reshape x: (k,) → (1, k)
    let x_2d = Tensor::from_fn((1, k), |idx| x[[idx[1]]]);
    
    // result_2d = x_2d @ A: (1, n)
    let result_2d = matmul(&x_2d, a);
    
    // Reshape back: (1, n) → (n,)
    Tensor::from_fn((n,), |idx| result_2d[[0, idx[0]]])
}
```

### 3. Transpose operations
```rust
/// Helper: transpose matrix
pub fn transpose(
    a: &Tensor<f64, (usize, usize)>
) -> Tensor<f64, (usize, usize)> {
    let (m, n) = *a.shape();
    Tensor::from_fn((n, m), |idx| a[[idx[1], idx[0]]])
}

/// Matrix^T @ Vector: y = A^T @ x
/// A: (m, k), x: (m,) → y: (k,)
pub fn matmul_t_vec(
    a: &Tensor<f64, (usize, usize)>,
    x: &Tensor<f64, (usize,)>,
) -> Tensor<f64, (usize,)> {
    // Option 1: explicit transpose
    let a_t = transpose(a);
    matvec(&a_t, x)
    
    // Option 2: reshape x and use matmul (avoid transpose)
    // vecmat(x, a) と同じ
}
```

## SVD Solver

```rust
/// Solve A @ x = b using SVD pseudoinverse
/// x = V @ diag(1/s) @ U^T @ b
pub fn solve_via_svd(
    u: &Tensor<f64, (usize, usize)>,
    s: &Tensor<f64, (usize,)>,
    v: &Tensor<f64, (usize, usize)>,
    b: &Tensor<f64, (usize,)>,
) -> Tensor<f64, (usize,)> {
    // Step 1: temp1 = U^T @ b
    let u_t = transpose(u);
    let temp1 = matvec(&u_t, b);
    
    // Step 2: temp2 = diag(1/s) @ temp1
    let temp2 = Tensor::from_fn((s.len(),), |idx| {
        temp1[[idx[0]]] / s[[idx[0]]]
    });
    
    // Step 3: x = V @ temp2
    matvec(v, &temp2)
}
```

## 完全なモジュール

```rust
// src/gemm.rs

//! Generic matrix multiplication and linear algebra utilities
//!
//! All operations are backed by Faer for high performance.

use mdarray::Tensor;
use mdarray_linalg::{MatMul, MatMulBuilder};
use mdarray_linalg_faer::Faer;

// ===== Core Operations =====

/// Matrix-Matrix multiplication: C = A @ B
pub fn matmul(
    a: &Tensor<f64, (usize, usize)>,
    b: &Tensor<f64, (usize, usize)>,
) -> Tensor<f64, (usize, usize)> {
    let (_, k1) = *a.shape();
    let (k2, _) = *b.shape();
    assert_eq!(k1, k2, "Inner dimensions must match: {} vs {}", k1, k2);
    
    Faer.matmul(a, b).eval()
}

/// Matrix transpose
pub fn transpose(
    a: &Tensor<f64, (usize, usize)>
) -> Tensor<f64, (usize, usize)> {
    let (m, n) = *a.shape();
    Tensor::from_fn((n, m), |idx| a[[idx[1], idx[0]]])
}

// ===== Convenience Wrappers =====

/// Matrix-Vector: y = A @ x
pub fn matvec(
    a: &Tensor<f64, (usize, usize)>,
    x: &Tensor<f64, (usize,)>,
) -> Tensor<f64, (usize,)> {
    let (m, k) = *a.shape();
    assert_eq!(k, x.len(), "Dimension mismatch");
    
    let x_2d = Tensor::from_fn((k, 1), |idx| x[[idx[0]]]);
    let result_2d = matmul(a, &x_2d);
    Tensor::from_fn((m,), |idx| result_2d[[idx[0], 0]])
}

/// Matrix^T @ Vector: y = A^T @ x
pub fn matmul_t_vec(
    a: &Tensor<f64, (usize, usize)>,
    x: &Tensor<f64, (usize,)>,
) -> Tensor<f64, (usize,)> {
    let a_t = transpose(a);
    matvec(&a_t, x)
}

// ===== SVD-based Solver =====

/// Solve A @ x = b using SVD pseudoinverse
/// x = V @ diag(1/s) @ U^T @ b
pub fn solve_via_svd(
    u: &Tensor<f64, (usize, usize)>,
    s: &Tensor<f64, (usize,)>,
    v: &Tensor<f64, (usize, usize)>,
    b: &Tensor<f64, (usize,)>,
) -> Tensor<f64, (usize,)> {
    let temp1 = matmul_t_vec(u, b);
    let temp2 = Tensor::from_fn((s.len(),), |idx| temp1[[idx[0]]] / s[[idx[0]]]);
    matvec(v, &temp2)
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_matmul_2x2() {
        let a = Tensor::from_fn((2, 2), |idx| {
            (idx[0] * 2 + idx[1] + 1) as f64
        });
        let b = Tensor::from_fn((2, 2), |idx| {
            (idx[0] * 2 + idx[1] + 5) as f64
        });
        
        let c = matmul(&a, &b);
        
        // [[1,2], [3,4]] @ [[5,6], [7,8]] = [[19,22], [43,50]]
        assert!((c[[0, 0]] - 19.0).abs() < 1e-10);
        assert!((c[[0, 1]] - 22.0).abs() < 1e-10);
        assert!((c[[1, 0]] - 43.0).abs() < 1e-10);
        assert!((c[[1, 1]] - 50.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_matvec() {
        let a = Tensor::from_fn((2, 3), |idx| {
            (idx[0] * 3 + idx[1] + 1) as f64
        });
        let x = Tensor::from_fn((3,), |idx| (idx[0] + 1) as f64);
        
        let y = matvec(&a, &x);
        
        // [[1,2,3], [4,5,6]] @ [1,2,3] = [14, 32]
        assert_eq!(y.len(), 2);
        assert!((y[[0]] - 14.0).abs() < 1e-10);
        assert!((y[[1]] - 32.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_transpose() {
        let a = Tensor::from_fn((2, 3), |idx| {
            (idx[0] * 3 + idx[1]) as f64
        });
        let a_t = transpose(&a);
        
        assert_eq!(*a_t.shape(), (3, 2));
        assert_eq!(a_t[[0, 0]], a[[0, 0]]);
        assert_eq!(a_t[[0, 1]], a[[1, 0]]);
        assert_eq!(a_t[[2, 1]], a[[1, 2]]);
    }
    
    #[test]
    fn test_solve_via_svd_simple() {
        // Simple 2x2 system
        let a = Tensor::from_fn((2, 2), |idx| {
            match (idx[0], idx[1]) {
                (0, 0) => 2.0,
                (0, 1) => 1.0,
                (1, 0) => 1.0,
                (1, 1) => 3.0,
                _ => unreachable!(),
            }
        });
        
        // Compute SVD
        let svd = xprec_svd::tsvd_f64(&a, 1e-15).unwrap();
        
        // b = [5, 7]
        let b = Tensor::from_fn((2,), |idx| {
            if idx[0] == 0 { 5.0 } else { 7.0 }
        });
        
        // Solve: A @ x = b
        let x = solve_via_svd(&svd.u, &svd.s, &svd.v, &b);
        
        // Verify: A @ x ≈ b
        let b_check = matvec(&a, &x);
        assert!((b_check[[0]] - 5.0).abs() < 1e-10);
        assert!((b_check[[1]] - 7.0).abs() < 1e-10);
    }
}
```

## 利点

### ✅ シンプル
- コア関数: `matmul()` のみ
- 他は全てラッパー（reshape + matmul）

### ✅ 高速
- Faerバックエンド常に使用
- SIMD最適化の恩恵

### ✅ 保守性
- 実装が1箇所に集約
- バグが少ない

## 実装サイズ

- **コア**: ~10行（matmul）
- **ラッパー**: ~50行（matvec, transpose, solve_via_svd等）
- **テスト**: ~100行
- **合計**: ~160行

## ファイル構成

```
src/
  gemm.rs          # ~160行
    - matmul()               ← コア（Faer使用）
    - transpose()
    - matvec()               ← reshape + matmul
    - matmul_t_vec()         ← transpose + matvec
    - solve_via_svd()        ← SVD pseudoinverse
    - tests
    
  sampling.rs      # ~200行（後で実装）
    - TauSampling
      - new()
      - evaluate()           ← gemm::matvec使用
      - fit()                ← gemm::solve_via_svd使用
    - tests
```

## sampling.rsでの使用

```rust
use crate::gemm;

impl TauSampling {
    pub fn evaluate(&self, coeffs: &Tensor<f64, (usize,)>) 
        -> Tensor<f64, (usize,)> 
    {
        // values = matrix @ coeffs
        gemm::matvec(&self.matrix, coeffs)
    }
    
    pub fn fit(&self, values: &Tensor<f64, (usize,)>) 
        -> Result<Tensor<f64, (usize,)>, Error> 
    {
        let svd = self.matrix_svd.as_ref().ok_or(Error::NoSVD)?;
        
        // coeffs = V @ diag(1/s) @ U^T @ values
        Ok(gemm::solve_via_svd(&svd.u, &svd.s, &svd.v, values))
    }
}
```

## 実装手順

### Step 1: gemm.rs作成 (30分)
```bash
# ✅ 実装
- matmul()      # Faerラッパー
- transpose()   # 単純なreshape

# ✅ テスト
- 2x2行列積
- 転置テスト
```

### Step 2: Vector操作追加 (30分)
```bash
# ✅ 実装
- matvec()        # reshape + matmul
- matmul_t_vec()  # transpose + matvec

# ✅ テスト
- matvecテスト
- A^T @ x テスト
```

### Step 3: SVDソルバー (30分)
```bash
# ✅ 実装
- solve_via_svd()

# ✅ テスト
- 簡単な連立方程式
- ラウンドトリップ（A @ x = b → solve → verify）
```

### Step 4: sampling.rs基本構造 (1-2時間)
```bash
# ✅ 実装
- TauSampling構造体
- new(), with_sampling_points()
- evaluate(), fit()

# ✅ テスト
- ラウンドトリップテスト
```

**合計推定時間: 2.5-3.5時間**

## 疑問点・議論ポイント

### Q1: transposeのコスト
```rust
// Option A: 毎回新しいTensor作成
let a_t = transpose(a);  // コピー発生

// Option B: Faer.matmul()にtransposeフラグを渡す？
// → mdarray-linalgのAPIを確認必要
```

### Q2: reshape vs view
```rust
// Option A: Tensor::from_fn (コピー)
let x_2d = Tensor::from_fn((k, 1), |idx| x[[idx[0]]]);

// Option B: view/reshape (ゼロコピー)
let x_2d = x.view().reshape((k, 1));  // mdarrayにこの機能ある？
```

**推奨**: まずOption Aで実装、必要ならOption Bで最適化

### Q3: エラーハンドリング
```rust
// Option A: panic (assert)
assert_eq!(k1, k2);

// Option B: Result
if k1 != k2 { return Err(...); }
```

**推奨**: panic（パフォーマンス重視、デバッグでキャッチ）

---

Generated: 2025-10-09
Status: Design v2 - Faer-first approach
