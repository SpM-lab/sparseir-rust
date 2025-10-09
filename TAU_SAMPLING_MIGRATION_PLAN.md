# TauSampling移植計画

## 概要

TauSamplingは、IR基底係数と虚時間のサンプリング点間の変換を行うクラス。

## 参照実装

### Julia実装 (SparseIR.jl)
```julia
struct TauSampling{T,TMAT,F,B} <: AbstractSampling{T,TMAT,F}
    sampling_points :: Vector{T}
    matrix          :: Matrix{TMAT}
    matrix_svd      :: F
    basis           :: B
end
```

### C++実装 (libsparseir)
```cpp
template <typename S>
class TauSampling : public AbstractSampling {
private:
    Eigen::VectorXd sampling_points_;
    Eigen::MatrixXd matrix_;           // (num_points, basis_size)
    std::shared_ptr<JacobiSVD<Eigen::MatrixXd>> matrix_svd_;
    
public:
    TauSampling(basis, sampling_points)
    // evaluate, fit methods
};
```

## Rust実装計画

### Phase 1: 基本構造体定義

```rust
// src/sampling.rs

use mdarray::Tensor;
use crate::basis::FiniteTempBasis;
use xprec_svd::SVDResult;

/// Sparse sampling in imaginary time
pub struct TauSampling {
    /// Sampling points in imaginary time (τ)
    sampling_points: Tensor<f64, (usize,)>,
    
    /// Evaluation matrix: (num_points, basis_size)
    /// matrix[i,l] = u_l(τ_i)
    matrix: Tensor<f64, (usize, usize)>,
    
    /// SVD decomposition of matrix for fitting
    matrix_svd: Option<SVDResult<f64>>,
    
    /// Reference to the basis
    basis: FiniteTempBasis,
}
```

### Phase 2: コンストラクタ

```rust
impl TauSampling {
    /// Create TauSampling with default sampling points
    pub fn new(basis: &FiniteTempBasis) -> Result<Self, Error> {
        let sampling_points = default_tau_sampling_points(basis);
        Self::with_sampling_points(basis, sampling_points)
    }
    
    /// Create TauSampling with custom sampling points
    pub fn with_sampling_points(
        basis: &FiniteTempBasis,
        sampling_points: Tensor<f64, (usize,)>
    ) -> Result<Self, Error> {
        // 1. Validate sampling points
        // 2. Compute evaluation matrix
        let matrix = eval_matrix_tau(basis, &sampling_points)?;
        
        // 3. Compute SVD for fitting
        let matrix_svd = compute_svd(&matrix)?;
        
        // 4. Check conditioning
        if is_poorly_conditioned(&matrix_svd) {
            warn!("Sampling matrix is poorly conditioned");
        }
        
        Ok(Self {
            sampling_points,
            matrix,
            matrix_svd: Some(matrix_svd),
            basis: basis.clone(),
        })
    }
}
```

### Phase 3: 主要メソッド

```rust
impl TauSampling {
    /// Evaluate basis coefficients at sampling points
    /// al[l] -> result[i] = Σ_l matrix[i,l] * al[l]
    pub fn evaluate(
        &self,
        coeffs: &Tensor<f64, (usize,)>
    ) -> Result<Tensor<f64, (usize,)>, Error> {
        // matrix @ coeffs
        matmul_1d(&self.matrix, coeffs)
    }
    
    /// Fit basis coefficients from sampling point values
    /// values[i] -> coeffs[l] = (matrix^+ @ values)[l]
    /// where matrix^+ is the pseudoinverse
    pub fn fit(
        &self,
        values: &Tensor<f64, (usize,)>
    ) -> Result<Tensor<f64, (usize,)>, Error> {
        let svd = self.matrix_svd.as_ref()
            .ok_or(Error::NoSVD)?;
        
        // Solve: matrix @ coeffs = values
        // Using SVD: coeffs = V @ diag(1/s) @ U^T @ values
        solve_via_svd(svd, values)
    }
    
    /// Get sampling points
    pub fn sampling_points(&self) -> &Tensor<f64, (usize,)> {
        &self.sampling_points
    }
    
    /// Get tau (alias for sampling_points)
    pub fn tau(&self) -> &Tensor<f64, (usize,)> {
        &self.sampling_points
    }
}
```

### Phase 4: ヘルパー関数

```rust
/// Get default tau sampling points for basis
/// Returns extrema of highest-order basis function
fn default_tau_sampling_points(
    basis: &FiniteTempBasis
) -> Tensor<f64, (usize,)> {
    // Find extrema of u_{size-1}(τ)
    let l = basis.size() - 1;
    find_extrema_of_basis_function(basis, l)
}

/// Evaluate matrix: matrix[i,l] = u_l(τ_i)
fn eval_matrix_tau(
    basis: &FiniteTempBasis,
    tau_points: &Tensor<f64, (usize,)>
) -> Result<Tensor<f64, (usize, usize)>, Error> {
    let num_points = tau_points.len();
    let basis_size = basis.size();
    
    Tensor::from_fn((num_points, basis_size), |idx| {
        let tau = tau_points[[idx[0]]];
        basis.u(l: idx[1], tau)
    })
}

/// Check if matrix is poorly conditioned
fn is_poorly_conditioned(svd: &SVDResult<f64>) -> bool {
    let cond = svd.s[[0]] / svd.s[[svd.s.len() - 1]];
    cond > 1e8
}

/// Solve matrix @ x = b using SVD
fn solve_via_svd(
    svd: &SVDResult<f64>,
    b: &Tensor<f64, (usize,)>
) -> Result<Tensor<f64, (usize,)>, Error> {
    // x = V @ diag(1/s) @ U^T @ b
    
    // 1. u_t_b = U^T @ b
    let u_t_b = matmul_1d(&transpose(&svd.u), b)?;
    
    // 2. s_inv_u_t_b = diag(1/s) @ u_t_b
    let s_inv_u_t_b = Tensor::from_fn((svd.s.len(),), |idx| {
        u_t_b[[idx[0]]] / svd.s[[idx[0]]]
    });
    
    // 3. x = V @ s_inv_u_t_b
    matmul_1d(&svd.v, &s_inv_u_t_b)
}
```

## 実装順序

### Step 1: 基本構造 (1-2時間)
- [ ] `src/sampling.rs` ファイル作成
- [ ] `TauSampling` 構造体定義
- [ ] `lib.rs` に `mod sampling` 追加

### Step 2: ヘルパー関数 (2-3時間)
- [ ] `eval_matrix_tau` 実装
- [ ] `default_tau_sampling_points` 実装
- [ ] `solve_via_svd` 実装

### Step 3: コンストラクタ (1-2時間)
- [ ] `TauSampling::new` 実装
- [ ] `TauSampling::with_sampling_points` 実装
- [ ] SVD計算の統合

### Step 4: 主要メソッド (2-3時間)
- [ ] `evaluate` 実装
- [ ] `fit` 実装
- [ ] ゲッターメソッド

### Step 5: テスト (3-4時間)
- [ ] 単体テスト作成
- [ ] Julia実装との比較テスト
- [ ] エッジケーステスト

**合計推定時間: 9-14時間**

## 依存関係

### 既存機能
- ✅ `FiniteTempBasis`
- ✅ `basis.u(tau)` 評価
- ✅ `xprec-svd` (SVD計算)

### 必要な新機能
- ❌ `default_tau_sampling_points` (extrema探索)
- ❌ 行列-ベクトル積ヘルパー
- ❌ SVDを使った線形方程式ソルバー

## 技術的課題

### 1. extrema探索
最高次基底関数の極値を見つける必要がある。
- オプション1: 数値微分 + 根探索
- オプション2: グリッドサーチ
- オプション3: 既知の解析解（もしあれば）

### 2. テンソル操作
- 1次元ベクトルと2次元行列の積
- SVD結果の利用
- mdarrayの制約内での実装

### 3. 多次元配列対応（後回し）
Juliaは多次元配列に対応：
```julia
evaluate(sampling, al; dim=1)  # al can be N-dimensional
```
Rust版は最初は1次元のみサポート。

## テストケース

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_tau_sampling_basic() {
        let basis = FiniteTempBasis::new(
            Statistics::Fermionic,
            1.0,   // beta
            10.0,  // wmax
            1e-6   // epsilon
        ).unwrap();
        
        let sampling = TauSampling::new(&basis).unwrap();
        assert_eq!(sampling.sampling_points().len(), basis.size());
    }
    
    #[test]
    fn test_evaluate_fit_roundtrip() {
        let basis = /* ... */;
        let sampling = TauSampling::new(&basis).unwrap();
        
        // Random coefficients
        let coeffs = random_tensor((basis.size(),));
        
        // Evaluate -> Fit should recover coefficients
        let values = sampling.evaluate(&coeffs).unwrap();
        let recovered = sampling.fit(&values).unwrap();
        
        assert_close(&coeffs, &recovered, 1e-10);
    }
}
```

## 次のステップ

Phase 1完了後:
1. MatsubaraSampling実装（より複雑）
2. 多次元配列サポート
3. パフォーマンス最適化

---

Generated: 2025-10-09
Branch: main
Status: Planning phase

## libsparseirのGEMMとevaluate/fit実装

### C++ gemm.hpp の実装

libsparseirは高度な行列演算を実装しています：

#### evaluate実装パターン
```cpp
// evaluate: result = matrix @ coeffs
// dim指定で任意の次元軸に沿って行列積

template <typename T1, typename T2, int N>
Eigen::Tensor<T_result, N> evaluate_dimx(
    const Eigen::MatrixXd &matrix,  // (n_points, basis_size)
    const Eigen::Tensor<T2, N> &coeffs,
    int dim = 0
) {
    // 1. coeffsをdim軸に沿ってreshape
    // 2. 行列積計算: matrix @ reshaped_coeffs
    // 3. 結果をreshape back
}
```

#### fit実装パターン
```cpp
// fit: coeffs = matrix^+ @ values
// SVDを使ったpseudoinverse

template <typename Scalar>
void fit_inplace_dim2(
    const JacobiSVD<Eigen::MatrixXd> &svd,
    const Eigen::MatrixXd &values,
    Eigen::MatrixXd &coeffs
) {
    // coeffs = V @ diag(1/s) @ U^T @ values
    // 1. temp = U^T @ values
    // 2. temp = diag(1/s) @ temp
    // 3. coeffs = V @ temp
}
```

#### 多次元対応
```cpp
// 3次元テンソル対応
template <typename Scalar, typename InputScalar, typename OutputScalar>
int evaluate_inplace_dim3(
    const Eigen::MatrixXd &matrix,
    const Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> &input,
    int dim,  // 0, 1, or 2
    Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> &output
) {
    // dimに応じて適切にcontract
    switch(dim) {
        case 0: // contract along first dimension
        case 1: // contract along second dimension  
        case 2: // contract along third dimension
    }
}
```

### Rust実装への示唆

#### 1. シンプルな1D実装から開始
```rust
/// evaluate: values = matrix @ coeffs
fn evaluate_1d(
    matrix: &Tensor<f64, (usize, usize)>,  // (n_points, basis_size)
    coeffs: &Tensor<f64, (usize,)>         // (basis_size,)
) -> Tensor<f64, (usize,)> {              // (n_points,)
    let (n_points, basis_size) = *matrix.shape();
    assert_eq!(basis_size, coeffs.len());
    
    Tensor::from_fn((n_points,), |idx| {
        let mut sum = 0.0;
        for l in 0..basis_size {
            sum += matrix[[idx[0], l]] * coeffs[[l]];
        }
        sum
    })
}

/// fit: coeffs = matrix^+ @ values (using SVD)
fn fit_1d(
    svd: &SVDResult<f64>,
    values: &Tensor<f64, (usize,)>
) -> Tensor<f64, (usize,)> {
    // 1. temp = U^T @ values
    let temp1 = Tensor::from_fn((svd.s.len(),), |idx| {
        let mut sum = 0.0;
        for i in 0..values.len() {
            sum += svd.u[[i, idx[0]]] * values[[i]];
        }
        sum
    });
    
    // 2. temp = diag(1/s) @ temp
    let temp2 = Tensor::from_fn((svd.s.len(),), |idx| {
        temp1[[idx[0]]] / svd.s[[idx[0]]]
    });
    
    // 3. coeffs = V @ temp
    let basis_size = svd.v.shape().0;
    Tensor::from_fn((basis_size,), |idx| {
        let mut sum = 0.0;
        for k in 0..svd.s.len() {
            sum += svd.v[[idx[0], k]] * temp2[[k]];
        }
        sum
    })
}
```

#### 2. mdarray-linalg-faerを活用
```rust
use mdarray_linalg::{MatMul, MatMulBuilder};
use mdarray_linalg_faer::Faer;

/// evaluate using Faer backend (faster)
fn evaluate_1d_faer(
    matrix: &Tensor<f64, (usize, usize)>,
    coeffs: &Tensor<f64, (usize,)>
) -> Tensor<f64, (usize,)> {
    // Reshape coeffs to (basis_size, 1) for matmul
    let coeffs_2d = reshape_1d_to_2d(coeffs);
    
    // result = matrix @ coeffs_2d
    let result_2d = Faer.matmul(matrix, &coeffs_2d).eval();
    
    // Reshape back to 1D
    reshape_2d_to_1d(&result_2d)
}
```

#### 3. 多次元対応は後回し
```rust
// Phase 2での実装
pub fn evaluate_nd<const N: usize>(
    &self,
    coeffs: &Tensor<f64, N>,
    dim: usize
) -> Result<Tensor<f64, N>, Error> {
    // TODO: implement multi-dimensional support
    unimplemented!("Multi-dimensional support coming in Phase 2")
}
```

### 実装優先順位

1. **Phase 1 (最小限)**:
   - ✅ 1D evaluate (手動ループ)
   - ✅ 1D fit (SVD pseudoinverse)
   - ✅ 基本的なテスト

2. **Phase 1.5 (最適化)**:
   - ⏳ mdarray-linalg-faerの利用
   - ⏳ パフォーマンスベンチマーク

3. **Phase 2 (拡張)**:
   - ⏳ 多次元配列対応
   - ⏳ complex対応（将来）

### パフォーマンス考慮事項

```rust
// 小さい行列: 手動ループでOK
if matrix.shape().0 * matrix.shape().1 < 10000 {
    evaluate_1d_manual(matrix, coeffs)
} else {
    // 大きい行列: BLASバックエンド使用
    evaluate_1d_faer(matrix, coeffs)
}
```

---

Updated: 2025-10-09
追加情報: libsparseir C++ gemm実装
