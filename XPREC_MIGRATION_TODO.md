# xprec-svd mdarray移行 TODO

## 優先順位付きファイルリスト

### 🔴 Critical (SVD本体)
1. ✅ svd/jacobi.rs (487行) - 開始済み
   - SVDResult構造体 ✅
   - jacobi_svd関数シグネチャ ✅
   - QR前処理部分 ✅
   - 残り: 行列操作本体（~400行）

### 🟡 High (QR分解)
2. qr/rrqr.rs - RRQR実装
3. qr/householder.rs - Householder変換
4. qr/truncate.rs - QR結果の切り詰め

### 🟢 Medium (ユーティリティ)
5. utils/norms.rs - ノルム計算
6. utils/pivoting.rs - ピボット操作
7. utils/validation.rs - 検証関数

### 🔵 Low (高レベルAPI)
8. tsvd.rs - 高レベルTSVD API

## 主な置き換えパターン

```rust
// Shape
.nrows() → .shape().0
.ncols() → .shape().1

// Creation
Array2::zeros((m, n)) → Tensor::from_elem((m, n), T::zero())
Array2::eye(n) → Tensor::from_fn((n, n), |idx| if idx[0]==idx[1] {T::one()} else {T::zero()})
Array1::from_vec(v) → Tensor::from_fn((n,), |idx| v[idx[0]])

// Access
arr[[i, j]] → arr[[i, j]] (同じ)
arr[i] → arr[[i]]

// Slicing (最大の変更点)
arr.slice(s![a..b, c..d]) → 手動コピー: Tensor::from_fn((b-a, d-c), |idx| arr[[a+idx[0], c+idx[1]]])

// Row/Column mutation
arr.row_mut(i) → 手動ループ: for j in 0..n { arr[[i, j]] = ... }
arr.column_mut(j) → 手動ループ: for i in 0..m { arr[[i, j]] = ... }

// Iteration
arr.iter() → arr.iter() (同じ)
```

## 現在の進捗

- Cargo.toml: ndarray削除、mdarray必須化 ✅
- lib.rs: 型エイリアス更新 ✅
- jacobi.rs: 30% 完了
  - SVDResult構造体 ✅
  - jacobi_svd signature ✅
  - QR前処理 ✅
  - 行列スケーリング開始中

## 次のステップ

1. jacobi.rsのメインループ移行
2. apply_givens_* 関数の移行
3. テスト実行・デバッグ
4. 次のファイルへ

