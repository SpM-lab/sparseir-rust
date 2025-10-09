# mdarray-linalg テンソルコントラクション対応状況

## 調査結果

### ❌ テンソルコントラクション機能なし

mdarray-linalgは**2階テンソル（行列）までの線形代数**に特化しています。

## サポート機能

### ✅ 対応済み
- **MatMul**: `A[i,j] * B[j,k] → C[i,k]` (行列積)
- **MatVec**: `A[i,j] * x[j] → y[i]` (行列ベクトル積)
- **SVD, Eig, LU, QR**: 標準的な行列分解

### ❌ 未対応
- **テンソルコントラクション**: `a[i,K] * b[j,K,k,l]` のような高階テンソル演算
- **einsum**: Einstein summation記法
- **tensordot**: NumPyのtensordot相当

## 代替手段

### 1. 手動実装
```rust
// a[i, K] * b[j, K, k, l] → result[i, j, k, l]
let (i_size, k_size) = *a.shape();
let (j_size, _, k_size2, l_size) = *b.shape();
assert_eq!(k_size, k_size2);

let result = Tensor::from_fn((i_size, j_size, k_size2, l_size), |idx| {
    let mut sum = 0.0;
    for k in 0..k_size {
        sum += a[[idx[0], k]] * b[[idx[1], k, idx[2], idx[3]]];
    }
    sum
});
```

### 2. reshape + matmul
```rust
// a[i, K] * b[j, K, k, l] の場合：
// 1. bを [j, K, k*l] にreshape
// 2. aとbで行列積: a[i,K] @ b_reshaped[K, j*k*l] → tmp[i, j*k*l]
// 3. tmpを [i, j, k, l] にreshape
```

### 3. 外部ライブラリ検討
- **ndarray**: `ndarray::linalg::general_mat_mul` + 手動reshape
- **faer直接**: より柔軟なテンソル操作が可能かも

## 推奨アプローチ

### ユースケース次第
1. **単純なケース**: 手動実装（ループ）
2. **パフォーマンス重視**: reshape + matmul
3. **複雑なケース**: 専用のテンソルライブラリ検討

## 参考: NumPyとの比較

```python
# NumPy
result = np.einsum('iK,jKkl->ijkl', a, b)
# または
result = np.tensordot(a, b, axes=([1], [1]))
```

↓

```rust
// mdarray-linalg: 未対応
// 手動実装が必要
```

## 結論

**mdarray-linalgは行列演算に特化**しており、
高階テンソルのコントラクションは**サポート外**です。

必要な場合は：
1. 手動実装
2. reshape技法で行列演算に還元
3. 他のライブラリ検討

---

Generated: 2025-10-09
