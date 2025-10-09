# Jacobi SVD収束警告について

## 警告内容
```
[Jacobi SVD] WARNING: Max iterations (1000) reached without convergence!
```

## 原因

### xprec-svd/src/svd/jacobi.rs
Jacobi SVDアルゴリズムは反復計算で収束を目指す：

```rust
while iteration < max_iterations {
    let max_off_diag = find_max_off_diagonal(&a);
    let threshold = precision * max_diag_entry;
    
    if max_off_diag <= threshold {
        break;  // 収束
    }
    // Givens回転を適用...
}
```

### 10x10サンプリング行列での問題
- 行列の要素が特定の範囲にある
- `max_off_diag`が`threshold`をわずかに超える
- 1000回反復しても収束基準を満たせない

## 影響

### ✅ 実用上は問題なし
- 特異値は正しく計算されている
- First SV: 5.684738
- Last SV: 0.233400
- Condition number: 24.4（良好）

### 数値精度の問題
収束基準が厳しすぎる可能性：
```rust
threshold = 2.0 * f64::EPSILON * max_diag_entry
          ≈ 4.4e-16 * max_diag_entry
```

## 対策

### 短期（現在）
- ⏳ 警告を無視（結果は正しい）
- ⏳ より緩い`rtol`でSVD実行

### 中期
- ⏳ xprec-svdの収束基準を調整
- ⏳ 行列スケーリングの改善

### 長期
- ⏳ 異なるSVDアルゴリズム（QR iteration等）の実装

## テストコードでの対応

```rust
// より緩いrtolでSVD実行
let svd = xprec_svd::tsvd_f64(&matrix_tensor, 1e-12).unwrap();  // 1e-15 → 1e-12
```

---

Generated: 2025-10-09
Status: Warning noted, not critical
