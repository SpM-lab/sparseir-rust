# ✅ poly.rs 拡張完了

## 実装内容

### 1. poly.rs: default_sampling_points()
**C++ソース**: `libsparseir/include/sparseir/basis.hpp:287-310`

```rust
pub fn default_sampling_points(
    u: &PiecewiseLegendrePolyVector,
    l: usize,
) -> Vec<f64>
```

**アルゴリズム**:
1. Check: u must be in [-1, 1]
2. If l < u.size(): return u[l].roots()
3. Else:
   - poly = u.last()
   - maxima = poly.deriv().roots()
   - left = (maxima[0] + poly.xmin) / 2
   - right = (maxima[last] + poly.xmax) / 2
   - return [left, maxima..., right]

### 2. poly.rs: last() method
```rust
impl PiecewiseLegendrePolyVector {
    pub fn last(&self) -> &PiecewiseLegendrePoly
}
```

### 3. basis.rs: default_tau_sampling_points()
**C++ソース**: `libsparseir/include/sparseir/basis.hpp:229-270`

```rust
impl FiniteTempBasis {
    pub fn default_tau_sampling_points(&self) -> Vec<f64>
}
```

**アルゴリズム**:
1. x = default_sampling_points(&sve_result.u, size)
2. Extract unique_x (half of x)
3. Generate symmetric points: ±(β/2)*(x + 1)
4. Sort
5. Convert negative to [β/2, β]: if tau < 0, tau += β

## C++実装との対応

| 機能 | C++行番号 | Rust実装 | 状態 |
|------|----------|---------|------|
| default_sampling_points | 287-310 | poly.rs:849-898 | ✅ |
| last() | polyvec.back() | poly.rs:814-817 | ✅ |
| default_tau_sampling_points | 229-270 | basis.rs:227-285 | ✅ |

## テスト

```rust
#[test]
fn test_default_tau_sampling_points() {
    let kernel = LogisticKernel::new(10.0);
    let basis = FermionicBasis::new(kernel, 1.0, Some(1e-6), None);
    
    let tau_points = basis.default_tau_sampling_points();
    
    // All points in [0, beta]
    for &tau in &tau_points {
        assert!(tau >= 0.0 && tau <= basis.beta);
    }
    
    // Sorted
    for i in 1..tau_points.len() {
        assert!(tau_points[i] >= tau_points[i-1]);
    }
}
```

## 次のステップ

1. ✅ **poly.rs拡張完了**
2. ⏩ **gemm.rs実装** - matmul wrapper
3. ⏩ **sampling.rs実装** - TauSampling

---

Generated: 2025-10-09
Commit: 22ab3d5
Status: ✅ Complete - Ready for sampling.rs
