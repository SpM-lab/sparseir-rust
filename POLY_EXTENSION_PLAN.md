# poly.rs 拡張計画

## 必要な機能（default_tau_sampling_points用）

### Julia実装（basis.jl:170-202）の移植

```julia
function default_sampling_points(u::PiecewiseLegendrePolyVector, L::Integer)
    if L < length(u)
        # L+1番目の多項式の根を使う
        x₀ = roots(u[L + 1])
    else
        # 最後の基底関数の微分の根（極値）を使う
        poly = last(u)
        maxima = roots(deriv(poly))
        
        # 端点を追加
        left  = (first(maxima) + xmin(poly)) / 2
        right = (last(maxima) + xmax(poly)) / 2
        x₀    = [left; maxima; right]
    end
    return x₀
end
```

## poly.rsの現状確認

### ✅ 実装済み
- `deriv()` - 微分
- `roots()` - 根探索

### ❓ 確認必要
- `xmin()`, `xmax()` - 定義域の端点
- `PiecewiseLegendrePolyVector::last()` - 最後の多項式取得
- `PiecewiseLegendrePolyVector::get()` - インデックスアクセス

## 必要な拡張

### 1. PiecewiseLegendrePoly
```rust
impl PiecewiseLegendrePoly {
    // 既存: deriv(), roots()
    
    /// 定義域の最小値
    pub fn xmin(&self) -> f64 {
        self.knots.first().copied().unwrap_or(0.0)
    }
    
    /// 定義域の最大値
    pub fn xmax(&self) -> f64 {
        self.knots.last().copied().unwrap_or(0.0)
    }
}
```

### 2. PiecewiseLegendrePolyVector
```rust
impl PiecewiseLegendrePolyVector {
    // 確認: 既存のメソッド
    
    /// 最後の多項式を取得
    pub fn last(&self) -> &PiecewiseLegendrePoly {
        self.polyvec.last()
            .expect("Empty PiecewiseLegendrePolyVector")
    }
    
    /// インデックスでアクセス
    pub fn get(&self, index: usize) -> &PiecewiseLegendrePoly {
        &self.polyvec[index]
    }
    
    /// 長さ
    pub fn len(&self) -> usize {
        self.polyvec.len()
    }
}
```

### 3. default_sampling_points関数
```rust
/// Get default sampling points in [-1, 1]
pub fn default_sampling_points(
    u: &PiecewiseLegendrePolyVector,
    l: usize
) -> Vec<f64> {
    assert_eq!(u.xmin(), -1.0);
    assert_eq!(u.xmax(), 1.0);
    
    let x0 = if l < u.len() {
        // Use roots of (L+1)-th polynomial
        u.get(l).roots()
    } else {
        // Use extrema of last basis function
        let poly = u.last();
        let poly_deriv = poly.deriv(1);
        let mut maxima = poly_deriv.roots();
        
        // Add endpoints
        let left = (maxima.first().unwrap() + poly.xmin()) / 2.0;
        let right = (maxima.last().unwrap() + poly.xmax()) / 2.0;
        
        let mut result = vec![left];
        result.extend_from_slice(&maxima);
        result.push(right);
        result
    };
    
    if x0.len() != l {
        eprintln!(
            "Warning: Expected {} sampling points, got {}",
            l, x0.len()
        );
    }
    
    x0
}
```

## 実装手順

### Step 1: poly.rsを読んで現状確認（15分）
- `xmin()`, `xmax()` の有無
- `PiecewiseLegendrePolyVector` のメソッド確認

### Step 2: 不足メソッド追加（30分）
- `xmin()`, `xmax()` 追加（必要なら）
- `last()`, `get()` 追加（必要なら）

### Step 3: default_sampling_points実装（30分）
- 新規関数追加
- テスト作成

### Step 4: default_tau_sampling_points実装（15分）
- スケーリング処理
- basis.rsに追加

**合計**: 1.5時間

---

Generated: 2025-10-09
Next: poly.rs確認 → 拡張実装
