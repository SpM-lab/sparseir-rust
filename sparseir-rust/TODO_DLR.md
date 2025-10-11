# DLR実装の作業予定

## ✅ 完了した機能

### 基盤
- [x] `Basis<S>` trait 実装
- [x] `FiniteTempBasis` に `Basis` trait 実装
- [x] `default_omega_sampling_points` 実装
- [x] `evaluate_tau`, `evaluate_matsubara`, `evaluate_omega` メソッド
- [x] `TauSampling`/`MatsubaraSampling` を `Basis` trait に対応

### DLR基本
- [x] `DiscreteLehmannRepresentation<S>` 構造体
- [x] `from_IR<T>` / `to_IR<T>` (ジェネリック: f64, Complex<f64>)
- [x] `from_IR_nd<T>` / `to_IR_nd<T>` (N次元対応)
- [x] DLR基本テスト (10個のテスト、全通過)

## 🚧 明日の作業予定

### 1. DLRの`Basis` trait実装 ⭐ **最優先**

DLRに`Basis<S>` traitを実装し、`TauSampling`/`MatsubaraSampling`で使えるようにする。

#### 必要な実装：

**a) `evaluate_tau` の実装**
- TauPoles基底関数: `u_i(τ) = -K(x, y_i)` where `x = 2τ/β - 1`, `y_i = pole_i/ωmax`
- Fermionic: `LogisticKernel`
- Bosonic: `RegularizedBoseKernel` (TODO: 実装確認必要)
- 拡張範囲対応: `-β < τ < 2β` (周期性/反周期性)

**b) `evaluate_matsubara` の実装**  
- MatsubaraPoles基底関数:
  - Fermionic: `u_i(iν) = 1 / (iν - pole_i)`
  - Bosonic: `u_i(iν) = tanh(β·pole_i/2) / (iν - pole_i)`

**c) その他の`Basis` traitメソッド**
- `beta()`, `wmax()`, `size()` - 簡単、既にフィールドあり
- `accuracy()`, `significance()` - DLRは全て1.0
- `default_tau_sampling_points()` - 元のbasisから取得？
- `default_matsubara_sampling_points()` - 元のbasisから取得？
- `default_omega_sampling_points()` - `self.poles.clone()`

### 2. TauSampling/MatsubaraSamplingのDLRテスト

**重要**: ジェネリック関数を使ってテスト実装を共通化

```rust
// 例: 既存のテストをジェネリック化
fn test_sampling_roundtrip_generic<B, S>(basis: &B)
where
    B: Basis<S>,
    S: StatisticsType + 'static,
{
    let sampling = TauSampling::new(basis);
    // ... roundtrip test ...
}

#[test]
fn test_tau_sampling_ir_fermionic() {
    let basis = FiniteTempBasis::new(...);
    test_sampling_roundtrip_generic(&basis);
}

#[test]
fn test_tau_sampling_dlr_fermionic() {
    let basis = FiniteTempBasis::new(...);
    let dlr = DiscreteLehmannRepresentation::new(&basis);
    test_sampling_roundtrip_generic(&dlr);
}
```

## ❓ 確認が必要な点

### A. RegularizedBoseKernel の実装
- Bosonic DLR の `evaluate_tau` に必要
- `kernel.rs` に実装されているか確認
- 未実装なら追加が必要

### B. 拡張τ範囲のサポート
- `evaluate_tau` で `-β ≤ τ ≤ β` に対応するか？
- Julia実装は `[0, β]` のみチェック
- Rust実装は拡張範囲対応済み（dlr.rsの単極Green関数）
- DLR基底関数も拡張範囲対応すべき？

### C. default sampling points
- DLRの `default_tau_sampling_points` / `default_matsubara_sampling_points`
- 元のIR basisから取得する？独自に定義する？
- Julia実装: DLRが元のbasisへの参照を保持

## 📋 実装優先順位

1. **高**: DLRの`Basis` trait実装 (evaluate_tau, evaluate_matsubara)
2. **高**: TauSampling/MatsubaraSamplingのDLRテスト（ジェネリック化）
3. **中**: RegularizedBoseKernel確認・実装
4. **低**: AugmentedBasis（実装しない予定）
5. **低**: FiniteTempBasisSet（複数基底管理）

## 🎯 最終目標

- [ ] DLRで`TauSampling`が動作
- [ ] DLRで`MatsubaraSampling`が動作  
- [ ] 全テストがIR/DLR両方でパス
- [ ] ドキュメント整備

## 📊 現在のテスト状況

- 全体: 76+ tests pass
- DLR基本: 10/10 tests pass
- Basis trait: 5/5 tests pass
- 次: DLRをsamplingでテスト
