# 明日の作業予定

## 現状まとめ（2025-10-11終了時点）

### ✅ 今日完了したこと

1. **Basis traitの導入**
   - `Basis<S>` trait定義 (evaluate_tau, evaluate_matsubara, evaluate_omega等)
   - `FiniteTempBasis`に実装
   - `TauSampling`/`MatsubaraSampling`をBasis trait対応に移行

2. **DLR基本実装完了**
   - `DiscreteLehmannRepresentation<S>` 構造体
   - `from_IR_nd<T>` / `to_IR_nd<T>` (real/complex ジェネリック)
   - `Basis<S>` trait実装
     - `evaluate_tau`: TauPoles基底（拡張範囲対応）
     - `evaluate_matsubara`: MatsubaraPoles基底
   - 8個のDLRテスト全通過

3. **Fitterの強化**
   - `RealMatrixFitter`に`fit_2d_generic<T>`/`evaluate_2d_generic<T>`追加
   - すべてのfitterでGEMM最適化完了

### 📊 テスト状況
- 全体: 80+ tests pass
- DLR: 8/8 tests pass
- `TauSampling`がDLRで動作確認済み

## 🚧 明日の最優先作業

### 1. DLRの統合テスト（ジェネリック化） ⭐⭐⭐

**目的**: 既存のsampling testsをジェネリック関数化し、IR/DLR両方でテスト

**実装方針**:
```rust
// tau_sampling_tests.rsに追加
fn test_tau_sampling_roundtrip_generic<B, T, S>(basis: &B)
where
    B: Basis<S>,
    T: /* usual bounds */,
    S: StatisticsType + 'static,
{
    let sampling = TauSampling::new(basis);
    // ... existing roundtrip test logic ...
}

#[test]
fn test_tau_sampling_dlr_fermionic_real() {
    let kernel = LogisticKernel::new(100.0);  // beta=10, wmax=10, epsilon=1e-6
    let basis_ir = FiniteTempBasis::new(...);
    let dlr = DiscreteLehmannRepresentation::new(&basis_ir);
    test_tau_sampling_roundtrip_generic::<_, f64, Fermionic>(&dlr);
}

// 同様にcomplex, bosonic, MatsubaraSamplingも
```

**テスト項目**:
- [ ] TauSampling × DLR (real, complex, fermionic, bosonic)  
- [ ] MatsubaraSampling × DLR (complex, fermionic, bosonic)
- [ ] evaluate_nd/fit_nd × DLR (dim=0,1,2)

### 2. default_tau/matsubara_sampling_pointsの実装

現在DLRは空配列を返すが、これを元のIR basisから取得するように修正。

**実装**:
```rust
impl DiscreteLehmannRepresentation<S> {
    // Add reference to original basis
    basis_ref: Option<Arc<dyn Basis<S>>>,  // or just store sampling points
}
```

または、Julia準拠で元のbasisへの参照を保持する。

### 3. RegularizedBoseKernelの確認

Bosonic DLRの`evaluate_tau`が正しく動作するか確認。
- LogisticKernelで暫定実装中
- 必要に応じてRegularizedBoseKernel実装

## 📋 その他のタスク（優先度低）

- [ ] AugmentedBasis（実装しない予定）
- [ ] FiniteTempBasisSet（複数基底管理）
- [ ] ドキュメント整備
- [ ] パフォーマンス最適化

## 🎯 最終ゴール

- [x] DLRで`TauSampling`が動作
- [ ] DLRで`MatsubaraSampling`が動作（テスト追加）
- [ ] 全テストがIR/DLR両方でパス
- [ ] ジェネリック化によるコードの共通化

## 💡 設計ノート

### `unsafe`使用について
- 現在: ジェネリック関数窓口で`TypeId`判定+`transmute`
- 内部: 型ごとに安全な実装（`*_real`, `*_complex`）
- 理由: Rustのジェネリック制約でf64/Complex<f64>の分岐が必要
- 将来: Traitによる抽象化も検討可能

### テストパラメータ統一
- `beta = 10.0`, `wmax = 10.0`, `epsilon = 1e-6`
- 計算時間とメモリのバランス（Λ=100）
