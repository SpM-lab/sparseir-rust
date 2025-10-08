# ndarray → mdarray 最終移行計画

## 現状評価（Phase 7完了後）

### 🎉 予想外の発見

**ndarray使用は既に最小限！**

実際のndarray依存箇所：
1. **sve/result.rs**: `s: Array1<f64>` (1箇所)
2. **basis.rs**: `s: Array1<f64>` (1箇所) 
3. **interpolation1d.rs**: `coeffs: Array1<T>`, 内部関数 (8箇所)
4. **interpolation2d.rs**: `coeffs: Array2<T>`, 内部関数 (9箇所)
5. **xprec-svd統合**: SVD入出力 (複数)

**重要**: poly.rsはndarray::Array2を内部で使うが、主にLegendre係数の内部表現のみ。

### 戦略の見直し

#### オプションA: 完全移行（大規模）
- 全てのndarray使用をmdarrayに置き換え
- xprec-svdもfork & 移行
- 推定時間: 20-30時間

#### オプションB: ハイブリッド運用（推奨）✅
- 新規コード: mdarray使用
- 既存コード: ndarray継続
- 変換レイヤー: `mdarray_compat`で橋渡し
- xprec-svd: ndarray版を継続使用
- 推定時間: 2-3時間（ほぼ完了）

#### オプションC: Faer直接統合
- xprec-svdの代わりにFaerのSVDを直接使用
- 高精度SVD（Jacobi方式）をFaerで再実装
- 推定時間: 10-15時間

## 推奨アプローチ：オプションB（ハイブリッド）

### 理由

1. **実用性**: 既存のテストとコードが動作する
2. **段階的**: 必要に応じて追加移行可能
3. **柔軟性**: xprec-svdの高精度SVDを継続利用
4. **パフォーマンス**: 重要な部分はFaerで最適化可能

### Phase 8-10の新計画

#### Phase 8: テスト確認（1時間）
- 既存テストが全てパスすることを確認
- mdarray版のテストを追加
- 互換性検証

#### Phase 9: Faerベンチマーク（2時間）
- Faer SVDの性能測定
- xprec-svd（Jacobi）と比較
- lambda=10, 100, 1000での計測
- 精度比較（特にlambda=1000）

#### Phase 10: 統合判断（1時間）
- Faer vs xprec-svdの性能・精度トレードオフ評価
- 最終的な統合戦略を決定
  - オプション1: xprec-svd継続（現状維持）
  - オプション2: Faer SVD採用（シンプル化）
  - オプション3: 両方サポート（feature flag）

### 実装済みの橋渡し機能

```rust
// mdarray_compat.rs で実装済み
pub fn array1_to_tensor<T: Clone>(arr: &Array1<T>) -> Tensor<T, (usize,)>
pub fn tensor_to_array1<T: Clone>(tensor: &Tensor<T, (usize,)>) -> Array1<T>
pub fn array2_to_tensor<T: Clone>(arr: &Array2<T>) -> Tensor<T, (usize, usize)>
pub fn tensor_to_array2<T: Clone>(tensor: &Tensor<T, (usize, usize)>) -> Array2<T>
```

使用例：
```rust
// SVEResult (ndarray版) から Tensor版へ
let s_tensor = mdarray_compat::array1_to_tensor(&sve_result.s);

// xprec-svdの結果をTensorで受け取る
let (u_nd, s_nd, v_nd) = xprec_svd::jacobi_svd(...);
let s_tensor = mdarray_compat::array1_to_tensor(&s_nd);
```

### 次のアクション

1. ✅ **Phase 8**: 全テスト実行・確認
2. 🔄 **Phase 9**: Faerベンチマーク実施
3. 🤔 **Phase 10**: 統合戦略決定

推定残り時間: **4時間**（当初予想の1/3）

---
**作成日**: 2025-01-08
**Phase 7見直し後の計画**
