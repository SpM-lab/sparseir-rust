# xprec-svd mdarray移行：作業中（WIP）

**ブランチ**: remove-ndarray  
**状態**: 作業中（進捗15%）  
**推定残り時間**: 5-6時間

## 完了した作業

### ✅ Import文更新（全ファイル）
- svd/jacobi.rs ✅
- qr/rrqr.rs ✅
- qr/householder.rs ✅
- qr/truncate.rs ✅
- tsvd.rs ✅
- utils/pivoting.rs ✅
- utils/validation.rs ✅
- utils/norms.rs ✅

### ✅ jacobi.rs 部分移行（30%）
- SVDResult構造体 ✅
- jacobi_svd signature ✅
- apply_givens_left ✅
- apply_givens_right ✅
- QR前処理 ✅
- 行列スケーリング ✅
- 結果抽出・ソート ✅
- テスト更新 ✅

## ❌ 未完了（残り作業）

### 各ファイルの内部実装移行

**規模**: 各ファイルで10-50箇所の置換

必要な置換：
1. **型シグネチャ**:
   - `Array2<T>` → `Tensor<T, (usize, usize)>` (~50箇所)
   - `Array1<T>` → `Tensor<T, (usize,)>` (~30箇所)
   - `ArrayView1<T>` → `&Tensor<T, (usize,)>` (~15箇所)
   - `ArrayView2<T>` → `&Tensor<T, (usize, usize)>` (~10箇所)
   - `ArrayViewMut*` → `&mut Tensor<...>` (~10箇所)

2. **メソッド呼び出し**:
   - `.nrows()` → `.shape().0` (~40箇所)
   - `.ncols()` → `.shape().1` (~40箇所)
   - `Array*::zeros((m,n))` → `Tensor::from_elem((m,n), T::zero())` (~20箇所)
   - `Array*::from_iter` → `Tensor::from_fn` (~10箇所)

3. **スライシング（最大の問題）**:
   - `matrix.slice(s![a..b, c..d])` → 手動コピー (~20箇所)
   - `matrix.slice_mut(s![...])` → 手動アクセスパターン (~15箇所)
   - `.row(i)`, `.column(j)` → 手動抽出 (~10箇所)
   - `.row_mut(i)`, `.column_mut(j)` → 手動ループ (~10箇所)

4. **その他**:
   - `.assign()` → 手動コピー (~5箇所)
   - `.view()` → スライス関数 (~5箇所)

**推定総数**: ~250箇所の置換

## 現在のエラー状況

```
error[E0412]: cannot find type `Array2` in this scope (×50)
error[E0412]: cannot find type `Array1` in this scope (×30)
error[E0412]: cannot find type `ArrayView*` in this scope (×20)
error[E0599]: no method named `.nrows` found (×20)
...etc
```

**総エラー数**: 66個

## 作業見積もり

| ファイル | 残り箇所 | 推定時間 |
|---------|---------|----------|
| qr/rrqr.rs | ~40 | 1-1.5h |
| qr/householder.rs | ~50 | 1.5-2h |
| qr/truncate.rs | ~15 | 30min |
| tsvd.rs | ~40 | 1-1.5h |
| utils/pivoting.rs | ~20 | 30min |
| utils/validation.rs | ~30 | 1h |
| utils/norms.rs | ~15 | 30min |
| テスト修正 | - | 30min |

**合計推定**: 5-6時間

## 判断ポイント

### 継続する場合
- ブランチ remove-ndarray で作業継続
- 6時間の集中作業でxprec-svdを完全mdarray化
- 利点: クリーンな実装、ndarray完全削除

### 一旦保留する場合
- 現在の変更をstash/commit
- mainブランチに戻る
- ハイブリッド運用を継続（ndarray + mdarray）
- 利点: 既に動作している実装を維持

## 推奨

**時間がある**: 継続（あと5-6時間）
**優先度が変わった**: mainに戻ってRegularizedBoseKernel実装へ

---
**現在のコミット**: 未保存（66 errors）  
**次のアクション**: ユーザーの判断待ち
