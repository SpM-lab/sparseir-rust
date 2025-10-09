# xprec-svd ndarray削除戦略

## 現状分析

xprec-svdのndarray使用箇所：
- svd/jacobi.rs: 16箇所（487行）
- qr/rrqr.rs: 12箇所
- qr/householder.rs: 14箇所
- qr/truncate.rs: 5箇所
- tsvd.rs: 14箇所
- utils/pivoting.rs: 9箇所
- utils/validation.rs: 18箇所
- utils/norms.rs: 6箇所

**合計**: 95箇所、~2200行

## 戦略オプション

### オプションA: 段階的移行（推奨）

**手順**:
1. ndarray-backendとmdarray-backendを両方サポート（現状）
2. jacobi.rsを条件付きコンパイルで両対応
3. 各ファイルを順次移行
4. 全テストパス後にndarray-backend削除

**利点**: 常にビルド可能、段階的検証
**所要時間**: 6-8時間

### オプションB: 一括移行（高速）

**手順**:
1. 全ファイルを一度にmdarrayに置き換え
2. コンパイルエラーを一括修正
3. テスト修正

**利点**: 早い
**デメリット**: リスク高、デバッグ困難
**所要時間**: 4-6時間（ただしデバッグで延長の可能性）

### オプションC: sparseir-rust側でラッパー作成

**手順**:
1. sparseir-rust/src/sve/ に xprec_svd_mdarray.rs 作成
2. mdarray → ndarray変換してxprec-svd呼び出し
3. 結果をndarray → mdarrayに変換

**利点**: xprec-svd無変更、影響範囲小
**デメリット**: 変換オーバーヘッド
**所要時間**: 1-2時間

## 推奨：オプションA（段階的移行）

### フェーズ1: jacobi.rsの完全移行（2-3時間）

最重要ファイル。以下を置き換え：
- `Array2<T>` → `Tensor<T, (usize, usize)>`
- `Array1<T>` → `Tensor<T, (usize,)>`
- `.nrows()` → `.shape().0`
- `.ncols()` → `.shape().1`
- `Array2::zeros((m, n))` → `Tensor::from_elem((m, n), T::zero())`
- `Array2::eye(n)` → `Tensor::from_fn((n, n), |idx| ...)`
- `s![a..b, c..d]` → 手動スライシング関数
- `.row_mut(i)` → 手動アクセス
- `.column_mut(j)` → 手動アクセス

### フェーズ2: qr/*.rs移行（2-3時間）

- qr/rrqr.rs
- qr/householder.rs
- qr/truncate.rs

### フェーズ3: utils/*.rs, tsvd.rs移行（1-2時間）

- utils/*
- tsvd.rs

### フェーズ4: ndarray-backend削除（30分）

- feature flag削除
- ndarray依存削除
- テスト確認

---
**推定総時間**: 6-8時間
**開始**: jacobi.rs
