# xprec-svd ndarray削除 - 状況まとめ

## ✅ 完全達成

### xprec-svd本体
- ✅ ndarray依存完全削除
- ✅ mdarrayのみで動作
- ✅ 47テストパス（ライブラリ22 + 統合25）
- ✅ apply_givens_left バグ修正完了

### 重要な発見とバグ修正

**クリティカルバグ**: `apply_givens_left`の公式が間違っていた
```rust
// ❌ 間違い（mdarray移行時に混入）
let new_xi = c * xi - s * yi;
let new_yi = s * xi + c * yi;

// ✅ 正しい（ndarray版と同じ）
let new_xi = c * xi + s * yi;
let new_yi = -s * xi + c * yi;
```

このバグにより：
- 余分なJacobi iterationが発生（3回 vs 2回）
- U行列の符号が間違っていた
- 復元テストが失敗していた

## ⚠️ sparseir-rust側の問題

sparseir-rustが**まだndarrayを使用**しているため：
- poly.rs
- interpolation*.rs
- その他のモジュール

### オプション

#### A. sparseir-rustもmdarray完全移行
- 時間: 3-4時間
- 全ndarrayをmdarrayに置換

#### B. sparseir-rustにndarray依存を追加
- 時間: 1分
- Cargo.tomlにndarray追加
- ハイブリッド運用継続

#### C. 以前のハイブリッドブランチ（main）に戻る
- mainブランチは既に動作確認済み
- ndarray + mdarray ハイブリッド

## 推奨

**オプションB**: sparseir-rustのCargo.tomlにndarray追加

理由：
1. xprec-svd（内部依存）はmdarray完全移行済み ✅
2. sparseir-rust本体のndarray使用は限定的
3. 段階的移行が安全

次のタスク：
1. sparseir-rustにndarray追加
2. テスト実行
3. RegularizedBoseKernel実装へ

