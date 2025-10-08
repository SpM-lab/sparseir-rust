# ndarray → mdarray 移行ステータス

**最終更新**: 2025-01-08

## ✅ 完了項目

### Phase 1-2: 準備完了
- ✅ Cargo.toml更新（mdarray + mdarray-linalg-faer追加）
- ✅ Rust 1.87 (edition 2024対応済み)
- ✅ mdarray, mdarray-linalg-faerのビルド確認

### Phase 3: SVEResult移行完了 🎉
- ✅ `result_mdarray.rs`作成
- ✅ `Array1<f64>` → `mdarray::Tensor<f64, (usize,)>` 変換
- ✅ 全テストパス (3/3)

#### 主な変更点

**型定義**:
```rust
// Before (ndarray)
pub s: Array1<f64>

// After (mdarray)
pub s: DTensor<f64, 1>  // = Tensor<f64, (usize,)>
```

**配列アクセス**:
```rust
// Before (ndarray)
self.s[0]
self.s.slice(s![..cut]).to_owned()

// After (mdarray)
self.s[[0]]
Tensor::from_fn((cut,), |idx| self.s[[idx[0]]])
```

**テスト結果**:
```
test sve::result_mdarray::tests::test_part_default_epsilon ... ok
test sve::result_mdarray::tests::test_part_with_max_size ... ok
test sve::result_mdarray::tests::test_part_with_threshold ... ok

test result: ok. 3 passed; 0 failed
```

## 🔄 進行中

- Phase 4: gauss.rs (30箇所) の移行準備中

## 📝 学んだこと

### mdarrayの特徴

1. **型アノテーション必要**:
   - `from_fn`はIntoShape traitの制約が厳しい
   - タプル形式 `(usize,)` が最も安定

2. **インデックスアクセス**:
   - `arr[i]` → `arr[[i]]`（配列で囲む）
   - `arr[[0]]` for 1D, `arr[[i, j]]` for 2D

3. **生成関数**:
   - `from_vec` は存在しない
   - `from_fn` を使用：`Tensor::from_fn((n,), |idx| ...)`

4. **イテレーション**:
   - `iter()` はそのまま使える
   - スライシングは `.view()` メソッド

### Faerバックエンド

- ビルド成功（15 warnings, 主に未使用変数）
- Pure Rust実装のため外部依存なし
- 次のPhaseでSVD機能をテスト予定

## 📊 統計

| 指標 | 値 |
|------|-----|
| 移行完了モジュール | 1/12 |
| 移行完了関数 | 12/126 |
| テスト成功率 | 100% (3/3) |
| ビルド時間 | ~2秒 |

## 🎯 次のステップ

Phase 4に進む予定：
- gauss.rs (30箇所)
- ルジャンドル多項式計算
- より複雑な配列操作のテスト

---
**移行ドキュメント**: `MIGRATION_NDARRAY_TO_MDARRAY.md`

### Phase 4: gauss.rs移行完了 🎉
- ✅ `gauss_mdarray.rs`作成
- ✅ `Rule<T>` 構造体の全フィールドを`Tensor<T, (usize,)>`に変換
- ✅ 全テストパス (3/3)

#### 主な変更点

**型定義**:
```rust
// Before (ndarray)
pub x: Array1<T>
pub w: Array1<T>

// After (mdarray)  
pub x: Tensor<T, (usize,)>
pub w: Tensor<T, (usize,)>
```

**mapv変換**:
```rust
// Before (ndarray)
self.x.mapv(|xi| xi - a)

// After (mdarray)
Tensor::from_fn((n,), |idx| self.x[[idx[0]]] - a)
```

**テスト結果**:
```
test gauss_mdarray::tests::test_rule_creation ... ok
test gauss_mdarray::tests::test_rule_reseat ... ok
test gauss_mdarray::tests::test_rule_scale ... ok

test result: ok. 3 passed; 0 failed
```

## 📊 統計（更新）

| 指標 | 値 |
|------|-----|
| 移行完了モジュール | 2/12 |
| 移行完了関数 | ~45/126 |
| テスト成功率 | 100% (6/6) |
| ビルド時間 | ~2秒 |

---
**最終更新**: 2025-01-08 (Phase 4完了)
