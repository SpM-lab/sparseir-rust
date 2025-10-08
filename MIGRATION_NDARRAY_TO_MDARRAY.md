# ndarray → mdarray 移行計画

## 概要

sparseir-rustをndarrayからmdarrayとmdarray-linalgに移行する。

## 理由

1. **Faerバックエンド優先**: Pure Rust実装、外部BLAS/LAPACK不要、高性能（SIMD最適化）
2. **クロスプラットフォーム対応**: 外部依存なしでどの環境でもビルド可能
3. **モダンなAPI設計**: Rust 2024 editionベース、より直感的なAPI
4. **パフォーマンス最適化**: SIMDとキャッシュ効率の最適化（faer-rsの実績）
5. **将来性**: 活発に開発中、C++ mdarray/mdspanからのインスパイア

## 現状分析

### ndarray使用箇所（2025-01-08時点）

| ファイル | Array1/2使用 | ndarray特有API | 優先度 |
|---------|-------------|----------------|--------|
| sve/result.rs | 12 | slice | **高** |
| gauss.rs | 30 | mapv, axis | **高** |
| kernelmatrix.rs | 11 | from_shape_fn | 中 |
| sve/utils.rs | 9 | slice, axis | 中 |
| sve/compute.rs | 17 | - | 中 |
| sve/strategy.rs | 16 | - | 中 |
| basis.rs | 3 | mapv | 低 |
| poly.rs | 12 | mapv | 低 |
| interpolation*.rs | 9+8 | - | 低 |
| numeric.rs | 4 | - | 低 |

**合計**: ~126箇所

## API対応表

### 基本型

| ndarray | mdarray | 備考 |
|---------|---------|------|
| `Array1<T>` | `DTensor<T, 1>` | 動的1次元配列 |
| `Array2<T>` | `DTensor<T, 2>` | 動的2次元配列 |
| `ArrayView1<T>` | `DSlice<T, 1, L>` | 1次元ビュー |
| `ArrayView2<T>` | `DSlice<T, 2, L>` | 2次元ビュー |
| `ArrayViewMut1<T>` | `DSlice<T, 1, L>` (mut) | 可変1次元ビュー |

### 生成

| ndarray | mdarray | 例 |
|---------|---------|-----|
| `Array1::zeros(n)` | `DTensor::<T, 1>::zeros([n])` | ゼロ配列 |
| `Array2::zeros((m, n))` | `DTensor::<T, 2>::zeros([m, n])` | ゼロ行列 |
| `Array1::from_vec(v)` | `DTensor::from_vec([n], v)` | ベクタから生成 |
| `array![...]` | `tensor![...]` | マクロ |
| `Array2::from_shape_fn((m,n), f)` | `DTensor::from_fn([m, n], f)` | 関数から生成 |

### 操作

| ndarray | mdarray | 備考 |
|---------|---------|------|
| `arr.mapv(f)` | `arr.expr().map(f).eval()` | 要素ごと変換 |
| `arr.slice(s![..n])` | `arr.view(.., ..n)` | スライス |
| `arr[[i, j]]` | `arr[[i, j]]` | インデックス（同じ） |
| `arr.iter()` | `arr.iter()` | イテレータ（同じ） |
| `arr.axis_iter(Axis(0))` | `arr.rows()` | 行イテレータ |
| `arr.axis_iter(Axis(1))` | `arr.cols()` | 列イテレータ |
| `arr.t()` | `arr.t()` | 転置（要確認） |

### 線形代数（新規：mdarray-linalg with Faer）

| 機能 | ndarray-linalg | mdarray-linalg (Faer優先) | バックエンド |
|------|---------------|---------------------------|--------------|
| SVD | `arr.svd()` | `Faer.svd(&mut arr)` | **Faer**, LAPACK |
| 固有値 | `arr.eig()` | `Faer.eig(&mut arr)` | **Faer**, LAPACK |
| 行列積 | `arr.dot(&b)` | `Faer.matmul(&arr, &b).eval()` | **Faer**, BLAS |
| QR分解 | `arr.qr()` | `Faer.qr(&mut arr)` | **Faer**, LAPACK |
| LU分解 | `arr.lu()` | `Faer.lu(&mut arr)` | **Faer**, LAPACK |
| 線形方程式 | `arr.solve(&b)` | `Faer.solve(&mut arr, &b)` | **Faer**, LAPACK |

## 移行手順

### Phase 1: 調査と準備（完了）✅

- [x] mdarrayとmdarray-linalgのsubmodule追加確認
- [x] API対応表の作成
- [x] 使用箇所の分析（126箇所特定）
- [x] 移行計画の策定

### Phase 2: 依存関係の更新（Faer優先）

```toml
[dependencies]
# mdarray (GitHub版を使用 - crates.io版は古い)
mdarray = { path = "../mdarray" }

# mdarray-linalg with Faer backend (Pure Rust実装)
mdarray-linalg = { path = "../mdarray-linalg/mdarray-linalg" }
mdarray-linalg-faer = { path = "../mdarray-linalg/mdarray-linalg-faer" }

# Faer本体（高性能線形代数）
faer = "0.23"

# オプション: BLAS/LAPACKバックエンド（パフォーマンス比較用）
# mdarray-linalg-blas = { path = "../mdarray-linalg/mdarray-linalg-blas" }
# mdarray-linalg-lapack = { path = "../mdarray-linalg/mdarray-linalg-lapack" }
# openblas-src = { version = "0.10", features = ["system"] }

# ndarray (段階的移行のため一時的に併存)
# ndarray = "0.15"  # 最終的には削除
```

**Faerを優先する理由**:
- ✅ Pure Rust実装（外部BLAS/LAPACK不要）
- ✅ クロスプラットフォーム（macOS/Linux/Windows）
- ✅ 高性能（SIMD最適化、並列処理対応）
- ✅ ビルド時間短縮（システムライブラリリンク不要）
- ✅ 依存関係のシンプル化

### Phase 3: 小規模モジュールでの試験移行

**対象**: `sve/result.rs` (12箇所、比較的独立)

1. `Array1<f64>` → `DTensor<f64, 1>`
2. `arr.slice(s![..n])` → `arr.view(.., ..n)`
3. テスト実行と修正
4. パフォーマンス測定

### Phase 4: gauss.rsの移行

**対象**: ガウス求積（30箇所、重要度高）

1. `Array1::from_shape_fn` → `DTensor::from_fn`
2. `mapv` → `expr().map().eval()`
3. ルジャンドル多項式計算の検証

### Phase 5: カーネル行列の移行

**対象**: `kernelmatrix.rs` (11箇所)

1. `Array2::from_shape_fn` → `DTensor::from_fn`
2. 行列生成ロジックの移行

### Phase 6: SVE計算周辺の移行

**対象**: `sve/utils.rs`, `sve/compute.rs`

⚠️ **注意**: xprec-svdとの統合問題
- 現在xprec-svdはndarrayベース
- オプション1: xprec-svdにmdarray対応を追加
- オプション2: SVD部分のみndarray←→mdarray変換層を挟む

### Phase 7: 残りのモジュール

**対象**: `basis.rs`, `poly.rs`, `interpolation*.rs`

順次移行、各モジュール移行後にテスト実行

### Phase 8: テスト修正

全テストケースの修正：
- インポート文の更新
- アサーションの調整
- 数値精度の検証

### Phase 9: パフォーマンス検証

ベンチマーク実施（Faer vs ndarray-linalg）：
- lambda=10, 100, 1000でのSVE計算時間
- メモリ使用量
- SVD精度の比較
- 並列化効果の測定（Faerのマルチスレッド対応）
- オプション: BLAS/LAPACKバックエンドとの比較

### Phase 10: クリーンアップ

- ndarray依存の削除
- ドキュメント更新
- Cargo.tomlの整理

## 課題と懸念事項

### 1. xprec-svd統合

**問題**: xprec-svdがndarrayベース

**解決策の選択肢**:
- A. xprec-svdをフォークしてmdarray対応版を作成
- B. 変換層を実装（ndarray ↔ mdarray）
- C. SVD部分だけmdarray-linalg-lapackを使用（精度要確認）

### 2. edition 2024要件

**問題**: mdarrayがRust 1.85 + edition 2024を要求

**解決策**:
- Rustツールチェーンを1.85以上に更新
- Cargo.tomlで`edition = "2024"`に変更

### 3. まだcrates.io未公開

**問題**: mdarray-linalgがcrates.io未公開

**解決策**:
- path依存で開発継続
- 将来的にsubmoduleまたはgit依存に切り替え

### 4. 既存のテストデータ

**問題**: Julia生成のテストデータとの互換性

**解決策**:
- 数値精度の検証を厳密に実施
- 必要に応じてテスト許容誤差を調整

## 推奨アプローチ

### 短期（1-2週間）

1. **Phase 2-3を実施**: 依存関係更新 + sve/result.rs移行
2. **動作確認**: 基本的な機能が動くことを確認
3. **問題点の洗い出し**: xprec-svd統合の方針決定

### 中期（1ヶ月）

1. **Phase 4-6を実施**: コア機能の移行
2. **テスト修正**: 主要なテストが通るように修正
3. **パフォーマンス測定**: ボトルネック特定

### 長期（2-3ヶ月）

1. **Phase 7-10を完了**: 全モジュール移行
2. **最適化**: BLAS/LAPACKバックエンドの最適化
3. **ドキュメント**: 移行ガイドと新API説明

## 次のステップ

**即座に実行可能**:

```bash
# 1. Rustツールチェーン確認
rustc --version  # 1.85以上が必要

# 2. edition 2024への更新
# Cargo.tomlで edition = "2024" に変更

# 3. 試験的なビルド
cd mdarray && cargo build
cd ../mdarray-linalg && cargo build

# 4. Phase 3の開始
# sve/result.rs の移行を試みる
```

---

**作成日**: 2025-01-08
**最終更新**: 2025-01-08

