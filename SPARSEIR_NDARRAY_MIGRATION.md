# sparseir-rust ndarray削除計画

## 現状分析

ndarray使用ファイル（12ファイル）：
1. src/mdarray_compat.rs - ndarray↔mdarray変換（削除対象）
2. src/sve/compute.rs - SVE計算
3. src/sve/utils.rs - SVEユーティリティ
4. src/basis.rs - 基底クラス
5. src/poly.rs - 多項式
6. src/sve/strategy.rs - SVE戦略
7. src/kernelmatrix.rs - カーネル行列
8. src/numeric.rs - 数値計算
9. src/sve/result.rs - SVE結果
10. src/interpolation2d.rs - 2D補間
11. src/gauss.rs - Gauss求積
12. src/interpolation1d.rs - 1D補間

## 戦略

### Phase 1: mdarray版モジュールを優先使用
- gauss_mdarray.rs → gauss.rs に統合
- kernelmatrix_mdarray.rs → kernelmatrix.rs に統合
- sve/result_mdarray.rs → sve/result.rs に統合

### Phase 2: 残りのndarray使用箇所を変換
- poly.rs
- interpolation*.rs
- numeric.rs
- basis.rs

### Phase 3: mdarray_compat.rs削除

推定時間: 2-3時間
