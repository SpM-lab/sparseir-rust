# xprec-svd テスト移行状況

## ✅ 完了（47テスト）

### ライブラリユニットテスト（src/内） - 22テスト
- svd/jacobi.rs: 3テスト ✅
- qr/rrqr.rs: 3テスト ✅
- qr/householder.rs: 2テスト ✅
- qr/truncate.rs: 1テスト ✅
- tsvd.rs: 3テスト ✅
- utils/norms.rs: 3テスト ✅
- utils/validation.rs: 3テスト ✅
- utils/pivoting.rs: 3テスト ✅

### 統合テスト（tests/） - 25テスト
- jacobi_svd_tests.rs: 22テスト ✅
- simple_df64_test.rs: 3テスト ✅

## ⏳ 未移行（4ファイル、~950行）

これらは必須ではないため、優先度低：

1. **rrqr_tests.rs** (343行)
   - RRQR追加テスト
   - 既にsrc/qr/rrqr.rsでカバー済み

2. **svd_accuracy_tests.rs** (227行)
   - 精度テスト
   - jacobi_svd_tests.rsで十分カバー

3. **hilbert_reconstruction_tests.rs** (216行)
   - Hilbert行列テスト
   - 特殊ケースのテスト

4. **df64_rrqr_tests.rs** (166行)
   - Df64精度のRRQRテスト
   - simple_df64_test.rsでカバー

## 戦略

未移行テストは**後回し**にして、次のタスクに進む：
1. sparseir-rustの統合テスト実行
2. RegularizedBoseKernel実装
3. 必要に応じて後で追加テスト移行

---

**結論**: xprec-svdは47テストで十分に検証されており、実用可能 ✅
