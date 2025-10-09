# xprec-svd mdarray移行の現状

## ⚠️ 大規模移行作業中

### 規模

- **総行数**: 2200行
- **ファイル数**: 14ファイル
- **ndarray使用**: 95箇所
- **推定作業時間**: 6-8時間

### 現在の進捗（5%）

✅ Cargo.toml: mdarray必須化
✅ lib.rs: 型エイリアス更新
🔄 jacobi.rs: 10% (SVDResult, jacobi_svd signature, apply_givens_*)
❌ 残り13ファイル: 未着手

### エラー状況

現在のエラー: 23個
- qr/*.rs からのunresolved import
- jacobi.rs内の未移行関数呼び出し
- Array型の不一致

### 作業戦略の選択

#### A. 完全移行を継続（6-8時間）✅

**現在採用中**

進行手順：
1. jacobi.rs完全移行（2-3h）
2. qr/*.rs移行（2-3h）
3. utils/*.rs移行（1-2h）
4. tsvd.rs移行（30min）
5. テスト修正（1h）

#### B. 一時的にndarray復帰（5分）

```bash
git checkout main
git branch -D remove-ndarray
```

ハイブリッド運用を維持し、将来的に移行。

#### C. ラッパーアプローチ（1-2時間）

xprec-svdはndarray維持、sparseir-rust側で変換層。

## 推奨

**時間がある場合**: オプションA継続
**急ぐ場合**: オプションB（mainに戻る）

現在は remove-ndarray ブランチで作業中。
commit してから判断可能。

