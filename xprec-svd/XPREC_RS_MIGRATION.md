# xprec-rs 置換計画

## 目的とスコープ
- `xprec-svd` が高精度演算用に依存している `twofloat` crate を `xprec-rs` の `Df64` 実装へ差し替え、将来的に libsparseir 系全体で共有できる純 Rust 拡張精度スタックを整備する。  
- 互換性維持は前提とせず、必要に応じて API 名称・型シグネチャ・feature 構成を整理し直す。  
- `xprec-rs` リポジトリ（同階層に存在）をソース依存として利用するが、上流公開 crate に切り替えられる構成を保つ。

## 現状整理（twofloat 利用箇所）
- `src/precision/mod.rs`：`TwoFloatPrecision` が `twofloat::TwoFloat` をラップし、`num_traits::Float` など多数のトレイトを手作業で実装。  
- `tsvd.rs`：公開 API `tsvd_twofloat`, `tsvd_twofloat_from_f64` が `TwoFloatPrecision` を通じて二重倍精度 SVD を提供。  
- テスト群（`tests/*twofloat*.rs`, `tests/simple_twofloat_test.rs`）と README サンプルが直接 `TwoFloatPrecision` を作成。  
- Cargo 依存：`twofloat = "0.2"` を直接宣言。  
- `sparseir-rust` は `xprec-svd` をパス依存し TwoFloat SVD を利用しているが、今回の移行で API 刷新やリネームを許容する。

## xprec-rs の特徴とギャップ
- `Df64` は `libxprec` に準拠した double-double 実装で、算術・初等関数・Gauss 求積などを完備。  
- 既に `Zero`, `One`, `Num`, `Signed`, `RealField` などを実装する一方、`num_traits::Float` は提供されない。  
- `Df64` は `Df64::EPSILON` など定数を型に紐付け。`Precision` トレイト要件（`num_traits::Float` 依存）を調整する必要がある。  
- `Df64` は `Df64::from(f64)` / `Into<f64>` を提供、`TwoFloat` 互換 API (`new_full`, `hi/lo` 取得) は存在しない → 変換ユーティリティ整備が必要。  
- `xprec-rs` 側で `serde`、`no_std` 対応など追加機能が存在するが、本置換では最小限の API 利用に集中する。

## フェーズ別移行計画
### フェーズ 0: 事前準備
- `xprec-rs` の `Df64` が必要な演算（`sqrt`, `hypot`, `sin`, `exp`, `powi` など）をすべてカバーしていることを確認し、欠落があれば upstream issue を起票。  
- 依存関係レビュー：`Cargo.toml` に `xprec-rs = { path = "../xprec-rs" }` を追加し、`twofloat` の段階的削除と代替 API へのリネーム方針を固める。  
- `Precision` トレイト要件洗い出し (`Float` 依存箇所、`AbsDiffEq`、`NumCast` など) と `Df64` 側に必要な shims を列挙。

### フェーズ 1: 抽象層の差し替え
- `precision/` モジュールに `Df64Precision`（仮称）を追加し、`TwoFloatPrecision` の置換となる API (`from_f64`, `to_f64`, `epsilon`, `min_positive` など) を提供。  
- `Precision` トレイトの制約を `num_traits::Float` 依存から `Precision + RealField + Signed + NumCast` 等に再設計し、f64 / Df64 双方が満たす形に拡張。  
- 古い `TwoFloatPrecision` 実装は段階的に削除し、`Df64` バックエンドのみを残す。  
- 公開 API 名も必要に応じて整理（例: `tsvd_twofloat` → `tsvd_df64` / `tsvd_extended`）し、呼び出し側の改修ガイドを同時に用意。

### フェーズ 2: パブリック API / テスト更新
- 新 backend を有効化した状態でユニットテスト・統合テストを走らせ、`Df64` での数値差分（許容誤差）を評価。  
- `tests/` と `README` のサンプルコードを新しい API 名・型に合わせて全面更新し、TwoFloat 前提のコードや記述は削除。  
- `sparseir-rust` など依存クレートの `Cargo.toml`／呼び出しコードを一括更新するタスクを用意し、必要な破壊的変更点を明示する。

### フェーズ 3: クリーンアップと検証
- `Cargo.toml` から `twofloat` を除去し、関連ソースファイル・テスト・ドキュメントを整理。  
- ベンチマーク（`tests/` または将来的な `benches/`）で性能比較を行い、xprec-rs 化による速度/精度変化を記録。  
- ドキュメント（`README`, `PLAN.md`, `CODING_RULES.md` 該当節）を更新し、xprec-rs ベースの構成・制約・依存関係を最新版として提示。  
- 依存リポジトリ向けに破壊的変更リリースノートを作成し、必要ならメジャーバージョンアップを伴う公開を調整。

## 評価・テスト戦略
- `cargo test --all-targets` を `Df64` backend で常時通す CI ジョブを追加。  
- 重要なテストケース（ヒルベルト行列再構成、RRQR 精度比較）で `twofloat` と `Df64` の誤差ログを取り、精度回帰をチェック。  
- 将来的に `xprec-rs` の MPFR 参照テストを取り込む場合に備え、`tests/` へベースライン比較ユーティリティを追加。

## リスクと対策
- **トレイト互換性**：`num_traits::Float` が欠落 → `Precision` トレイトの再設計と `Df64` 向けラッパー実装で吸収。  
- **性能回帰**：`xprec-rs` の演算コストが未知 → ベンチ結果を `AUTO_MIGRATE.sh` 等に記録し、必要ならホットパスに最適化パッチを upstream へ提案。  
- **依存分散**：`xprec-rs` と `xprec-svd` が循環参照しないように注意し、将来 crates.io 公開時にバージョン整合戦略を準備。  
- **ドキュメント負債**：破壊的変更を伴うため、依存プロジェクトへのアナウンスと移行ガイドを整備しないと混乱が生じる。

## 未決事項 / 次のアクション
1. `xprec-rs` に `num_traits::Float` 相当の実装を追加するか、`xprec-svd` 側で抽象を調整するかの方針決定。  
2. `tsvd_twofloat` → `tsvd_df64`（仮）などの名称変更ポリシーを確定し、依存プロジェクトへの周知手順を策定。  
3. 破壊的変更に伴うバージョニング戦略（メジャーバンプ / release branch）と CI matrix 更新方針を決める。  
4. 上流共有を視野に入れた crate publish 手順（versioning, docs.rs, license 表記）を別ドキュメントで策定。
