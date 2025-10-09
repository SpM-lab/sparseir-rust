# ソート問題のデバッグ

## 出力結果
```
tau[0] = 0.5779494626
tau[1] = 0.7250037514
tau[2] = 0.8480393048
tau[3] = 0.9373226374
tau[4] = 0.9880318582
...
tau[9] = 0.4220505374  ← 最後が最小！
```

## C++実装の再確認が必要

### 疑問点
1. ソートのタイミングは？
2. 負の値変換の後に再ソート必要？

### 修正案

**Option A**: 負の値変換後に再ソート
```rust
for tau in &mut smpl_taus {
    if *tau < 0.0 {
        *tau += self.beta;
    }
}
// 再ソート追加
smpl_taus.sort_by(|a, b| a.partial_cmp(b).unwrap());
```

**Option B**: C++実装を精密に確認

---

Generated: 2025-10-09
Status: Debug needed
