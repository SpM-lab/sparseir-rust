//! ユニット構造体のデモ

use sparseir_rust::*;

fn main() {
    println!("=== ユニット構造体のデモ ===\n");
    
    // 1. ユニット構造体の作成
    println!("1. ユニット構造体の作成");
    let fermionic = Fermionic;
    let bosonic = Bosonic;
    
    println!("   Fermionic: {:?}", fermionic);
    println!("   Bosonic: {:?}", bosonic);
    
    // 2. メモリサイズ
    println!("\n2. メモリサイズ");
    println!("   Fermionic size: {} bytes", std::mem::size_of::<Fermionic>());
    println!("   Bosonic size: {} bytes", std::mem::size_of::<Bosonic>());
    println!("   Statistics enum size: {} bytes", std::mem::size_of::<Statistics>());
    
    // 3. 型情報
    println!("\n3. 型情報");
    println!("   Fermionic TypeId: {:?}", std::any::TypeId::of::<Fermionic>());
    println!("   Bosonic TypeId: {:?}", std::any::TypeId::of::<Bosonic>());
    
    // 4. 定数値
    println!("\n4. 定数値");
    println!("   Fermionic::STATISTICS: {:?}", Fermionic::STATISTICS);
    println!("   Bosonic::STATISTICS: {:?}", Bosonic::STATISTICS);
    
    // 5. カーネルでの使用
    println!("\n5. カーネルでの使用");
    let kernel = LogisticKernel::new(10.0);
    
    // 型パラメータとして使用
    let w_fermi = kernel.weight::<Fermionic>(1.0, 1.0);
    let w_bose = kernel.weight::<Bosonic>(1.0, 1.0);
    
    println!("   Fermionic weight: {}", w_fermi);
    println!("   Bosonic weight: {}", w_bose);
    
    // 6. パターンマッチング
    println!("\n6. パターンマッチング");
    match Fermionic::STATISTICS {
        Statistics::Fermionic => println!("   Fermionic statistics detected"),
        Statistics::Bosonic => println!("   Bosonic statistics detected"),
    }
    
    match Bosonic::STATISTICS {
        Statistics::Fermionic => println!("   Fermionic statistics detected"),
        Statistics::Bosonic => println!("   Bosonic statistics detected"),
    }
    
    // 7. 比較
    println!("\n7. 比較");
    println!("   Fermionic == Fermionic: {}", fermionic == fermionic);
    println!("   Fermionic == Bosonic: {}", fermionic == bosonic);
    println!("   Bosonic == Bosonic: {}", bosonic == bosonic);
    
    // 8. ハッシュ値
    println!("\n8. ハッシュ値");
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    fermionic.hash(&mut hasher);
    println!("   Fermionic hash: {}", hasher.finish());
    
    let mut hasher = DefaultHasher::new();
    bosonic.hash(&mut hasher);
    println!("   Bosonic hash: {}", hasher.finish());
    
    println!("\n=== ユニット構造体の利点 ===");
    println!("1. ゼロサイズ: メモリ効率が良い");
    println!("2. 型安全性: コンパイル時に型を区別");
    println!("3. パフォーマンス: 実行時オーバーヘッドなし");
    println!("4. 柔軟性: トレイト実装で様々な動作を定義");
}
