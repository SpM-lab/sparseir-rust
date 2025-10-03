//! ライフタイム境界のデモ

use sparseir_rust::*;

// 'staticライフタイムを持つ型
struct MyStaticType;

// 'staticライフタイムを持たない型
struct MyNonStaticType<'a> {
    data: &'a str,
}

// テスト用のトレイト
trait MyTrait {}

impl MyTrait for MyStaticType {}
impl<'a> MyTrait for MyNonStaticType<'a> {}

fn demonstrate_type_id() {
    println!("=== TypeId::of のデモ ===");
    
    // 'static型はOK
    let id1 = std::any::TypeId::of::<MyStaticType>();
    println!("MyStaticType ID: {:?}", id1);
    
    // 関数内で'static型を作成してテスト
    fn test_static_type() {
        let id = std::any::TypeId::of::<MyStaticType>();
        println!("Static type ID in function: {:?}", id);
    }
    test_static_type();
    
    // 'static境界がない場合のエラー例（コメントアウト）
    /*
    fn test_non_static_type<'a>() {
        // これはコンパイルエラーになる
        let id = std::any::TypeId::of::<MyNonStaticType<'a>>();
    }
    */
    
    println!("'static境界が必要な理由:");
    println!("- 型情報はプログラム全体で一意である必要がある");
    println!("- 参照を含む型は実行時に生存期間が変わる");
    println!("- TypeId::ofは型の一意性を保証する必要がある");
}

fn demonstrate_kernel_usage() {
    println!("\n=== カーネルでの使用例 ===");
    
    let kernel = LogisticKernel::new(10.0);
    
    // これらは'static型なので正常に動作
    let w_fermi = kernel.weight::<Fermionic>(1.0, 1.0);
    let w_bose = kernel.weight::<Bosonic>(1.0, 1.0);
    
    println!("Fermionic weight: {}", w_fermi);
    println!("Bosonic weight: {}", w_bose);
    
    println!("Fermionic と Bosonic は 'static 型:");
    println!("- プログラム全体で生存");
    println!("- 参照を含まない");
    println!("- TypeId::of で安全に型情報を取得可能");
}

fn main() {
    demonstrate_type_id();
    demonstrate_kernel_usage();
    
    println!("\n=== まとめ ===");
    println!("'+'static' は以下を保証:");
    println!("1. 型がプログラム全体で生存");
    println!("2. 参照を含まない");
    println!("3. TypeId::of で安全に使用可能");
    println!("4. 型レベルでの安全性");
}
