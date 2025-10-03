//! 関連定数のデモ

use sparseir_rust::*;

// 独自の統計型を定義
struct MyCustomStatistics;

impl StatisticsType for MyCustomStatistics {
    const STATISTICS: Statistics = Statistics::Fermionic;
}

fn demonstrate_associated_constants() {
    println!("=== 関連定数のデモ ===\n");
    
    // 1. 直接アクセス
    println!("1. 直接アクセス");
    println!("   Fermionic::STATISTICS = {:?}", Fermionic::STATISTICS);
    println!("   Bosonic::STATISTICS = {:?}", Bosonic::STATISTICS);
    println!("   MyCustomStatistics::STATISTICS = {:?}", MyCustomStatistics::STATISTICS);
    
    // 2. ジェネリック関数での使用
    println!("\n2. ジェネリック関数での使用");
    print_statistics_type::<Fermionic>();
    print_statistics_type::<Bosonic>();
    print_statistics_type::<MyCustomStatistics>();
    
    // 3. パターンマッチング
    println!("\n3. パターンマッチング");
    match_statistics::<Fermionic>();
    match_statistics::<Bosonic>();
    
    // 4. カーネルでの実際の使用
    println!("\n4. カーネルでの実際の使用");
    let kernel = LogisticKernel::new(10.0);
    
    // これらの呼び出しで内部でS::STATISTICSが使用される
    let w_fermi = kernel.weight::<Fermionic>(1.0, 1.0);
    let w_bose = kernel.weight::<Bosonic>(1.0, 1.0);
    
    println!("   Fermionic weight: {}", w_fermi);
    println!("   Bosonic weight: {}", w_bose);
    
    // 5. 型推論との関係
    println!("\n5. 型推論との関係");
    demonstrate_type_inference();
}

// ジェネリック関数で関連定数を使用
fn print_statistics_type<S: StatisticsType>() {
    println!("   {}::STATISTICS = {:?}", 
        std::any::type_name::<S>(), 
        S::STATISTICS);
}

// パターンマッチングでの使用
fn match_statistics<S: StatisticsType>() {
    match S::STATISTICS {
        Statistics::Fermionic => {
            println!("   {} is Fermionic", std::any::type_name::<S>());
        }
        Statistics::Bosonic => {
            println!("   {} is Bosonic", std::any::type_name::<S>());
        }
    }
}

// 型推論のデモ
fn demonstrate_type_inference() {
    // コンパイル時に型が決定される
    let result1 = get_statistics_value::<Fermionic>();
    let result2 = get_statistics_value::<Bosonic>();
    
    println!("   Fermionic value: {}", result1);
    println!("   Bosonic value: {}", result2);
}

// 関連定数を使った関数
fn get_statistics_value<S: StatisticsType>() -> &'static str {
    match S::STATISTICS {
        Statistics::Fermionic => "fermionic",
        Statistics::Bosonic => "bosonic",
    }
}

// 関連定数と通常の値の比較
fn compare_approaches() {
    println!("\n=== 関連定数 vs 通常の値 ===");
    
    // 関連定数（コンパイル時）
    println!("関連定数（コンパイル時決定）:");
    println!("  Fermionic::STATISTICS = {:?}", Fermionic::STATISTICS);
    
    // 通常の値（実行時）
    println!("通常の値（実行時決定）:");
    let runtime_stats = Statistics::Fermionic;
    println!("  runtime_stats = {:?}", runtime_stats);
    
    // パフォーマンス比較
    println!("\nパフォーマンス:");
    println!("  関連定数: コンパイル時に最適化される");
    println!("  通常の値: 実行時に評価される");
}

fn main() {
    demonstrate_associated_constants();
    compare_approaches();
    
    println!("\n=== まとめ ===");
    println!("S::STATISTICS は:");
    println!("1. 型Sに関連付けられた定数");
    println!("2. コンパイル時に値が決定される");
    println!("3. ジェネリック関数で型情報を取得");
    println!("4. パターンマッチングで型に応じた処理");
    println!("5. 実行時オーバーヘッドなし");
}
