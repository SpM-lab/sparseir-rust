//! Rust vs C++ テンプレートの違いを説明

use sparseir_rust::*;

// === Rustの制限と解決策 ===

// 1. コンパイル時型判定の制限
trait MyTrait {
    const VALUE: i32;
}

struct TypeA;
impl MyTrait for TypeA {
    const VALUE: i32 = 10;
}

struct TypeB;
impl MyTrait for TypeB {
    const VALUE: i32 = 20;
}

// これができない理由：実行時には型情報が消える
fn bad_example<T: MyTrait>(x: i32) -> i32 {
    // if T::VALUE == 10 {  // コンパイル時に決定できない
    //     x * 2
    // } else {
    //     x * 3
    // }
    x // 仮の実装
}

// 解決策1: トレイトメソッドを使用
trait ComputeTrait {
    fn compute(&self, x: i32) -> i32;
}

impl ComputeTrait for TypeA {
    fn compute(&self, x: i32) -> i32 {
        x * 2  // TypeA固有の処理
    }
}

impl ComputeTrait for TypeB {
    fn compute(&self, x: i32) -> i32 {
        x * 3  // TypeB固有の処理
    }
}

// 解決策2: 定数を使った分岐
fn good_example<T: MyTrait>(x: i32) -> i32 {
    match T::VALUE {
        10 => x * 2,
        20 => x * 3,
        _ => x,
    }
}

// === カーネルの場合の解決策 ===

// 解決策1: トレイトメソッド（推奨）
trait WeightComputation {
    fn compute_weight(&self, beta: f64, omega: f64) -> f64;
}

impl WeightComputation for Fermionic {
    fn compute_weight(&self, _beta: f64, _omega: f64) -> f64 {
        1.0
    }
}

impl WeightComputation for Bosonic {
    fn compute_weight(&self, _beta: f64, omega: f64) -> f64 {
        1.0 / (0.5 * _beta * omega).tanh()
    }
}

// 解決策2: 定数を使った分岐
fn weight_by_const<S: StatisticsType>(beta: f64, omega: f64) -> f64 {
    match S::STATISTICS {
        Statistics::Fermionic => 1.0,
        Statistics::Bosonic => 1.0 / (0.5 * beta * omega).tanh(),
    }
}

// 解決策3: 現在の実装（TypeId使用）
fn weight_by_typeid<S: StatisticsType + 'static>(beta: f64, omega: f64) -> f64 {
    if std::any::TypeId::of::<S>() == std::any::TypeId::of::<Fermionic>() {
        1.0
    } else {
        1.0 / (0.5 * beta * omega).tanh()
    }
}

// === パフォーマンス比較 ===
fn benchmark_approaches() {
    println!("=== パフォーマンス比較 ===");
    
    let beta = 1.0;
    let omega = 1.0;
    let iterations = 1_000_000;
    
    // 定数分岐（最も高速）
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = weight_by_const::<Fermionic>(beta, omega);
    }
    let const_time = start.elapsed();
    
    // TypeId分岐（現在の実装）
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = weight_by_typeid::<Fermionic>(beta, omega);
    }
    let typeid_time = start.elapsed();
    
    println!("定数分岐: {:?}", const_time);
    println!("TypeId分岐: {:?}", typeid_time);
    println!("比率: {:.2}x", 
        typeid_time.as_nanos() as f64 / const_time.as_nanos() as f64);
}

fn main() {
    println!("=== Rust vs C++ テンプレートの違い ===\n");
    
    // 1. 定数を使った分岐
    println!("1. 定数を使った分岐（推奨）:");
    println!("   Fermionic: {}", weight_by_const::<Fermionic>(1.0, 1.0));
    println!("   Bosonic: {}", weight_by_const::<Bosonic>(1.0, 1.0));
    
    // 2. TypeIdを使った分岐
    println!("\n2. TypeIdを使った分岐（現在の実装）:");
    println!("   Fermionic: {}", weight_by_typeid::<Fermionic>(1.0, 1.0));
    println!("   Bosonic: {}", weight_by_typeid::<Bosonic>(1.0, 1.0));
    
    // 3. パフォーマンス比較
    println!();
    benchmark_approaches();
    
    println!("\n=== なぜC++スタイルが使えないのか ===");
    println!("1. 型消去: 実行時には型情報が失われる");
    println!("2. 安全性: Rustは型安全性を優先");
    println!("3. 明示性: どの方法を使うか明示的に選択");
    println!("4. パフォーマンス: 定数分岐が最も高速");
}
