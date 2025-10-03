//! Arc（原子参照カウント）のデモ

use std::sync::Arc;
use std::thread;

fn main() {
    println!("=== Arc（原子参照カウント）のデモ ===\n");
    
    // 1. 基本的な使用
    println!("1. 基本的な使用");
    let data = Arc::new(vec![1, 2, 3, 4, 5]);
    println!("   Original data: {:?}", data);
    
    let data1 = Arc::clone(&data);
    let data2 = Arc::clone(&data);
    
    println!("   data1: {:?}", data1);
    println!("   data2: {:?}", data2);
    println!("   All point to the same memory location");
    
    // 2. 参照カウントの確認
    println!("\n2. 参照カウントの確認");
    println!("   Reference count: {}", Arc::strong_count(&data));
    
    {
        let data3 = Arc::clone(&data);
        println!("   After cloning: {}", Arc::strong_count(&data));
    } // data3がドロップされる
    
    println!("   After data3 dropped: {}", Arc::strong_count(&data));
    
    // 3. スレッド間での共有
    println!("\n3. スレッド間での共有");
    let shared_data = Arc::new(vec!["Hello", "World", "Rust"]);
    
    let handles: Vec<_> = (0..3)
        .map(|i| {
            let data = Arc::clone(&shared_data);
            thread::spawn(move || {
                println!("   Thread {}: {:?}", i, data);
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // 4. カスタム構造体での使用
    println!("\n4. カスタム構造体での使用");
    let polynomial_data = Arc::new(PolynomialData {
        coefficients: vec![1.0, 2.0, 3.0],
        intervals: vec![(0.0, 1.0), (1.0, 2.0)],
    });
    
    let poly1 = PiecewiseLegendrePoly {
        data: Arc::clone(&polynomial_data),
        index: 0,
    };
    
    let poly2 = PiecewiseLegendrePoly {
        data: Arc::clone(&polynomial_data),
        index: 1,
    };
    
    println!("   poly1: {:?}", poly1);
    println!("   poly2: {:?}", poly2);
    println!("   Reference count: {}", Arc::strong_count(&polynomial_data));
    
    // 5. メモリ効率のデモ
    println!("\n5. メモリ効率のデモ");
    demonstrate_memory_efficiency();
    
    println!("\n=== Arcの利点 ===");
    println!("1. 複数の所有者が同じデータを共有");
    println!("2. スレッドセーフ（Send + Sync）");
    println!("3. 自動メモリ管理（参照カウント）");
    println!("4. ゼロコスト抽象化（参照カウントのみのオーバーヘッド）");
    println!("5. 循環参照の検出（Weak<T>と組み合わせ）");
}

// 多項式データの例
#[derive(Debug)]
struct PolynomialData {
    coefficients: Vec<f64>,
    intervals: Vec<(f64, f64)>,
}

#[derive(Debug)]
struct PiecewiseLegendrePoly {
    data: Arc<PolynomialData>,
    index: usize,
}

fn demonstrate_memory_efficiency() {
    let large_data = Arc::new(vec![0; 1000]);  // 1000個の要素
    
    println!("   Large data size: {} bytes", std::mem::size_of_val(&*large_data));
    println!("   Arc overhead: {} bytes", std::mem::size_of::<Arc<Vec<i32>>>());
    
    // 複数のクローンを作成
    let clones: Vec<_> = (0..10)
        .map(|_| Arc::clone(&large_data))
        .collect();
    
    println!("   Created {} clones", clones.len());
    println!("   Total memory usage: Still just one copy of the data!");
    println!("   Reference count: {}", Arc::strong_count(&large_data));
}

// Arc vs Box の比較
#[allow(dead_code)]
fn compare_arc_vs_box() {
    println!("\n=== Arc vs Box の比較 ===");
    
    // Box: 単一所有者
    let boxed_data = Box::new(vec![1, 2, 3]);
    // let boxed_copy = boxed_data;  // 所有権が移動（コピーではない）
    // println!("{:?}", boxed_data);  // エラー！所有権が移動済み
    
    // Arc: 複数所有者
    let arc_data = Arc::new(vec![1, 2, 3]);
    let arc_copy = Arc::clone(&arc_data);  // 参照カウントが増加
    println!("   Arc original: {:?}", arc_data);
    println!("   Arc copy: {:?}", arc_copy);
    println!("   Both are valid!");
}
