use ndarray::{Array2, arr2};

fn main() {
    // テスト用の行列を作成
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
    
    // .t() メソッド（転置）
    let a_transpose = a.t();
    println!("A^T = {:?}", a_transpose);
    
    // .dot() メソッド（行列の積）
    let c = a.dot(&b);
    println!("A * B = {:?}", c);
    
    println!("Success! .dot() and .t() work with ndarray-linalg!");
}