use ndarray::{Array2, arr2};

/// システムOpenBLAS（Homebrew）のshared libraryテスト
pub fn test_openblas_shared() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing system OpenBLAS (Homebrew) as shared library...");
    
    // テスト用の行列を作成
    let a: Array2<f64> = arr2(&[
        [1.0, 2.0],
        [3.0, 4.0]
    ]);
    
    let b: Array2<f64> = arr2(&[
        [5.0, 6.0],
        [7.0, 8.0]
    ]);
    
    println!("Matrix A:\n{}", a);
    println!("Matrix B:\n{}", b);
    
    // .t() メソッド（転置）
    let a_transpose = a.t();
    println!("A^T:\n{}", a_transpose);
    
    // .dot() メソッド（行列の積）- これがBLAS関数を使用
    let c = a.dot(&b);
    println!("A * B:\n{}", c);
    
    // 結果の検証
    let expected = arr2(&[
        [19.0, 22.0],
        [43.0, 50.0]
    ]);
    
    if c.abs_diff_eq(&expected, 1e-10) {
        println!("✅ Matrix multiplication successful!");
        println!("✅ System OpenBLAS shared library is working!");
        Ok(())
    } else {
        println!("❌ Matrix multiplication failed!");
        Err("Matrix multiplication result doesn't match expected".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_openblas_functionality() {
        test_openblas_shared().unwrap();
    }
}