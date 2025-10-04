use ndarray::{Array2, arr2};
use xprec_svd::*;

fn main() {
    println!("Testing xprec-svd with system OpenBLAS...");
    
    // テスト用の行列を作成
    let matrix: Array2<f64> = arr2(&[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]);
    
    println!("Original matrix:\n{}", matrix);
    
    // xprec-svdを使用してSVDを実行
    let config = TSVDConfig::new(1e-12);
    match tsvd_f64(&matrix, config.rtol) {
        Ok(result) => {
            println!("SVD successful!");
            println!("Singular values: {:?}", result.s);
            println!("U shape: {:?}", result.u.shape());
            println!("V shape: {:?}", result.v.shape());
        }
        Err(e) => {
            println!("SVD failed: {}", e);
        }
    }
    
    // ndarrayの基本操作もテスト
    let transpose = matrix.t();
    println!("Transpose:\n{}", transpose);
    
    let dot_product = matrix.dot(&transpose);
    println!("Matrix * Transpose:\n{}", dot_product);
    
    println!("Test completed with system OpenBLAS!");
}