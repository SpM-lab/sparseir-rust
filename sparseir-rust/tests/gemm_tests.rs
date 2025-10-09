use mdarray::{tensor, DTensor};
use sparseir_rust::gemm::matmul_par;

#[test]
fn test_matmul_par_basic() {
    let a: DTensor<f64, 2> = tensor![[1.0, 2.0], [3.0, 4.0]];
    let b: DTensor<f64, 2> = tensor![[5.0, 6.0], [7.0, 8.0]];
    let c = matmul_par(&a, &b);
    
    // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //         = [[19, 22], [43, 50]]
    assert!((c[[0, 0]] - 19.0).abs() < 1e-10);
    assert!((c[[0, 1]] - 22.0).abs() < 1e-10);
    assert!((c[[1, 0]] - 43.0).abs() < 1e-10);
    assert!((c[[1, 1]] - 50.0).abs() < 1e-10);
}

#[test]
fn test_matmul_par_non_square() {
    let a: DTensor<f64, 2> = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2x3
    let b: DTensor<f64, 2> = tensor![[7.0], [8.0], [9.0]]; // 3x1
    let c = matmul_par(&a, &b);
    
    // Expected: [[1*7+2*8+3*9], [4*7+5*8+6*9]]
    //         = [[50], [122]]
    assert!((c[[0, 0]] - 50.0).abs() < 1e-10);
    assert!((c[[1, 0]] - 122.0).abs() < 1e-10);
}

