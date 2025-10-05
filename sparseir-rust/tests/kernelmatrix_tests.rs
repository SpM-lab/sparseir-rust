//! Tests for kernel matrix discretization

use sparseir_rust::gauss::legendre;
use sparseir_rust::kernel::LogisticKernel;
use sparseir_rust::kernel::SymmetryType;
use sparseir_rust::kernelmatrix::matrix_from_gauss;

#[test]
fn test_matrix_from_gauss_basic() {
    // 2x2の小さな行列で基本動作確認
    let kernel = LogisticKernel::new(1.0);
    let gauss_x = legendre::<f64>(2).reseat(0.0, 1.0);
    let gauss_y = legendre::<f64>(2).reseat(0.0, 1.0);
    let matrix = matrix_from_gauss(&kernel, &gauss_x, &gauss_y, SymmetryType::Even);
    
    assert_eq!(matrix.nrows(), 2);
    assert_eq!(matrix.ncols(), 2);
}

#[test]
fn test_matrix_from_gauss_sizes() {
    let kernel = LogisticKernel::new(1.0);
    
    for n in [2, 4, 8] {
        let gauss_x = legendre::<f64>(n).reseat(0.0, 1.0);
        let gauss_y = legendre::<f64>(n).reseat(0.0, 1.0);
        let matrix = matrix_from_gauss(&kernel, &gauss_x, &gauss_y, SymmetryType::Even);
        
        assert_eq!(matrix.nrows(), n);
        assert_eq!(matrix.ncols(), n);
    }
}
