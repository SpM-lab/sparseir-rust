use ndarray::Array2;
use xprec_svd::svd::jacobi::jacobi_svd;

#[test]
fn debug_direct_svd_ndarray() {
    let a = Array2::from_elem((2, 2), 1.0);
    
    println!("=== NDARRAY: Direct jacobi_svd on 2x2 matrix ===");
    println!("Input:");
    for i in 0..2 {
        println!("  [{}, {}]", a[[i, 0]], a[[i, 1]]);
    }
    
    let result = jacobi_svd(&a);
    
    println!("\nU ({}x{}):", result.u.nrows(), result.u.ncols());
    for i in 0..2 {
        println!("  [{:.10}, {:.10}]", result.u[[i, 0]], result.u[[i, 1]]);
    }
    
    println!("\nS: [{:.10}, {:.10}]", result.s[0], result.s[1]);
    
    println!("\nV ({}x{}):", result.v.nrows(), result.v.ncols());
    for i in 0..2 {
        println!("  [{:.10}, {:.10}]", result.v[[i, 0]], result.v[[i, 1]]);
    }
}
