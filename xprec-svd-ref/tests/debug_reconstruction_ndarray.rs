use ndarray::Array2;
use xprec_svd::svd::jacobi::jacobi_svd;

#[test]
fn debug_ndarray_reconstruction() {
    let a = Array2::from_elem((2, 2), 1.0);
    
    println!("=== NDARRAY VERSION ===");
    println!("Original matrix:");
    for i in 0..2 {
        println!("  [{}, {}]", a[[i, 0]], a[[i, 1]]);
    }
    
    let result = jacobi_svd(&a);
    
    println!("\nU matrix ({}x{}):", result.u.nrows(), result.u.ncols());
    for i in 0..2 {
        println!("  [{:.10}, {:.10}]", result.u[[i, 0]], result.u[[i, 1]]);
    }
    
    println!("\nS: [{:.10}, {:.10}]", result.s[0], result.s[1]);
    
    println!("\nV matrix ({}x{}):", result.v.nrows(), result.v.ncols());
    for i in 0..2 {
        println!("  [{:.10}, {:.10}]", result.v[[i, 0]], result.v[[i, 1]]);
    }
    
    // V^T using .t()
    let v_t = result.v.t();
    println!("\nV^T (using .t()):");
    for i in 0..2 {
        println!("  [{:.10}, {:.10}]", v_t[[i, 0]], v_t[[i, 1]]);
    }
    
    // Reconstruct
    let s_diag = Array2::from_diag(&result.s);
    let reconstructed = result.u.dot(&s_diag).dot(&result.v.t());
    
    println!("\nReconstructed (U * S * V^T):");
    for i in 0..2 {
        println!("  [{:.10}, {:.10}]", reconstructed[[i, 0]], reconstructed[[i, 1]]);
    }
    
    // Manual calculation for A[1,0]
    let a10_manual = result.u[[1, 0]] * result.s[0] * v_t[[0, 0]]
                   + result.u[[1, 1]] * result.s[1] * v_t[[1, 0]];
    println!("\nManual A[1,0] = U[1,0]*S[0]*V^T[0,0] + U[1,1]*S[1]*V^T[1,0]");
    println!("              = {} * {} * {} + {} * {} * {}",
             result.u[[1,0]], result.s[0], v_t[[0,0]],
             result.u[[1,1]], result.s[1], v_t[[1,0]]);
    println!("              = {:.10}", a10_manual);
}
