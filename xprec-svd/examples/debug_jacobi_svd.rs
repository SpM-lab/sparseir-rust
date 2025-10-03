//! Debug Jacobi SVD implementation

use ndarray::array;
use xprec_svd::svd::jacobi::jacobi_svd;

fn main() {
    println!("=== Debug Jacobi SVD ===");
    
    // Test case: rank-one matrix (all elements are 1)
    let a = array![
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ];
    
    println!("Input matrix A:");
    println!("{}", a);
    
    // Compute SVD
    let result = jacobi_svd(&a);
    
    println!("\nSVD Results:");
    println!("Singular values: {:?}", result.s);
    println!("Rank: {}", result.rank);
    
    println!("\nU matrix (left singular vectors):");
    println!("{}", result.u);
    
    println!("\nV matrix (right singular vectors):");
    println!("{}", result.v);
    
    // Check reconstruction
    let u_s_vt = result.u.dot(&ndarray::Array2::from_diag(&result.s)).dot(&result.v.t());
    println!("\nReconstruction U * S * V^T:");
    println!("{}", u_s_vt);
    
    println!("\nReconstruction error:");
    let error = &a - &u_s_vt;
    println!("{}", error);
    println!("Max error: {}", error.iter().map(|x| x.abs()).fold(0.0, f64::max));
    
    // Expected: only first singular value should be non-zero (around 3.0)
    println!("\nExpected: s[0] ≈ 3.0, s[1] = 0.0, s[2] = 0.0");
    println!("Actual: s[0] = {}, s[1] = {}, s[2] = {}", 
             result.s[0], result.s[1], result.s[2]);
}
