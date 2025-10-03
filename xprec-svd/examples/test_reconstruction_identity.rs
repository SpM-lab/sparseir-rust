use ndarray::Array2;
use xprec_svd::svd::jacobi::jacobi_svd;

fn main() {
    println!("=== Test 1: Identity Matrix ===");
    let a = Array2::<f64>::eye(3);
    
    println!("Original matrix A:");
    println!("{}", a);
    
    let result = jacobi_svd(&a);
    
    println!("\nSingular values: {:?}", result.s);
    println!("\nU matrix:");
    println!("{}", result.u);
    println!("\nV matrix:");
    println!("{}", result.v);
    
    // Reconstruct: A = U * S * V^T
    let s_diag = Array2::from_diag(&result.s);
    println!("\nS (diagonal matrix):");
    println!("{}", s_diag);
    
    let reconstructed = result.u.dot(&s_diag).dot(&result.v.t());
    println!("\nReconstructed matrix (U * S * V^T):");
    println!("{}", reconstructed);
    
    let error = &a - &reconstructed;
    println!("\nReconstruction error:");
    println!("{}", error);
    
    let max_error = error.mapv(|x| x.abs()).fold(0.0_f64, |acc, &x| acc.max(x));
    println!("\nMax absolute error: {:.2e}", max_error);
    
    if max_error < 1e-10 {
        println!("✓ Reconstruction SUCCESS!");
    } else {
        println!("✗ Reconstruction FAILED!");
    }
}

