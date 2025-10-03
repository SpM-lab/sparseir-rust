use ndarray::array;
use xprec_svd::svd::jacobi::jacobi_svd;

fn main() {
    println!("=== Debug Rectangular Matrix ===");
    let a = array![
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ];
    
    println!("Original matrix A (3x2):");
    println!("{}", a);
    
    let result = jacobi_svd(&a);
    
    println!("\nSingular values: {:?}", result.s);
    println!("\nU matrix ({}x{}):", result.u.nrows(), result.u.ncols());
    println!("{}", result.u);
    println!("\nV matrix ({}x{}):", result.v.nrows(), result.v.ncols());
    println!("{}", result.v);
    
    // For rectangular matrix A (m x n), SVD gives U(m x k), S(k x k), V(n x k)
    // where k = min(m, n)
    let k = result.s.len();
    
    println!("\nk={}", k);
    println!("U is {}x{}, V is {}x{}", result.u.nrows(), result.u.ncols(), result.v.nrows(), result.v.ncols());
    println!("S diagonal should be {}x{}", k, k);
    
    // Create S as a diagonal matrix k x k
    let s_diag = ndarray::Array2::from_diag(&result.s);
    
    println!("\nS matrix:");
    println!("{}", s_diag);
    
    let reconstructed = result.u.dot(&s_diag).dot(&result.v.t());
    println!("\nReconstructed matrix:");
    println!("{}", reconstructed);
    
    let error = &a - &reconstructed;
    println!("\nReconstruction error:");
    println!("{}", error);
    
    let max_error = error.mapv(|x| x.abs()).fold(0.0_f64, |acc, &x| acc.max(x));
    println!("\nMax absolute error: {:.2e}", max_error);
}

