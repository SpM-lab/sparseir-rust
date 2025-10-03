use xprec_svd::svd::jacobi::jacobi_svd;
use ndarray::array;

fn main() {
    println!("=== Debug Off-Diagonal Matrix ===");
    
    // Test case: [0, 1], [1, 0]
    let a = array![
        [0.0, 1.0],
        [1.0, 0.0]
    ];
    
    println!("Input matrix A:\n{}", a);
    
    let result = jacobi_svd(&a);
    
    println!("\nSVD Results:");
    println!("Singular values: {}", result.s);
    println!("U matrix:\n{}", result.u);
    println!("V matrix:\n{}", result.v);
    
    let s_diag = ndarray::Array2::from_diag(&result.s);
    let reconstructed = result.u.dot(&s_diag).dot(&result.v.t());
    println!("\nReconstruction U * S * V^T:\n{}", reconstructed);
    
    let error = &a - &reconstructed;
    println!("\nReconstruction error:\n{}", error);
    println!("Max error: {}", error.mapv(|x| x.abs()).fold(0.0_f64, |acc, x| acc.max(*x)));
    
    println!("\nExpected: s[0] = 1.0, s[1] = 1.0");
    println!("Actual: s[0] = {}, s[1] = {}", result.s[0], result.s[1]);
}
