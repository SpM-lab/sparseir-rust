//! Debug Jacobi SVD with 2x2 matrix

use ndarray::array;
use xprec_svd::svd::jacobi::jacobi_svd;

fn main() {
    println!("=== Debug Jacobi SVD 2x2 ===");
    
    // Test case 1: Simple 2x2 matrix
    let a1 = array![
        [1.0, 1.0],
        [1.0, 1.0]
    ];
    
    println!("Test 1: Identity matrix");
    println!("Input matrix A:");
    println!("{}", a1);
    
    let result1 = jacobi_svd(&a1);
    println!("Singular values: {:?}", result1.s);
    println!("U:\n{}", result1.u);
    println!("V:\n{}", result1.v);
    
    // Test case 2: Rank-one 2x2 matrix
    let a2 = array![
        [1.0, 1.0],
        [1.0, 1.0]
    ];
    
    println!("\nTest 2: Rank-one matrix");
    println!("Input matrix A:");
    println!("{}", a2);
    
    let result2 = jacobi_svd(&a2);
    println!("Singular values: {:?}", result2.s);
    println!("U:\n{}", result2.u);
    println!("V:\n{}", result2.v);
    
    // Test case 3: Simple diagonal matrix
    let a3 = array![
        [2.0, 0.0],
        [0.0, 3.0]
    ];
    
    println!("\nTest 3: Diagonal matrix");
    println!("Input matrix A:");
    println!("{}", a3);
    
    let result3 = jacobi_svd(&a3);
    println!("Singular values: {:?}", result3.s);
    println!("U:\n{}", result3.u);
    println!("V:\n{}", result3.v);
    
    // Test case 4: Simple off-diagonal matrix
    let a4 = array![
        [0.0, 1.0],
        [1.0, 0.0]
    ];
    
    println!("\nTest 4: Off-diagonal matrix");
    println!("Input matrix A:");
    println!("{}", a4);
    
    let result4 = jacobi_svd(&a4);
    println!("Singular values: {:?}", result4.s);
    println!("U:\n{}", result4.u);
    println!("V:\n{}", result4.v);
}
