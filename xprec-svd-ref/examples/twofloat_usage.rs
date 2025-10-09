//! Example usage of TwoFloat precision in xprec-svd
//!
//! This example demonstrates how to use TwoFloat precision for high-precision SVD computations.

use xprec_svd::precision::TwoFloatPrecision;
use xprec_svd::{tsvd_twofloat, tsvd_twofloat_from_f64};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("TwoFloat Precision SVD Example");
    println!("==============================");
    
    // Example 1: Using f64 input with automatic conversion
    println!("\n1. f64 input with automatic conversion:");
    let matrix_f64 = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ];
    
    let result_f64 = tsvd_twofloat_from_f64(&matrix_f64, 1e-12)?;
    println!("   Rank: {}", result_f64.rank);
    println!("   Singular values: {:?}", 
        result_f64.s.iter().map(|&x| x.to_f64()).collect::<Vec<f64>>());
    
    // Example 2: Direct TwoFloat input
    println!("\n2. Direct TwoFloat input:");
    let matrix_tf = array![
        [TwoFloatPrecision::from_f64(1.0), TwoFloatPrecision::from_f64(2.0)],
        [TwoFloatPrecision::from_f64(3.0), TwoFloatPrecision::from_f64(4.0)]
    ];
    
    let rtol_tf = TwoFloatPrecision::from_f64(1e-12);
    let result_tf = tsvd_twofloat(&matrix_tf, rtol_tf)?;
    
    println!("   Rank: {}", result_tf.rank);
    println!("   Singular values: {:?}", 
        result_tf.s.iter().map(|&x| x.to_f64()).collect::<Vec<f64>>());
    
    // Example 3: Precision comparison
    println!("\n3. Precision comparison:");
    let matrix_small = array![
        [1e-10, 2e-10],
        [2e-10, 4e-10]
    ];
    
    // f64 precision
    let result_f64_small = xprec_svd::tsvd_f64(&matrix_small, 1e-12)?;
    
    // TwoFloat precision
    let result_tf_small = tsvd_twofloat_from_f64(&matrix_small, 1e-12)?;
    
    println!("   f64 singular values: {:?}", result_f64_small.s);
    println!("   TwoFloat singular values: {:?}", 
        result_tf_small.s.iter().map(|&x| x.to_f64()).collect::<Vec<f64>>());
    
    // Example 4: High-precision computation
    println!("\n4. High-precision computation:");
    let matrix_precise = array![
        [1.0 + 1e-15, 2.0 + 2e-15],
        [3.0 + 3e-15, 4.0 + 4e-15]
    ];
    
    let result_precise = tsvd_twofloat_from_f64(&matrix_precise, 1e-15)?;
    
    println!("   Input matrix with small perturbations:");
    for i in 0..2 {
        for j in 0..2 {
            println!("     [{}, {}] = {}", i, j, matrix_precise[[i, j]]);
        }
    }
    
    println!("   TwoFloat singular values: {:?}", 
        result_precise.s.iter().map(|&x| x.to_f64()).collect::<Vec<f64>>());
    
    println!("\nTwoFloat precision provides higher accuracy for small values!");
    
    Ok(())
}
