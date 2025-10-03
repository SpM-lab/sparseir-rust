//! Debug SVD 2x2 calculation step by step

use ndarray::array;
// svd_2x2 function has been replaced with real_2x2_jacobi_svd

fn main() {
    println!("=== Debug SVD 2x2 Step by Step ===");
    
    // Test case: [1, 1], [1, 1]
    let a: f64 = 1.0;
    let b: f64 = 1.0;
    let c: f64 = 1.0;
    let d: f64 = 1.0;
    
    println!("Input matrix:");
    println!("[{}, {}]", a, b);
    println!("[{}, {}]", c, d);
    
    // Manual calculation
    println!("\nManual calculation:");
    let a_sq = a * a;
    let b_sq = b * b;
    let c_sq = c * c;
    let d_sq = d * d;
    
    println!("a² = {}, b² = {}, c² = {}, d² = {}", a_sq, b_sq, c_sq, d_sq);
    
    let trace = a_sq + b_sq + c_sq + d_sq;
    let det = (a * d - b * c).abs();
    
    println!("trace = a² + b² + c² + d² = {}", trace);
    println!("det = |ad - bc| = |{}| = {}", a * d - b * c, det);
    
    let discriminant = trace * trace - 4.0 * det * det;
    println!("discriminant = trace² - 4*det² = {} - 4*{} = {}", 
             trace * trace, det * det, discriminant);
    
    let sqrt_disc = discriminant.sqrt();
    println!("√discriminant = {}", sqrt_disc);
    
    let s1_sq = (trace + sqrt_disc) / 2.0;
    let s2_sq = (trace - sqrt_disc) / 2.0;
    
    println!("s1² = (trace + √disc) / 2 = ({} + {}) / 2 = {}", 
             trace, sqrt_disc, s1_sq);
    println!("s2² = (trace - √disc) / 2 = ({} - {}) / 2 = {}", 
             trace, sqrt_disc, s2_sq);
    
    let s1 = s1_sq.sqrt();
    let s2 = s2_sq.sqrt();
    
    println!("s1 = √s1² = √{} = {}", s1_sq, s1);
    println!("s2 = √s2² = √{} = {}", s2_sq, s2);
    
    // Call the actual function
    println!("\nActual function result:");
    // Note: svd_2x2 has been replaced with real_2x2_jacobi_svd
    // This example is now for reference only
    println!("Function has been updated to use Eigen3's real_2x2_jacobi_svd approach");
}
