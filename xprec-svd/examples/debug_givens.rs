use ndarray::array;
use xprec_svd::svd::jacobi::{apply_givens_left, apply_givens_right};

fn main() {
    println!("=== Debug Givens Rotations ===");
    
    // Test case: [0, 1], [1, 0]
    let mut a = array![
        [0.0, 1.0],
        [1.0, 0.0]
    ];
    
    println!("Initial matrix A:\n{}", a);
    
    // Simulate Eigen3's real_2x2_jacobi_svd
    // From Eigen3 debug output:
    // rot1: c = 1, s = 0 (identity)
    // j_right: c = 0.707107, s = 0.707107
    // j_left: c = 0.707107, s = -0.707107
    
    let c_left = 0.707107;
    let s_left = -0.707107;
    let c_right = 0.707107;
    let s_right = 0.707107;
    
    println!("\nApplying left rotation: c = {}, s = {}", c_left, s_left);
    apply_givens_left(&mut a, 0, 1, c_left, s_left);
    println!("After left rotation:\n{}", a);
    
    println!("\nApplying right rotation: c = {}, s = {}", c_right, s_right);
    apply_givens_right(&mut a, 0, 1, c_right, s_right);
    println!("After right rotation:\n{}", a);
    
    println!("\nExpected diagonal: [1, 1] or [-1, -1] (signs may vary)");
    println!("Actual diagonal: [{}, {}]", a[[0, 0]], a[[1, 1]]);
    println!("Off-diagonal: [{}, {}]", a[[0, 1]], a[[1, 0]]);
}
