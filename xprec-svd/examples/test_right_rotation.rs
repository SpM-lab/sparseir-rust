use ndarray::array;
use xprec_svd::svd::jacobi::apply_givens_right;

fn main() {
    println!("=== Rust Right Rotation Test ===");
    
    // Test matrix after left rotation: [[1.414, 0], [0, 0]]
    let mut a = array![
        [1.414213562373095, 0.0],
        [0.0, 0.0]
    ];
    
    println!("Initial matrix A:\n{}", a);
    
    // Right rotation: c = 0.707, s = 0.707
    let c = 0.7071067811865475;
    let s = 0.7071067811865475;
    
    println!("Right rotation: c = {}, s = {}", c, s);
    
    // Apply right rotation to columns 0 and 1
    apply_givens_right(&mut a, 0, 1, c, s);
    
    println!("After right rotation:\n{}", a);
    println!("A[0,0] = {}, A[1,1] = {}", a[[0, 0]], a[[1, 1]]);
}
