use ndarray::array;
use xprec_svd::svd::jacobi::jacobi_svd;

fn main() {
    println!("=== Debug U and V Columns ===");
    let a = array![
        [1.0, 1.0],
        [1.0, 1.0]
    ];
    
    let result = jacobi_svd(&a);
    
    println!("Singular values: {:?}", result.s);
    
    println!("\nU matrix:");
    println!("{}", result.u);
    println!("\nU columns:");
    println!("  u_0 = {:?}", result.u.column(0).to_vec());
    println!("  u_1 = {:?}", result.u.column(1).to_vec());
    
    println!("\nV matrix:");
    println!("{}", result.v);
    println!("\nV columns:");
    println!("  v_0 = {:?}", result.v.column(0).to_vec());
    println!("  v_1 = {:?}", result.v.column(1).to_vec());
    
    // Manual reconstruction
    let s0 = result.s[0];
    let u0 = result.u.column(0);
    let v0 = result.v.column(0);
    
    println!("\nManual reconstruction:");
    println!("s_0 = {}", s0);
    println!("u_0 = {:?}", u0.to_vec());
    println!("v_0 = {:?}", v0.to_vec());
    
    let reconstructed_manual = s0 * array![[u0[0] * v0[0], u0[0] * v0[1]], [u0[1] * v0[0], u0[1] * v0[1]]];
    println!("\nManual: s_0 * u_0 * v_0^T =");
    println!("{}", reconstructed_manual);
}

