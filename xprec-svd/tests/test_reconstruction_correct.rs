use mdarray::Tensor;
use xprec_svd::svd::jacobi::jacobi_svd;

#[test]
fn verify_reconstruction_formula() {
    let a = Tensor::from_fn((2, 2), |_| 1.0);
    let result = jacobi_svd(&a);
    
    println!("U matrix:");
    for i in 0..2 {
        println!("  [{:.6}, {:.6}]", result.u[[i, 0]], result.u[[i, 1]]);
    }
    
    println!("S: [{:.6}, {:.6}]", result.s[[0]], result.s[[1]]);
    
    println!("V matrix:");
    for i in 0..2 {
        println!("  [{:.6}, {:.6}]", result.v[[i, 0]], result.v[[i, 1]]);
    }
    
    // Test 1: A[0,0] = U[0,0]*S[0]*V[0,0] + U[0,1]*S[1]*V[0,1]
    let a00_recon = result.u[[0, 0]] * result.s[[0]] * result.v[[0, 0]]
                  + result.u[[0, 1]] * result.s[[1]] * result.v[[0, 1]];
    println!("\nA[0,0] reconstruction:");
    println!("  U[0,0]*S[0]*V[0,0] = {} * {} * {} = {}", 
             result.u[[0,0]], result.s[[0]], result.v[[0,0]], 
             result.u[[0, 0]] * result.s[[0]] * result.v[[0, 0]]);
    println!("  U[0,1]*S[1]*V[0,1] = {} * {} * {} = {}", 
             result.u[[0,1]], result.s[[1]], result.v[[0,1]],
             result.u[[0, 1]] * result.s[[1]] * result.v[[0, 1]]);
    println!("  Total = {}, Expected = 1.0", a00_recon);
    
    // Test 2: A[1,0] = U[1,0]*S[0]*V[0,0] + U[1,1]*S[1]*V[0,1]
    let a10_recon = result.u[[1, 0]] * result.s[[0]] * result.v[[0, 0]]
                  + result.u[[1, 1]] * result.s[[1]] * result.v[[0, 1]];
    println!("\nA[1,0] reconstruction:");
    println!("  U[1,0]*S[0]*V[0,0] = {} * {} * {} = {}", 
             result.u[[1,0]], result.s[[0]], result.v[[0,0]], 
             result.u[[1, 0]] * result.s[[0]] * result.v[[0, 0]]);
    println!("  U[1,1]*S[1]*V[0,1] = {} * {} * {} = {}", 
             result.u[[1,1]], result.s[[1]], result.v[[0,1]],
             result.u[[1, 1]] * result.s[[1]] * result.v[[0, 1]]);
    println!("  Total = {}, Expected = 1.0", a10_recon);
    
    assert!((a00_recon - 1.0).abs() < 1e-10);
    assert!((a10_recon - 1.0).abs() < 1e-10);
}
