use mdarray::Tensor;
use xprec_svd::svd::jacobi::jacobi_svd;

#[test]
fn debug_direct_svd_mdarray() {
    let a = Tensor::from_fn((2, 2), |_| 1.0);
    
    println!("=== MDARRAY: Direct jacobi_svd on 2x2 matrix ===");
    println!("Input:");
    for i in 0..2 {
        println!("  [{}, {}]", a[[i, 0]], a[[i, 1]]);
    }
    
    let result = jacobi_svd(&a);
    
    let u_shape = *result.u.shape();
    let v_shape = *result.v.shape();
    println!("\nU ({}x{}):", u_shape.0, u_shape.1);
    for i in 0..2 {
        println!("  [{:.10}, {:.10}]", result.u[[i, 0]], result.u[[i, 1]]);
    }
    
    println!("\nS: [{:.10}, {:.10}]", result.s[[0]], result.s[[1]]);
    
    println!("\nV ({}x{}):", v_shape.0, v_shape.1);
    for i in 0..2 {
        println!("  [{:.10}, {:.10}]", result.v[[i, 0]], result.v[[i, 1]]);
    }
}
