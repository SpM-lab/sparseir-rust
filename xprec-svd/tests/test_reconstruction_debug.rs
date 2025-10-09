use mdarray::Tensor;
use xprec_svd::svd::jacobi::jacobi_svd;

fn reconstruct_svd(
    u: &Tensor<f64, (usize, usize)>,
    s: &Tensor<f64, (usize,)>,
    v: &Tensor<f64, (usize, usize)>,
) -> Tensor<f64, (usize, usize)> {
    let u_shape = *u.shape();
    let v_shape = *v.shape();
    
    println!("U shape: {:?}, S len: {}, V shape: {:?}", u_shape, s.len(), v_shape);
    
    Tensor::from_fn((u_shape.0, v_shape.0), |idx| {
        let mut sum = 0.0;
        for k in 0..s.len() {
            sum = sum + u[[idx[0], k]] * s[[k]] * v[[idx[1], k]];
        }
        sum
    })
}

#[test]
fn debug_reconstruction() {
    let a = Tensor::from_fn((2, 2), |_| 1.0);
    let result = jacobi_svd(&a);
    
    println!("U:");
    for i in 0..2 {
        println!("  [{:.6}, {:.6}]", result.u[[i, 0]], result.u[[i, 1]]);
    }
    
    println!("S: [{:.6}, {:.6}]", result.s[[0]], result.s[[1]]);
    
    println!("V:");
    for i in 0..2 {
        println!("  [{:.6}, {:.6}]", result.v[[i, 0]], result.v[[i, 1]]);
    }
    
    let reconstructed = reconstruct_svd(&result.u, &result.s, &result.v);
    
    println!("Reconstructed:");
    for i in 0..2 {
        println!("  [{:.6}, {:.6}]", reconstructed[[i, 0]], reconstructed[[i, 1]]);
    }
    
    println!("Original:");
    for i in 0..2 {
        println!("  [{:.6}, {:.6}]", a[[i, 0]], a[[i, 1]]);
    }
}
