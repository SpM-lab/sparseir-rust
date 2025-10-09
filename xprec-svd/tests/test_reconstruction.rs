use mdarray::Tensor;
use xprec_svd::svd::jacobi::jacobi_svd;

fn reconstruct_svd<T: xprec_svd::precision::Precision>(
    u: &Tensor<T, (usize, usize)>,
    s: &Tensor<T, (usize,)>,
    v: &Tensor<T, (usize, usize)>,
) -> Tensor<T, (usize, usize)> {
    let u_shape = *u.shape();
    let v_shape = *v.shape();
    
    Tensor::from_fn((u_shape.0, v_shape.0), |idx| {
        let mut sum = T::zero();
        for k in 0..s.len() {
            sum = sum + u[[idx[0], k]] * s[[k]] * v[[idx[1], k]];
        }
        sum
    })
}

#[test]
fn test_reconstruction_2x2() {
    let a = Tensor::from_fn((2, 2), |_| 1.0);
    println!("Original matrix:");
    for i in 0..2 {
        println!("{:?}", (a[[i, 0]], a[[i, 1]]));
    }
    
    let result = jacobi_svd(&a);
    println!("\nSingular values: {:?}", (result.s[[0]], result.s[[1]]));
    println!("U shape: {:?}, V shape: {:?}", result.u.shape(), result.v.shape());
    
    let reconstructed = reconstruct_svd(&result.u, &result.s, &result.v);
    println!("\nReconstructed:");
    for i in 0..2 {
        println!("{:?}", (reconstructed[[i, 0]], reconstructed[[i, 1]]));
    }
}
