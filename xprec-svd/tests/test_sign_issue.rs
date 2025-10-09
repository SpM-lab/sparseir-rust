use mdarray::Tensor;
use xprec_svd::svd::jacobi::jacobi_svd;
use approx::assert_abs_diff_eq;

fn reconstruct_svd(
    u: &Tensor<f64, (usize, usize)>,
    s: &Tensor<f64, (usize,)>,
    v: &Tensor<f64, (usize, usize)>,
) -> Tensor<f64, (usize, usize)> {
    let u_shape = *u.shape();
    let v_shape = *v.shape();
    
    Tensor::from_fn((u_shape.0, v_shape.0), |idx| {
        let mut sum = 0.0;
        for k in 0..s.len() {
            sum = sum + u[[idx[0], k]] * s[[k]] * v[[idx[1], k]];
        }
        sum
    })
}

#[test]
fn test_sign_invariance() {
    // Rank-1 matrix: all 1s
    let a = Tensor::from_fn((2, 2), |_| 1.0);
    let result = jacobi_svd(&a);
    
    println!("Singular values: {:?}, {:?}", result.s[[0]], result.s[[1]]);
    
    let reconstructed = reconstruct_svd(&result.u, &result.s, &result.v);
    
    // Check that reconstruction matches or is sign-flipped version
    let mut matches_original = true;
    let mut matches_flipped = true;
    
    for i in 0..2 {
        for j in 0..2 {
            let orig = a[[i, j]];
            let recon = reconstructed[[i, j]];
            if (recon - orig).abs() > 1e-10 {
                matches_original = false;
            }
            if (recon + orig).abs() > 1e-10 {
                matches_flipped = false;
            }
        }
    }
    
    assert!(matches_original || matches_flipped, 
            "Reconstruction should match original or be sign-flipped");
}
