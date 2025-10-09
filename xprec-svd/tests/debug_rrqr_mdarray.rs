use mdarray::Tensor;
use xprec_svd::qr::{rrqr, truncate_qr_result};
use xprec_svd::svd::jacobi::jacobi_svd;

#[test]
fn debug_rrqr_then_svd_mdarray() {
    let mut a = Tensor::from_fn((2, 2), |_| 1.0);
    
    println!("=== MDARRAY: STEP 0 - Original matrix ===");
    for i in 0..2 {
        println!("  [{}, {}]", a[[i, 0]], a[[i, 1]]);
    }
    
    // Step 1: RRQR
    let (qr, rank) = rrqr(&mut a, 1e-10);
    println!("\n=== MDARRAY: STEP 1 - After RRQR ===");
    println!("Rank: {}", rank);
    println!("Matrix A (modified by RRQR):");
    for i in 0..2 {
        println!("  [{:.10}, {:.10}]", a[[i, 0]], a[[i, 1]]);
    }
    
    // Step 2: Truncate QR
    let (q_trunc, r_trunc) = truncate_qr_result(&qr, rank);
    println!("\n=== MDARRAY: STEP 2 - After truncation ===");
    let q_shape = *q_trunc.shape();
    let r_shape = *r_trunc.shape();
    println!("Q_trunc ({}x{}):", q_shape.0, q_shape.1);
    for i in 0..q_shape.0 {
        for j in 0..rank {
            print!("  {:.10}", q_trunc[[i, j]]);
        }
        println!();
    }
    
    println!("\nR_trunc ({}x{}):", r_shape.0, r_shape.1);
    for i in 0..rank {
        println!("  [{:.10}, {:.10}]", r_trunc[[i, 0]], r_trunc[[i, 1]]);
    }
    
    // Step 3: SVD of R^T
    let r_t = Tensor::from_fn((r_shape.1, r_shape.0), |idx| r_trunc[[idx[1], idx[0]]]);
    println!("\n=== MDARRAY: STEP 3 - R^T for SVD ===");
    println!("R^T ({}x{}):", r_shape.1, r_shape.0);
    for i in 0..r_shape.1 {
        for j in 0..r_shape.0 {
            print!("  {:.10}", r_t[[i, j]]);
        }
        println!();
    }
    
    let svd_result = jacobi_svd(&r_t);
    println!("\n=== MDARRAY: STEP 4 - SVD of R^T ===");
    let u_svd_shape = *svd_result.u.shape();
    let v_svd_shape = *svd_result.v.shape();
    println!("U_svd ({}x{}):", u_svd_shape.0, u_svd_shape.1);
    for i in 0..u_svd_shape.0 {
        for j in 0..u_svd_shape.1 {
            print!("  {:.10}", svd_result.u[[i, j]]);
        }
        println!();
    }
    println!("S_svd:");
    for i in 0..svd_result.s.len() {
        println!("  [{:.10}]", svd_result.s[[i]]);
    }
    println!("V_svd ({}x{}):", v_svd_shape.0, v_svd_shape.1);
    for i in 0..v_svd_shape.0 {
        for j in 0..v_svd_shape.1 {
            print!("  {:.10}", svd_result.v[[i, j]]);
        }
        println!();
    }
    
    // Step 5: Final U = Q_trunc * V_svd (manual matmul)
    println!("\n=== MDARRAY: STEP 5 - Final U = Q_trunc * V_svd ===");
    let u_final = Tensor::from_fn((q_shape.0, v_svd_shape.1), |idx| {
        let mut sum = 0.0;
        for k in 0..q_shape.1 {
            sum += q_trunc[[idx[0], k]] * svd_result.v[[k, idx[1]]];
        }
        sum
    });
    println!("U_final ({}x{}):", q_shape.0, v_svd_shape.1);
    for i in 0..q_shape.0 {
        for j in 0..v_svd_shape.1 {
            print!("  {:.10}", u_final[[i, j]]);
        }
        println!();
    }
}
