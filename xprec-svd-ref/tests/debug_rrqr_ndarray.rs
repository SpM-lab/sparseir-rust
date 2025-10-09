use ndarray::Array2;
use xprec_svd::qr::{rrqr, truncate_qr_result};
use xprec_svd::svd::jacobi::jacobi_svd;

#[test]
fn debug_rrqr_then_svd_ndarray() {
    let mut a = Array2::from_elem((2, 2), 1.0);
    
    println!("=== NDARRAY: STEP 0 - Original matrix ===");
    for i in 0..2 {
        println!("  [{}, {}]", a[[i, 0]], a[[i, 1]]);
    }
    
    // Step 1: RRQR
    let (qr, rank) = rrqr(&mut a, 1e-10);
    println!("\n=== NDARRAY: STEP 1 - After RRQR ===");
    println!("Rank: {}", rank);
    println!("Matrix A (modified by RRQR):");
    for i in 0..2 {
        println!("  [{:.10}, {:.10}]", a[[i, 0]], a[[i, 1]]);
    }
    
    // Step 2: Truncate QR
    let (q_trunc, r_trunc) = truncate_qr_result(&qr, rank);
    println!("\n=== NDARRAY: STEP 2 - After truncation ===");
    println!("Q_trunc ({}x{}):", q_trunc.nrows(), q_trunc.ncols());
    for i in 0..2 {
        for j in 0..rank {
            print!("  {:.10}", q_trunc[[i, j]]);
        }
        println!();
    }
    
    println!("\nR_trunc ({}x{}):", r_trunc.nrows(), r_trunc.ncols());
    for i in 0..rank {
        println!("  [{:.10}, {:.10}]", r_trunc[[i, 0]], r_trunc[[i, 1]]);
    }
    
    // Step 3: SVD of R^T
    let r_t = r_trunc.t().to_owned();
    println!("\n=== NDARRAY: STEP 3 - R^T for SVD ===");
    println!("R^T ({}x{}):", r_t.nrows(), r_t.ncols());
    for i in 0..r_t.nrows() {
        for j in 0..r_t.ncols() {
            print!("  {:.10}", r_t[[i, j]]);
        }
        println!();
    }
    
    let svd_result = jacobi_svd(&r_t);
    println!("\n=== NDARRAY: STEP 4 - SVD of R^T ===");
    println!("U_svd ({}x{}):", svd_result.u.nrows(), svd_result.u.ncols());
    for i in 0..svd_result.u.nrows() {
        for j in 0..svd_result.u.ncols() {
            print!("  {:.10}", svd_result.u[[i, j]]);
        }
        println!();
    }
    println!("S_svd: [{:.10}, {:.10}]", svd_result.s[0], svd_result.s[1]);
    println!("V_svd ({}x{}):", svd_result.v.nrows(), svd_result.v.ncols());
    for i in 0..svd_result.v.nrows() {
        for j in 0..svd_result.v.ncols() {
            print!("  {:.10}", svd_result.v[[i, j]]);
        }
        println!();
    }
    
    // Step 5: Final U and V
    let u_final = q_trunc.dot(&svd_result.v);
    println!("\n=== NDARRAY: STEP 5 - Final U = Q_trunc * V_svd ===");
    for i in 0..u_final.nrows() {
        for j in 0..u_final.ncols() {
            print!("  {:.10}", u_final[[i, j]]);
        }
        println!();
    }
}
