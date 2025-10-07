// Debug test for uhat evaluation
use sparseir_rust::*;
use kernel::LogisticKernel;
use basis::FermionicBasis;
use num_complex::Complex64;

#[test]
fn debug_uhat_evaluation() {
    let beta = 1.0;
    let lambda = 10.0;
    let epsilon = 1e-6;
    
    let kernel = LogisticKernel::new(lambda);
    let basis = FermionicBasis::new(kernel, beta, Some(epsilon), None);
    
    println!("\n=== Debugging uhat evaluation ===");
    println!("Basis size: {}", basis.size());
    
    // Evaluate uhat[0] at n=1
    let uhat_rust = basis.uhat.polyvec[0].evaluate_at_n(1);
    let uhat_julia = Complex64::new(1.7177310195820194e-17, 0.543551067457796);
    
    println!("\nuhat[0](n=1):");
    println!("  Rust:  {:.6e} + {:.6e}i", uhat_rust.re, uhat_rust.im);
    println!("  Julia: {:.6e} + {:.6e}i", uhat_julia.re, uhat_julia.im);
    println!("  |diff| = {:.6e}", (uhat_rust - uhat_julia).norm());
    
    // Check if values are swapped
    let swapped = Complex64::new(uhat_rust.im, uhat_rust.re);
    println!("\nIf swapped (im, re):");
    println!("  Swapped: {:.6e} + {:.6e}i", swapped.re, swapped.im);
    println!("  |diff| = {:.6e}", (swapped - uhat_julia).norm());
    
    // Check the underlying polynomial
    println!("\n--- Underlying polynomial (from sve_result.u) ---");
    let poly_sve = &basis.sve_result.u.polyvec[0];
    println!("  symm = {} (should be 1 for even, -1 for odd)", poly_sve.symm);
    println!("  polyorder = {}", poly_sve.polyorder);
    println!("  segments = {}", poly_sve.knots.len() - 1);
    println!("  First basis function should have EVEN symmetry (symm=1)!");
    
    // Check the scaled polynomial (in uhat_full)
    println!("\n--- Scaled polynomial (in uhat_full) ---");
    let poly_uhat = &basis.uhat_full.polyvec[0].poly;
    println!("  symm = {}", poly_uhat.symm);
    println!("  polyorder = {}", poly_uhat.polyorder);
    println!("  segments = {}", poly_uhat.knots.len() - 1);
    println!("  data[0,0] = {:.6e}", poly_uhat.data[[0, 0]]);
    println!("  data[1,0] = {:.6e}", poly_uhat.data[[1, 0]]);
}

#[test]
fn check_matsubara_freq_type() {
    use freq::{FermionicFreq, MatsubaraFreq};
    
    println!("\n=== MatsubaraFreq type check ===");
    
    match FermionicFreq::new(1) {
        Ok(freq) => {
            println!("FermionicFreq::new(1) succeeded");
            println!("  get_n() = {}", freq.get_n());
        },
        Err(e) => println!("Error: {}", e),
    }
    
    match FermionicFreq::new(2) {
        Ok(_) => println!("FermionicFreq::new(2) succeeded (WRONG!)"),
        Err(e) => println!("FermionicFreq::new(2) failed as expected: {}", e),
    }
}

