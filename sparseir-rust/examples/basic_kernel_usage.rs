//! Basic kernel usage example for sparseir-rust

use sparseir_rust::*;
use twofloat::TwoFloat;

fn main() {
    println!("=== SparseIR Rust Kernel Examples ===\n");
    
    // Example 1: Logistic Kernel
    println!("1. Logistic Kernel Example");
    let logistic_kernel = LogisticKernel::new(10.0);
    println!("   Lambda: {}", logistic_kernel.lambda());
    println!("   X range: {:?}", logistic_kernel.xrange());
    println!("   Y range: {:?}", logistic_kernel.yrange());
    
    // Compute kernel value at (0, 0)
    let x = TwoFloat::from(0.0);
    let y = TwoFloat::from(0.0);
    let k_value = logistic_kernel.compute(x, y);
    println!("   K(0, 0) = {}", Into::<f64>::into(k_value));
    
    // Test weight functions
    let beta = 1.0;
    let omega = 1.0;
    
    let w_fermi = logistic_kernel.weight::<Fermionic>(beta, omega);
    let w_bose = logistic_kernel.weight::<Bosonic>(beta, omega);
    println!("   Fermionic weight: {}", w_fermi);
    println!("   Bosonic weight: {}", w_bose);
    
    // Test safe inverse weight functions
    let inv_w_fermi = logistic_kernel.inv_weight::<Fermionic>(beta, omega);
    let inv_w_bose = logistic_kernel.inv_weight::<Bosonic>(beta, omega);
    println!("   Fermionic inv_weight: {}", inv_w_fermi);
    println!("   Bosonic inv_weight: {}", inv_w_bose);
    
    // Test weighted computation
    let result_fermi = logistic_kernel.compute_weighted::<Fermionic>(x, y, beta, omega);
    let result_bose = logistic_kernel.compute_weighted::<Bosonic>(x, y, beta, omega);
    println!("   Fermionic weighted result: {}", Into::<f64>::into(result_fermi));
    println!("   Bosonic weighted result: {}", Into::<f64>::into(result_bose));
    
    println!();
    
    // Example 2: Regularized Bose Kernel
    println!("2. Regularized Bose Kernel Example");
    let bose_kernel = RegularizedBoseKernel::new(10.0);
    println!("   Lambda: {}", bose_kernel.lambda());
    println!("   X range: {:?}", bose_kernel.xrange());
    println!("   Y range: {:?}", bose_kernel.yrange());
    
    // Test bosonic weight functions
    let w_bose = bose_kernel.weight::<Bosonic>(beta, omega);
    let inv_w_bose = bose_kernel.inv_weight::<Bosonic>(beta, omega);
    println!("   Bosonic weight: {}", w_bose);
    println!("   Bosonic inv_weight: {}", inv_w_bose);
    
    // Test weighted computation
    let result_bose = bose_kernel.compute_weighted::<Bosonic>(x, y, beta, omega);
    println!("   Bosonic weighted result: {}", Into::<f64>::into(result_bose));
    
    println!();
    
    // Example 3: Safety at omega=0
    println!("3. Safety Test at omega=0");
    let omega_zero = 0.0;
    
    // This should not panic and should handle omega=0 gracefully
    let inv_w_bose_safe = logistic_kernel.inv_weight::<Bosonic>(beta, omega_zero);
    println!("   Bosonic inv_weight at omega=0: {}", inv_w_bose_safe);
    
    let result_safe = logistic_kernel.compute_weighted::<Bosonic>(x, y, beta, omega_zero);
    println!("   Bosonic weighted result at omega=0: {}", Into::<f64>::into(result_safe));
    
    println!();
    
    // Example 4: f64 utility function
    println!("4. f64 Utility Function Example");
    let result_f64 = compute_f64(&logistic_kernel, 0.5, -0.3);
    println!("   f64 result at (0.5, -0.3): {}", result_f64);
    
    println!("\n=== All examples completed successfully! ===");
}
