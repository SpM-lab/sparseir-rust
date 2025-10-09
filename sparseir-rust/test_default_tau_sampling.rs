//! Test default_tau_sampling_points implementation

use sparseir_rust::basis::FermionicBasis;
use sparseir_rust::kernel::LogisticKernel;

fn main() {
    // Create fermionic basis (beta=1.0, wmax=10.0)
    let kernel = LogisticKernel::new(10.0);
    let basis = FermionicBasis::new(kernel, 1.0, Some(1e-6), None);
    
    println!("Basis size: {}", basis.size());
    println!("Beta: {}", basis.beta);
    
    // Get default tau sampling points
    let tau_points = basis.default_tau_sampling_points();
    
    println!("\nDefault tau sampling points ({} points):", tau_points.len());
    for (i, &tau) in tau_points.iter().enumerate() {
        println!("  tau[{}] = {:.6}", i, tau);
    }
    
    // Verify they are in [0, beta]
    for &tau in &tau_points {
        assert!(tau >= 0.0 && tau <= basis.beta, 
                "tau={} out of range [0, {}]", tau, basis.beta);
    }
    
    println!("\n✅ All tau points in valid range [0, {}]", basis.beta);
}
