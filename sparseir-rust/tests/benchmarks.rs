//! Benchmark tests for sparseir-rust
//!
//! These tests measure performance and ensure that the implementation
//! is reasonably efficient.

use sparseir_rust::*;
use twofloat::TwoFloat;
use std::time::Instant;

#[test]
fn benchmark_kernel_computation() {
    let kernel = LogisticKernel::new(10.0);
    let num_iterations = 10000;
    
    // Benchmark f64 computation
    let start = Instant::now();
    for _ in 0..num_iterations {
        let _ = compute_f64(&kernel, 0.5, -0.3);
    }
    let f64_duration = start.elapsed();
    
    // Benchmark TwoFloat computation
    let start = Instant::now();
    for _ in 0..num_iterations {
        let _ = kernel.compute(TwoFloat::from(0.5), TwoFloat::from(-0.3));
    }
    let twofloat_duration = start.elapsed();
    
    println!("f64 computation: {:?} for {} iterations", f64_duration, num_iterations);
    println!("TwoFloat computation: {:?} for {} iterations", twofloat_duration, num_iterations);
    
    // TwoFloat should be slower but not dramatically so
    let ratio = twofloat_duration.as_nanos() as f64 / f64_duration.as_nanos() as f64;
    println!("TwoFloat/f64 ratio: {:.2}", ratio);
    
    // TwoFloat should not be more than 10x slower
    assert!(ratio < 10.0, "TwoFloat computation is too slow: {:.2}x slower", ratio);
}

#[test]
fn benchmark_weight_computation() {
    let kernel = LogisticKernel::new(10.0);
    let num_iterations = 10000;
    
    // Benchmark weight computation
    let start = Instant::now();
    for _ in 0..num_iterations {
        let _ = kernel.weight::<Fermionic>(1.0, 1.0);
        let _ = kernel.weight::<Bosonic>(1.0, 1.0);
    }
    let weight_duration = start.elapsed();
    
    // Benchmark inv_weight computation
    let start = Instant::now();
    for _ in 0..num_iterations {
        let _ = kernel.inv_weight::<Fermionic>(1.0, 1.0);
        let _ = kernel.inv_weight::<Bosonic>(1.0, 1.0);
    }
    let inv_weight_duration = start.elapsed();
    
    println!("Weight computation: {:?} for {} iterations", weight_duration, num_iterations);
    println!("Inv_weight computation: {:?} for {} iterations", inv_weight_duration, num_iterations);
    
    // Both should be fast (microseconds per operation)
    let weight_per_op = weight_duration.as_nanos() as f64 / (num_iterations * 2) as f64;
    let inv_weight_per_op = inv_weight_duration.as_nanos() as f64 / (num_iterations * 2) as f64;
    
    println!("Weight per operation: {:.0} ns", weight_per_op);
    println!("Inv_weight per operation: {:.0} ns", inv_weight_per_op);
    
    // Should be fast (less than 1000ns per operation)
    assert!(weight_per_op < 1000.0, "Weight computation too slow: {:.0} ns", weight_per_op);
    assert!(inv_weight_per_op < 1000.0, "Inv_weight computation too slow: {:.0} ns", inv_weight_per_op);
}

#[test]
fn benchmark_kernel_creation() {
    let num_iterations = 10000;
    
    let start = Instant::now();
    for _ in 0..num_iterations {
        let _ = LogisticKernel::new(10.0);
        let _ = RegularizedBoseKernel::new(10.0);
    }
    let duration = start.elapsed();
    
    println!("Kernel creation: {:?} for {} iterations", duration, num_iterations);
    
    let per_op = duration.as_nanos() as f64 / (num_iterations * 2) as f64;
    println!("Kernel creation per operation: {:.0} ns", per_op);
    
    // Kernel creation should be very fast (less than 100ns per operation)
    assert!(per_op < 100.0, "Kernel creation too slow: {:.0} ns", per_op);
}
