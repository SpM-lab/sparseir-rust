//! Comparison tests between Rust and Julia SVE implementations

use sparseir_rust::{compute_sve, LogisticKernel, TworkType};
use ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Load reference singular values from Julia
fn load_reference_singular_values(filepath: &str) -> Array1<f64> {
    let file = File::open(filepath).expect(&format!("Failed to open {}", filepath));
    let reader = BufReader::new(file);
    let values: Vec<f64> = reader
        .lines()
        .map(|line| line.unwrap().trim().parse().unwrap())
        .collect();
    Array1::from(values)
}

/// Load reference matrix (u or v values) from Julia
fn load_reference_matrix(filepath: &str) -> Array2<f64> {
    let file = File::open(filepath).expect(&format!("Failed to open {}", filepath));
    let reader = BufReader::new(file);
    
    let mut data = Vec::new();
    let mut nrows = 0;
    let mut ncols = 0;
    
    for line in reader.lines() {
        let line = line.unwrap();
        let values: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        
        if ncols == 0 {
            ncols = values.len();
        }
        data.extend(values);
        nrows += 1;
    }
    
    Array2::from_shape_vec((nrows, ncols), data).unwrap()
}

#[test]
fn test_sve_singular_values_lambda_5() {
    let lambda = 5.0;
    let epsilon = 1e-6;
    
    println!("Testing SVE for λ={}, ε={}", lambda, epsilon);
    
    let kernel = LogisticKernel::new(lambda);
    let result = compute_sve(
        kernel,
        epsilon,
        None,  // cutoff
        None,  // max_num_svals
        TworkType::Auto,
    );
    
    // Load reference values from Julia
    let s_ref = load_reference_singular_values(
        "tests/reference_data/sve_lambda_5.0_eps_1.0e-6_svals.txt"
    );
    
    println!("Rust: {} singular values", result.s.len());
    println!("Julia: {} singular values", s_ref.len());
    
    // Print all Rust singular values
    println!("\nAll Rust singular values:");
    for (i, &s) in result.s.iter().enumerate() {
        println!("  Rust s[{}] = {:.6e}", i, s);
    }
    
    // Filter significant singular values in Rust result
    let threshold = epsilon * result.s[0];
    println!("\nThreshold = {} * {} = {:.6e}", epsilon, result.s[0], threshold);
    
    let significant_rust: Vec<f64> = result.s
        .iter()
        .filter(|&&s| s > threshold)
        .copied()
        .collect();
    
    println!("Rust significant (> threshold): {}", significant_rust.len());
    println!("First Rust s[0] = {}", result.s[0]);
    
    // Compare number of significant singular values
    assert_eq!(
        significant_rust.len(),
        s_ref.len(),
        "Number of significant singular values mismatch"
    );
    
    // Compare singular values
    println!("\nComparing singular values:");
    for i in 0..significant_rust.len() {
        let rust_val = significant_rust[i];
        let julia_val = s_ref[i];
        let abs_error = (rust_val - julia_val).abs();
        let rel_error = abs_error / julia_val;
        
        println!(
            "  s[{}]: rust={:.6e}, julia={:.6e}, rel_err={:.2e}",
            i, rust_val, julia_val, rel_error
        );
        
        assert!(
            rel_error < 1e-10,
            "Singular value {} mismatch: rust={}, julia={}, rel_error={}",
            i, rust_val, julia_val, rel_error
        );
    }
    
    println!("\n✓ All {} singular values matched within 1e-10 relative error", 
             significant_rust.len());
}

#[test]
fn test_sve_singular_functions_lambda_5() {
    let lambda = 5.0;
    let epsilon = 1e-6;
    
    println!("Testing SVE singular functions for λ={}, ε={}", lambda, epsilon);
    
    let kernel = LogisticKernel::new(lambda);
    let result = compute_sve(kernel, epsilon, None, None, TworkType::Auto);
    
    // Test points
    let x_test = vec![-0.9, -0.5, 0.0, 0.5, 0.9];
    
    // Load reference u and v values
    let u_ref = load_reference_matrix("tests/reference_data/sve_lambda_5.0_eps_1.0e-6_u.txt");
    let v_ref = load_reference_matrix("tests/reference_data/sve_lambda_5.0_eps_1.0e-6_v.txt");
    
    let n_funcs = u_ref.ncols();
    println!("Comparing first {} singular functions", n_funcs);
    
    // Compare u functions
    println!("\nComparing u functions:");
    for i in 0..n_funcs {
        println!("  u[{}]:", i);
        let mut max_error: f64 = 0.0;
        for (j, &x) in x_test.iter().enumerate() {
            let u_rust = result.u.get_polys()[i].evaluate(x);
            let u_julia = u_ref[[j, i]];
            let abs_error = (u_rust - u_julia).abs();
            println!("    x={:5.2}: rust={:12.6e}, julia={:12.6e}, error={:.2e}", 
                     x, u_rust, u_julia, abs_error);
            max_error = max_error.max(abs_error);
        }
        
        println!("    max_abs_error = {:.2e}", max_error);
        
        assert!(
            max_error < 1e-10,
            "u[{}] error too large: {:.2e}",
            i, max_error
        );
    }
    
    // Compare v functions
    println!("\nComparing v functions:");
    for i in 0..n_funcs {
        let max_error = x_test.iter().enumerate().map(|(j, &x)| {
            let v_rust = result.v.get_polys()[i].evaluate(x);
            let v_julia = v_ref[[j, i]];
            (v_rust - v_julia).abs()
        }).fold(0.0, f64::max);
        
        println!("  v[{}]: max_abs_error = {:.2e}", i, max_error);
        
        assert!(
            max_error < 1e-10,
            "v[{}] error too large: {:.2e}",
            i, max_error
        );
    }
    
    println!("\n✓ All singular functions matched within 1e-10 absolute error");
}

