//! Piecewise Legendre polynomial usage examples

use sparseir_rust::*;

fn main() {
    println!("=== Piecewise Legendre Polynomial Examples ===\n");
    
    // Example 1: Single polynomial
    example_single_polynomial();
    
    // Example 2: Multiple polynomials in a vector
    example_polynomial_vector();
    
    // Example 3: Memory efficiency demonstration
    example_memory_efficiency();
    
    // Example 4: High precision evaluation
    example_high_precision_evaluation();
}

fn example_single_polynomial() {
    println!("1. Single Polynomial Example");
    println!("   Creating P(x) = 1.0 + 2.0*x + 3.0*x^2 on interval [0, 2]");
    
    let poly = PiecewiseLegendrePoly::new(
        vec![1.0, 2.0, 3.0],  // coefficients
        vec![(0.0, 2.0)],     // single interval
        2,                     // degree
    );
    
    println!("   Coefficients: {:?}", poly.coefficients());
    println!("   Intervals: {:?}", poly.intervals());
    println!("   Degree: {}", poly.degree());
    println!("   Domain: {:?}", poly.domain());
    
    // Evaluate at different points
    let points = vec![0.0, 0.5, 1.0, 1.5, 2.0];
    println!("   Evaluation:");
    for &x in &points {
        let value = poly.evaluate_f64(x);
        println!("     P({}) = {:.6}", x, value);
    }
    
    println!();
}

fn example_polynomial_vector() {
    println!("2. Polynomial Vector Example");
    println!("   Creating vector with multiple polynomials");
    
    // Create individual polynomials
    let poly1 = PiecewiseLegendrePoly::new(
        vec![1.0, 2.0],           // P1(x) = 1 + 2x
        vec![(0.0, 1.0)],         // on [0, 1]
        1,
    );
    
    let poly2 = PiecewiseLegendrePoly::new(
        vec![3.0, 4.0, 5.0],      // P2(x) = 3 + 4x + 5x^2
        vec![(1.0, 2.0)],         // on [1, 2]
        2,
    );
    
    let poly3 = PiecewiseLegendrePoly::new(
        vec![6.0, 7.0],           // P3(x) = 6 + 7x
        vec![(2.0, 3.0)],         // on [2, 3]
        1,
    );
    
    // Create vector
    let vector = PiecewiseLegendrePolyVector::new(vec![poly1, poly2, poly3]);
    
    println!("   Vector length: {}", vector.len());
    println!("   Vector is empty: {}", vector.is_empty());
    
    // Access by index
    println!("   Access by index:");
    for i in 0..vector.len() {
        let poly = &vector[i];
        println!("     Polynomial {}: degree = {}, domain = {:?}", 
                 i, poly.degree(), poly.domain());
    }
    
    // Extract individual polynomials
    println!("   Extracting polynomials:");
    let extracted = vector.extract(1).unwrap();
    println!("     Extracted polynomial degree: {}", extracted.degree());
    
    // Iterate over all polynomials
    println!("   Iteration:");
    for (i, poly) in vector.iter().enumerate() {
        let value = poly.evaluate_f64(0.5);
        println!("     P{}(0.5) = {:.6}", i, value);
    }
    
    println!();
}

fn example_memory_efficiency() {
    println!("3. Memory Efficiency Example");
    println!("   Demonstrating shared data storage");
    
    // Create polynomials with overlapping data
    let poly1 = PiecewiseLegendrePoly::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![(0.0, 1.0), (1.0, 2.0)],
        4,
    );
    
    let poly2 = PiecewiseLegendrePoly::new(
        vec![6.0, 7.0, 8.0],
        vec![(2.0, 3.0)],
        2,
    );
    
    let vector = PiecewiseLegendrePolyVector::new(vec![poly1, poly2]);
    
    // Extract multiple polynomials
    let extracted1 = vector.extract(0).unwrap();
    let extracted2 = vector.extract(1).unwrap();
    
    println!("   Original vector length: {}", vector.len());
    // Check if they share the same underlying data by comparing shared data references
    let shared_data1 = extracted1.shared_data();
    let shared_data2 = extracted2.shared_data();
    println!("   Extracted polynomials share data: {}", 
             std::ptr::eq(shared_data1, shared_data2));
    
    // Show that data is shared (Arc reference counting)
    let shared_data = vector.shared_data();
    println!("   Shared data contains {} coefficients", 
             shared_data.all_coefficients.len());
    println!("   Shared data contains {} intervals", 
             shared_data.all_intervals.len());
    println!("   Shared data contains {} polynomial info entries", 
             shared_data.polynomial_info.len());
    
    println!();
}

fn example_high_precision_evaluation() {
    println!("4. High Precision Evaluation Example");
    println!("   Comparing f64 vs TwoFloat precision");
    
    // Create a polynomial with coefficients that might lose precision
    let poly = PiecewiseLegendrePoly::new(
        vec![1.0e-15, 2.0e-10, 3.0e-5, 4.0],
        vec![(0.0, 1.0)],
        3,
    );
    
    let x = 0.5;
    
    // Evaluate with f64 precision
    let value_f64 = poly.evaluate_f64(x);
    
    // Evaluate with high precision
    let value_high = poly.evaluate(TwoFloat::from(x));
    let value_high_f64: f64 = value_high.into();
    
    println!("   Polynomial: P(x) = 1e-15 + 2e-10*x + 3e-5*x^2 + 4*x^3");
    println!("   Evaluation at x = {}", x);
    println!("   f64 precision: {:.15e}", value_f64);
    println!("   TwoFloat precision: {:.15e}", value_high_f64);
    println!("   Difference: {:.2e}", (value_high_f64 - value_f64).abs());
    
    println!();
}

fn example_piecewise_evaluation() {
    println!("5. Piecewise Evaluation Example");
    println!("   Polynomial with multiple intervals");
    
    // Create a piecewise polynomial
    let poly = PiecewiseLegendrePoly::new(
        vec![1.0, 2.0],  // coefficients
        vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],  // multiple intervals
        1,  // linear polynomial
    );
    
    println!("   Polynomial: P(x) = 1 + 2*x on multiple intervals");
    println!("   Intervals: {:?}", poly.intervals());
    
    // Evaluate at points across all intervals
    let points = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    println!("   Evaluation across intervals:");
    
    for &x in &points {
        let value = poly.evaluate_f64(x);
        println!("     P({}) = {:.6}", x, value);
    }
    
    // Test boundary conditions
    println!("   Boundary evaluation:");
    println!("     P(1.0) = {:.6}", poly.evaluate_f64(1.0));
    println!("     P(2.0) = {:.6}", poly.evaluate_f64(2.0));
    println!("     P(3.0) = {:.6}", poly.evaluate_f64(3.0));
    
    println!();
}

fn example_error_handling() {
    println!("6. Error Handling Example");
    println!("   Testing edge cases and error conditions");
    
    // Empty vector
    let empty_vector = PiecewiseLegendrePolyVector::new(vec![]);
    println!("   Empty vector length: {}", empty_vector.len());
    println!("   Empty vector get(0): {:?}", empty_vector.get(0));
    println!("   Empty vector extract(0): {:?}", empty_vector.extract(0));
    
    // Vector with one polynomial
    let poly = PiecewiseLegendrePoly::new(vec![1.0], vec![(0.0, 1.0)], 0);
    let vector = PiecewiseLegendrePolyVector::new(vec![poly]);
    
    // Valid access
    println!("   Valid access get(0): {:?}", vector.get(0).is_some());
    println!("   Valid extraction extract(0): {:?}", vector.extract(0).is_some());
    
    // Invalid access
    println!("   Invalid access get(1): {:?}", vector.get(1).is_some());
    println!("   Invalid extraction extract(1): {:?}", vector.extract(1).is_some());
    
    // Evaluation outside domain
    let poly = PiecewiseLegendrePoly::new(vec![1.0], vec![(0.0, 1.0)], 0);
    println!("   Evaluation inside domain P(0.5): {:.6}", poly.evaluate_f64(0.5));
    println!("   Evaluation outside domain P(2.0): {:.6}", poly.evaluate_f64(2.0));
    
    println!();
}
