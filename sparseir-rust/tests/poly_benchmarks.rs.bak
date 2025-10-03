//! Benchmarks for the poly module

use sparseir_rust::*;
use std::time::Instant;

#[test]
fn benchmark_polynomial_creation() {
    let start = Instant::now();
    
    // Create 1000 individual polynomials
    for i in 0..1000 {
        let _poly = PiecewiseLegendrePoly::new(
            vec![i as f64, (i + 1) as f64, (i + 2) as f64],
            vec![(0.0, 1.0), (1.0, 2.0)],
            2,
        );
    }
    
    let duration = start.elapsed();
    println!("Individual polynomial creation (1000): {:?}", duration);
    
    // Create 1000 polynomials in a vector
    let start = Instant::now();
    let polynomials: Vec<_> = (0..1000)
        .map(|i| {
            PiecewiseLegendrePoly::new(
                vec![i as f64, (i + 1) as f64, (i + 2) as f64],
                vec![(0.0, 1.0), (1.0, 2.0)],
                2,
            )
        })
        .collect();
    let _vector = PiecewiseLegendrePolyVector::new(polynomials);
    
    let duration = start.elapsed();
    println!("Vector polynomial creation (1000): {:?}", duration);
}

#[test]
fn benchmark_polynomial_evaluation() {
    let poly = PiecewiseLegendrePoly::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![(0.0, 1.0), (1.0, 2.0)],
        4,
    );
    
    let points: Vec<f64> = (0..10000).map(|i| i as f64 / 5000.0).collect();
    
    // Benchmark f64 evaluation
    let start = Instant::now();
    let mut _sum = 0.0;
    for &x in &points {
        _sum += poly.evaluate_f64(x);
    }
    let f64_duration = start.elapsed();
    
    // Benchmark TwoFloat evaluation
    let start = Instant::now();
    let mut _sum_high = TwoFloat::from(0.0);
    for &x in &points {
        _sum_high += poly.evaluate(TwoFloat::from(x));
    }
    let high_precision_duration = start.elapsed();
    
    println!("f64 evaluation (10000 points): {:?}", f64_duration);
    println!("TwoFloat evaluation (10000 points): {:?}", high_precision_duration);
    println!("TwoFloat/f64 ratio: {:.2}x", 
        high_precision_duration.as_nanos() as f64 / f64_duration.as_nanos() as f64);
}

#[test]
fn benchmark_vector_operations() {
    // Create a large vector
    let polynomials: Vec<_> = (0..1000)
        .map(|i| {
            PiecewiseLegendrePoly::new(
                vec![i as f64, (i + 1) as f64],
                vec![(0.0, 1.0)],
                1,
            )
        })
        .collect();
    let vector = PiecewiseLegendrePolyVector::new(polynomials);
    
    // Benchmark indexing
    let start = Instant::now();
    let mut _sum = 0.0;
    for i in 0..1000 {
        let poly = &vector[i];
        _sum += poly.evaluate_f64(0.5);
    }
    let indexing_duration = start.elapsed();
    
    // Benchmark extraction
    let start = Instant::now();
    let mut _extracted = Vec::new();
    for i in 0..100 {
        _extracted.push(vector.extract(i).unwrap());
    }
    let extraction_duration = start.elapsed();
    
    // Benchmark iteration
    let start = Instant::now();
    let mut _sum = 0.0;
    for poly in vector.iter() {
        _sum += poly.evaluate_f64(0.5);
    }
    let iteration_duration = start.elapsed();
    
    println!("Vector indexing (1000): {:?}", indexing_duration);
    println!("Vector extraction (100): {:?}", extraction_duration);
    println!("Vector iteration (1000): {:?}", iteration_duration);
}

#[test]
fn benchmark_memory_efficiency() {
    // Create polynomials with different data sizes
    let sizes = vec![10, 100, 1000];
    
    for size in sizes {
        let start = Instant::now();
        
        let polynomials: Vec<_> = (0..size)
            .map(|i| {
                PiecewiseLegendrePoly::new(
                    vec![i as f64; 10], // 10 coefficients each
                    vec![(0.0, 1.0); 5], // 5 intervals each
                    9,
                )
            })
            .collect();
        let vector = PiecewiseLegendrePolyVector::new(polynomials);
        
        let creation_duration = start.elapsed();
        
        // Extract all polynomials to test sharing efficiency
        let start = Instant::now();
        let _extracted: Vec<_> = (0..size)
            .map(|i| vector.extract(i).unwrap())
            .collect();
        let extraction_duration = start.elapsed();
        
        println!("Size {} - Creation: {:?}, Extraction: {:?}", 
                 size, creation_duration, extraction_duration);
    }
}

#[test]
fn benchmark_piecewise_evaluation() {
    // Create a polynomial with many intervals
    let intervals: Vec<_> = (0..100)
        .map(|i| (i as f64, (i + 1) as f64))
        .collect();
    
    let poly = PiecewiseLegendrePoly::new(
        vec![1.0, 2.0, 3.0],
        intervals,
        2,
    );
    
    let points: Vec<f64> = (0..10000).map(|i| i as f64 / 100.0).collect();
    
    let start = Instant::now();
    let mut _sum = 0.0;
    for &x in &points {
        _sum += poly.evaluate_f64(x);
    }
    let duration = start.elapsed();
    
    println!("Piecewise evaluation (100 intervals, 10000 points): {:?}", duration);
}

#[test]
fn benchmark_shared_data_access() {
    // Create a vector with shared data
    let polynomials: Vec<_> = (0..100)
        .map(|i| {
            PiecewiseLegendrePoly::new(
                vec![i as f64, (i + 1) as f64, (i + 2) as f64],
                vec![(0.0, 1.0)],
                2,
            )
        })
        .collect();
    let vector = PiecewiseLegendrePolyVector::new(polynomials);
    
    // Benchmark shared data access
    let start = Instant::now();
    let shared_data = vector.shared_data();
    let mut _sum = 0.0;
    for i in 0..100 {
        let coeffs = shared_data.get_coefficients(i).unwrap();
        _sum += coeffs.iter().sum::<f64>();
    }
    let shared_access_duration = start.elapsed();
    
    // Benchmark individual polynomial access
    let start = Instant::now();
    let mut _sum = 0.0;
    for i in 0..100 {
        let poly = &vector[i];
        let coeffs = poly.coefficients();
        _sum += coeffs.iter().sum::<f64>();
    }
    let individual_access_duration = start.elapsed();
    
    println!("Shared data access (100): {:?}", shared_access_duration);
    println!("Individual access (100): {:?}", individual_access_duration);
    println!("Shared/Individual ratio: {:.2}x", 
        shared_access_duration.as_nanos() as f64 / individual_access_duration.as_nanos() as f64);
}
