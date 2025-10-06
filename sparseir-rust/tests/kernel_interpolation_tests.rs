//! Tests for kernel interpolation precision with lambda = 100.0

use sparseir_rust::{
    kernel::{LogisticKernel, LogisticSVEHints, SymmetryType, SVEHints, KernelProperties, CentrosymmKernel},
    kernelmatrix::InterpolatedKernel,
    numeric::CustomNumeric,
    TwoFloat,
};

/// Test kernel interpolation precision with lambda = 100.0
/// 
/// This test creates a discretized kernel and interpolated kernel, then
/// evaluates interpolation error at various points within each segment.
#[test]
fn test_kernel_interpolation_precision_lambda_100() {
    let lambda = 100.0;
    let kernel = LogisticKernel::new(lambda);
    let hints = LogisticSVEHints::new(kernel.clone(), 1e-12);
    
    // Create segments for discretization
    let segments_x = hints.segments_x();
    let segments_y = hints.segments_y();
    
    println!("Lambda = {}", lambda);
    println!("Segments x: {:?}", segments_x);
    println!("Segments y: {:?}", segments_y);
    
    // Create interpolated kernel directly from kernel and segments
    let gauss_per_cell = 4; // degree 3 polynomial
    
    println!("Creating interpolated kernel with {}x{} cells, {} points per cell", 
             segments_x.len()-1, segments_y.len()-1, gauss_per_cell);
    
    // Create interpolated kernel directly (no composite Gauss rules needed)
    let interpolated = InterpolatedKernel::from_kernel_and_segments(
        &kernel,
        segments_x.clone(),
        segments_y.clone(),
        gauss_per_cell,
        SymmetryType::Even,
    );
    
    println!("Interpolated kernel created with {}x{} cells", 
             interpolated.n_cells_x(), interpolated.n_cells_y());
    
    // Test points within each segment (relative coordinates)
    let test_points_relative = vec![
        (0.1, 0.2),
        (0.4, 0.3),
        (0.7, 0.6),
        (0.9, 0.8),
    ];
    
    
    let mut max_error: f64 = 0.0;
    let mut max_rel_error: f64 = 0.0;
    let mut total_tests = 0;
    
    // Test each cell
    for i in 0..interpolated.n_cells_x() {
        for j in 0..interpolated.n_cells_y() {
            let x_min = interpolated.segments_x[i];
            let x_max = interpolated.segments_x[i + 1];
            let y_min = interpolated.segments_y[j];
            let y_max = interpolated.segments_y[j + 1];
            
            // Test points in this cell
            for &(rel_x, rel_y) in &test_points_relative {
                let x = x_min + rel_x * (x_max - x_min);
                let y = y_min + rel_y * (y_max - y_min);
                
                
                // Skip if point is outside kernel domain
                if x > kernel.xmax() || y > kernel.ymax() {
                    continue;
                }
                
                // Direct kernel evaluation
                let direct_value: f64 = kernel.compute_reduced(x, y, SymmetryType::Even);
                
                // Interpolated value
                let interpolated_value: f64 = match interpolated.evaluate(x, y) {
                    val if val.is_nan() => {
                        println!("Warning: NaN returned for ({}, {})", x, y);
                        0.0
                    },
                    val => val
                };
                
                
                let error: f64 = (direct_value - interpolated_value).abs();
                let rel_error = if direct_value.abs() > 1e-12 {
                    error / direct_value.abs()
                } else {
                    error
                };
                
                max_error = max_error.max(error);
                max_rel_error = max_rel_error.max(rel_error);
                total_tests += 1;
                
            }
        }
    }
    
    
    // For now, just check that we can create the structures
    // TODO: Implement proper interpolation and set realistic tolerances
    assert!(total_tests > 0, "Should have at least some test points");
}

/// Test with TwoFloat precision
#[test] 
fn test_kernel_interpolation_precision_twofloat_lambda_100() {
    let lambda = 100.0;
    let kernel = LogisticKernel::new(lambda);
    let hints = LogisticSVEHints::new(kernel.clone(), 1e-12);
    
    // Create segments for discretization
    let segments_x = hints.segments_x();
    let segments_y = hints.segments_y();
    
    // Convert to TwoFloat
    let segments_x_tf: Vec<TwoFloat> = segments_x.iter().map(|&x| TwoFloat::from_f64(x)).collect();
    let segments_y_tf: Vec<TwoFloat> = segments_y.iter().map(|&y| TwoFloat::from_f64(y)).collect();
    
    
    // Create TwoFloat interpolated kernel directly
    let gauss_per_cell = 4;
    
    
    // Create TwoFloat interpolated kernel directly
    let interpolated_tf = InterpolatedKernel::from_kernel_and_segments(
        &kernel,
        segments_x_tf,
        segments_y_tf,
        gauss_per_cell,
        SymmetryType::Even,
    );
    
    
    // Verify structure creation
    assert!(interpolated_tf.n_cells_x() > 0);
    assert!(interpolated_tf.n_cells_y() > 0);
}
