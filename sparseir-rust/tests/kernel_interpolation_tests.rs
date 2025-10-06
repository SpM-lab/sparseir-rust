//! Tests for kernel interpolation precision

use sparseir_rust::{
    kernel::{LogisticKernel, LogisticSVEHints, SymmetryType, SVEHints, KernelProperties, CentrosymmKernel},
    kernelmatrix::InterpolatedKernel,
    numeric::CustomNumeric,
    TwoFloat,
};


/// Generic test for kernel interpolation precision
/// 
/// This test creates a discretized kernel and interpolated kernel, then
/// evaluates interpolation error at various points within each segment.
fn test_kernel_interpolation_precision_generic<T: CustomNumeric + Clone + 'static>(
    lambda: f64,
    epsilon: f64,
    tolerance_abs: f64,
    tolerance_rel: f64,
    symmetry_type: SymmetryType,
) where
    T: std::fmt::Debug,
{
    let kernel = LogisticKernel::new(lambda);
    let hints = LogisticSVEHints::new(kernel.clone(), epsilon);
    
    // Create segments for discretization
    let segments_x_f64 = hints.segments_x();
    let segments_y_f64 = hints.segments_y();
    
    // Convert to generic type T
    let segments_x: Vec<T> = segments_x_f64.iter().map(|&x| T::from_f64(x)).collect();
    let segments_y: Vec<T> = segments_y_f64.iter().map(|&y| T::from_f64(y)).collect();
    
    println!("Lambda = {}", lambda);
    println!("Segments x: {:?}", segments_x_f64);
    println!("Segments y: {:?}", segments_y_f64);
    
    // Create interpolated kernel directly from kernel and segments
    let gauss_per_cell = hints.ngauss(); // Get polynomial degree from SVE hints
    
    println!("Creating interpolated kernel with {}x{} cells, {} points per cell", 
             segments_x.len()-1, segments_y.len()-1, gauss_per_cell);
    
    // Create interpolated kernel directly (no composite Gauss rules needed)
    let interpolated = InterpolatedKernel::from_kernel_and_segments(
        &kernel,
        segments_x.clone(),
        segments_y.clone(),
        gauss_per_cell,
        symmetry_type,
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
    
    
    let mut max_error: T = T::from_f64(0.0);
    let mut max_rel_error: T = T::from_f64(0.0);
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
                let x = x_min + T::from_f64(rel_x) * (x_max - x_min);
                let y = y_min + T::from_f64(rel_y) * (y_max - y_min);
                
                
                // Skip if point is outside kernel domain
                if x > T::from_f64(kernel.xmax()) || y > T::from_f64(kernel.ymax()) {
                    continue;
                }
                
                // Direct kernel evaluation with specified symmetry type
                let direct_value: T = kernel.compute_reduced(x, y, symmetry_type);
                
                // Interpolated value
                let interpolated_value: T = interpolated.evaluate(x, y);
                
                
                let error: T = (direct_value - interpolated_value).abs();
                let rel_error = if direct_value.abs() > T::from_f64(1e-12) {
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
    
    
    println!("Max absolute error: {:.6e}", max_error.to_f64());
    println!("Max relative error: {:.6e}", max_rel_error.to_f64());
    println!("Total test points: {}", total_tests);
    
    println!("Tolerance: absolute={:.0e}, relative={:.0e}", tolerance_abs, tolerance_rel);
    
    assert!(max_error < T::from_f64(tolerance_abs), 
            "Max absolute error {:.6e} exceeds tolerance {:.0e}", max_error.to_f64(), tolerance_abs);
    assert!(max_rel_error < T::from_f64(tolerance_rel),
            "Max relative error {:.6e} exceeds tolerance {:.0e}", max_rel_error.to_f64(), tolerance_rel);
    assert!(total_tests > 0, "Should have at least some test points");
}


/// Test kernel interpolation precision with f64
#[test]
fn test_kernel_interpolation_precision_f64_lambda_100() {
    println!("Testing f64 with Even symmetry");
    test_kernel_interpolation_precision_generic::<f64>(
        100.0,    // lambda
        1e-12,    // epsilon
        1e-12,    // tolerance_abs
        1e-10,    // tolerance_rel
        SymmetryType::Even,
    );
    
    println!("Testing f64 with Odd symmetry");
    test_kernel_interpolation_precision_generic::<f64>(
        100.0,    // lambda
        1e-12,    // epsilon
        1e-12,    // tolerance_abs
        1e-10,    // tolerance_rel
        SymmetryType::Odd,
    );
}

/// Test kernel interpolation precision with TwoFloat
#[test] 
fn test_kernel_interpolation_precision_twofloat_lambda_100() {
    println!("Testing TwoFloat with Even symmetry");
    test_kernel_interpolation_precision_generic::<TwoFloat>(
        100.0,    // lambda
        1e-12,    // epsilon
        1e-11,    // tolerance_abs 
        1e-10,    // tolerance_rel
        SymmetryType::Even,
    );
    
    println!("Testing TwoFloat with Odd symmetry");
    test_kernel_interpolation_precision_generic::<TwoFloat>(
        100.0,    // lambda
        1e-12,    // epsilon
        1e-11,    // tolerance_abs 
        1e-10,    // tolerance_rel
        SymmetryType::Odd,
    );
}
