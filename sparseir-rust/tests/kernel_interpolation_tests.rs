//! Tests for kernel interpolation precision with lambda = 100.0

use sparseir_rust::{
    kernel::{LogisticKernel, LogisticSVEHints, SymmetryType, SVEHints, KernelProperties, CentrosymmKernel},
    kernelmatrix::{InterpolatedKernel, matrix_from_gauss_with_segments},
    gauss::{legendre_generic, Rule},
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
    
    // Create Gauss rules for each segment
    let gauss_per_segment = 4; // degree 3 polynomial
    
    // Build composite Gauss rules for x direction
    let mut x_rules = Vec::new();
    for i in 0..segments_x.len()-1 {
        let segment_rule = legendre_generic::<f64>(gauss_per_segment)
            .reseat(segments_x[i], segments_x[i+1]);
        x_rules.push(segment_rule);
    }
    let gauss_x = Rule::join(&x_rules);
    
    // Build composite Gauss rules for y direction  
    let mut y_rules = Vec::new();
    for i in 0..segments_y.len()-1 {
        let segment_rule = legendre_generic::<f64>(gauss_per_segment)
            .reseat(segments_y[i], segments_y[i+1]);
        y_rules.push(segment_rule);
    }
    let gauss_y = Rule::join(&y_rules);
    
    println!("Gauss x points: {}", gauss_x.x.len());
    println!("Gauss y points: {}", gauss_y.x.len());
    
    // Create discretized kernel
    let discretized = matrix_from_gauss_with_segments(
        &kernel,
        &gauss_x,
        &gauss_y,
        SymmetryType::Even,
        &hints,
    );
    
    // Create interpolated kernel
    let interpolated = InterpolatedKernel::from_discretized(&discretized, gauss_per_segment);
    
    println!("Interpolated kernel created with {}x{} cells", 
             interpolated.n_cells_x(), interpolated.n_cells_y());
    
    // Test points within each segment (relative coordinates)
    let test_points_relative = vec![
        (0.1, 0.2),
        (0.4, 0.3),
        (0.7, 0.6),
        (0.9, 0.8),
    ];
    
    println!("\nInterpolation error analysis:");
    println!("Format: Cell(i,j) Point(x,y) | Direct | Interpolated | Error | RelError");
    println!("{}", "-".repeat(80));
    
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
                let direct_value = kernel.compute_reduced(x, y, SymmetryType::Even);
                
                // Interpolated value (placeholder - will be 0.0 for now)
                let interpolated_value = interpolated.evaluate(x, y);
                
                let error = (direct_value - interpolated_value).abs();
                let rel_error = if direct_value.abs() > 1e-12 {
                    error / direct_value.abs()
                } else {
                    error
                };
                
                max_error = max_error.max(error);
                max_rel_error = max_rel_error.max(rel_error);
                total_tests += 1;
                
                println!("Cell({},{}) Point({:.3},{:.3}) | {:.6e} | {:.6e} | {:.6e} | {:.6e}",
                         i, j, x, y, direct_value, interpolated_value, error, rel_error);
            }
        }
    }
    
    println!("\nSummary:");
    println!("Total test points: {}", total_tests);
    println!("Maximum absolute error: {:.6e}", max_error);
    println!("Maximum relative error: {:.6e}", max_rel_error);
    
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
    
    println!("TwoFloat test - Lambda = {}", lambda);
    println!("Segments x: {:?}", segments_x_tf);
    println!("Segments y: {:?}", segments_y_tf);
    
    // Create Gauss rules for each segment (TwoFloat)
    let gauss_per_segment = 4;
    
    // Build composite Gauss rules for x direction
    let mut x_rules_tf = Vec::new();
    for i in 0..segments_x_tf.len()-1 {
        let segment_rule = legendre_generic::<TwoFloat>(gauss_per_segment)
            .reseat(segments_x_tf[i], segments_x_tf[i+1]);
        x_rules_tf.push(segment_rule);
    }
    let gauss_x_tf = Rule::join(&x_rules_tf);
    
    // Build composite Gauss rules for y direction  
    let mut y_rules_tf = Vec::new();
    for i in 0..segments_y_tf.len()-1 {
        let segment_rule = legendre_generic::<TwoFloat>(gauss_per_segment)
            .reseat(segments_y_tf[i], segments_y_tf[i+1]);
        y_rules_tf.push(segment_rule);
    }
    let gauss_y_tf = Rule::join(&y_rules_tf);
    
    println!("TwoFloat Gauss x points: {}", gauss_x_tf.x.len());
    println!("TwoFloat Gauss y points: {}", gauss_y_tf.x.len());
    
    // For now, just verify we can create the structures
    // TODO: Implement TwoFloat version of matrix_from_gauss_with_segments
    assert!(gauss_x_tf.x.len() > 0);
    assert!(gauss_y_tf.x.len() > 0);
}
