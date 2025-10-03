use ndarray::array;

fn main() {
    println!("=== Debug Threshold Calculation ===");
    
    // Test case: [0, 1], [1, 0]
    let a = array![
        [0.0_f64, 1.0_f64],
        [1.0_f64, 0.0_f64]
    ];
    
    println!("Input matrix A:\n{}", a);
    
    // Calculate initial max_diag_entry
    let m = a.nrows();
    let n = a.ncols();
    let mut max_diag_entry = 0.0_f64;
    for i in 0..m {
        for j in 0..n {
            max_diag_entry = max_diag_entry.max(a[[i, j]].abs());
        }
    }
    
    println!("\nInitial max_diag_entry = {}", max_diag_entry);
    
    let consider_as_zero = std::f64::MIN_POSITIVE;
    let precision = 2.0 * std::f64::EPSILON;
    
    println!("consider_as_zero = {}", consider_as_zero);
    println!("precision = {}", precision);
    
    let threshold = consider_as_zero.max(precision * max_diag_entry);
    
    println!("\nthreshold = max(consider_as_zero, precision * max_diag_entry)");
    println!("          = max({}, {} * {})", consider_as_zero, precision, max_diag_entry);
    println!("          = max({}, {})", consider_as_zero, precision * max_diag_entry);
    println!("          = {}", threshold);
    
    println!("\nOff-diagonal elements:");
    println!("|a[0,1]| = {}", a[[0, 1]].abs());
    println!("|a[1,0]| = {}", a[[1, 0]].abs());
    
    println!("\nComparison:");
    println!("|a[0,1]| <= threshold? {} <= {} = {}", a[[0, 1]].abs(), threshold, a[[0, 1]].abs() <= threshold);
    println!("|a[1,0]| <= threshold? {} <= {} = {}", a[[1, 0]].abs(), threshold, a[[1, 0]].abs() <= threshold);
    
    if a[[0, 1]].abs() <= threshold && a[[1, 0]].abs() <= threshold {
        println!("\n=> Would SKIP Jacobi SVD (threshold too high!)");
    } else {
        println!("\n=> Would PROCESS Jacobi SVD (correct)");
    }
}
