use ndarray::array;

fn main() {
    println!("=== Rust Jacobi SVD Step by Step ===");
    
    let a = array![
        [0.0, 1.0],
        [1.0, 0.0]
    ];
    
    println!("Input matrix A:");
    println!("{}", a);
    
    // Initialize
    let m = a.nrows();
    let n = a.ncols();
    let k = m.min(n);
    
    let mut u = ndarray::Array2::<f64>::eye(m);
    let mut v = ndarray::Array2::<f64>::eye(n);
    let mut a_work = a.clone();
    
    println!("\nInitial U:");
    println!("{}", u);
    println!("\nInitial V:");
    println!("{}", v);
    println!("\nInitial A_work:");
    println!("{}", a_work);
    
    // Calculate threshold
    let consider_as_zero = std::f64::MIN_POSITIVE;
    let precision = 2.0 * std::f64::EPSILON;
    let mut max_diag_entry = 0.0_f64;
    for i in 0..m {
        for j in 0..n {
            let val: f64 = a_work[[i, j]];
            max_diag_entry = max_diag_entry.max(val.abs());
        }
    }
    
    println!("\nmax_diag_entry = {}", max_diag_entry);
    println!("precision = {}", precision);
    println!("consider_as_zero = {}", consider_as_zero);
    
    let max_iter = 30;
    
    for iter in 0..max_iter {
        println!("\n=== Iteration {} ===", iter);
        let mut converged = true;
        
        for i in 0..k {
            for j in (i + 1)..k {
                let threshold = consider_as_zero.max(precision * max_diag_entry);
                
                println!("\nProcessing block ({}, {})", i, j);
                println!("  a_work[{},{}] = {}", i, j, a_work[[i, j]]);
                println!("  a_work[{},{}] = {}", j, i, a_work[[j, i]]);
                println!("  threshold = {}", threshold);
                let aij: f64 = a_work[[i, j]];
                let aji: f64 = a_work[[j, i]];
                println!("  |a[i,j]| = {}, |a[j,i]| = {}", aij.abs(), aji.abs());
                
                if aij.abs() <= threshold && aji.abs() <= threshold {
                    println!("  => Skipping (below threshold)");
                    continue;
                }
                
                println!("  => Processing!");
                converged = false;
                
                // Print 2x2 submatrix
                println!("  2x2 submatrix:");
                println!("    [{}, {}]", a_work[[i, i]], a_work[[i, j]]);
                println!("    [{}, {}]", a_work[[j, i]], a_work[[j, j]]);
                
                // Here we would call real_2x2_jacobi_svd
                // For now, just print that we would process it
                println!("  Would call real_2x2_jacobi_svd here");
                
                break; // Exit early for debugging
            }
            if !converged {
                break; // Exit early for debugging
            }
        }
        
        if converged {
            println!("\n=> Converged after {} iterations", iter);
            break;
        }
        
        break; // Exit after first iteration for debugging
    }
    
    println!("\n=== Final State ===");
    println!("A_work:");
    println!("{}", a_work);
    println!("\nU:");
    println!("{}", u);
    println!("\nV:");
    println!("{}", v);
}
