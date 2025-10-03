use xprec_svd::precision::Precision;

fn main() {
    println!("=== Debug 2x2 Jacobi SVD Step by Step ===");
    
    // Test case: [0, 1], [1, 0]
    let a: f64 = 0.0;
    let b: f64 = 1.0;
    let c: f64 = 1.0;
    let d: f64 = 0.0;
    
    println!("Input matrix:");
    println!("[{}, {}]", a, b);
    println!("[{}, {}]", c, d);
    
    // Manual step-by-step calculation
    let mut m = [[a, b], [c, d]];
    println!("\nInitial m:");
    println!("[{}, {}]", m[0][0], m[0][1]);
    println!("[{}, {}]", m[1][0], m[1][1]);
    
    // First rotation to eliminate off-diagonal elements
    let t = m[0][0] + m[1][1];
    let d_val = m[1][0] - m[0][1];
    
    println!("\nt = m[0][0] + m[1][1] = {} + {} = {}", m[0][0], m[1][1], t);
    println!("d_val = m[1][0] - m[0][1] = {} - {} = {}", m[1][0], m[0][1], d_val);
    
    let (c1, s1) = if d_val.abs() < f64::EPSILON {
        println!("d_val is too small, using identity rotation");
        (1.0, 0.0)
    } else {
        let u = t / d_val;
        let tmp = (1.0 + u * u).sqrt();
        let c_val = u / tmp;
        let s_val = 1.0 / tmp;
        println!("u = t / d_val = {} / {} = {}", t, d_val, u);
        println!("tmp = sqrt(1 + u*u) = sqrt(1 + {}*{}) = {}", u, u, tmp);
        println!("c1 = u / tmp = {} / {} = {}", u, tmp, c_val);
        println!("s1 = 1 / tmp = 1 / {} = {}", tmp, s_val);
        (c_val, s_val)
    };
    
    // Apply left rotation
    println!("\nApply left rotation:");
    let temp1 = c1 * m[0][0] - s1 * m[1][0];
    let temp2 = c1 * m[0][1] - s1 * m[1][1];
    m[1][0] = s1 * m[0][0] + c1 * m[1][0];
    m[1][1] = s1 * m[0][1] + c1 * m[1][1];
    m[0][0] = temp1;
    m[0][1] = temp2;
    
    println!("After left rotation:");
    println!("[{}, {}]", m[0][0], m[0][1]);
    println!("[{}, {}]", m[1][0], m[1][1]);
    
    // Second rotation to diagonalize
    let n = (m[0][0] * m[0][0] + m[1][0] * m[1][0]).sqrt();
    println!("\nn = sqrt(m[0][0]^2 + m[1][0]^2) = sqrt({}^2 + {}^2) = {}", m[0][0], m[1][0], n);
    
    let (c2, s2) = if n.abs() < f64::EPSILON {
        println!("n is too small, using identity rotation");
        (1.0, 0.0)
    } else {
        let c_val = m[0][0] / n;
        let s_val = m[1][0] / n;
        println!("c2 = m[0][0] / n = {} / {} = {}", m[0][0], n, c_val);
        println!("s2 = m[1][0] / n = {} / {} = {}", m[1][0], n, s_val);
        (c_val, s_val)
    };
    
    // Apply right rotation
    println!("\nApply right rotation:");
    let temp1 = c2 * m[0][0] - s2 * m[0][1];
    let temp2 = c2 * m[1][0] - s2 * m[1][1];
    m[0][1] = s2 * m[0][0] + c2 * m[0][1];
    m[1][1] = s2 * m[1][0] + c2 * m[1][1];
    m[0][0] = temp1;
    m[1][0] = temp2;
    
    println!("After right rotation:");
    println!("[{}, {}]", m[0][0], m[0][1]);
    println!("[{}, {}]", m[1][0], m[1][1]);
    
    let s1_final = m[0][0].abs();
    let s2_final = m[1][1].abs();
    
    println!("\nFinal singular values:");
    println!("s1 = |m[0][0]| = |{}| = {}", m[0][0], s1_final);
    println!("s2 = |m[1][1]| = |{}| = {}", m[1][1], s2_final);
    
    println!("\nExpected: s1 = 1.0, s2 = 1.0");
}
