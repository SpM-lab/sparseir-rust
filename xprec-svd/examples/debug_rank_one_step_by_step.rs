use xprec_svd::precision::Precision;
use xprec_svd::svd::jacobi::{real_2x2_jacobi_svd, JacobiRotation, make_jacobi, apply_rotation_left};

fn main() {
    println!("=== Debug Rank-One Matrix Step by Step ===");
    
    let a = 1.0;
    let b = 1.0;
    let c = 1.0;
    let d = 1.0;
    
    println!("Input matrix:");
    println!("[{}, {}]", a, b);
    println!("[{}, {}]", c, d);
    
    let mut m = [[a, b], [c, d]];
    println!("\nInitial m:");
    println!("[{}, {}]", m[0][0], m[0][1]);
    println!("[{}, {}]", m[1][0], m[1][1]);
    
    let t = m[0][0] + m[1][1];
    let d_val = m[1][0] - m[0][1];
    println!("\nt = m[0][0] + m[1][1] = {} + {} = {}", m[0][0], m[1][1], t);
    println!("d_val = m[1][0] - m[0][1] = {} - {} = {}", m[1][0], m[0][1], d_val);
    
    let rot1 = if Precision::abs(d_val) < f64::EPSILON {
        println!("d_val is too small, using identity rotation");
        JacobiRotation::identity()
    } else {
        let u = t / d_val;
        let tmp = Precision::sqrt(1.0 + u * u);
        let rot = JacobiRotation::new(u / tmp, 1.0 / tmp);
        println!("rot1: c = {}, s = {}", rot.c, rot.s);
        rot
    };
    
    println!("\nApply left rotation:");
    apply_rotation_left(&mut m, 0, 1, &rot1);
    println!("After left rotation:");
    println!("[{}, {}]", m[0][0], m[0][1]);
    println!("[{}, {}]", m[1][0], m[1][1]);
    
    println!("\nMake Jacobi rotation (rot2):");
    let rot2 = make_jacobi(&m, 0, 1);
    println!("x = m[0][0] = {}", m[0][0]);
    println!("y = m[0][1] = {}", m[0][1]);
    println!("z = m[1][1] = {}", m[1][1]);
    
    let two = 2.0;
    let deno = two * Precision::abs(m[0][1]);
    println!("deno = 2 * |y| = 2 * {} = {}", Precision::abs(m[0][1]), deno);
    
    if deno < f64::EPSILON {
        println!("deno is too small, using identity rotation");
    } else {
        let tau = (m[0][0] - m[1][1]) / deno;
        let w = Precision::sqrt(tau * tau + 1.0);
        println!("tau = (x - z) / deno = ({} - {}) / {} = {}", m[0][0], m[1][1], deno, tau);
        println!("w = sqrt(tau^2 + 1) = sqrt({}^2 + 1) = {}", tau, w);
        
        let t = if tau > 0.0 {
            1.0 / (tau + w)
        } else {
            1.0 / (tau - w)
        };
        println!("t = 1 / (tau + w) = 1 / ({} + {}) = {}", tau, w, t);
        
        let sign_t = if t > 0.0 { 1.0 } else { -1.0 };
        let n = 1.0 / Precision::sqrt(t * t + 1.0);
        let sign_y = if m[0][1] >= 0.0 { 1.0 } else { -1.0 };
        
        println!("sign_t = {}, n = {}, sign_y = {}", sign_t, n, sign_y);
        
        let c_val = n;
        let s_val = -sign_t * sign_y * Precision::abs(t) * n;
        println!("c_val = n = {}", c_val);
        println!("s_val = -sign_t * sign_y * |t| * n = -{} * {} * {} * {} = {}", sign_t, sign_y, Precision::abs(t), n, s_val);
    }
    
    println!("rot2: c = {}, s = {}", rot2.c, rot2.s);
    
    let left_rot = rot1.compose(&rot2.transpose());
    println!("\nleft_rot = rot1.compose(rot2.transpose())");
    println!("rot2.transpose: c = {}, s = {}", rot2.transpose().c, rot2.transpose().s);
    println!("left_rot: c = {}, s = {}", left_rot.c, left_rot.s);
}
