// V^Tのインデックスを確認

use mdarray::Tensor;

#[test]
fn verify_transpose_index() {
    // V matrix (2x2):
    // [[a, b],
    //  [c, d]]
    
    let v = Tensor::from_fn((2, 2), |idx| {
        let vals = [[1.0, 2.0], [3.0, 4.0]];
        vals[idx[0]][idx[1]]
    });
    
    println!("V matrix:");
    for i in 0..2 {
        println!("  [{}, {}]", v[[i, 0]], v[[i, 1]]);
    }
    
    println!("\nV^T should be:");
    println!("  [[a, c],   [[1, 3],");
    println!("   [b, d]]    [2, 4]]");
    
    println!("\nSo V^T[k,j] = V[j,k]:");
    println!("  V^T[0,0] = V[0,0] = {}", v[[0, 0]]);
    println!("  V^T[0,1] = V[1,0] = {}", v[[1, 0]]);
    println!("  V^T[1,0] = V[0,1] = {}", v[[0, 1]]);
    println!("  V^T[1,1] = V[1,1] = {}", v[[1, 1]]);
    
    println!("\nIn reconstruction A[i,j] = Σ_k U[i,k] * S[k] * V^T[k,j]");
    println!("  For j=0, k=1: V^T[1,0] = V[0,1] → use v[[0, 1]]");
    println!("  For j=1, k=0: V^T[0,1] = V[1,0] → use v[[1, 0]]");
    println!("\nTherefore: v[[idx[1], k]] is CORRECT for V^T[k,j] = V[j,k]");
}
