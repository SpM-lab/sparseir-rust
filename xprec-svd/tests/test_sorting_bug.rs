use mdarray::Tensor;

#[test]
fn test_column_reordering() {
    // Original U
    let u = Tensor::from_fn((2, 2), |idx| {
        [[0.707, -0.707], [0.707, 0.707]][idx[0]][idx[1]]
    });
    
    println!("Original U:");
    for i in 0..2 {
        println!("  [{:.3}, {:.3}]", u[[i, 0]], u[[i, 1]]);
    }
    
    // Sort indices [0, 1] (no reordering)
    let indices = vec![0, 1];
    
    // Current (wrong?) implementation
    let u_sorted_current = Tensor::from_fn((2, 2), |idx| u[[idx[0], indices[idx[1]]]]);
    println!("\nCurrent implementation: u[[idx[0], indices[idx[1]]]]");
    for i in 0..2 {
        println!("  [{:.3}, {:.3}]", u_sorted_current[[i, 0]], u_sorted_current[[i, 1]]);
    }
    
    // Check what we're actually getting
    println!("\nDetailed:");
    println!("  u_sorted[0,0] = u[[0, indices[0]]] = u[[0, {}]] = {:.3}", indices[0], u[[0, indices[0]]]);
    println!("  u_sorted[0,1] = u[[0, indices[1]]] = u[[0, {}]] = {:.3}", indices[1], u[[0, indices[1]]]);
    println!("  u_sorted[1,0] = u[[1, indices[0]]] = u[[1, {}]] = {:.3}", indices[0], u[[1, indices[0]]]);
    println!("  u_sorted[1,1] = u[[1, indices[1]]] = u[[1, {}]] = {:.3}", indices[1], u[[1, indices[1]]]);
}
