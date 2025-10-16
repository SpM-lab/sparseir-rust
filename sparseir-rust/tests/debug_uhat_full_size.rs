use sparseir_rust::*;

#[test]
fn test_uhat_full_size() {
    let beta = 100.0;
    let wmax = 2.0;
    let epsilon = 1e-6;
    
    let kernel = kernel::LogisticKernel::new(beta * wmax);
    let sve_result = sve::compute_sve(&kernel, epsilon);
    
    eprintln!("SVE result: s.len() = {}", sve_result.s.len());
    eprintln!("SVE result: u.polyvec.len() = {}", sve_result.u.polyvec.len());
    
    let basis = basis::FiniteTempBasis::<kernel::LogisticKernel, traits::Fermionic>::from_sve_result(
        kernel,
        beta,
        sve_result,
        Some(epsilon),
        None,
    );
    
    eprintln!("Basis size: {}", basis.size());
    eprintln!("uhat_full.len(): {}", basis.uhat_full.len());
    eprintln!("uhat.len(): {}", basis.uhat.len());
    
    assert_eq!(basis.size(), basis.uhat.len());
    // Check if uhat_full is larger
    eprintln!("uhat_full.len() >= basis.size(): {}", basis.uhat_full.len() >= basis.size());
}

