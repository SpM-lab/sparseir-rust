use sparseir_rust::normalize_tau;
use sparseir_rust::traits::{Fermionic, Bosonic};

#[test]
fn test_normalize_tau_generic_fermionic() {
    let beta = 1.0;

    // Normal range
    let (tau_norm, sign) = normalize_tau::<Fermionic>(0.5, beta);
    assert!((tau_norm - 0.5).abs() < 1e-14);
    assert!((sign - 1.0).abs() < 1e-14);

    // At β (interpreted as β-)
    let (tau_norm, sign) = normalize_tau::<Fermionic>(beta, beta);
    assert!((tau_norm - beta).abs() < 1e-14);
    assert!((sign - 1.0).abs() < 1e-14);

    // Negative range
    let (tau_norm, sign) = normalize_tau::<Fermionic>(-0.3, beta);
    assert!((tau_norm - 0.7).abs() < 1e-14);
    assert!((sign - (-1.0)).abs() < 1e-14);

    // Extended range
    let (tau_norm, sign) = normalize_tau::<Fermionic>(1.2, beta);
    assert!((tau_norm - 0.2).abs() < 1e-14);
    assert!((sign - (-1.0)).abs() < 1e-14);
    
    // Test -β (interpreted as -β + 0, wraps to normal range)
    let (tau_norm, sign) = normalize_tau::<Fermionic>(-beta, beta);
    assert!(tau_norm.abs() < 1e-14);  // wraps to 0
    assert!((sign - (-1.0)).abs() < 1e-14);
}

#[test]
fn test_normalize_tau_generic_bosonic() {
    let beta = 1.0;

    // Normal range
    let (tau_norm, sign) = normalize_tau::<Bosonic>(0.5, beta);
    assert!((tau_norm - 0.5).abs() < 1e-14);
    assert!((sign - 1.0).abs() < 1e-14);

    // At β (interpreted as β-)
    let (tau_norm, sign) = normalize_tau::<Bosonic>(beta, beta);
    assert!((tau_norm - beta).abs() < 1e-14);
    assert!((sign - 1.0).abs() < 1e-14);

    // Negative range
    let (tau_norm, sign) = normalize_tau::<Bosonic>(-0.3, beta);
    assert!((tau_norm - 0.7).abs() < 1e-14);
    assert!((sign - 1.0).abs() < 1e-14);

    // Extended range
    let (tau_norm, sign) = normalize_tau::<Bosonic>(1.2, beta);
    assert!((tau_norm - 0.2).abs() < 1e-14);
    assert!((sign - 1.0).abs() < 1e-14);
    
    // Test -β (interpreted as -β + 0, wraps to normal range)
    let (tau_norm, sign) = normalize_tau::<Bosonic>(-beta, beta);
    assert!(tau_norm.abs() < 1e-14);  // wraps to 0
    assert!((sign - 1.0).abs() < 1e-14);
}

#[test]
fn test_normalize_tau_wrapper_functions() {
    use sparseir_rust::{normalize_tau_fermionic, normalize_tau_bosonic};
    
    let beta = 1.0;
    
    // Test fermionic wrapper
    let (tau_norm_f, sign_f) = normalize_tau_fermionic(0.5, beta);
    let (tau_norm_generic, sign_generic) = normalize_tau::<Fermionic>(0.5, beta);
    assert!((tau_norm_f - tau_norm_generic).abs() < 1e-14);
    assert!((sign_f - sign_generic).abs() < 1e-14);
    
    // Test bosonic wrapper
    let tau_norm_b = normalize_tau_bosonic(0.5, beta);
    let (tau_norm_generic2, _) = normalize_tau::<Bosonic>(0.5, beta);
    assert!((tau_norm_b - tau_norm_generic2).abs() < 1e-14);
}

