use sparseir_rust::dlr::{gtau_single_pole, fermionic_single_pole, bosonic_single_pole};
use sparseir_rust::traits::{StatisticsType, Fermionic, Bosonic};

/// Generic test for periodicity/anti-periodicity
fn test_periodicity_generic<S: StatisticsType>(expected_sign: f64, stat_name: &str) {
    let beta = 1.0;
    let omega = 5.0;
    
    // Use interior points, avoiding boundaries
    for tau in [0.1, 0.3, 0.7] {
        let g_tau = gtau_single_pole::<S>(tau, omega, beta);
        let g_tau_plus_beta = gtau_single_pole::<S>(tau + beta, omega, beta);
        
        // For fermions: G(τ+β) = -G(τ) → sign = -1
        // For bosons: G(τ+β) = G(τ) → sign = 1
        let expected = expected_sign * g_tau;
        
        assert!(
            (expected - g_tau_plus_beta).abs() < 1e-14,
            "{} periodicity violated at τ={}: G(τ)={}, G(τ+β)={}, expected={}",
            stat_name, tau, g_tau, g_tau_plus_beta, expected
        );
    }
}

#[test]
fn test_fermionic_antiperiodicity() {
    // Fermions: G(τ+β) = -G(τ)
    test_periodicity_generic::<Fermionic>(-1.0, "Fermionic");
}

#[test]
fn test_bosonic_periodicity() {
    // Bosons: G(τ+β) = G(τ)
    test_periodicity_generic::<Bosonic>(1.0, "Bosonic");
}

#[test]
fn test_generic_function_matches_specific() {
    let beta = 1.0;
    let omega = 5.0;
    let tau = 0.5;
    
    // Test that generic function matches specific functions
    let g_f_specific = fermionic_single_pole(tau, omega, beta);
    let g_f_generic = gtau_single_pole::<Fermionic>(tau, omega, beta);
    
    let g_b_specific = bosonic_single_pole(tau, omega, beta);
    let g_b_generic = gtau_single_pole::<Bosonic>(tau, omega, beta);
    
    assert!(
        (g_f_specific - g_f_generic).abs() < 1e-14,
        "Fermionic: specific={}, generic={}", g_f_specific, g_f_generic
    );
    assert!(
        (g_b_specific - g_b_generic).abs() < 1e-14,
        "Bosonic: specific={}, generic={}", g_b_specific, g_b_generic
    );
}

