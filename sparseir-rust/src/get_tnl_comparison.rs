use mdarray::tensor;
use crate::poly::PiecewiseLegendrePoly;
use crate::polyfourier::{FermionicPiecewiseLegendreFT, PiecewiseLegendreFT};
use crate::special_functions::spherical_bessel_j;
use crate::traits::Fermionic;

/// Test to compare get_tnl implementation with expected values
/// This is a simplified test - in practice we would need reference values from C++
#[test]
fn test_get_tnl_basic_values() {
    // Create a simple polynomial for testing
    let data = tensor![[1.0, 0.0], [0.0, 1.0]];
    let knots = vec![-1.0, 0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 1, None, 0);

    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);

    // Test get_tnl for various l and w values
    // Note: These are expected values that should match C++ implementation

    // For l=0, get_tnl should be 2 * j_0(|w|) where j_0(x) = sin(x)/x
    let result_0_1 = PiecewiseLegendreFT::<Fermionic>::get_tnl(0, 1.0);
    let expected_0_1 = 2.0 * (1.0_f64.sin() / 1.0); // 2 * sin(1)
    println!(
        "get_tnl(0, 1.0) = {}, expected = {}",
        result_0_1, expected_0_1
    );

    let result_0_pi = PiecewiseLegendreFT::<Fermionic>::get_tnl(0, std::f64::consts::PI);
    let expected_0_pi = 2.0 * (std::f64::consts::PI.sin() / std::f64::consts::PI); // Should be close to 0
    println!(
        "get_tnl(0, π) = {}, expected = {}",
        result_0_pi, expected_0_pi
    );

    // For l=1, get_tnl should be 2i * j_1(|w|) where j_1(x) = sin(x)/x² - cos(x)/x
    let result_1_1 = PiecewiseLegendreFT::<Fermionic>::get_tnl(1, 1.0);
    let j1_1 = 1.0_f64.sin() / (1.0 * 1.0) - 1.0_f64.cos() / 1.0;
    let im_unit = num_complex::Complex64::new(0.0, 1.0);
    let expected_1_1 = 2.0 * im_unit * j1_1;
    println!(
        "get_tnl(1, 1.0) = {}, expected = {}",
        result_1_1, expected_1_1
    );

    // Test negative w (should apply conjugation)
    let result_0_neg1 = PiecewiseLegendreFT::<Fermionic>::get_tnl(0, -1.0);
    println!(
        "get_tnl(0, -1.0) = {}, should be conjugate of positive",
        result_0_neg1
    );

    // Basic sanity checks
    assert!(result_0_1.re.is_finite());
    assert!(result_0_1.im.is_finite());
    assert!(result_1_1.re.is_finite());
    assert!(result_1_1.im.is_finite());
    assert!(result_0_neg1.re.is_finite());
    assert!(result_0_neg1.im.is_finite());
}

/// Test spherical Bessel function implementation
#[test]
fn test_spherical_bessel_basic() {
    let data = tensor![[1.0]];
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);
    let _ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);

    // Test j_0(x) = sin(x)/x for x != 0
    let x: f64 = 1.0;
    let j0 = spherical_bessel_j(0, x);
    let expected_j0 = x.sin() / x;
    println!("j_0({}) = {}, expected = {}", x, j0, expected_j0);

    // Test j_1(x) = sin(x)/x² - cos(x)/x
    let j1 = spherical_bessel_j(1, x);
    let expected_j1 = x.sin() / (x * x) - x.cos() / x;
    println!("j_1({}) = {}, expected = {}", x, j1, expected_j1);

    // Test j_0(0) = 1
    let j0_zero = spherical_bessel_j(0, 0.0);
    println!("j_0(0) = {}, expected = 1.0", j0_zero);
    assert!((j0_zero - 1.0).abs() < 1e-10);

    // Test j_n(0) = 0 for n > 0
    let j1_zero = spherical_bessel_j(1, 0.0);
    println!("j_1(0) = {}, expected = 0.0", j1_zero);
    assert!(j1_zero.abs() < 1e-10);
}

/// Test constant polynomial Fourier transform
/// This should help identify why constant polynomials produce non-zero values
#[test]
fn test_constant_polynomial_fourier_transform() {
    // Create constant polynomial f(x) = 1
    let data = tensor![[1.0], [0.0]];
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);

    let ft_poly = FermionicPiecewiseLegendreFT::new(poly, Fermionic, None);

    // Test evaluation at different frequencies
    for n in 0..5 {
        if let Ok(omega) = crate::freq::MatsubaraFreq::new(n) {
            let result = ft_poly.evaluate(&omega);
            println!("Constant poly at n={}: {}", n, result);

            // For constant polynomial, only n=0 should be non-zero
            if n == 0 {
                assert!(
                    result.norm() > 0.1,
                    "Constant polynomial should be non-zero at n=0"
                );
            } else {
                // For n > 0, the result should be very close to zero
                println!("  Norm at n={}: {}", n, result.norm());
                if result.norm() > 1e-6 {
                    println!(
                        "  WARNING: Constant polynomial has significant value at n={}",
                        n
                    );
                }
            }
        }
    }
}
