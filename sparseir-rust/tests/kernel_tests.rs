//! Precision tests for kernel implementations
//!
//! This module provides precision tests comparing TwoFloat implementations
//! against high-precision DBig calculations for kernel functions.

use dashu_base::{Abs, Approximation};
use dashu_float::{round::mode::HalfAway, Context, DBig};
use sparseir_rust::{
    compute_logistic_kernel, CentrosymmKernel, CustomNumeric, LogisticKernel, SymmetryType,
};
use std::str::FromStr;
use twofloat::TwoFloat;

// Configuration for precision tests
const DBIG_DIGITS: usize = 100;
const TOLERANCE_F64: f64 = 1e-12;
const TOLERANCE_TWOFLOAT: f64 = 1e-12; // TODO: MAKE IT TIGHTER

/// Convert f64 to DBig with high precision
fn f64_to_dbig(val: f64, precision: usize) -> DBig {
    let val_str = format!("{:.17e}", val);
    DBig::from_str(&val_str)
        .unwrap()
        .with_precision(precision)
        .unwrap()
}

/// Extract f64 from Approximation
fn extract_f64(approx: Approximation<f64, dashu_float::round::Rounding>) -> f64 {
    match approx {
        Approximation::Exact(val) => val,
        Approximation::Inexact(val, _) => val,
    }
}

/// High-precision logistic kernel implementation using DBig
fn logistic_kernel_dbig(lambda: DBig, x: DBig, y: DBig, ctx: &Context<HalfAway>) -> DBig {
    // K(x, y) = exp(-Λy(x + 1)/2) / (1 + exp(-Λy))
    let one = f64_to_dbig(1.0, ctx.precision());
    let two = f64_to_dbig(2.0, ctx.precision());

    let numerator = (-lambda.clone() * y.clone() * (x + one.clone()) / two).exp();
    let denominator = one + (-lambda * y).exp();

    numerator / denominator
}

/// Test precision of logistic kernel implementation for any CustomNumeric type
fn test_logistic_kernel_compute_precision<T: CustomNumeric>(
    lambda: f64,
    x: f64,
    y: f64,
    tolerance: f64,
) {
    // Convert inputs to type T
    let x_t: T = T::from_f64(x);
    let y_t: T = T::from_f64(y);
    let result_t = compute_logistic_kernel(lambda, x_t, y_t);

    // DBig version (high precision reference)
    let ctx = Context::<HalfAway>::new(DBIG_DIGITS);
    let lambda_dbig = f64_to_dbig(lambda, ctx.precision());
    let x_dbig = f64_to_dbig(x, ctx.precision());
    let y_dbig = f64_to_dbig(y, ctx.precision());
    let result_dbig = logistic_kernel_dbig(lambda_dbig, x_dbig, y_dbig, &ctx);

    // Convert T result to DBig for high-precision comparison
    let result_t_dbig = result_t.to_dbig(ctx.precision());

    // Absolute error comparison (T vs DBig) - high precision
    let error_t_dbig = (result_t_dbig - result_dbig.clone()).abs();
    let error_t_dbig_f64 = extract_f64(error_t_dbig.to_f64());
    assert!(
        error_t_dbig_f64 <= tolerance,
        "lambda={}, x={}, y={}: {}={:.15e}, DBig={:.15e}, error={:.15e} > tolerance={:.15e}",
        lambda,
        x,
        y,
        std::any::type_name::<T>(),
        result_t.to_f64(),
        extract_f64(result_dbig.to_f64()),
        error_t_dbig_f64,
        tolerance
    );
}

#[test]
fn test_logistic_kernel_compute_different_lambdas() {
    let lambdas = [10.0, 1e2, 1e4]; // [10, 100, 10000]

    for lambda in lambdas {
        for x in [-1.0, 0.0, 1.0] {
            for y in [-1.0, 0.0, 1.0] {
                // Test both f64 and TwoFloat implementations
                test_logistic_kernel_compute_precision::<f64>(lambda, x, y, TOLERANCE_F64);
                test_logistic_kernel_compute_precision::<TwoFloat>(
                    lambda,
                    x,
                    y,
                    TOLERANCE_TWOFLOAT,
                );
            }
        }
    }
}

/// High-precision compute_reduced implementation using DBig
fn compute_reduced_dbig(
    lambda: DBig,
    x: DBig,
    y: DBig,
    symmetry: SymmetryType,
    ctx: &Context<HalfAway>,
) -> DBig {
    match symmetry {
        SymmetryType::Even => {
            // K(x, y) + K(x, -y)
            let k_plus = logistic_kernel_dbig(lambda.clone(), x.clone(), y.clone(), ctx);
            let k_minus = logistic_kernel_dbig(lambda, x, -y, ctx);
            k_plus + k_minus
        }
        SymmetryType::Odd => {
            // K(x, y) - K(x, -y) (DBigで力押し)
            let k_plus = logistic_kernel_dbig(lambda.clone(), x.clone(), y.clone(), ctx);
            let k_minus = logistic_kernel_dbig(lambda, x, -y, ctx);
            k_plus - k_minus
        }
    }
}

/// Test precision of compute_reduced implementation for any CustomNumeric type
fn test_logistic_kernel_compute_reduced_precision<T: CustomNumeric>(
    lambda: f64,
    x: f64,
    y: f64,
    symmetry: SymmetryType,
    tolerance: f64,
) {
    // Convert inputs to type T
    let x_t: T = T::from_f64(x);
    let y_t: T = T::from_f64(y);

    // For compute_reduced, we need to call the trait method
    let kernel = LogisticKernel::new(lambda);
    let result_reduced_t = kernel.compute_reduced(x_t, y_t, symmetry);

    // DBig version (high precision reference)
    let ctx = Context::<HalfAway>::new(DBIG_DIGITS);
    let lambda_dbig = f64_to_dbig(lambda, ctx.precision());
    let x_dbig = f64_to_dbig(x, ctx.precision());
    let y_dbig = f64_to_dbig(y, ctx.precision());
    let result_reduced_dbig = compute_reduced_dbig(lambda_dbig, x_dbig, y_dbig, symmetry, &ctx);

    // Convert T result to DBig for high-precision comparison
    let result_reduced_t_dbig = result_reduced_t.to_dbig(ctx.precision());

    // Absolute error comparison (T vs DBig) - high precision
    let error_t_dbig = (result_reduced_t_dbig - result_reduced_dbig.clone()).abs();
    let error_t_dbig_f64 = extract_f64(error_t_dbig.to_f64());
    assert!(
        error_t_dbig_f64 <= tolerance,
        "lambda={}, x={}, y={}, symmetry={:?}: {}={:.15e}, DBig={:.15e}, error={:.15e} > tolerance={:.15e}",
        lambda, x, y, symmetry, std::any::type_name::<T>(), result_reduced_t.to_f64(), extract_f64(result_reduced_dbig.to_f64()), error_t_dbig_f64, tolerance
    );
}

#[test]
fn test_logistic_kernel_compute_reduced_different_lambdas() {
    let lambdas = [10.0, 1e2, 1e4]; // [10, 100, 10000]

    for lambda in lambdas {
        for x in [0.0, 0.5, 1.0] {
            // x >= 0
            for y in [0.0, 0.5, 1.0] {
                // y >= 0
                for symmetry in [SymmetryType::Even, SymmetryType::Odd] {
                    // Test both f64 and TwoFloat implementations
                    test_logistic_kernel_compute_reduced_precision::<f64>(
                        lambda,
                        x,
                        y,
                        symmetry,
                        TOLERANCE_F64,
                    );
                    test_logistic_kernel_compute_reduced_precision::<TwoFloat>(
                        lambda,
                        x,
                        y,
                        symmetry,
                        TOLERANCE_TWOFLOAT,
                    );
                }
            }
        }
    }
}

#[test]
fn test_compute_reduced_negative_y_works() {
    let kernel = LogisticKernel::new(1.0);

    // compute_reduced should work for negative y (implementation allows it)
    let x = TwoFloat::from(0.5);
    let y_negative = TwoFloat::from(-0.5);

    let result = kernel.compute_reduced(x, y_negative, SymmetryType::Even);
    assert!(
        result.is_finite(),
        "compute_reduced should work for negative y"
    );
}

#[test]
fn test_compute_reduced_positive_y_works() {
    let kernel = LogisticKernel::new(1.0);

    // compute_reduced should work for positive y
    let x = TwoFloat::from(0.5);
    let y_positive = TwoFloat::from(0.5);

    let result = kernel.compute_reduced(x, y_positive, SymmetryType::Even);
    assert!(
        result.is_finite(),
        "compute_reduced should work for positive y"
    );
}

#[test]
fn test_compute_reduced_symmetry_types() {
    let kernel = LogisticKernel::new(1.0);
    let x = TwoFloat::from(0.5);
    let y = TwoFloat::from(0.5);

    // Test both symmetry types
    let result_even = kernel.compute_reduced(x, y, SymmetryType::Even);
    let result_odd = kernel.compute_reduced(x, y, SymmetryType::Odd);

    assert!(result_even.is_finite(), "Even symmetry should work");
    assert!(result_odd.is_finite(), "Odd symmetry should work");

    // Results should be different for even and odd symmetry
    assert_ne!(
        result_even, result_odd,
        "Even and odd results should be different"
    );
}
