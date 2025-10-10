//! Tests for DiscreteLehmannRepresentation

use sparseir_rust::{
    DiscreteLehmannRepresentation, LogisticKernel, FiniteTempBasis, 
    Fermionic, Bosonic
};
use mdarray::Tensor;
use num_complex::Complex;

mod common;

#[test]
fn test_dlr_construction_fermionic() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;
    
    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None);
    
    // Create DLR with default poles
    let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(&basis);
    
    assert_eq!(dlr.poles.len(), basis.size());
    assert_eq!(dlr.beta, beta);
    assert_eq!(dlr.wmax, wmax);
    
    // Poles should be in [-wmax, wmax]
    for &pole in &dlr.poles {
        assert!(pole.abs() <= wmax, "pole = {} exceeds wmax = {}", pole, wmax);
    }
}

/// Generic test for from_IR/to_IR roundtrip
fn test_dlr_from_to_ir_roundtrip_generic<T, S>()
where
    T: num_complex::ComplexFloat + faer_traits::ComplexField + From<f64> + Copy + Default + 'static 
        + common::ErrorNorm + common::ConvertFromReal,
    S: sparseir_rust::StatisticsType + 'static,
{
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;
    
    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<LogisticKernel, S>::new(kernel, beta, Some(epsilon), None);
    
    // Create DLR
    let dlr = DiscreteLehmannRepresentation::<S>::new(&basis);
    
    // Create IR coefficients
    let gl: Vec<T> = (0..basis.size())
        .map(|i| {
            let mag = ((i + 1) as f64).powi(-2);
            T::from_real(mag)
        })
        .collect();
    
    // IR → DLR → IR
    let g_dlr = dlr.from_IR::<T>(&gl);
    let gl_reconst = dlr.to_IR::<T>(&g_dlr);
    
    // Check roundtrip accuracy
    let max_error = gl.iter()
        .zip(gl_reconst.iter())
        .map(|(a, b)| (*a - *b).error_norm())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    
    println!("DLR {:?} {} roundtrip error: {:.2e}", S::STATISTICS, std::any::type_name::<T>(), max_error);
    assert!(max_error < 1e-7, "Roundtrip error too large: {:.2e}", max_error);
}

#[test]
fn test_dlr_from_to_IR_roundtrip_real_fermionic() {
    test_dlr_from_to_ir_roundtrip_generic::<f64, Fermionic>();
}

#[test]
fn test_dlr_from_to_IR_roundtrip_complex_fermionic() {
    test_dlr_from_to_ir_roundtrip_generic::<Complex<f64>, Fermionic>();
}

#[test]
fn test_dlr_from_to_IR_roundtrip_real_bosonic() {
    test_dlr_from_to_ir_roundtrip_generic::<f64, Bosonic>();
}

#[test]
fn test_dlr_from_to_IR_roundtrip_complex_bosonic() {
    test_dlr_from_to_ir_roundtrip_generic::<Complex<f64>, Bosonic>();
}

#[test]
fn test_dlr_with_custom_poles() {
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;
    
    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<LogisticKernel, Bosonic>::new(kernel, beta, Some(epsilon), None);
    
    // Custom poles within [-wmax, wmax]
    let poles = vec![-8.0, -3.0, 0.0, 3.0, 8.0];
    
    let dlr = DiscreteLehmannRepresentation::<Bosonic>::with_poles(&basis, poles.clone());
    
    assert_eq!(dlr.poles, poles);
    assert_eq!(dlr.beta, beta);
}

/// Generic test for from_IR_nd/to_IR_nd roundtrip
fn test_dlr_nd_roundtrip_generic<T, S>()
where
    T: num_complex::ComplexFloat + faer_traits::ComplexField + From<f64> + Copy + Default + 'static 
        + common::ErrorNorm + common::ConvertFromReal,
    S: sparseir_rust::StatisticsType + 'static,
{
    let beta = 10.0;
    let wmax = 10.0;
    let epsilon = 1e-6;
    
    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<LogisticKernel, S>::new(kernel, beta, Some(epsilon), None);
    
    let dlr = DiscreteLehmannRepresentation::<S>::new(&basis);
    
    let basis_size = basis.size();
    
    // Create reference 3D tensor with basis_size at dim=0
    let shape_ref = vec![basis_size, 3, 4];
    let gl_ref = {
        let mut tensor = Tensor::<T, mdarray::DynRank>::zeros(&shape_ref[..]);
        for l in 0..basis_size {
            for i in 0..3 {
                for j in 0..4 {
                    let mag = ((l + 1) as f64).powi(-2) * (i + j + 1) as f64;
                    tensor[&[l, i, j][..]] = T::from_real(mag);
                }
            }
        }
        tensor
    };
    
    // Test transformation along each dimension
    for dim in 0..3 {
        // Move basis dimension from 0 to dim
        let gl_3d = common::movedim(&gl_ref, 0, dim);
        
        // Transform: IR → DLR → IR
        let g_dlr = dlr.from_IR_nd::<T>(&gl_3d, dim);
        let gl_reconst = dlr.to_IR_nd::<T>(&g_dlr, dim);
        
        // Move back to dim=0 for comparison
        let gl_reconst_dim0 = common::movedim(&gl_reconst, dim, 0);
        
        // Check shape
        assert_eq!(gl_reconst_dim0.rank(), gl_ref.rank());
        
        // Check roundtrip - compare with reference
        let mut max_error = 0.0;
        for l in 0..basis_size {
            for i in 0..3 {
                for j in 0..4 {
                    let val_orig = gl_ref[&[l, i, j][..]];
                    let val_reconst = gl_reconst_dim0[&[l, i, j][..]];
                    let error = (val_orig - val_reconst).error_norm();
                    if error > max_error {
                        max_error = error;
                    }
                }
            }
        }
        
        println!("DLR {:?} {} ND roundtrip (dim={}): error = {:.2e}", 
                 S::STATISTICS, std::any::type_name::<T>(), dim, max_error);
        assert!(max_error < 1e-7, "ND roundtrip error too large for dim {}: {:.2e}", dim, max_error);
    }
}

#[test]
fn test_dlr_nd_roundtrip_real_fermionic() {
    test_dlr_nd_roundtrip_generic::<f64, Fermionic>();
}

#[test]
fn test_dlr_nd_roundtrip_complex_fermionic() {
    test_dlr_nd_roundtrip_generic::<Complex<f64>, Fermionic>();
}

#[test]
fn test_dlr_nd_roundtrip_real_bosonic() {
    test_dlr_nd_roundtrip_generic::<f64, Bosonic>();
}

#[test]
fn test_dlr_nd_roundtrip_complex_bosonic() {
    test_dlr_nd_roundtrip_generic::<Complex<f64>, Bosonic>();
}
