use sparseir_rust::basis::FiniteTempBasis;
use sparseir_rust::kernel::LogisticKernel;
use sparseir_rust::sampling::TauSampling;
use sparseir_rust::traits::Fermionic;
use num_complex::Complex;
use mdarray::Shape;

#[test]
fn test_fit_complex_basic() {
    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);
    
    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, epsilon, None);
    let sampling = TauSampling::new(&basis);
    
    // Create complex coefficients: real part from singular values, imaginary part constant
    let coeffs_real: Vec<f64> = basis.s.iter().copied().collect();
    let coeffs_complex: Vec<Complex<f64>> = coeffs_real.iter()
        .map(|&re| Complex::new(re, re * 0.1))
        .collect();
    
    // Evaluate: real matrix × complex coeffs → complex values
    let values_complex: Vec<Complex<f64>> = sampling.sampling_points()
        .iter()
        .enumerate()
        .map(|(_i, &tau)| {
            // Manually compute: sum_l coeffs[l] * u_l(tau)
            coeffs_complex.iter()
                .enumerate()
                .map(|(l, &c)| c * Complex::new(basis.u[l].evaluate(tau), 0.0))
                .sum()
        })
        .collect();
    
    // Convert to 1D Tensor for fit_nd
    let n_points = sampling.n_sampling_points();
    let mut values_tensor: mdarray::Tensor<Complex<f64>, mdarray::DynRank> = 
        mdarray::Tensor::zeros(&[n_points][..]);
    for (i, &val) in values_complex.iter().enumerate() {
        values_tensor[&[i][..]] = val;
    }
    
    // Fit: complex values → complex coeffs
    let fitted_coeffs_tensor = sampling.fit_nd_complex(&values_tensor, 0);
    
    // Check roundtrip
    for (i, &orig) in coeffs_complex.iter().enumerate() {
        let fitted = fitted_coeffs_tensor[&[i][..]];
        let abs_error = (orig - fitted).norm();
        assert!(
            abs_error < 1e-10,
            "Complex fit error: orig={}, fitted={}, error={}",
            orig,
            fitted,
            abs_error
        );
    }
}

#[test]
fn test_fit_complex_physical() {
    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);
    
    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, epsilon, None);
    let sampling = TauSampling::new(&basis);
    
    // Physical test: Retarded Green's function in imaginary time
    // G^R(τ) = -θ(τ) e^(-ωτ) → complex poles in frequency
    let omega = 5.0;
    let gamma = 0.5;  // damping
    
    // Complex Green's function with pole at ω - iγ
    let g_values: Vec<Complex<f64>> = sampling.sampling_points()
        .iter()
        .map(|&tau| {
            let re = -(-omega * tau).exp() * (gamma * tau).cos();
            let im = -(-omega * tau).exp() * (gamma * tau).sin();
            Complex::new(re, im) / (1.0 + (-beta * omega).exp())
        })
        .collect();
    
    // Convert to 1D Tensor for fit_nd
    let n_points = sampling.n_sampling_points();
    let mut g_values_tensor: mdarray::Tensor<Complex<f64>, mdarray::DynRank> = 
        mdarray::Tensor::zeros(&[n_points][..]);
    for (i, &val) in g_values.iter().enumerate() {
        g_values_tensor[&[i][..]] = val;
    }
    
    // Fit to IR basis
    let coeffs_tensor = sampling.fit_nd_complex(&g_values_tensor, 0);
    
    // Evaluate back
    let fitted_values: Vec<Complex<f64>> = sampling.sampling_points()
        .iter()
        .map(|&tau| {
            (0..sampling.basis_size())
                .map(|l| coeffs_tensor[&[l][..]] * Complex::new(basis.u[l].evaluate(tau), 0.0))
                .sum()
        })
        .collect();
    
    // Check accuracy
    for (orig, fitted) in g_values.iter().zip(fitted_values.iter()) {
        let abs_error = (orig - fitted).norm();
        assert!(
            abs_error < 1e-10,
            "Physical complex Green's function error: {}",
            abs_error
        );
    }
}

#[test]
fn test_fit_real_vs_complex() {
    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);
    
    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, epsilon, None);
    let sampling = TauSampling::new(&basis);
    
    // Real values: constant value at all sampling points
    let value: f64 = basis.s.iter()
        .enumerate()
        .map(|(l, &s)| s * basis.u[l].evaluate(0.5 * beta))
        .sum();
    let n_points = sampling.n_sampling_points();
    
    // Create real tensor
    let mut values_real_tensor: mdarray::Tensor<f64, mdarray::DynRank> = 
        mdarray::Tensor::zeros(&[n_points][..]);
    for i in 0..n_points {
        values_real_tensor[&[i][..]] = value;
    }
    
    // Create complex tensor (zero imaginary part)
    let mut values_complex_tensor: mdarray::Tensor<Complex<f64>, mdarray::DynRank> = 
        mdarray::Tensor::zeros(&[n_points][..]);
    for i in 0..n_points {
        values_complex_tensor[&[i][..]] = Complex::new(value, 0.0);
    }
    
    // Fit both
    let coeffs_real_tensor = sampling.fit_nd(&values_real_tensor, 0);
    let coeffs_complex_tensor = sampling.fit_nd_complex(&values_complex_tensor, 0);
    
    // Should give same results (complex with zero imaginary part)
    for i in 0..sampling.basis_size() {
        let real = coeffs_real_tensor[&[i][..]];
        let complex = coeffs_complex_tensor[&[i][..]];
        let diff_re = (real - complex.re).abs();
        let diff_im = complex.im.abs();
        
        assert!(diff_re < 1e-12, "Real part mismatch: {}", diff_re);
        assert!(diff_im < 1e-12, "Imaginary part should be ~0: {}", diff_im);
    }
}

#[test]
fn test_evaluate_nd_complex() {
    use mdarray::{Tensor, DynRank};
    
    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);
    
    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, epsilon, None);
    let sampling = TauSampling::new(&basis);
    
    let basis_size = basis.size();
    let n_k = 5;
    let n_omega = 7;
    
    // Create 3D complex coefficients: (basis_size, n_k, n_omega)
    let mut coeffs: Tensor<Complex<f64>, DynRank> = 
        Tensor::zeros(&[basis_size, n_k, n_omega][..]);
    
    for l in 0..basis_size {
        for k in 0..n_k {
            for omega in 0..n_omega {
                let flat_idx = l * n_k * n_omega + k * n_omega + omega;
                let re = (flat_idx as f64) * 0.1;
                let im = (flat_idx as f64) * 0.05;
                coeffs[&[l, k, omega][..]] = (basis.s[l]/basis.s[0]) * Complex::new(re, im);
            }
        }
    }
    
    // Evaluate along dim=0 (basis dimension)
    let values = sampling.evaluate_nd_complex(&coeffs, 0);
    
    // Check shape
    assert_eq!(values.shape().dim(0), sampling.n_sampling_points());
    assert_eq!(values.shape().dim(1), n_k);
    assert_eq!(values.shape().dim(2), n_omega);
    
    // Fit back
    let fitted_coeffs = sampling.fit_nd_complex(&values, 0);
    
    // Check roundtrip
    for k in 0..n_k {
        for omega in 0..n_omega {
            for l in 0..basis_size {
                let orig = coeffs[&[l, k, omega][..]];
                let fitted = fitted_coeffs[&[l, k, omega][..]];
                let abs_error = (orig - fitted).norm();
                
                assert!(
                    abs_error < 1e-10,
                    "ND complex roundtrip error at ({},{},{}): orig={}, fitted={}, error={}",
                    l, k, omega, orig, fitted, abs_error
                );
            }
        }
    }
}

#[test]
fn test_fit_nd_complex_different_dim() {
    use mdarray::{Tensor, DynRank};
    
    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);
    
    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, epsilon, None);
    let sampling = TauSampling::new(&basis);
    
    let n_points = sampling.n_sampling_points();
    let basis_size = basis.size();
    let n_k = 4;
    
    // Create 2D complex values: (n_k, n_points)
    let mut values: Tensor<Complex<f64>, DynRank> = 
        Tensor::zeros(&[n_k, n_points][..]);
    
    for k in 0..n_k {
        for i in 0..n_points {
            let flat_idx = k * n_points + i;
            values[&[k, i][..]] = Complex::new((flat_idx as f64) * 0.2, (flat_idx as f64) * 0.1);
        }
    }
    
    // Fit along dim=1 (sampling points dimension)
    let coeffs = sampling.fit_nd_complex(&values, 1);
    
    // Check shape
    assert_eq!(coeffs.shape().dim(0), n_k);
    assert_eq!(coeffs.shape().dim(1), basis_size);
    
    // Evaluate back
    let fitted_values = sampling.evaluate_nd_complex(&coeffs, 1);
    
    // Check roundtrip
    for k in 0..n_k {
        for i in 0..n_points {
            let orig = values[&[k, i][..]];
            let fitted = fitted_values[&[k, i][..]];
            let abs_error = (orig - fitted).norm();
            
            assert!(
                abs_error < 1e-10,
                "ND complex roundtrip (dim=1) error at ({},{}): error={}",
                k, i, abs_error
            );
        }
    }
}

#[test]
fn test_nd_real_vs_complex() {
    use mdarray::{Tensor, DynRank};
    
    let beta = 1.0;
    let wmax = 10.0;
    let epsilon = Some(1e-6);
    
    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, epsilon, None);
    let sampling = TauSampling::new(&basis);
    
    let basis_size = basis.size();
    let n_extra = 8;
    
    // Create real 2D coefficients
    let mut coeffs_real: Tensor<f64, DynRank> = 
        Tensor::zeros(&[basis_size, n_extra][..]);
    
    for l in 0..basis_size {
        for j in 0..n_extra {
            let flat_idx = l * n_extra + j;
            coeffs_real[&[l, j][..]] = (flat_idx as f64) * 0.15;
        }
    }
    
    // Same as complex (zero imaginary)
    let mut coeffs_complex: Tensor<Complex<f64>, DynRank> = 
        Tensor::zeros(&[basis_size, n_extra][..]);
    
    for l in 0..basis_size {
        for j in 0..n_extra {
            let flat_idx = l * n_extra + j;
            coeffs_complex[&[l, j][..]] = Complex::new((flat_idx as f64) * 0.15, 0.0);
        }
    }
    
    // Evaluate both
    let values_real = sampling.evaluate_nd(&coeffs_real, 0);
    let values_complex = sampling.evaluate_nd_complex(&coeffs_complex, 0);
    
    // Compare
    let n_points = sampling.n_sampling_points();
    for i in 0..n_points {
        for j in 0..n_extra {
            let real = values_real[&[i, j][..]];
            let complex = values_complex[&[i, j][..]];
            
            let diff_re = (real - complex.re).abs();
            let diff_im = complex.im.abs();
            
            assert!(diff_re < 1e-12, "Real evaluate_nd mismatch: {}", diff_re);
            assert!(diff_im < 1e-12, "Imaginary part should be ~0: {}", diff_im);
        }
    }
}

