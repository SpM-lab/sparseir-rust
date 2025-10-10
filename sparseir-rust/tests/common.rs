//! Common test utilities

use num_complex::Complex;
use mdarray::{Tensor, DynRank};

/// Move axis from position src to position dst
///
/// Equivalent to numpy.moveaxis or libsparseir's movedim.
pub fn movedim<T: Clone>(arr: &Tensor<T, DynRank>, src: usize, dst: usize) -> Tensor<T, DynRank> {
    let rank = arr.rank();
    assert!(src < rank && dst < rank, "src={}, dst={} must be < rank={}", src, dst, rank);
    
    if src == dst {
        return arr.clone();
    }
    
    // Create permutation: move src to dst
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.remove(src);
    perm.insert(dst, src);
    
    arr.permute(&perm[..]).to_tensor()
}

/// Simple deterministic pseudo-random number generator (LCG)
///
/// Linear Congruential Generator for reproducible random numbers in tests.
/// Uses the common parameters: a = 1664525, c = 1013904223 (from Numerical Recipes)
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Create a new RNG with the given seed
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    
    /// Generate next f64 in range [0, 1) (base method)
    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        // Map to [0, 1) using upper 48 bits
        ((self.state >> 16) as f64) / ((1u64 << 48) as f64)
    }
    
    /// Generate next value of type T (generic)
    pub fn next<T: RandomGenerate>(&mut self) -> T {
        T::generate(self)
    }
}

/// Trait for types that can be randomly generated from SimpleRng
pub trait RandomGenerate {
    fn generate(rng: &mut SimpleRng) -> Self;
}

impl RandomGenerate for f64 {
    fn generate(rng: &mut SimpleRng) -> Self {
        rng.next_f64()
    }
}

impl RandomGenerate for Complex<f64> {
    fn generate(rng: &mut SimpleRng) -> Self {
        // Generate independent real and imaginary parts
        let re = rng.next_f64();
        let im = rng.next_f64();
        Complex::new(re, im)
    }
}

/// Trait for computing error magnitude (real value)
pub trait ErrorNorm {
    /// Compute the error magnitude as f64
    fn error_norm(self) -> f64;
}

/// Trait for converting f64 to T (identity for f64, promote for Complex<f64>)
pub trait ConvertFromReal {
    fn from_real(value: f64) -> Self;
}

impl ConvertFromReal for f64 {
    fn from_real(value: f64) -> Self {
        value
    }
}

impl ConvertFromReal for Complex<f64> {
    fn from_real(value: f64) -> Self {
        Complex::new(value, 0.0)
    }
}

impl ErrorNorm for f64 {
    fn error_norm(self) -> f64 {
        self.abs()
    }
}

impl ErrorNorm for Complex<f64> {
    fn error_norm(self) -> f64 {
        self.norm()  // sqrt(re^2 + im^2)
    }
}

/// Generate test data for both τ and Matsubara sampling
///
/// Creates random coefficients and computes corresponding Green's function values
/// at both τ sampling points and Matsubara frequencies using a single-pole model.
///
/// # Type Parameters
/// * `T` - Element type for coefficients (f64 or Complex<f64>)
/// * `S` - Statistics type (Fermionic or Bosonic)
///
/// # Arguments
/// * `basis` - Finite temperature basis
/// * `tau_points` - τ sampling points
/// * `matsubara_freqs` - Matsubara frequency sampling points
/// * `seed` - Random seed for reproducible test data
///
/// # Returns
/// Tuple of (coefficients, τ values, Matsubara values)
pub fn generate_test_data_tau_and_matsubara<T, S, K>(
    basis: &sparseir_rust::FiniteTempBasis<K, S>,
    tau_points: &[f64],
    matsubara_freqs: &[sparseir_rust::freq::MatsubaraFreq<S>],
    seed: u64,
) -> (Vec<T>, Vec<T>, Vec<Complex<f64>>)
where
    T: RandomGenerate + ConvertFromReal + Clone + std::ops::Mul<f64, Output = T>,
    S: sparseir_rust::traits::StatisticsType,
    K: sparseir_rust::kernel::KernelProperties + sparseir_rust::kernel::CentrosymmKernel + Clone + 'static,
{
    use sparseir_rust::giwn_single_pole;
    
    let mut rng = SimpleRng::new(seed);
    let basis_size = basis.size();
    let beta = basis.beta;
    let wmax = basis.kernel.lambda() / beta;
    
    // Choose a pole position (arbitrary choice within reasonable range)
    let omega = wmax * 0.5; // Pole at half of wmax
    
    // Generate random coefficients scaled by singular values
    let coeffs: Vec<T> = (0..basis_size)
        .map(|l| rng.next::<T>() * basis.s[l])
        .collect();
    
    // Compute G(τ) at τ sampling points
    let gtau_values: Vec<T> = tau_points
        .iter()
        .map(|&tau| {
            let g = sparseir_rust::gtau_single_pole::<S>(tau, omega, beta);
            T::from_real(g)
        })
        .collect();
    
    // Compute G(iωn) at Matsubara frequencies
    let giwn_values: Vec<Complex<f64>> = matsubara_freqs
        .iter()
        .map(|freq| giwn_single_pole::<S>(freq, omega, beta))
        .collect();
    
    (coeffs, gtau_values, giwn_values)
}

