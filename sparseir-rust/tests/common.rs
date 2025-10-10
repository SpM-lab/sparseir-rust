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

