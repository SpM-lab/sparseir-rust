//! # sparseir-rust: Rust implementation of SparseIR functionality
//!
//! A high-performance implementation of the SparseIR (Sparse Intermediate Representation)
//! library in Rust, providing analytical continuation and sparse representation
//! functionality for quantum many-body physics calculations.

pub mod freq;
pub mod gauss;
pub mod gemm;  // Matrix multiplication utilities (Faer backend)
pub mod interpolation1d;
pub mod interpolation2d;
pub mod kernel;
pub mod kernelmatrix;
pub mod numeric;
pub mod poly;
pub mod polyfourier;
pub mod sampling;  // Sparse sampling in imaginary time
pub mod matsubara_sampling;  // Sparse sampling in Matsubara frequencies
pub mod special_functions;
pub mod traits;
pub mod sve;
pub mod basis_trait;  // Common trait for basis representations
pub mod basis;
pub mod dlr;  // Discrete Lehmann Representation utilities
pub mod fitter;  // Least-squares fitters (real/complex matrices)

// Re-export commonly used types and traits
pub use freq::{
    create_statistics, fermionic_sign, is_less, is_zero, sign, zero, BosonicFreq, FermionicFreq,
    MatsubaraFreq,
};
pub use gauss::{legendre, legendre_custom, legendre_twofloat, Rule};
pub use kernel::{
    compute_logistic_kernel, CentrosymmKernel, KernelProperties, LogisticKernel, LogisticSVEHints,
    SVEHints, SymmetryType,
};
pub use numeric::CustomNumeric;
pub use poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
pub use polyfourier::{
    BosonicPiecewiseLegendreFT, BosonicPiecewiseLegendreFTVector, FermionicPiecewiseLegendreFT,
    FermionicPiecewiseLegendreFTVector, PiecewiseLegendreFT, PiecewiseLegendreFTVector, PowerModel,
};
pub use traits::{Bosonic, Fermionic, Statistics, StatisticsMarker, StatisticsType};
pub use interpolation1d::Interpolate1D;
pub use interpolation2d::Interpolate2D;
pub use kernelmatrix::{DiscretizedKernel, InterpolatedKernel, matrix_from_gauss, matrix_from_gauss_with_segments};
pub use sve::{SVEResult, TworkType, SVDStrategy, SamplingSVE, CentrosymmSVE, SVEStrategy, compute_sve, compute_svd, truncate};
pub use basis_trait::Basis;
pub use basis::{FiniteTempBasis, FermionicBasis, BosonicBasis};
pub use dlr::{
    DiscreteLehmannRepresentation,
    gtau_single_pole, fermionic_single_pole, bosonic_single_pole, giwn_single_pole
};
pub use fitter::{RealMatrixFitter, ComplexToRealFitter, ComplexMatrixFitter};
pub use sampling::TauSampling;
pub use matsubara_sampling::{MatsubaraSampling, MatsubaraSamplingPositiveOnly};

// Re-export external dependencies for convenience
pub use twofloat::TwoFloat;
pub use mdarray::{DTensor, Tensor, DynRank};
