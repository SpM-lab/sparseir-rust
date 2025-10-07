//! # sparseir-rust: Rust implementation of SparseIR functionality
//!
//! A high-performance implementation of the SparseIR (Sparse Intermediate Representation)
//! library in Rust, providing analytical continuation and sparse representation
//! functionality for quantum many-body physics calculations.

pub mod freq;
pub mod gauss;
pub mod interpolation1d;
pub mod interpolation2d;
pub mod kernel;
pub mod kernelmatrix;
pub mod numeric;
pub mod poly;
pub mod polyfourier;
pub mod special_functions;
pub mod traits;
pub mod sve;

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
pub use numeric::{CustomNumeric, TwoFloatArrayOps};
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

// Re-export external dependencies for convenience
pub use ndarray::{Array1, Array2};
pub use twofloat::TwoFloat;
