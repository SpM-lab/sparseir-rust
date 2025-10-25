//! QR decomposition with column pivoting (RRQR)

pub mod householder;
pub mod rrqr;
pub mod truncate;

pub use rrqr::{QRPivoted, rrqr, rrqr_with_options};
pub use truncate::truncate_qr_result;
