//! QR decomposition with column pivoting (RRQR)


pub mod rrqr;
pub mod householder;
pub mod truncate;

pub use rrqr::{QRPivoted, rrqr};
pub use truncate::truncate_qr_result;
