//! # nangila-hpc
//!
//! HPC-domain implementations for scientific data compression with strict error bounds.
//!
//! Key types:
//! - [`LinearPredictor`]: First-order extrapolation (2·S_t - S_{t-1})
//! - [`ErrorBoundedQuantizer`]: Strict ε-guarantee via i16 quantization (SZ-basic style)
//! - [`RunLengthQuantizer`]: RLE for zero-heavy residuals
//! - [`BlockSparseQuantizer`]: Bitmask-based sparse i16 encoding
//! - [`BlockSparseRleQuantizer`]: Bitmask + mask-RLE for very sparse data

pub mod linear;
pub mod error_bounded;
pub mod rle;
pub mod block_sparse;
pub mod block_sparse_rle;
// pub mod checkpoint;

pub use error_bounded::ErrorBoundedQuantizer;
pub use linear::LinearPredictor;
pub use rle::RunLengthQuantizer;
pub use block_sparse::BlockSparseQuantizer;
pub use block_sparse_rle::BlockSparseRleQuantizer;
