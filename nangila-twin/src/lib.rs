//! # nangila-twin
//!
//! Digital-twin domain: edge-cloud sync via compressed residual streaming.
//!
//! Key types:
//! - [`EdgeNode`]: Sensor-side node that predicts, computes residuals, and sends compressed packets
//! - [`CloudNode`]: Cloud-side node that receives, decompresses, and reconstructs state
//!
//! Both nodes use **closed-loop feedback** — updating their predictor with the
//! *reconstructed* value (not raw data) to stay drift-free.

pub mod edge_sync;

pub use edge_sync::{EdgeNode, CloudNode, SyncError};
