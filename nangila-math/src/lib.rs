//! # nangila-math
//!
//! Deterministic Q8.23 fixed-point arithmetic for the Nangila compression framework.
//!
//! This crate provides [`FixedPointBuffer`] — a vector of `i32` values representing
//! fixed-point numbers with 23 fractional bits (~1e-7 precision, ±256.0 range).
//! All operations use saturating arithmetic to prevent overflow-induced divergence.
//!
//! **Zero external dependencies** (besides `thiserror` for error types) — auditable in isolation.

pub mod fixed_point;
pub mod ops;

// Re-export FixedPointBuffer and Error for easier access
pub use fixed_point::{FixedPointBuffer, FixedPointError};
