//! # nangila-core
//!
//! Core trait abstractions for Nangila's predictive-residual compression.
//!
//! This crate defines the two fundamental traits:
//! - [`Predictor`]: generate predictions from temporal history, update state
//! - [`Quantizer`]: compress residuals to bytes and decompress back
//!
//! Also provides [`UnifiedQuantizer`] (runtime mode-switching dispatcher),
//! [`TopologyMask`] (Driver/Passenger layer classification), and
//! [`QuantizationMode`] (Stochastic / ErrorBounded / TopK selection).

pub mod predictor;
pub mod quantizer;
pub mod topology;
pub mod unified_quantizer;

pub use predictor::Predictor;
pub use quantizer::Quantizer;
pub use topology::TopologyMask;
pub use unified_quantizer::{UnifiedQuantizer, QuantizationMode};
