//! # nangila-ai
//!
//! AI-domain implementations for gradient compression in distributed training.
//!
//! Key types:
//! - [`MomentumPredictor`]: EMA-based gradient predictor (β·state + (1-β)·observation)
//! - [`StochasticQuantizer`]: Unbiased INT4/INT8 quantization with hash-based PRNG
//! - [`TopKQuantizer`]: Sparse gradient compression — keeps top k% by magnitude

pub mod momentum;
pub mod quantizer;
pub mod topk;
// pub mod ddp_hook;

pub use momentum::MomentumPredictor;
pub use quantizer::{StochasticQuantizer, RefCellQuantizer};
pub use topk::TopKQuantizer;
