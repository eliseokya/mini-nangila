use nangila_math::FixedPointBuffer;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PredictorError {
    #[error("Shape mismatch in predictor")]
    ShapeMismatch,
    #[error("Math error: {0}")]
    MathError(#[from] nangila_math::FixedPointError),
}

pub trait Predictor {
    /// Generate prediction based on internal state
    fn predict(&self) -> Result<FixedPointBuffer, PredictorError>;
    
    /// Update internal state with new observation (g_t)
    fn update(&mut self, observation: &FixedPointBuffer) -> Result<(), PredictorError>;
    
    /// Reset state (e.g., start of epoch)
    fn reset(&mut self);
}
