use nangila_core::{Predictor, Quantizer};
use nangila_math::{FixedPointBuffer, FixedPointError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SyncError {
    #[error("Predictor error: {0}")]
    PredictorError(#[from] nangila_core::predictor::PredictorError),
    #[error("Math error: {0}")]
    MathError(#[from] FixedPointError),
}

/// Represents a unidirectional sync channel (Edge -> Cloud)
/// This struct holds the logic for both sides for simulation purposes
/// or just the Edge side logic.
pub struct EdgeNode<P: Predictor, Q: Quantizer> {
    predictor: P,
    quantizer: Q,
}

impl<P: Predictor, Q: Quantizer> EdgeNode<P, Q> {
    pub fn new(predictor: P, quantizer: Q) -> Self {
        Self { predictor, quantizer }
    }

    /// Process a new sensor reading
    /// Returns: (compressed_residual, scale_factor)
    /// Side-effect: Updates internal predictor state
    pub fn send(&mut self, sensor_data: &FixedPointBuffer) -> Result<(Vec<u8>, f32), SyncError> {
        // 1. Generate prediction from previous state
        let prediction = self.predictor.predict()?;
        
        // 2. Calculate residual: r_t = x_t - hat_x_t
        // Note: careful with shapes. If prediction is init (0), it works.
        // If shapes mismatch, math error.
        let residual = if prediction.is_empty() {
            sensor_data.clone()
        } else {
            sensor_data.sub(&prediction)?
        };

        // 3. Quantize residual
        let (compressed, scale) = self.quantizer.quantize(&residual);
        
        // 4. Update predictor with *reconstructed* value to stay in sync with receiver
        // Crucial: The receiver only sees (prediction + dequantized_residual).
        // So the sender must act as if it is the receiver to avoid drift.
        // x_hat = prediction + dequantized_residual
        let dequantized_res = self.quantizer.dequantize(&compressed, scale);
        
        // If prediction is empty, reconstruction is just residual
        let reconstruction = if prediction.is_empty() {
            dequantized_res
        } else {
            prediction.add(&dequantized_res)?
        };
        
        // Update local predictor with the reconstruction (not raw sensor data!)
        // This is "Error Feedback" or "Closed Loop" prediction
        self.predictor.update(&reconstruction)?;
        
        Ok((compressed, scale))
    }
}

pub struct CloudNode<P: Predictor, Q: Quantizer> {
    predictor: P,
    quantizer: Q, // Needed for dequantize
    current_state: Option<FixedPointBuffer>,
}

impl<P: Predictor, Q: Quantizer> CloudNode<P, Q> {
    pub fn new(predictor: P, quantizer: Q) -> Self {
        Self { 
            predictor, 
            quantizer,
            current_state: None,
        }
    }

    /// Process incoming compressed packet
    /// Returns: Reconstructed state
    pub fn receive(&mut self, compressed: &[u8], scale: f32) -> Result<FixedPointBuffer, SyncError> {
        // 1. Generate prediction (mirrors EdgeNode)
        let prediction = self.predictor.predict()?;
        
        // 2. Dequantize residual
        let residual = self.quantizer.dequantize(compressed, scale);
        
        // 3. Reconstruct: x_hat = prediction + residual
        let reconstruction = if prediction.is_empty() {
             residual
        } else {
            prediction.add(&residual)?
        };
        
        // 4. Update predictor
        self.predictor.update(&reconstruction)?;
        self.current_state = Some(reconstruction.clone());
        
        Ok(reconstruction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nangila_ai::MomentumPredictor;
    use nangila_hpc::ErrorBoundedQuantizer; // Use HPC quantizer for predictable results
    
    #[test]
    fn test_end_to_end_sync() {
        let p_edge = MomentumPredictor::new(0.9);
        let q_edge = ErrorBoundedQuantizer::new(0.05); // EPS=0.05 -> Max range approx +/- 12.7
        let mut edge = EdgeNode::new(p_edge, q_edge);
        
        let p_cloud = MomentumPredictor::new(0.9);
        let q_cloud = ErrorBoundedQuantizer::new(0.05);
        let mut cloud = CloudNode::new(p_cloud, q_cloud);
        
        // Simulate a simple signal: constant 10.0
        let sensor_data = FixedPointBuffer::from_f32(&[10.0]);
        
        // Step 1
        let (packet, scale) = edge.send(&sensor_data).unwrap();
        let reconstructed = cloud.receive(&packet, scale).unwrap();
        
        let diff = (reconstructed.to_f32()[0] - 10.0).abs();
        assert!(diff < 0.06, "Step 1 reconstruction error: {}", diff);
        
        // Step 2
        let (packet2, scale2) = edge.send(&sensor_data).unwrap();
        let reconstructed2 = cloud.receive(&packet2, scale2).unwrap();
        assert!((reconstructed2.to_f32()[0] - 10.0).abs() < 0.06);
        
        // Residual should decrease as predictor learns the constant
        // Initial residual was 10.0 (pred=0).
        // Next prediction should be close to 10.0, so residual ~0.
        // Scale should decrease technically or be small.
    }
}
