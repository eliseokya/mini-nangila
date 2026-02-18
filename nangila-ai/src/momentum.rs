use nangila_core::Predictor;
use nangila_core::predictor::PredictorError;
use nangila_math::FixedPointBuffer;

/// Momentum-based predictor (EMA)
/// Formula: hat_g_t = beta * hat_g_{t-1} + (1-beta) * g_{t-1}
/// Or more simply as tracked state: m_t = beta * m_{t-1} + g_t
/// Prediction for next step is often just m_t
pub struct MomentumPredictor {
    beta: f32,
    buffer: Option<FixedPointBuffer>, // Current momentum state
}

impl MomentumPredictor {
    pub fn new(beta: f32) -> Self {
        Self {
            beta,
            buffer: None,
        }
    }
}

impl Predictor for MomentumPredictor {
    fn predict(&self) -> Result<FixedPointBuffer, PredictorError> {
        match &self.buffer {
            Some(buf) => Ok(buf.clone()), // Predict signal will continue as-is
            None => Ok(FixedPointBuffer::new(0)), // Should ideally handle initialization sizing better
        }
    }

    fn update(&mut self, observation: &FixedPointBuffer) -> Result<(), PredictorError> {
        match &mut self.buffer {
            Some(state) => {
                // state = state * beta + observation * (1 - beta) ?
                // Standard SGD momentum: v = mu * v + g
                // Nesterov/Predictive usually predicts g_{t+1} approx g_t + momentum 
                
                // Let's implement simple EMA for now as per whitepaper spec:
                // hat_g = beta * hat_g_prev + (1-beta) * g_current
                
                // 1. Decay old state
                let decayed = state.mul_scalar(self.beta);
                
                // 2. Scale observation
                let scaled_obs = observation.mul_scalar(1.0 - self.beta);
                
                // 3. Add
                *state = decayed.add(&scaled_obs)?;
            }
            None => {
                // Initialize with first observation
                // Usually we just set state = observation for t=0
                self.buffer = Some(observation.clone());
            }
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.buffer = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_momentum_update() {
        let mut pred = MomentumPredictor::new(0.9);
        
        let g1 = FixedPointBuffer::from_f32(&[10.0]);
        pred.update(&g1).unwrap();
        
        // Step 1: state = 10.0 (init)
        let p1 = pred.predict().unwrap();
        assert!((p1.to_f32()[0] - 10.0).abs() < 1e-6);

        let g2 = FixedPointBuffer::from_f32(&[10.0]);
        pred.update(&g2).unwrap();
        
        // Step 2: state = 0.9 * 10 + 0.1 * 10 = 10.0
        let p2 = pred.predict().unwrap();
        assert!((p2.to_f32()[0] - 10.0).abs() < 1e-6);
        
        let g3 = FixedPointBuffer::from_f32(&[20.0]);
        pred.update(&g3).unwrap();
        
        // Step 3: state = 0.9 * 10 + 0.1 * 20 = 9 + 2 = 11.0
        let p3 = pred.predict().unwrap();
        assert!((p3.to_f32()[0] - 11.0).abs() < 1e-6);
    }
}
