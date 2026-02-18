use nangila_core::Predictor;
use nangila_core::predictor::PredictorError;
use nangila_math::FixedPointBuffer;

/// Linear Extrapolator Predictor
///
/// Formula: `ŝ_{t+1} = S_t + (S_t - S_{t-1}) · dt`
///
/// Assuming uniform timesteps (dt=1): `ŝ_{t+1} = 2·S_t - S_{t-1}`
#[derive(Default)]
pub struct LinearPredictor {
    s_t: Option<FixedPointBuffer>,     // Current state
    s_t_minus_1: Option<FixedPointBuffer>, // Previous state
}

impl LinearPredictor {
    pub fn new() -> Self {
        Self {
            s_t: None,
            s_t_minus_1: None,
        }
    }
}

impl Predictor for LinearPredictor {
    fn predict(&self) -> Result<FixedPointBuffer, PredictorError> {
        match (&self.s_t, &self.s_t_minus_1) {
            (Some(curr), Some(prev)) => {
                // Predict: 2 * curr - prev
                // = curr + (curr - prev)
                
                // 1. Calculate diff = curr - prev
                let diff = curr.sub(prev)?;
                
                // 2. Add diff to curr
                let pred = curr.add(&diff)?;
                
                Ok(pred)
            }
            (Some(curr), None) => {
                // Only one point, predict same as current (zero velocity assumption)
                Ok(curr.clone())
            }
            _ => Ok(FixedPointBuffer::new(0)), // No history
        }
    }

    fn update(&mut self, observation: &FixedPointBuffer) -> Result<(), PredictorError> {
        // Shift history: 
        // s_{t-1} <- s_t
        // s_t <- observation
        
        self.s_t_minus_1 = self.s_t.take();
        self.s_t = Some(observation.clone());
        
        Ok(())
    }

    fn reset(&mut self) {
        self.s_t = None;
        self.s_t_minus_1 = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_extrapolation() {
        let mut pred = LinearPredictor::new();
        
        // t=0: S_0 = 10.0
        let s0 = FixedPointBuffer::from_f32(&[10.0]);
        pred.update(&s0).unwrap();
        
        // Predict: S_1 ?? (Only S_0 known -> assume S_1 = S_0 = 10.0)
        let p1 = pred.predict().unwrap();
        assert!((p1.to_f32()[0] - 10.0).abs() < 1e-6);
        
        // t=1: S_1 = 12.0
        let s1 = FixedPointBuffer::from_f32(&[12.0]);
        pred.update(&s1).unwrap();
        
        // Predict S_2: 2*S_1 - S_0 = 2*12 - 10 = 14
        let p2 = pred.predict().unwrap();
        assert!((p2.to_f32()[0] - 14.0).abs() < 1e-6, "Expected 14.0, got {:?}", p2.to_f32());
        
        // t=2: S_2 = 14.0 (Exact prediction!)
        let s2 = FixedPointBuffer::from_f32(&[14.0]);
        pred.update(&s2).unwrap();
        
        // Predict S_3: 2*14 - 12 = 16
        let p3 = pred.predict().unwrap();
        assert!((p3.to_f32()[0] - 16.0).abs() < 1e-6);
    }
}
