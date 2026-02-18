use super::{FixedPointBuffer, FixedPointError};
use crate::fixed_point::FRACTIONAL_BITS;

impl FixedPointBuffer {
    /// Element-wise saturating addition
    pub fn add(&self, other: &Self) -> Result<Self, FixedPointError> {
        if self.len() != other.len() {
            return Err(FixedPointError::ShapeMismatch(self.len(), other.len()));
        }

        let data = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a.saturating_add(b))
            .collect();
            
        Ok(Self { data })
    }

    /// Element-wise saturating subtraction
    pub fn sub(&self, other: &Self) -> Result<Self, FixedPointError> {
        if self.len() != other.len() {
            return Err(FixedPointError::ShapeMismatch(self.len(), other.len()));
        }

        let data = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a.saturating_sub(b))
            .collect();
            
        Ok(Self { data })
    }

    /// Scalar multiplication (saturating)
    /// Used for momentum decay: state = state * beta
    pub fn mul_scalar(&self, scalar: f32) -> Self {
        // Convert scalar to fixed point representation
        // We multiply i32 * i32 which can overflow i64, then shift back
        let scalar_fixed = (scalar * (1 << FRACTIONAL_BITS) as f32) as i64;
        
        let data = self.data.iter().map(|&val| {
            let prod = (val as i64) * scalar_fixed;
            // Shift back to Q8.23
            let shifted = prod >> FRACTIONAL_BITS;
            // Saturate to i32
            shifted.max(i32::MIN as i64).min(i32::MAX as i64) as i32
        }).collect();
        
        Self { data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = FixedPointBuffer::from_f32(&[1.0, 2.0]);
        let b = FixedPointBuffer::from_f32(&[0.5, -0.5]);
        let c = a.add(&b).unwrap();
        
        let out = c.to_f32();
        assert!((out[0] - 1.5).abs() < 1e-6);
        assert!((out[1] - 1.5).abs() < 1e-6);
    }
    
    #[test]
    fn test_mul_scalar() {
        let a = FixedPointBuffer::from_f32(&[10.0, -10.0]);
        let b = a.mul_scalar(0.5);
        
        let out = b.to_f32();
        assert!((out[0] - 5.0).abs() < 1e-6);
        assert!((out[1] - -5.0).abs() < 1e-6);
    }
}
