#![no_std]

// Minimal no_std utilities for edge environments.
// Provides a tiny fixed-point wrapper and scale helpers without std.

extern crate alloc;
use alloc::vec::Vec;

/// Minimal fixed-point buffer using Q8.23 encoding backed by i32.
pub struct FixedQ {
    pub data: Vec<i32>,
}

impl FixedQ {
    pub const FRACTIONAL_BITS: i32 = 23;

    #[inline]
    pub fn from_f32(slice: &[f32]) -> Self {
        let scale = (1 << Self::FRACTIONAL_BITS) as f32;
        let mut v = Vec::with_capacity(slice.len());
        for &x in slice {
            // Round-to-nearest with trunc cast: add +/-0.5 then cast
            let s = x * scale;
            let y = if s >= (i32::MAX as f32) {
                i32::MAX
            } else if s <= (i32::MIN as f32) {
                i32::MIN
            } else {
                let r = if s >= 0.0 { s + 0.5 } else { s - 0.5 };
                r as i32
            };
            v.push(y);
        }
        Self { data: v }
    }

    #[inline]
    pub fn to_f32(&self) -> Vec<f32> {
        let inv = 1.0f32 / ((1 << Self::FRACTIONAL_BITS) as f32);
        self.data.iter().map(|&y| (y as f32) * inv).collect()
    }
}
