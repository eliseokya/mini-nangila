use serde::{Deserialize, Serialize};

/// Static topology mask for Driver/Passenger layer classification
/// 
/// In the Nangila framework:
/// - **Drivers**: Layers that must be transmitted (high variance, critical for convergence)
/// - **Passengers**: Layers that can be skipped (low variance, correlated with Drivers)
/// 
/// This is a simplified, variance-based implementation for the open-core.
/// The proprietary Sculptor uses advanced correlation analysis and learned heuristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyMask {
    /// Indices of Driver layers (must transmit)
    pub driver_indices: Vec<usize>,
    
    /// Indices of Passenger layers (can skip)
    pub passenger_indices: Vec<usize>,
    
    /// Threshold used to generate this mask
    pub threshold: f32,
    
    /// Total number of layers
    pub total_layers: usize,
}

impl TopologyMask {
    /// Create a new topology mask from layer statistics
    /// 
    /// # Algorithm (Variance-Based)
    /// 1. Compute variance for each layer
    /// 2. Sort layers by variance (descending)
    /// 3. Mark top (1 - drop_percent) as Drivers
    /// 4. Mark bottom drop_percent as Passengers
    /// 
    /// # Arguments
    /// * `layer_variances` - Variance of each layer (e.g., gradient variance over time)
    /// * `drop_percent` - Fraction of layers to mark as Passengers (0.0 to 1.0)
    /// 
    /// # Example
    /// ```
    /// use nangila_core::TopologyMask;
    /// 
    /// // Layer variances (e.g., from 100 training steps)
    /// let variances = vec![0.5, 0.1, 0.8, 0.05, 0.6];
    /// 
    /// // Drop bottom 40% (2 layers)
    /// let mask = TopologyMask::from_variance_threshold(&variances, 0.4);
    /// 
    /// // Indices are sorted for deterministic ordering
    /// assert_eq!(mask.driver_indices, vec![0, 2, 4]);  // High variance
    /// assert_eq!(mask.passenger_indices, vec![1, 3]);  // Low variance
    /// // 5 total / 3 drivers ≈ 1.67
    /// assert!((mask.compression_factor() - 1.67).abs() < 0.01);
    /// ```
    pub fn from_variance_threshold(layer_variances: &[f32], drop_percent: f32) -> Self {
        let total_layers = layer_variances.len();
        
        // Handle edge cases
        if total_layers == 0 {
            return Self {
                driver_indices: vec![],
                passenger_indices: vec![],
                threshold: 0.0,
                total_layers: 0,
            };
        }
        
        let drop_percent = drop_percent.clamp(0.0, 1.0);
        
        // Create (index, variance) pairs and sort by variance descending
        let mut indexed_variances: Vec<(usize, f32)> = layer_variances
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        
        indexed_variances.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Determine split point
        let num_drivers = ((total_layers as f32) * (1.0 - drop_percent)).ceil() as usize;
        let num_drivers = num_drivers.max(1).min(total_layers); // At least 1 driver
        
        // Split into drivers and passengers
        let mut driver_indices: Vec<usize> = indexed_variances[..num_drivers]
            .iter()
            .map(|(i, _)| *i)
            .collect();
        
        let mut passenger_indices: Vec<usize> = indexed_variances[num_drivers..]
            .iter()
            .map(|(i, _)| *i)
            .collect();
        
        // Sort indices for deterministic ordering
        driver_indices.sort_unstable();
        passenger_indices.sort_unstable();
        
        // Threshold is the minimum variance among drivers
        let threshold = if num_drivers < total_layers {
            indexed_variances[num_drivers - 1].1
        } else {
            0.0
        };
        
        Self {
            driver_indices,
            passenger_indices,
            threshold,
            total_layers,
        }
    }
    
    /// Check if a layer is a Driver (must transmit)
    pub fn is_driver(&self, layer_id: usize) -> bool {
        self.driver_indices.binary_search(&layer_id).is_ok()
    }
    
    /// Check if a layer is a Passenger (can skip)
    pub fn is_passenger(&self, layer_id: usize) -> bool {
        self.passenger_indices.binary_search(&layer_id).is_ok()
    }
    
    /// Get the compression factor from topology masking alone
    /// 
    /// Returns: total_layers / num_drivers
    pub fn compression_factor(&self) -> f32 {
        if self.driver_indices.is_empty() {
            1.0
        } else {
            self.total_layers as f32 / self.driver_indices.len() as f32
        }
    }
    
    /// Get the percentage of layers marked as Passengers
    pub fn passenger_percentage(&self) -> f32 {
        if self.total_layers == 0 {
            0.0
        } else {
            (self.passenger_indices.len() as f32 / self.total_layers as f32) * 100.0
        }
    }
    
    /// Create a mask that marks all layers as Drivers (no compression)
    pub fn all_drivers(num_layers: usize) -> Self {
        Self {
            driver_indices: (0..num_layers).collect(),
            passenger_indices: vec![],
            threshold: 0.0,
            total_layers: num_layers,
        }
    }
    
    /// Serialize mask to bytes (for storage/transmission)
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| format!("Serialization error: {}", e))
    }
    
    /// Deserialize mask from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        bincode::deserialize(bytes).map_err(|e| format!("Deserialization error: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variance_threshold_basic() {
        let variances = vec![0.5, 0.1, 0.8, 0.05, 0.6];
        let mask = TopologyMask::from_variance_threshold(&variances, 0.4);
        
        // Should keep top 60% (3 layers): indices 2, 4, 0
        assert_eq!(mask.driver_indices, vec![0, 2, 4]);
        assert_eq!(mask.passenger_indices, vec![1, 3]);
        assert_eq!(mask.total_layers, 5);
        
        // Check compression factor: 5 / 3 ≈ 1.67
        assert!((mask.compression_factor() - 1.67).abs() < 0.01);
        
        // Check passenger percentage: 40%
        assert!((mask.passenger_percentage() - 40.0).abs() < 0.1);
    }
    
    #[test]
    fn test_is_driver_passenger() {
        let variances = vec![0.5, 0.1, 0.8, 0.05, 0.6];
        let mask = TopologyMask::from_variance_threshold(&variances, 0.4);
        
        assert!(mask.is_driver(0));
        assert!(mask.is_passenger(1));
        assert!(mask.is_driver(2));
        assert!(mask.is_passenger(3));
        assert!(mask.is_driver(4));
    }
    
    #[test]
    fn test_edge_case_drop_all() {
        let variances = vec![0.5, 0.1, 0.8];
        let mask = TopologyMask::from_variance_threshold(&variances, 1.0);
        
        // Should keep at least 1 driver
        assert_eq!(mask.driver_indices.len(), 1);
        assert_eq!(mask.driver_indices[0], 2); // Highest variance
    }
    
    #[test]
    fn test_edge_case_drop_none() {
        let variances = vec![0.5, 0.1, 0.8];
        let mask = TopologyMask::from_variance_threshold(&variances, 0.0);
        
        // Should keep all as drivers
        assert_eq!(mask.driver_indices, vec![0, 1, 2]);
        assert_eq!(mask.passenger_indices.len(), 0);
        assert_eq!(mask.compression_factor(), 1.0);
    }
    
    #[test]
    fn test_edge_case_empty() {
        let variances: Vec<f32> = vec![];
        let mask = TopologyMask::from_variance_threshold(&variances, 0.5);
        
        assert_eq!(mask.driver_indices.len(), 0);
        assert_eq!(mask.passenger_indices.len(), 0);
        assert_eq!(mask.total_layers, 0);
    }
    
    #[test]
    fn test_all_drivers() {
        let mask = TopologyMask::all_drivers(10);
        
        assert_eq!(mask.driver_indices.len(), 10);
        assert_eq!(mask.passenger_indices.len(), 0);
        assert_eq!(mask.compression_factor(), 1.0);
        
        for i in 0..10 {
            assert!(mask.is_driver(i));
            assert!(!mask.is_passenger(i));
        }
    }
    
    #[test]
    fn test_serialization() {
        let variances = vec![0.5, 0.1, 0.8, 0.05, 0.6];
        let mask = TopologyMask::from_variance_threshold(&variances, 0.4);
        
        let bytes = mask.to_bytes().unwrap();
        let restored = TopologyMask::from_bytes(&bytes).unwrap();
        
        assert_eq!(mask.driver_indices, restored.driver_indices);
        assert_eq!(mask.passenger_indices, restored.passenger_indices);
        assert_eq!(mask.threshold, restored.threshold);
        assert_eq!(mask.total_layers, restored.total_layers);
    }
    
    #[test]
    fn test_deterministic_ordering() {
        // Same variances should produce same mask
        let variances = vec![0.3, 0.7, 0.2, 0.9, 0.1];
        
        let mask1 = TopologyMask::from_variance_threshold(&variances, 0.4);
        let mask2 = TopologyMask::from_variance_threshold(&variances, 0.4);
        
        assert_eq!(mask1.driver_indices, mask2.driver_indices);
        assert_eq!(mask1.passenger_indices, mask2.passenger_indices);
    }
}
