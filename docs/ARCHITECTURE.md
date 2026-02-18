# Mini-Nangila Architecture

**A trait-based, modular compression framework for AI, HPC, and Digital Twins**

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Crate Structure](#crate-structure)
3. [Core Abstractions](#core-abstractions)
4. [Fixed-Point Arithmetic](#fixed-point-arithmetic)
5. [Compression Pipeline](#compression-pipeline)
6. [Domain-Specific Implementations](#domain-specific-implementations)
7. [Design Decisions](#design-decisions)
8. [Extension Points](#extension-points)

---

## Design Philosophy

Mini-Nangila follows three core principles:

### 1. Separation of Concerns
```
Math Layer (nangila-math)
    ↓ Provides deterministic arithmetic
Core Layer (nangila-core)
    ↓ Defines abstractions (Predictor, Quantizer)
Domain Layer (nangila-ai, nangila-hpc, nangila-twin)
    ↓ Implements domain-specific strategies
```

Each layer has a single responsibility and minimal dependencies.

### 2. Trait-Based Polymorphism
Instead of inheritance, we use Rust traits for extensibility:

```rust
pub trait Predictor {
    fn predict(&self) -> Result<FixedPointBuffer, PredictorError>;
    fn update(&mut self, observation: &FixedPointBuffer) -> Result<(), PredictorError>;
    fn reset(&mut self);
}

pub trait Quantizer {
    fn quantize(&self, residual: &FixedPointBuffer) -> (Vec<u8>, f32);
    fn dequantize(&self, compressed: &[u8], scale: f32) -> FixedPointBuffer;
}
```

This allows:
- Mix-and-match predictors and quantizers
- Zero-cost abstractions (no vtables unless needed)
- Easy testing and mocking

### 3. CPU-First, GPU-Later
All algorithms are implemented for CPU first:
- Easier to debug and validate
- Accessible to any researcher
- Compression ratios are hardware-agnostic

GPU acceleration (proprietary) provides computational speedup without changing algorithms.

---

## Crate Structure

### Dependency Graph

```
nangila-math (0 dependencies)
    ↓
nangila-core (depends on: nangila-math)
    ↓
    ├─ nangila-ai (depends on: nangila-core, nangila-math)
    ├─ nangila-hpc (depends on: nangila-core, nangila-math)
    └─ nangila-twin (depends on: all above)
```

### Crate Responsibilities

| Crate | Purpose | Key Types | Dependencies |
|-------|---------|-----------|--------------|
| `nangila-math` | Fixed-point arithmetic | `FixedPointBuffer` | None |
| `nangila-core` | Compression abstractions | `Predictor`, `Quantizer`, `TopologyMask` | `nangila-math` |
| `nangila-ai` | AI gradient compression | `MomentumPredictor`, `TopKQuantizer` | `nangila-core` |
| `nangila-hpc` | Scientific data compression | `LinearPredictor`, `ErrorBoundedQuantizer` | `nangila-core` |
| `nangila-twin` | Edge-cloud synchronization | `EdgeNode`, `CloudNode` | All above |

---

## Core Abstractions

### 1. FixedPointBuffer (nangila-math)

**Purpose**: Deterministic, cross-platform arithmetic

```rust
pub struct FixedPointBuffer {
    pub data: Vec<i32>,  // Q*.23 format (23 fractional bits)
}
```

**Why fixed-point?**
- Bit-exact results across x86_64, ARM64, GPU
- No floating-point non-determinism
- Predictable rounding behavior

**Format: Q8.23**
```
┌─────────┬──────────────────────────┐
│ 8 bits  │      23 bits             │
│ integer │      fractional          │
└─────────┴──────────────────────────┘
  (signed)

Range: [-256, 256)
Precision: 2^-23 ≈ 1.19e-7
```

**Operations**:
```rust
impl FixedPointBuffer {
    pub fn from_f32(data: &[f32]) -> Self;
    pub fn to_f32(&self) -> Vec<f32>;
    pub fn add(&self, other: &Self) -> Result<Self, FixedPointError>;
    pub fn sub(&self, other: &Self) -> Result<Self, FixedPointError>;
    pub fn mul_scalar(&self, scalar: f32) -> Self;
}
```

### 2. Predictor Trait (nangila-core)

**Purpose**: Predict next value based on history

```rust
pub trait Predictor {
    /// Generate prediction from internal state
    fn predict(&self) -> Result<FixedPointBuffer, PredictorError>;
    
    /// Update state with new observation
    fn update(&mut self, observation: &FixedPointBuffer) -> Result<(), PredictorError>;
    
    /// Reset state (e.g., start of epoch)
    fn reset(&mut self);
}
```

**Key Insight**: Predictors enable entropy reduction
```
H(residual) = H(data - prediction) < H(data)
```

**Implementations**:
- `MomentumPredictor` (AI): EMA-based gradient prediction
- `LinearPredictor` (HPC): First-order extrapolation

### 3. Quantizer Trait (nangila-core)

**Purpose**: Compress residuals to bytes

```rust
pub trait Quantizer {
    /// Compress residual to bytes + scale factor
    fn quantize(&self, residual: &FixedPointBuffer) -> (Vec<u8>, f32);
    
    /// Reconstruct from compressed bytes
    fn dequantize(&self, compressed: &[u8], scale: f32) -> FixedPointBuffer;
}
```

**Key Insight**: Quantization trades precision for size
```
FP32 (4 bytes) → INT4 (0.5 bytes) = 8× compression
```

**Implementations**:
- `StochasticQuantizer` (AI): Unbiased randomized rounding
- `ErrorBoundedQuantizer` (HPC): Strict ε-guarantee
- `TopKQuantizer` (AI): Sparse compression
- `UnifiedQuantizer` (Core): Runtime mode switching

Top‑K encoding format (AI):

```
[u32 n_elems]
repeat per 32‑wide block:
  [u32 mask]        // bit i indicates presence
  [i8 payload...]   // only for set bits
```

Decoders use `n_elems` to reconstruct exact lengths, including the last partial block.

### 4. TopologyMask (nangila-core)

**Purpose**: Classify layers as Driver (transmit) or Passenger (skip)

```rust
pub struct TopologyMask {
    pub driver_indices: Vec<usize>,
    pub passenger_indices: Vec<usize>,
    pub threshold: f32,
    pub total_layers: usize,
}
```

**Key Insight**: Not all layers are equally important
```
Compression = total_layers / num_drivers
```

**Algorithm** (Variance-Based):
1. Compute variance for each layer
2. Sort by variance (descending)
3. Mark top (1 - drop_percent) as Drivers
4. Mark bottom drop_percent as Passengers

---

## Fixed-Point Arithmetic

### Q8.23 Format Details

**Conversion: Float → Fixed**
```rust
fn from_f32(val: f32) -> i32 {
    let scaled = val * (1 << 23);  // Scale by 2^23
    let clamped = scaled.clamp(i32::MIN as f32, i32::MAX as f32);
    clamped.round() as i32
}
```

**Conversion: Fixed → Float**
```rust
fn to_f32(val: i32) -> f32 {
    (val as f32) / (1 << 23) as f32
}
```

**Addition** (Exact)
```rust
fn add(a: i32, b: i32) -> i32 {
    a.saturating_add(b)  // Clamp on overflow
}
```

**Multiplication by Scalar** (Fixed-Point)
```rust
fn mul_scalar(val: i32, scalar: f32) -> i32 {
    let scalar_fixed = (scalar * (1 << 23) as f32) as i64;
    let result = (val as i64 * scalar_fixed) >> 23;
    result.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}
```

### Why Not IEEE 754 Float?

| Property | Q8.23 Fixed | IEEE 754 Float |
|----------|-------------|----------------|
| Determinism | ✅ Bit-exact | ❌ Platform-dependent |
| Overflow | ✅ Saturating | ❌ NaN/Inf |
| Range | [-256, 256) | [-3.4e38, 3.4e38] |
| Precision | Uniform (1.19e-7) | Variable |
| Performance | ✅ Integer ops | ✅ Hardware support |

**Trade-off**: We sacrifice range for determinism and predictability.

---

## Compression Pipeline

### Standard Pipeline (3 Stages)

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Prediction (Entropy Reduction)                │
│  ────────────────────────────────────────────────────   │
│  Data ──▶ [Predictor] ──▶ Prediction                    │
│           (from history)                                 │
│                                                          │
│  Residual = Data - Prediction                           │
│  H(Residual) < H(Data)  ← Key insight                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Stage 2: Quantization (Precision Reduction)            │
│  ────────────────────────────────────────────────────   │
│  Residual ──▶ [Quantizer] ──▶ Compressed Bytes          │
│               (FP32 → INT4/8)                            │
│                                                          │
│  Compression: 4 bytes → 0.5-1 bytes (4-8×)              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Stage 3: Topology Masking (Layer Selection)            │
│  ────────────────────────────────────────────────────   │
│  Layers ──▶ [TopologyMask] ──▶ Drivers only             │
│             (skip Passengers)                            │
│                                                          │
│  Compression: N layers → M drivers (N/M ×)              │
└─────────────────────────────────────────────────────────┘

Total Compression = Stage1 × Stage2 × Stage3
                  = (1.5-3×) × (4-8×) × (1.2-2×)
                  = 7-48× typical
```

### Reconstruction Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  Receiver Side (Inverse Operations)                     │
│  ────────────────────────────────────────────────────   │
│  Compressed ──▶ [Dequantize] ──▶ Residual               │
│                                                          │
│  Prediction ──▶ [Predictor] (same state as sender)      │
│                                                          │
│  Reconstructed = Prediction + Residual                  │
│                                                          │
│  [Update Predictor] ← Reconstructed (closed-loop)       │
└─────────────────────────────────────────────────────────┘
```

**Critical**: Both sender and receiver must update predictors with reconstructed values (not raw data) to avoid drift.

---

## Domain-Specific Implementations

### AI Domain (nangila-ai)

**Goal**: Maximize compression for distributed training

**Components**:
1. **MomentumPredictor**
   ```rust
   // EMA-based gradient prediction
   m_t = β * m_{t-1} + (1-β) * g_t
   prediction = m_t
   ```
   - Typical β: 0.9
   - Exploits temporal correlation in gradients

2. **StochasticQuantizer**
   ```rust
   // Unbiased INT4 quantization
   q = floor(x) + (random < frac(x) ? 1 : 0)
   ```
   - Maintains gradient expectation
   - Hash-based PRNG for determinism

3. **TopKQuantizer**
   ```rust
   // Keep top K% by magnitude
   threshold = percentile(|x|, 1-K)
   transmit only x where |x| > threshold
   ```
   - Exploits gradient sparsity
   - Block-based encoding (32 values/block)

**Typical Compression**: 20-40×

### HPC Domain (nangila-hpc)

**Goal**: Strict error bounds for scientific data

**Components**:
1. **LinearPredictor**
   ```rust
   // First-order extrapolation
   S_{t+1} = S_t + Δt * (S_t - S_{t-1}) / Δt
          = 2*S_t - S_{t-1}
   ```
   - Works well for smooth trajectories
   - Low computational cost

2. **ErrorBoundedQuantizer**
   ```rust
   // Guarantee |error| < ε
   scale = 2 * ε
   q = round(x / scale)
   reconstructed = q * scale
   ```
   - Uses INT16 for range
   - Mathematical guarantee: |x - reconstructed| < ε

3. **RunLengthQuantizer**
   ```rust
   // Compress zero-heavy residuals
   encode: [0,0,0,5,0,0] → [(3,0), (1,5), (2,0)]
   ```
   - Exploits predictability (many zeros)
   - High compression when predictor works well

**Typical Compression**: 50-100×

### Digital Twin Domain (nangila-twin)

**Goal**: Real-time edge-cloud synchronization

**Components**:
1. **EdgeNode**
   ```rust
   pub struct EdgeNode<P: Predictor, Q: Quantizer> {
       predictor: P,
       quantizer: Q,
   }
   
   impl EdgeNode {
       pub fn send(&mut self, sensor_data: &FixedPointBuffer) 
           -> Result<(Vec<u8>, f32), SyncError>;
   }
   ```
   - Compresses sensor data before transmission
   - Updates predictor with reconstructed values

2. **CloudNode**
   ```rust
   pub struct CloudNode<P: Predictor, Q: Quantizer> {
       predictor: P,
       quantizer: Q,
       current_state: Option<FixedPointBuffer>,
   }
   
   impl CloudNode {
       pub fn receive(&mut self, compressed: &[u8], scale: f32) 
           -> Result<FixedPointBuffer, SyncError>;
   }
   ```
   - Reconstructs sensor data from compressed packets
   - Maintains synchronized predictor state

**Typical Compression**: 80-120×

---

## Design Decisions

### 1. Why Traits Instead of Enums?

**Option A: Enum Dispatch**
```rust
enum Predictor {
    Momentum(MomentumPredictor),
    Linear(LinearPredictor),
}
```
❌ Closed set (can't add new predictors without modifying core)  
❌ Larger binary size (all variants included)

**Option B: Trait Objects**
```rust
Box<dyn Predictor>
```
❌ Heap allocation  
❌ Virtual dispatch overhead  
✅ Open set (extensible)

**Option C: Generic Traits** (Our Choice)
```rust
fn compress<P: Predictor, Q: Quantizer>(p: P, q: Q) { ... }
```
✅ Zero-cost abstraction (monomorphization)  
✅ Open set (extensible)  
✅ No heap allocation  
✅ Compile-time dispatch

### 2. Why Closed-Loop Prediction?

**Open-Loop** (Wrong):
```rust
// Sender
prediction = predictor.predict();
residual = data - prediction;
compressed = quantizer.quantize(residual);
predictor.update(data);  // ← Uses raw data

// Receiver
prediction = predictor.predict();
residual = quantizer.dequantize(compressed);
reconstructed = prediction + residual;
predictor.update(reconstructed);  // ← Uses reconstructed

// Problem: Predictors diverge over time!
```

**Closed-Loop** (Correct):
```rust
// Sender
prediction = predictor.predict();
residual = data - prediction;
compressed = quantizer.quantize(residual);
reconstructed_residual = quantizer.dequantize(compressed);
reconstructed = prediction + reconstructed_residual;
predictor.update(reconstructed);  // ← Uses reconstructed

// Receiver
prediction = predictor.predict();
residual = quantizer.dequantize(compressed);
reconstructed = prediction + residual;
predictor.update(reconstructed);  // ← Uses reconstructed

// Both predictors stay synchronized!
```

### 3. Why UnifiedQuantizer?

**Before**: Separate types for each mode
```rust
let stochastic = StochasticQuantizer::new(42);
let error_bounded = ErrorBoundedQuantizer::new(1e-3);

// Can't switch at runtime without Box<dyn Quantizer>
```

**After**: Single type with mode enum
```rust
let mut quantizer = UnifiedQuantizer::new(
    QuantizationMode::Stochastic { seed: 42, bits: 4 }
);

// Switch modes with zero overhead
quantizer.set_mode(QuantizationMode::ErrorBounded { epsilon: 1e-3 });
```

Benefits:
- ✅ Runtime mode switching
- ✅ Zero overhead (enum dispatch)
- ✅ Serializable configuration
- ✅ Single API to learn

---

## Extension Points

### Adding a New Predictor

```rust
// 1. Implement the Predictor trait
pub struct MyPredictor {
    state: Vec<f32>,
}

impl Predictor for MyPredictor {
    fn predict(&self) -> Result<FixedPointBuffer, PredictorError> {
        // Your prediction logic
    }
    
    fn update(&mut self, observation: &FixedPointBuffer) -> Result<(), PredictorError> {
        // Update internal state
    }
    
    fn reset(&mut self) {
        self.state.clear();
    }
}

// 2. Use it with any quantizer
let predictor = MyPredictor { state: vec![] };
let quantizer = UnifiedQuantizer::new(QuantizationMode::Stochastic { seed: 42, bits: 4 });

// Works immediately!
```

### Adding a New Quantizer

```rust
// 1. Implement the Quantizer trait
pub struct MyQuantizer {
    config: MyConfig,
}

impl Quantizer for MyQuantizer {
    fn quantize(&self, residual: &FixedPointBuffer) -> (Vec<u8>, f32) {
        // Your compression logic
    }
    
    fn dequantize(&self, compressed: &[u8], scale: f32) -> FixedPointBuffer {
        // Your decompression logic
    }
}

// 2. Use it with any predictor
let predictor = MomentumPredictor::new(0.9);
let quantizer = MyQuantizer { config: MyConfig::default() };

// Works immediately!
```

### Adding a New Domain

```rust
// Create a new crate: nangila-robotics
// Cargo.toml:
// [dependencies]
// nangila-core = { path = "../nangila-core" }
// nangila-math = { path = "../nangila-math" }

// Implement domain-specific predictors/quantizers
pub struct KalmanPredictor { ... }
impl Predictor for KalmanPredictor { ... }

pub struct AdaptiveQuantizer { ... }
impl Quantizer for AdaptiveQuantizer { ... }

// Use existing infrastructure!
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `FixedPointBuffer::from_f32` | O(n) | Linear scan + conversion |
| `Predictor::predict` | O(n) | Depends on implementation |
| `Predictor::update` | O(n) | Depends on implementation |
| `Quantizer::quantize` | O(n) | Linear scan + encoding |
| `Quantizer::dequantize` | O(n) | Linear scan + decoding |
| `TopologyMask::from_variance` | O(n log n) | Sorting dominates |
| `TopologyMask::is_driver` | O(log n) | Binary search |

### Space Complexity

| Type | Size | Notes |
|------|------|-------|
| `FixedPointBuffer` | 4n bytes | Vec<i32> |
| `MomentumPredictor` | 4n bytes | Stores momentum buffer |
| `LinearPredictor` | 8n bytes | Stores 2 previous states |
| `TopologyMask` | ~8m bytes | m = num_layers |
| `UnifiedQuantizer` | ~40 bytes | QuantizationMode enum + seed + counter |

### Throughput (CPU)

| Domain | Compression | Decompression | Bottleneck |
|--------|-------------|---------------|------------|
| AI | 100-200 MB/s | 150-300 MB/s | Topology mask lookup |
| HPC | 200-400 MB/s | 300-500 MB/s | Predictor update |
| Twin | 50-100 MB/s | 100-200 MB/s | Small batch overhead |

---

## Conclusion

Mini-Nangila's architecture achieves three goals:

1. **Modularity**: Clean separation between math, abstractions, and domain logic
2. **Extensibility**: Trait-based design allows easy addition of new predictors/quantizers
3. **Performance**: Zero-cost abstractions with compile-time dispatch

The design proves the "science" (compression algorithms) while keeping the "scale" (GPU optimization) for the proprietary version.

**Key Takeaway**: Compression ratios are determined by algorithms, not hardware. Mini-Nangila validates the algorithms on CPU; the proprietary version accelerates them on GPU.
