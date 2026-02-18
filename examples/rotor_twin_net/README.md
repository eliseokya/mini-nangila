## Rotor Twin (gRPC) — Feature-gated Example

This example demonstrates CPU-only edge→cloud streaming using gRPC (tonic):
- Edge (client): simulates IMU, applies prediction+quantization, transmits residuals
- Cloud (server): dequantizes, reconstructs, updates predictor, logs metrics

Note: This crate is outside the workspace and not built by default. It requires `protoc` for code generation. Enable feature `net` when building.

### Prerequisites
- Rust toolchain
- `protoc` installed and on PATH (e.g., `brew install protobuf` on macOS)

### Build

```
cd mini-nangila/examples/rotor_twin_net
cargo run --bin server --features net -- --listen 127.0.0.1:50051 --metrics server_metrics.csv
```

In another terminal:

```
cd mini-nangila/examples/rotor_twin_net
cargo run --bin client --features net -- --server http://127.0.0.1:50051 --rate-hz 50 --duration 5 --metrics client_metrics.csv
```

### Plot

```
python mini-nangila/python/snippets/plot_rotor_net.py mini-nangila/examples/rotor_twin_net/client_metrics.csv mini-nangila/examples/rotor_twin_net/server_metrics.csv
```

This produces `rotor_net_ratio.png` showing compression ratio over ticks.

