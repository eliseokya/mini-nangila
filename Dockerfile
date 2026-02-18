# Multi-stage Dockerfile for Mini-Nangila
# Optimized for reproducible builds and fast iteration

# Stage 1: Builder
FROM rust:1.75-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy manifests first (for better caching)
COPY Cargo.toml Cargo.lock ./
COPY nangila-math/Cargo.toml ./nangila-math/
COPY nangila-core/Cargo.toml ./nangila-core/
COPY nangila-ai/Cargo.toml ./nangila-ai/
COPY nangila-hpc/Cargo.toml ./nangila-hpc/
COPY nangila-twin/Cargo.toml ./nangila-twin/
COPY nangila-edge/Cargo.toml ./nangila-edge/
COPY nangila-checkpoint/Cargo.toml ./nangila-checkpoint/
COPY examples/*/Cargo.toml ./examples/

# Create dummy source files to cache dependencies
RUN mkdir -p nangila-math/src nangila-core/src nangila-ai/src nangila-hpc/src \
    nangila-twin/src nangila-edge/src nangila-checkpoint/src && \
    echo "fn main() {}" > nangila-math/src/lib.rs && \
    echo "fn main() {}" > nangila-core/src/lib.rs && \
    echo "fn main() {}" > nangila-ai/src/lib.rs && \
    echo "fn main() {}" > nangila-hpc/src/lib.rs && \
    echo "fn main() {}" > nangila-twin/src/lib.rs && \
    echo "fn main() {}" > nangila-edge/src/lib.rs && \
    echo "fn main() {}" > nangila-checkpoint/src/lib.rs

# Build dependencies (cached layer)
RUN cargo build --release --workspace || true

# Copy actual source code
COPY . .

# Build for real
RUN cargo build --release --workspace

# Run tests
RUN cargo test --release --workspace

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 nangila

# Copy binaries from builder
COPY --from=builder /build/target/release/nano-gpt /usr/local/bin/
COPY --from=builder /build/target/release/lammps-compress /usr/local/bin/
COPY --from=builder /build/target/release/rotor-twin /usr/local/bin/
COPY --from=builder /build/target/release/topology-calibration /usr/local/bin/
COPY --from=builder /build/target/release/mode-switching /usr/local/bin/
COPY --from=builder /build/target/release/throughput-bench /usr/local/bin/

# Copy example data
COPY --from=builder /build/examples/lammps_compress/sample.dump /data/sample.dump

# Set user
USER nangila
WORKDIR /home/nangila

# Default command
CMD ["lammps-compress", "--help"]
