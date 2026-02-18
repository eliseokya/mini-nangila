#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=verification
mkdir -p "$OUT"

echo "[1/6] Build + tests"
cargo build >/dev/null
cargo test >"$OUT/tests.txt"

echo "[2/6] Determinism hash"
cargo run -q -p nangila-math --example determinism_hash >"$OUT/determinism_hash.txt"
cat "$OUT/determinism_hash.txt"

echo "[3/6] AI throughput bench (CPU)"
cargo run --release --manifest-path examples/throughput_bench/Cargo.toml >"$OUT/ai_throughput.txt"

echo "[4/6] HPC compress bench (CPU)"
cargo run --release --manifest-path examples/hpc_compress_bench/Cargo.toml >"$OUT/hpc_bench.txt"

echo "[5/6] Topology calibration"
cargo run --release --manifest-path examples/topology_calibration/Cargo.toml >"$OUT/topology.txt"

echo "[6/6] Rotor twin demo (short)"
cargo run --release --manifest-path examples/rotor_twin/Cargo.toml -- --duration 3 --rate-hz 50 >"$OUT/rotor.txt"

echo "Verification outputs in $OUT/"
