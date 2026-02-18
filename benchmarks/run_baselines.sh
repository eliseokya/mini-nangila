#!/usr/bin/env bash
# Baseline benchmark script for LAMMPS compression
# Compares Mini‑Nangila (ErrorBounded and ErrorBounded+RLE) against gzip/bzip2/xz,
# and optionally SZ/ZFP if CLI tools are available.
#
# To enable SZ/ZFP via CLI, set one of the following env vars before running:
#   ZFP_CMD='zfp -i {in} -o {out} -r {eps}'
#   SZ_CMD='sz -i {in} -o {out} -M ABS {eps}'
# Replace with your local CLI syntax. Placeholders: {in}, {out}, {eps}

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Mini-Nangila Baseline Benchmark"
echo "========================================="
echo ""

# Configuration
DATA_FILE="benchmarks/data/large.dump"
RESULTS_DIR="benchmarks/results"
RESULTS_FILE="$RESULTS_DIR/baseline_comparison.txt"
RESULTS_CSV="$RESULTS_DIR/baseline_comparison.csv"

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}Error: $DATA_FILE not found${NC}"
    echo "Generating synthetic dataset..."
    cargo run --release --bin lammps-compress -- \
        --generate "$DATA_FILE" \
        --particles 100000 \
        --steps 100
fi

# Get original size
ORIGINAL_SIZE=$(stat -f%z "$DATA_FILE" 2>/dev/null || stat -c%s "$DATA_FILE")
ORIGINAL_MB=$(echo "scale=2; $ORIGINAL_SIZE / 1048576" | bc)

echo -e "${GREEN}Original file: $DATA_FILE${NC}"
echo "Size: $ORIGINAL_MB MB ($ORIGINAL_SIZE bytes)"
echo ""

# Initialize results file
mkdir -p "$RESULTS_DIR"
echo "Baseline Compression Comparison" > "$RESULTS_FILE"
echo "Generated: $(date)" >> "$RESULTS_FILE"
echo "Original size: $ORIGINAL_MB MB ($ORIGINAL_SIZE bytes)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Initialize CSV
mkdir -p "$RESULTS_DIR"
echo "tool,epsilon,codec,compressed_bytes,ratio,comp_time_s,throughput_MBps,status" > "$RESULTS_CSV"

# Function to run compression and measure
compress_and_measure() {
    local tool=$1
    local command=$2
    local output=$3
    local label=$4
    
    echo -e "${YELLOW}Testing: $label${NC}"
    
    # Remove old output
    rm -f "$output"
    
    # Measure compression time
    START=$(date +%s%N)
    eval "$command" > /dev/null 2>&1
    END=$(date +%s%N)
    
    COMP_TIME=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
    COMP_SIZE=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output")
    COMP_MB=$(echo "scale=2; $COMP_SIZE / 1048576" | bc)
    RATIO=$(echo "scale=2; $ORIGINAL_SIZE / $COMP_SIZE" | bc)
    THROUGHPUT=$(echo "scale=2; $ORIGINAL_MB / $COMP_TIME" | bc)
    
    echo "  Compressed: $COMP_MB MB ($COMP_SIZE bytes)"
    echo "  Ratio: ${RATIO}×"
    echo "  Time: ${COMP_TIME}s"
    echo "  Throughput: ${THROUGHPUT} MB/s"
    echo ""
    
    # Write to results
    echo "$label:" >> "$RESULTS_FILE"
    echo "  Compressed size: $COMP_MB MB ($COMP_SIZE bytes)" >> "$RESULTS_FILE"
    echo "  Compression ratio: ${RATIO}×" >> "$RESULTS_FILE"
    echo "  Compression time: ${COMP_TIME}s" >> "$RESULTS_FILE"
    echo "  Throughput: ${THROUGHPUT} MB/s" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    # CSV row
    # tool label, epsilon N/A unless provided by caller via global CSV_EPS
    local eps=${CSV_EPS:-}
    local codec=${CSV_CODEC:-}
    echo "$label,$eps,$codec,$COMP_SIZE,$RATIO,$COMP_TIME,$THROUGHPUT,OK" >> "$RESULTS_CSV"
}

# Test compressors
echo "========================================="
echo "Running Compressions..."
echo "========================================="
echo ""

# gzip (level 1 - fast)
compress_and_measure "gzip1" \
    "gzip -1 -c $DATA_FILE > $RESULTS_DIR/large.dump.gz1" \
    "$RESULTS_DIR/large.dump.gz1" \
    "gzip -1 (fast)"

# gzip (level 6 - default)
compress_and_measure "gzip6" \
    "gzip -6 -c $DATA_FILE > $RESULTS_DIR/large.dump.gz6" \
    "$RESULTS_DIR/large.dump.gz6" \
    "gzip -6 (default)"

# gzip (level 9 - best)
compress_and_measure "gzip9" \
    "gzip -9 -c $DATA_FILE > $RESULTS_DIR/large.dump.gz9" \
    "$RESULTS_DIR/large.dump.gz9" \
    "gzip -9 (best)"

# bzip2
compress_and_measure "bzip2" \
    "bzip2 -c $DATA_FILE > $RESULTS_DIR/large.dump.bz2" \
    "$RESULTS_DIR/large.dump.bz2" \
    "bzip2"

# xz
compress_and_measure "xz" \
    "xz -c $DATA_FILE > $RESULTS_DIR/large.dump.xz" \
    "$RESULTS_DIR/large.dump.xz" \
    "xz"

# Mini‑Nangila (epsilon=0.001, ErrorBounded)
echo -e "${YELLOW}Testing: Mini-Nangila (ε=0.001, EB)${NC}"
START=$(date +%s%N)
cargo run --release --bin lammps-compress -- \
    --input "$DATA_FILE" \
    --epsilon 0.001 \
    --output "$RESULTS_DIR/large.nz.001" 2>&1 | grep "DONE" | tee -a "$RESULTS_DIR/mini_nangila_001.log"
END=$(date +%s%N)

COMP_TIME=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
COMP_SIZE=$(stat -f%z "$RESULTS_DIR/large.nz.001" 2>/dev/null || stat -c%s "$RESULTS_DIR/large.nz.001")
COMP_MB=$(echo "scale=2; $COMP_SIZE / 1048576" | bc)
RATIO=$(echo "scale=2; $ORIGINAL_SIZE / $COMP_SIZE" | bc)
THROUGHPUT=$(echo "scale=2; $ORIGINAL_MB / $COMP_TIME" | bc)

echo "  Compressed: $COMP_MB MB ($COMP_SIZE bytes)"
echo "  Ratio: ${RATIO}×"
echo "  Time: ${COMP_TIME}s"
echo "  Throughput: ${THROUGHPUT} MB/s"
echo ""

echo "Mini-Nangila (ε=0.001, EB):" >> "$RESULTS_FILE"
echo "  Compressed size: $COMP_MB MB ($COMP_SIZE bytes)" >> "$RESULTS_FILE"
echo "  Compression ratio: ${RATIO}×" >> "$RESULTS_FILE"
echo "  Compression time: ${COMP_TIME}s" >> "$RESULTS_FILE"
echo "  Throughput: ${THROUGHPUT} MB/s" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# CSV row
CSV_EPS=0.001 CSV_CODEC=EB \
    echo "Mini-Nangila (EB),0.001,EB,$COMP_SIZE,$RATIO,$COMP_TIME,$THROUGHPUT,OK" >> "$RESULTS_CSV"

# Mini‑Nangila (epsilon=0.001, ErrorBounded+RLE)
echo -e "${YELLOW}Testing: Mini-Nangila (ε=0.001, EB+RLE)${NC}"
START=$(date +%s%N)
cargo run --release --bin lammps-compress -- \
    --input "$DATA_FILE" \
    --epsilon 0.001 \
    --use-rle \
    --output "$RESULTS_DIR/large.nz.001.rle" 2>&1 | grep "DONE" | tee -a "$RESULTS_DIR/mini_nangila_001_rle.log"
END=$(date +%s%N)

COMP_TIME=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
COMP_SIZE=$(stat -f%z "$RESULTS_DIR/large.nz.001.rle" 2>/dev/null || stat -c%s "$RESULTS_DIR/large.nz.001.rle")
COMP_MB=$(echo "scale=2; $COMP_SIZE / 1048576" | bc)
RATIO=$(echo "scale=2; $ORIGINAL_SIZE / $COMP_SIZE" | bc)
THROUGHPUT=$(echo "scale=2; $ORIGINAL_MB / $COMP_TIME" | bc)

echo "  Compressed: $COMP_MB MB ($COMP_SIZE bytes)"
echo "  Ratio: ${RATIO}×"
echo "  Time: ${COMP_TIME}s"
echo "  Throughput: ${THROUGHPUT} MB/s"
echo ""

echo "Mini-Nangila (ε=0.001, EB+RLE):" >> "$RESULTS_FILE"
echo "  Compressed size: $COMP_MB MB ($COMP_SIZE bytes)" >> "$RESULTS_FILE"
echo "  Compression ratio: ${RATIO}×" >> "$RESULTS_FILE"
echo "  Compression time: ${COMP_TIME}s" >> "$RESULTS_FILE"
echo "  Throughput: ${THROUGHPUT} MB/s" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# CSV row
CSV_EPS=0.001 CSV_CODEC=EB+RLE \
    echo "Mini-Nangila (EB+RLE),0.001,EB+RLE,$COMP_SIZE,$RATIO,$COMP_TIME,$THROUGHPUT,OK" >> "$RESULTS_CSV"

# Mini‑Nangila (epsilon=0.001, BlockSparse)
echo -e "${YELLOW}Testing: Mini-Nangila (ε=0.001, BlockSparse)${NC}"
START=$(date +%s%N)
cargo run --release --bin lammps-compress -- \
    --input "$DATA_FILE" \
    --epsilon 0.001 \
    --use-block-sparse \
    --output "$RESULTS_DIR/large.nz.001.bs" 2>&1 | grep "DONE" | tee -a "$RESULTS_DIR/mini_nangila_001_bs.log"
END=$(date +%s%N)

COMP_TIME=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
COMP_SIZE=$(stat -f%z "$RESULTS_DIR/large.nz.001.bs" 2>/dev/null || stat -c%s "$RESULTS_DIR/large.nz.001.bs")
COMP_MB=$(echo "scale=2; $COMP_SIZE / 1048576" | bc)
RATIO=$(echo "scale=2; $ORIGINAL_SIZE / $COMP_SIZE" | bc)
THROUGHPUT=$(echo "scale=2; $ORIGINAL_MB / $COMP_TIME" | bc)

echo "  Compressed: $COMP_MB MB ($COMP_SIZE bytes)"
echo "  Ratio: ${RATIO}×"
echo "  Time: ${COMP_TIME}s"
echo "  Throughput: ${THROUGHPUT} MB/s"
echo ""

echo "Mini-Nangila (ε=0.001, BlockSparse):" >> "$RESULTS_FILE"
echo "  Compressed size: $COMP_MB MB ($COMP_SIZE bytes)" >> "$RESULTS_FILE"
echo "  Compression ratio: ${RATIO}×" >> "$RESULTS_FILE"
echo "  Compression time: ${COMP_TIME}s" >> "$RESULTS_FILE"
echo "  Throughput: ${THROUGHPUT} MB/s" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# CSV row
CSV_EPS=0.001 CSV_CODEC=BlockSparse \
    echo "Mini-Nangila (BlockSparse),0.001,BlockSparse,$COMP_SIZE,$RATIO,$COMP_TIME,$THROUGHPUT,OK" >> "$RESULTS_CSV"

# Mini‑Nangila (epsilon=0.001, BlockSparse+RLE)
echo -e "${YELLOW}Testing: Mini-Nangila (ε=0.001, BlockSparse+RLE)${NC}"
START=$(date +%s%N)
cargo run --release --bin lammps-compress -- \
    --input "$DATA_FILE" \
    --epsilon 0.001 \
    --use-block-sparse-rle \
    --output "$RESULTS_DIR/large.nz.001.bsrle" 2>&1 | grep "DONE" | tee -a "$RESULTS_DIR/mini_nangila_001_bsrle.log"
END=$(date +%s%N)

COMP_TIME=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
COMP_SIZE=$(stat -f%z "$RESULTS_DIR/large.nz.001.bsrle" 2>/dev/null || stat -c%s "$RESULTS_DIR/large.nz.001.bsrle")
COMP_MB=$(echo "scale=2; $COMP_SIZE / 1048576" | bc)
RATIO=$(echo "scale=2; $ORIGINAL_SIZE / $COMP_SIZE" | bc)
THROUGHPUT=$(echo "scale=2; $ORIGINAL_MB / $COMP_TIME" | bc)

echo "  Compressed: $COMP_MB MB ($COMP_SIZE bytes)"
echo "  Ratio: ${RATIO}×"
echo "  Time: ${COMP_TIME}s"
echo "  Throughput: ${THROUGHPUT} MB/s"
echo ""

echo "Mini-Nangila (ε=0.001, BlockSparse+RLE):" >> "$RESULTS_FILE"
echo "  Compressed size: $COMP_MB MB ($COMP_SIZE bytes)" >> "$RESULTS_FILE"
echo "  Compression ratio: ${RATIO}×" >> "$RESULTS_FILE"
echo "  Compression time: ${COMP_TIME}s" >> "$RESULTS_FILE"
echo "  Throughput: ${THROUGHPUT} MB/s" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# CSV row
CSV_EPS=0.001 CSV_CODEC=BlockSparse+RLE \
    echo "Mini-Nangila (BlockSparse+RLE),0.001,BlockSparse+RLE,$COMP_SIZE,$RATIO,$COMP_TIME,$THROUGHPUT,OK" >> "$RESULTS_CSV"

# Mini‑Nangila (epsilon=0.01, EB)
echo -e "${YELLOW}Testing: Mini-Nangila (ε=0.01, EB)${NC}"
START=$(date +%s%N)
cargo run --release --bin lammps-compress -- \
    --input "$DATA_FILE" \
    --epsilon 0.01 \
    --output "$RESULTS_DIR/large.nz.01" 2>&1 | grep "DONE" | tee -a "$RESULTS_DIR/mini_nangila_01.log"
END=$(date +%s%N)

COMP_TIME=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
COMP_SIZE=$(stat -f%z "$RESULTS_DIR/large.nz.01" 2>/dev/null || stat -c%s "$RESULTS_DIR/large.nz.01")
COMP_MB=$(echo "scale=2; $COMP_SIZE / 1048576" | bc)
RATIO=$(echo "scale=2; $ORIGINAL_SIZE / $COMP_SIZE" | bc)
THROUGHPUT=$(echo "scale=2; $ORIGINAL_MB / $COMP_TIME" | bc)

echo "  Compressed: $COMP_MB MB ($COMP_SIZE bytes)"
echo "  Ratio: ${RATIO}×"
echo "  Time: ${COMP_TIME}s"
echo "  Throughput: ${THROUGHPUT} MB/s"
echo ""

echo "Mini-Nangila (ε=0.01, EB):" >> "$RESULTS_FILE"
echo "  Compressed size: $COMP_MB MB ($COMP_SIZE bytes)" >> "$RESULTS_FILE"
echo "  Compression ratio: ${RATIO}×" >> "$RESULTS_FILE"
echo "  Compression time: ${COMP_TIME}s" >> "$RESULTS_FILE"
echo "  Throughput: ${THROUGHPUT} MB/s" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# CSV row
CSV_EPS=0.01 CSV_CODEC=EB \
    echo "Mini-Nangila (EB),0.01,EB,$COMP_SIZE,$RATIO,$COMP_TIME,$THROUGHPUT,OK" >> "$RESULTS_CSV"

# Mini‑Nangila (epsilon=0.01, EB+RLE)
echo -e "${YELLOW}Testing: Mini-Nangila (ε=0.01, EB+RLE)${NC}"
START=$(date +%s%N)
cargo run --release --bin lammps-compress -- \
    --input "$DATA_FILE" \
    --epsilon 0.01 \
    --use-rle \
    --output "$RESULTS_DIR/large.nz.01.rle" 2>&1 | grep "DONE" | tee -a "$RESULTS_DIR/mini_nangila_01_rle.log"
END=$(date +%s%N)

COMP_TIME=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
COMP_SIZE=$(stat -f%z "$RESULTS_DIR/large.nz.01.rle" 2>/dev/null || stat -c%s "$RESULTS_DIR/large.nz.01.rle")
COMP_MB=$(echo "scale=2; $COMP_SIZE / 1048576" | bc)
RATIO=$(echo "scale=2; $ORIGINAL_SIZE / $COMP_SIZE" | bc)
THROUGHPUT=$(echo "scale=2; $ORIGINAL_MB / $COMP_TIME" | bc)

echo "  Compressed: $COMP_MB MB ($COMP_SIZE bytes)"
echo "  Ratio: ${RATIO}×"
echo "  Time: ${COMP_TIME}s"
echo "  Throughput: ${THROUGHPUT} MB/s"
echo ""

echo "Mini-Nangila (ε=0.01, EB+RLE):" >> "$RESULTS_FILE"
echo "  Compressed size: $COMP_MB MB ($COMP_SIZE bytes)" >> "$RESULTS_FILE"
echo "  Compression ratio: ${RATIO}×" >> "$RESULTS_FILE"
echo "  Compression time: ${COMP_TIME}s" >> "$RESULTS_FILE"
echo "  Throughput: ${THROUGHPUT} MB/s" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# CSV row
CSV_EPS=0.01 CSV_CODEC=EB+RLE \
    echo "Mini-Nangila (EB+RLE),0.01,EB+RLE,$COMP_SIZE,$RATIO,$COMP_TIME,$THROUGHPUT,OK" >> "$RESULTS_CSV"

# Optional: SZ and ZFP (if commands configured)
if [ -n "${ZFP_CMD:-}" ]; then
  echo -e "${YELLOW}Testing: ZFP (ε=0.001)${NC}"
  OUT="$RESULTS_DIR/large.dump.zfp"
  rm -f "$OUT"
  START=$(date +%s%N)
  eval "${ZFP_CMD//\{in\}/$DATA_FILE}" | sed "s|{out}|$OUT|g" | sed "s|{eps}|0.001|g" >/dev/null 2>&1 || true
  END=$(date +%s%N)
  if [ -f "$OUT" ]; then
    COMP_TIME=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
    COMP_SIZE=$(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT")
    COMP_MB=$(echo "scale=2; $COMP_SIZE / 1048576" | bc)
    RATIO=$(echo "scale=2; $ORIGINAL_SIZE / $COMP_SIZE" | bc)
    THROUGHPUT=$(echo "scale=2; $ORIGINAL_MB / $COMP_TIME" | bc)
    echo "ZFP (ε=0.001):" >> "$RESULTS_FILE"
    echo "  Compressed size: $COMP_MB MB ($COMP_SIZE bytes)" >> "$RESULTS_FILE"
    echo "  Compression ratio: ${RATIO}×" >> "$RESULTS_FILE"
    echo "  Compression time: ${COMP_TIME}s" >> "$RESULTS_FILE"
    echo "  Throughput: ${THROUGHPUT} MB/s" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    echo "ZFP,0.001,NA,$COMP_SIZE,$RATIO,$COMP_TIME,$THROUGHPUT,OK" >> "$RESULTS_CSV"
  else
    echo "ZFP,0.001,NA,0,0,0,0,SKIPPED" >> "$RESULTS_CSV"
  fi
fi

if [ -n "${SZ_CMD:-}" ]; then
  echo -e "${YELLOW}Testing: SZ (ε=0.001)${NC}"
  OUT="$RESULTS_DIR/large.dump.sz"
  rm -f "$OUT"
  START=$(date +%s%N)
  eval "${SZ_CMD//\{in\}/$DATA_FILE}" | sed "s|{out}|$OUT|g" | sed "s|{eps}|0.001|g" >/dev/null 2>&1 || true
  END=$(date +%s%N)
  if [ -f "$OUT" ]; then
    COMP_TIME=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
    COMP_SIZE=$(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT")
    COMP_MB=$(echo "scale=2; $COMP_SIZE / 1048576" | bc)
    RATIO=$(echo "scale=2; $ORIGINAL_SIZE / $COMP_SIZE" | bc)
    THROUGHPUT=$(echo "scale=2; $ORIGINAL_MB / $COMP_TIME" | bc)
    echo "SZ (ε=0.001):" >> "$RESULTS_FILE"
    echo "  Compressed size: $COMP_MB MB ($COMP_SIZE bytes)" >> "$RESULTS_FILE"
    echo "  Compression ratio: ${RATIO}×" >> "$RESULTS_FILE"
    echo "  Compression time: ${COMP_TIME}s" >> "$RESULTS_FILE"
    echo "  Throughput: ${THROUGHPUT} MB/s" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    echo "SZ,0.001,NA,$COMP_SIZE,$RATIO,$COMP_TIME,$THROUGHPUT,OK" >> "$RESULTS_CSV"
  else
    echo "SZ,0.001,NA,0,0,0,0,SKIPPED" >> "$RESULTS_CSV"
  fi
fi

echo "========================================="
echo -e "${GREEN}Benchmark Complete!${NC}"
echo "========================================="
echo ""
echo "Results saved to: $RESULTS_FILE"
echo "CSV: $RESULTS_CSV"
echo ""
echo "Summary:"
cat "$RESULTS_FILE"
