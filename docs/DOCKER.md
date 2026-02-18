# Docker Usage

Mini-Nangila includes Docker support for reproducible builds and testing.

## Quick Start

### Build the Image

```bash
docker build -t mini-nangila .
```

### Run an Example

```bash
# LAMMPS compression
docker run --rm mini-nangila lammps-compress --help

# With mounted data
docker run --rm -v $(pwd)/data:/data mini-nangila \
  lammps-compress --input /data/sample.dump --output /output/compressed.nz
```

## Using Docker Compose

### Run Tests

```bash
docker-compose run test
```

### Run Benchmarks

```bash
docker-compose run benchmark
```

### Interactive Shell

```bash
docker-compose run --rm mini-nangila bash
```

## Build Stages

The Dockerfile uses multi-stage builds:

1. **Builder stage**: Compiles all binaries with caching
2. **Runtime stage**: Minimal Debian image with only binaries

This keeps the final image small (~100MB) while caching dependencies for fast rebuilds.

## CI/CD Integration

The main repository includes GitHub Actions CI that:
- Tests on Ubuntu and macOS
- Runs clippy lints
- Builds all examples
- Caches dependencies

See [../.github/workflows/ci.yml](../.github/workflows/ci.yml) for details.

## Reproduc ibility

The Docker image ensures:
- ✅ Pinned Rust version (1.75)
- ✅ Pinned dependencies (Cargo.lock)
- ✅ Consistent build environment
- ✅ All tests pass before image creation

This guarantees that builds are reproducible across machines and time.
