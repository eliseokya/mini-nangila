# Phase 1 (Weeks 1–2): Core Alignment and Hardening

Scope aligns with Open‑Core Spec (WHITEPAPER_v2):

## Goals
- Fix docs/tests mismatches; ensure determinism story is tight.
- Finalize unified quantizer RNG determinism approach.

## Tasks

1) Doctests and README
- [x] TopologyMask doctest ordering and approx ratio check
- [x] README: default `cargo test`; document optional deps (HDF5/libtorch); update examples list

2) Q8.23 Determinism
- [x] Add determinism tests on exact rationals and integer ops
- [ ] Cross‑platform CI matrix (x86_64, ARM64) to validate bit‑exactness
- [ ] Clarify Q8.23 range semantics in docs vs implementation

3) Quantizer RNG Design
- [x] Unify guidance: deterministic RNG via RefCell or hash‑counter PRNG
- [x] Silence unused param warning in unified quantizer
- [x] Clarify `Quantizer` trait docs re determinism
- [ ] Consider `&mut self` variant or step counter argument for future versions

## Deliverables
- Green `cargo test` (workspace) without optional system deps
- Passing doctests
- Determinism tests in `nangila-math`
- Updated README and quantizer trait docs

## Out‑of‑Scope (tracked for later phases)
- PyTorch DDP comm hook + 2‑GPU NanoGPT example (Phase 1–2 bridge)
- HDF5 plugin crate and LAMMPS end‑to‑end (Phase 2)
- Rotor twin real‑time demo (Phase 2)
- Whitepaper alignment + tutorials + release packaging (Phase 3)

