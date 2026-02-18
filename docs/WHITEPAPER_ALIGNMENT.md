## Whitepaper Alignment Checklist (Mini-Nangila)

- Terminology and equations match WHITEPAPER_v2.md (Predictor, Residual, Quantizer)
- AI (Stochastic): Momentum predictor, INT4 stochastic quantizer, static topology masking
- HPC (Deterministic): Linear predictor, error-bounded quantizer (ε), RLE for sparsity
- Twin: Closed-loop edge/cloud synchronization

Open Items:
- PyTorch DDP comm hook with real bandwidth reduction (INT4 transport)
- HDF5 filter plugin and LAMMPS integration
- Rotor twin real-time demo metrics and plots
- Cross-platform determinism CI (x86_64/ARM64)
- Science tutorials and reproducibility artifacts
