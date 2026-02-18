## Releasing v0.1.0 (Open-Core)

This checklist covers crates and the Python DDP hook.

### Preflight
- Ensure CI is green (build/tests; determinism hash artifacts; bench logs)
- Update CHANGELOG.md and README references
- Verify CPU verification script: `./scripts/verify.sh`

### Crates (Cargo)
1. Set versions in Cargo.toml for:
   - `nangila-math`, `nangila-core`, `nangila-ai`, `nangila-hpc`, `nangila-twin`, `nangila-checkpoint`, `nangila-edge`
2. Tag the repo: `git tag v0.1.0 && git push origin v0.1.0`
3. Publish (order respects dependencies):
   - `cargo publish -p nangila-math`
   - `cargo publish -p nangila-core`
   - `cargo publish -p nangila-ai`
   - `cargo publish -p nangila-hpc`
   - `cargo publish -p nangila-twin`
   - `cargo publish -p nangila-checkpoint`
   - `cargo publish -p nangila-edge`

### Python (Wheel)
1. Create `pyproject.toml` in `mini-nangila/python` (see template below)
2. Build: `python -m build`
3. Upload: `twine upload dist/*`

### pyproject.toml (template)
```
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "nangila-ddp"
version = "0.1.0"
description = "DDP INT4 communication hook for predictive-residual compression (CPU/GPU)"
authors = [{ name="Nangila" }]
requires-python = ">=3.8"
dependencies = ["torch", "matplotlib"]
readme = "README.md"
license = { text = "Apache-2.0" }

[tool.setuptools]
packages = ["nangila_ddp", "nangila_ddp.examples"]
```

