# Installation

## Requirements

- Python 3.11
- JAX-compatible runtime (CPU or GPU)

## Install from source

```bash
git clone https://github.com/bibymaths/phoscrosstalk.git
cd phoscrosstalk
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

## Optional extras

| Extra   | Purpose                         |
|---------|---------------------------------|
| `dev`   | Testing, linting (pytest, ruff) |
| `docs`  | Documentation (MkDocs, Material)|

## Verify installation

```bash
phoscrosstalk --help
```

!!! tip "Smoke test"
    Use `--n-starts 1 --max-steps 50` to verify end-to-end execution quickly
    without waiting for full convergence.
