# Developer Guide

## Setup

```bash
git clone https://github.com/bibymaths/phoscrosstalk.git
cd phoscrosstalk
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

## Running tests

```bash
pytest
```

With coverage report:

```bash
pytest --cov=phoscrosstalk --cov-report=term-missing
```

Tests live in `tests/`. The test suite covers config loading, data loading,
simulation, derived rates, RNA handling, and CLI smoke tests.

## Linting and formatting

```bash
# Check for lint issues
ruff check .

# Check formatting (dry run)
ruff format --check .

# Auto-format
ruff format .
```

Ruff is configured in `pyproject.toml` under `[tool.ruff]`.

## Building documentation

```bash
# Local dev server with live reload
mkdocs serve

# Strict build (fails on warnings)
mkdocs build --strict
```

Documentation source is in `docs/`. MkDocs configuration is in `mkdocs.yml`.

## Building the package

```bash
pip install --upgrade build
python -m build
```

Distribution artifacts are written to `dist/`.

## Module structure

| Module             | Purpose                                                     |
|--------------------|-------------------------------------------------------------|
| `main.py`          | CLI entry point and pipeline orchestration                  |
| `config.py`        | `ModelDims`, `load_config()`, `DEFAULT_TIMEPOINTS`         |
| `simulation.py`    | `simulate()`: Diffrax ODE solver wrapper                |
| `jax_mechanisms.py`| JAX RHS kernels for dist/seq/rand mechanisms                |
| `core_mechanisms.py`| `decode_theta()`, `clip_scalar()`                          |
| `optimization.py`  | `NetworkProblem`, scalar loss, `create_bounds()`            |
| `multistarts.py`   | Multi-start wrapper, best-solution selection                |
| `derived_rates.py` | `make_k_act_fn()`, `make_s_prod_fn()` closures             |
| `data_loader.py`   | Input parsing, scaling, matrix construction                 |
| `analysis.py`      | Post-optimization analysis, file export, plotting           |
| `post_processing.py`| Network export, clustermap, metadata                       |
| `steadystate.py`   | Post-optimization steady-state simulation                   |
| `knockouts.py`     | In-silico kinase knockout screening                         |
| `sensitivity.py`   | Global Sobol sensitivity analysis                           |
| `equations.py`     | ODE equation report generation                              |
| `app.py`           | Streamlit interactive dashboard                             |
| `weighting.py`     | Data weight matrix construction                             |

## GitHub Workflows

All workflows run **only on tag pushes** (e.g. `git tag v1.0.0 && git push --tags`).
They require the tag commit to be reachable from `origin/main`.

| Workflow      | File               | Purpose                          |
|---------------|--------------------|----------------------------------|
| CI            | `ci.yml`           | Run pytest with coverage         |
| Ruff          | `ruff.yml`         | Lint and format check            |
| Docs          | `docs.yml`         | Build and deploy MkDocs site     |
| Release       | `release.yml`      | Build package, create GitHub release |

## Adding a new mechanism

1. Implement the kinetic equations in `jax_mechanisms.py` as a JAX-compatible function.
2. Add the mechanism key to the `choices` list in `main.py` (`--mechanism`).
3. Add a unit test in `tests/test_simulation.py`.
4. Update the mechanism table in `docs/model.md`.

!!! warning "JAX compatibility"
    All RHS functions must be JAX-traceable (no Python control flow over
    traced values). Use `jnp` operations and `jax.lax` conditionals.
