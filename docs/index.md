# PhosCrosstalk

**Global phospho-network ODE modeling with PTM crosstalk, kinase-site priors, and TF/mRNA integration.**

PhosCrosstalk reconstructs protein activation, kinase activity, and phosphosite kinetics
across a signalling network. It fits large parameter sets to time-series phosphoproteomics
data using gradient-based multi-start optimisation via [JAX](https://github.com/google/jax),
[Diffrax](https://github.com/patrick-kidger/diffrax), and
[Optimistix](https://github.com/patrick-kidger/optimistix).

## What PhosCrosstalk does

- Models global phospho-network dynamics with five coupled ODE state variables
- Integrates PTMcode2 intra/inter-protein crosstalk priors
- Uses kinase-site interaction priors (KEA/KS or custom TSV)
- Incorporates a TF→mRNA regulatory network to drive protein activation
- Fits a single scalar loss to phosphosite, protein abundance, and mRNA time-series
- Supports distributive, sequential, and random/cooperative kinase mechanisms
- Offers downstream analyses: steady-state, in-silico knockouts, global sensitivity
- Provides an interactive Streamlit dashboard for result exploration

## State variables

| Variable   | Description                        | Dimension |
|------------|------------------------------------|-----------|
| `R(t)`     | mRNA level (ODE state)             | K         |
| `S(t)`     | Protein activation / signalling    | K         |
| `A(t)`     | Protein abundance                  | K         |
| `Kdyn(t)`  | Kinase activity                    | M         |
| `p(t)`     | Relative phosphosite signal        | N         |

Full state vector: `y = [R, S, A, Kdyn, p]`, dimension `3K + M + N`.

## Scalar objective

```
total_loss =
    w_phospho   × phosphosite_loss
  + w_abundance × protein_abundance_loss
  + w_rna       × rna_loss
  + w_reg       × regularization_loss
```

Weights are configured in `config.toml` under `[loss_weights]`.

!!! note "Note"
    `k_act(t)` and `s_prod(t)` are derived rates computed from data, not optimized parameters.

!!! note "Phosphosite state interpretation"
    Phosphosite state `p` is modeled as a nonnegative relative signal, not a
    fractional occupancy.  Occupancy-like regulation uses `q = p/(1+p)`.  This
    allows fitting fold-change or relative-intensity phosphoproteomics values
    above 1 while retaining bounded regulatory feedback.

## Quick start

```bash
# Sync dependencies and run from repo root
uv sync
uv run phoscrosstalk --config config.toml

# Launch the Streamlit dashboard
uv run streamlit run phoscrosstalk/app.py
```

All data paths, model settings, and analysis flags live in `config.toml`.
The CLI only accepts `--config <path>` (plus `--help` and `--version`).

See [Running PhosCrosstalk](running.md) for all command variants.
