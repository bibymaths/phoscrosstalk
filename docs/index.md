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
| `p(t)`     | Phosphosite occupancy              | N         |

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

## Quick start

```bash
phoscrosstalk \
  --config config.toml \
  --data data_timeseries/filtered_input1.csv \
  --rna-data data_timeseries/filtered_input3.csv \
  --tf-net data_interactions/tf_mrna.csv \
  --ptm-intra data_curated/processed/ptm_intra.db \
  --ptm-inter data_curated/processed/ptm_inter.db \
  --kinase-tsv data_interactions/kinase_sites.tsv \
  --outdir results/experiment_01 \
  --mechanism dist \
  --n-starts 3 \
  --max-steps 20000
```

See [Running PhosCrosstalk](running.md) for all command variants.
