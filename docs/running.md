# Running PhosCrosstalk

All runtime options are configured in `config.toml`.
The CLI only accepts `--config <path>` (plus `--help` and `--version`).

## Base command

```bash
phoscrosstalk --config config.toml
```

## Minimal `config.toml`

```toml
[paths]
data      = "data_timeseries/input1.csv"
ptm_intra = "data_curated/processed/ptm_intra.db"
ptm_inter = "data_curated/processed/ptm_inter.db"
output_dir = "results/experiment_01"

# kinase-site prior (at least one of these is recommended)
kinase_tsv = "data_interactions/kinase_sites.tsv"
# kea_ks_table = "data_curated/processed/ks_psite_table.tsv"

# optional RNA / TF-network inputs
# rna_data = "data_timeseries/input3.csv"
# tf_net   = "data_interactions/tf_mrna.csv"

[model]
mechanism = "dist"   # dist | seq | rand

[optimisation]
n_starts  = 3
max_steps = 20000
```

## Mechanism variants

Create one config per mechanism and change the `mechanism` field:

```toml
# Distributive (default)
[model]
mechanism = "dist"

# Sequential
# mechanism = "seq"

# Random / cooperative
# mechanism = "rand"
```

Then run:

```bash
phoscrosstalk --config config.toml
```

## With downstream analyses

Set flags in `config.toml`:

```toml
[analysis]
run_steadystate = true
run_knockouts   = true
run_sensitivity = true
```

## With KEA kinase-substrate table

```toml
[paths]
kea_ks_table = "data_curated/processed/ks_psite_table.tsv"
# kinase_tsv = ""   # omit or leave empty to use kea_ks_table instead
```

## Smoke test

```toml
[optimisation]
n_starts  = 1
max_steps = 50
```

```bash
phoscrosstalk --config smoke_config.toml
```

## Extended TF-as-proteins mode

```toml
[paths]
data     = "data_timeseries/input1.csv"   # full (unfiltered) time-series
rna_data = "data_timeseries/input3.csv"
tf_net   = "data_interactions/tf_mrna.csv"

[model]
include_tfs_as_proteins = true
```

!!! warning
    Pass **full (unfiltered)** time-series files when `include_tfs_as_proteins = true`.
    Files with `filtered` in their name may have had TF proteins removed before modeling.

## Dashboard

```bash
streamlit run phoscrosstalk/app.py
```

Point the dashboard at an existing results directory to explore fitted
trajectories, parameter distributions, and kinase activity.

## Help

```bash
phoscrosstalk --help
phoscrosstalk --version
```
