<p align="center">
  <img src="docs/assets/logo.png" width="300" alt="PhosCrosstalk logo">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/license-BSD--3--Clause-green" alt="BSD 3-Clause license">
  <img src="https://img.shields.io/badge/backend-JAX%2FDiffrax-blueviolet" alt="JAX/Diffrax">
  <img src="https://img.shields.io/badge/status-work%20in%20progress-orange" alt="Work in Progress">
</p>

# PhosCrosstalk

**Global phospho-network ODE modeling with PTM crosstalk, kinase-site priors, and TF/mRNA integration.**

PhosCrosstalk reconstructs protein activation, kinase activity, and phosphosite kinetics
across a signalling network. It fits large parameter sets to time-series phosphoproteomics
data using gradient-based multi-start optimisation via [JAX](https://github.com/google/jax),
[Diffrax](https://github.com/patrick-kidger/diffrax), and
[Optimistix](https://github.com/patrick-kidger/optimistix).

---

## What PhosCrosstalk does

- Models global phospho-network dynamics with five coupled ODE state variables
- Integrates PTMcode2 intra/inter-protein crosstalk priors
- Uses kinase-site interaction priors (KEA/KS or custom TSV)
- Incorporates a TF→mRNA regulatory network to drive protein activation
- Fits a single scalar loss to phosphosite, protein abundance, and mRNA time-series
- Supports distributive, sequential, and random/cooperative kinase mechanisms
- Offers downstream analyses: steady-state, in-silico knockouts, global sensitivity
- Provides an interactive Streamlit dashboard for result exploration

---

## Model overview

### State variables

The ODE state vector is:

```text
y = [R, S, A, Kdyn, p]    (dimension: 3K + M + N)

R(t)    – mRNA level (ODE state)              shape (K,)
S(t)    – protein activation / signalling     shape (K,)
A(t)    – protein abundance                   shape (K,)
Kdyn(t) – kinase activity                     shape (M,)
p(t)    – relative phosphosite signal      shape (N,)
```

where **K** = number of model proteins, **M** = number of kinases,
**N** = number of phosphosites.

> **Phosphosite state interpretation:** Phosphosite state `p` is modeled as a
> nonnegative relative signal, not a fractional occupancy.  Occupancy-like
> regulation uses `q = p/(1+p)`.  This allows fitting fold-change or
> relative-intensity phosphoproteomics values above 1 while retaining bounded
> regulatory feedback.

### Derived rates

Two rates are derived from data and **not** optimized:

- `k_act(t)` – protein activation rate, derived from the TF→mRNA regulatory signal
- `s_prod(t)` – kinase production rate, derived from kinase/protein signals

If `rna_data` and `tf_net` are absent from `config.toml`, `k_act` defaults to a constant vector of 1.0.

### Single scalar objective

```text
total_loss =
    w_phospho   × phosphosite_loss
  + w_abundance × protein_abundance_loss
  + w_rna       × rna_loss          (only when rna_data is set in config.toml)
  + w_reg       × regularization_loss
```

Weights are set in `config.toml` under `[loss_weights]`.

---

## Inputs

All input paths are configured in `config.toml` under `[paths]`:

| `config.toml` key | Description | Example |
|-------------------|-------------|---------|
| `data` | Protein/phosphosite time-series CSV | `data_timeseries/input1.csv` |
| `rna_data` | mRNA time-series CSV (GeneID, x1…x9) | `data_timeseries/input3.csv` |
| `tf_net` | TF→mRNA network CSV (Source, Target, Weight) | `data_interactions/tf_mrna.csv` |
| `ptm_intra` | PTMcode2 intra-protein SQLite DB | `data_curated/processed/ptm_intra.db` |
| `ptm_inter` | PTMcode2 inter-protein SQLite DB | `data_curated/processed/ptm_inter.db` |
| `kinase_tsv` | Kinase-site prior TSV (Site, Kinase, weight) | `data_interactions/kinase_sites.tsv` |
| `kea_ks_table` | Alternative KEA/KS table | `data_curated/processed/ks_psite_table.tsv` |
| `unified_graph_pkl` | Kinase-kinase graph for Laplacian regularization | `data_curated/processed/unified_kinase_graph.gpickle` |

### Input sanity checks

```bash
head data_timeseries/input1.csv
head data_timeseries/input3.csv
head data_interactions/tf_mrna.csv
head data_interactions/kinase_sites.tsv
```

Check TF network symbols against RNA data:

```bash
awk -F',' '
  NR==FNR { if (FNR > 1) rna[$1]; next }
  FNR > 1 {
    if (!($1 in rna)) missing[$1]
    if (!($2 in rna)) missing[$2]
  }
  END { for (g in missing) print g }
' data_timeseries/filtered_input3.csv data_interactions/tf_mrna.csv | sort
```

---

## Installation

```bash
git clone https://github.com/bibymaths/phoscrosstalk.git
cd phoscrosstalk
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

---

## Configuration

All runtime options live in `config.toml`.
Copy the template and edit it for your dataset:

```toml
[paths]
data      = "data_timeseries/input1.csv"
ptm_intra = "data_curated/processed/ptm_intra.db"
ptm_inter = "data_curated/processed/ptm_inter.db"
output_dir = "results/experiment_01"
kinase_tsv = "data_interactions/kinase_sites.tsv"

# optional
# rna_data = "data_timeseries/input3.csv"
# tf_net   = "data_interactions/tf_mrna.csv"

[model]
mechanism = "dist"   # dist | seq | rand

[optimisation]
n_starts  = 3
max_steps = 500
lambda_net = 0.0001
reg_lambda = 0.0001

[loss_weights]
phospho   = 1.0
abundance = 1.0
mrna      = 1.0
reg       = 1.0

[solver]
rtol = 1e-6
atol = 1e-9
max_steps = 16384

[time]
mrna_time_points = [4, 8, 15, 30, 60, 120, 240, 480, 960]
interpolation    = "piecewise_constant"

[analysis]
tune            = false
run_steadystate = false
run_knockouts   = false
run_sensitivity = false
```

---

## Running PhosCrosstalk

### Recommended: run from repo root with `uv`

```bash
# Sync dependencies
uv sync

# Run the pipeline with a config file
uv run phoscrosstalk --config config.toml

# Launch the Streamlit dashboard
uv run streamlit run phoscrosstalk/app.py
```

### Base command (installed environment)

```bash
phoscrosstalk --config config.toml
```

The CLI only accepts `--config <path>` (plus `--help` and `--version`).
All data paths, model settings, optimisation parameters, and analysis flags
live in `config.toml`.

### Smoke test

```toml
# smoke_config.toml
[optimisation]
n_starts  = 1
max_steps = 50
```

```bash
uv run phoscrosstalk --config smoke_config.toml
```

### Dashboard

```bash
uv run streamlit run phoscrosstalk/app.py
```

The dashboard asks for a results directory path in its sidebar.
Enter the path to a completed run directory (e.g. `test_results_dist`).

> **Note:** The dashboard does not accept a `--results-dir` CLI flag.
> The results directory is configured interactively via the sidebar text input.

---

## Outputs

| File | Description |
|------|-------------|
| `fitted_params.npz` | Optimized parameter vector and decoded arrays |
| `fit_timeseries.tsv` | Simulated vs observed phosphosite and protein time-series |
| `mrna_fit_timeseries.tsv` | Simulated `R(t)` vs observed mRNA (only when RNA ODE state is active) |
| `mrna_diagnostics.tsv` | Per-gene mRNA fit diagnostics |
| `internal_states.tsv` | Simulated `S(t)` and `Kdyn(t)` |
| `parameter_summary_*.tsv` | Per-protein, per-kinase, per-site parameters |
| `pareto_front.npz` | All multi-start losses and parameters |
| `preopt_snapshot/` | Full input snapshot before optimization |
| `network_nodes.tsv`, `network_edges.tsv` | Cytoscape-compatible network export |
| `equations/` | ODE equation reports |
| `knockouts/` | Knockout screening results |
| `sensitivity/` | Sobol sensitivity indices |
| `steadystate/` | Steady-state simulation results |

> **Note:** `mrna_fit_timeseries.tsv` contains model-simulated `R(t)` values,
> not a copy of the input mRNA data. It is only written when `rna_data` is
> set in `config.toml` and at least one gene symbol matches a model protein.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|-------------|
| Warning: No RNA genes matched | Gene symbols differ between `rna_data` and `data` files |
| `mrna_fit_timeseries.tsv` absent | RNA data not provided, or no symbol match |
| Loss is `NaN` | Solver non-finite; check `rtol`/`atol`, try smoke test |
| `IndexError` decoding parameters | Stale `fitted_params.npz` from a run with different dimensions |
| Wrong TF direction | `Source` and `Target` columns swapped in the TF network file |
| Empty kinase-site matrix | Site labels in `kinase_tsv` do not match model site labels |
| Validation error at startup | Required `[paths]` fields missing or files not found in `config.toml` |

---

## Developer workflow

```bash
pytest
pytest --cov=phoscrosstalk --cov-report=term-missing
ruff check .
ruff format --check .
mkdocs serve
mkdocs build --strict
```

---

## License

BSD 3-Clause — see [LICENSE](LICENSE).
