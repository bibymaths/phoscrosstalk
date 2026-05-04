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
p(t)    – phosphosite occupancy               shape (N,)
```

where **K** = number of model proteins, **M** = number of kinases,
**N** = number of phosphosites.

### Derived rates

Two rates are derived from data and **not** optimized:

- `k_act(t)` – protein activation rate, derived from the TF→mRNA regulatory signal
- `s_prod(t)` – kinase production rate, derived from kinase/protein signals

If `--rna-data` and `--tf-net` are absent, `k_act` defaults to a constant vector of 1.0.

### Single scalar objective

```text
total_loss =
    w_phospho   × phosphosite_loss
  + w_abundance × protein_abundance_loss
  + w_rna       × rna_loss          (only when --rna-data is provided)
  + w_reg       × regularization_loss
```

Weights are set in `config.toml` under `[loss_weights]`.

---

## Inputs

| Flag | Description | Example |
|------|-------------|---------|
| `--data` | Protein/phosphosite time-series CSV | `data_timeseries/filtered_input1.csv` |
| `--rna-data` | mRNA time-series CSV (GeneID, x1…x9) | `data_timeseries/filtered_input3.csv` |
| `--tf-net` | TF→mRNA network CSV (Source, Target, Weight) | `data_interactions/tf_mrna.csv` |
| `--ptm-intra` | PTMcode2 intra-protein SQLite DB | `data_curated/processed/ptm_intra.db` |
| `--ptm-inter` | PTMcode2 inter-protein SQLite DB | `data_curated/processed/ptm_inter.db` |
| `--kinase-tsv` | Kinase-site prior TSV (Site, Kinase, weight) | `data_interactions/kinase_sites.tsv` |
| `--kea-ks-table` | Alternative KEA/KS table | `data_curated/processed/ks_psite_table.tsv` |
| `--unified-graph-pkl` | Kinase-kinase graph for Laplacian regularization | `data_curated/processed/unified_kinase_graph.gpickle` |

### Input sanity checks

```bash
head data_timeseries/filtered_input1.csv
head data_timeseries/filtered_input3.csv
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

Copy and edit `config.toml`:

```toml
[paths]
data_dir   = "data_timeseries"
output_dir = "results"

[model]
mechanism    = "dist"      # dist | seq | rand
scale_mode   = "none"
length_scale = 50.0
weight_scheme = "uniform"

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
```

---

## Running PhosCrosstalk

### Base command

```bash
phoscrosstalk \
  --config config.toml \
  --data data_timeseries/filtered_input1.csv \
  --rna-data data_timeseries/filtered_input3.csv \
  --tf-net data_interactions/tf_mrna.csv \
  --ptm-intra data_curated/processed/ptm_intra.db \
  --ptm-inter data_curated/processed/ptm_inter.db \
  --kinase-tsv data_interactions/kinase_sites.tsv \
  --unified-graph-pkl data_curated/processed/unified_kinase_graph.gpickle \
  --outdir results/experiment_01 \
  --mechanism dist \
  --n-starts 3 \
  --max-steps 20000
```

### Mechanism variants

```bash
# Distributive (default)
phoscrosstalk ... --mechanism dist --outdir results/dist_run

# Sequential
phoscrosstalk ... --mechanism seq --outdir results/seq_run

# Random / cooperative
phoscrosstalk ... --mechanism rand --outdir results/rand_run
```

### With downstream analyses

```bash
phoscrosstalk ... \
  --run-steadystate \
  --run-knockouts \
  --run-sensitivity
```

### With KEA kinase-substrate table

```bash
phoscrosstalk ... \
  --kea-ks-table data_curated/processed/ks_psite_table.tsv
```

### Smoke test

```bash
phoscrosstalk ... --n-starts 1 --max-steps 50 --outdir results/smoke
```

### Dashboard

```bash
streamlit run phoscrosstalk/app.py
```

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
> not a copy of the input mRNA data. It is only written when `--rna-data` is
> provided and at least one gene symbol matches a model protein.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|-------------|
| Warning: No RNA genes matched | Gene symbols differ between `--rna-data` and `--data` |
| `mrna_fit_timeseries.tsv` absent | RNA data not provided, or no symbol match |
| Loss is `NaN` | Solver non-finite; check `rtol`/`atol`, try smoke test |
| `IndexError` decoding parameters | Stale `fitted_params.npz` from a run with different dimensions |
| Wrong TF direction | `Source` and `Target` columns swapped in `tf_mrna.csv` |
| Empty kinase-site matrix | Site labels in `--kinase-tsv` do not match model site labels |

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
