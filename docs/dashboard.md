# Dashboard

PhosCrosstalk includes an interactive Streamlit dashboard for exploring
optimization results without rerunning the full pipeline.

## Launch

```bash
streamlit run phoscrosstalk/app.py
```

The dashboard starts at `http://localhost:8501` by default.

## Features

The dashboard loads results from a completed optimization run directory and
provides:

- **Trajectory plots** – simulated vs observed phosphosite, protein abundance,
  and (if available) mRNA time courses
- **Kinase activity** – `Kdyn(t)` time courses per kinase
- **Protein activity** – `S(t)` time courses per protein
- **Network visualization** – interactive kinase-substrate graph via `gravis`
- **Parameter explorer** – fitted parameter distributions across multi-start solutions
- **Knockout comparison** – if `knockouts/` directory is present

## Requirements

The dashboard reads the following files from the results directory:

| File                    | Required |
|-------------------------|----------|
| `fitted_params.npz`     | Yes      |
| `preopt_snapshot/`      | Yes      |
| `fit_timeseries.tsv`    | Yes      |
| `mrna_fit_timeseries.tsv` | No (optional) |
| `knockouts/`            | No (optional) |

!!! warning "Missing files"
    The dashboard will display a warning and skip panels for which required
    files are absent. It will not crash when optional outputs (e.g. mRNA fit)
    are missing.

## Configuration

Point the dashboard at a different results directory by editing the path in the
sidebar or passing it as a query parameter (if implemented).
