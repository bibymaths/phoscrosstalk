# Outputs

After a successful run, PhosCrosstalk writes all results to `--outdir`.

## Core output files

| File                        | Description                                         |
|-----------------------------|-----------------------------------------------------|
| `fitted_params.npz`         | Optimized parameter vector and decoded per-element arrays |
| `fit_timeseries.tsv`        | Simulated vs observed phosphosite and protein time-series |
| `internal_states.tsv`       | Simulated `S(t)` (protein activity) and `Kdyn(t)` (kinase activity) |
| `parameter_summary_proteins.tsv` | Per-protein parameters (`k_deact`, `d_deg`)   |
| `parameter_summary_kinases.tsv`  | Per-kinase parameters (`alpha`, `kK_act`, `kK_deact`) |
| `parameter_summary_sites.tsv`    | Per-site phosphatase rate (`k_off`)           |
| `parameter_summary_global.txt`   | Global coupling parameters (`beta_g`, `beta_l`) |
| `biological_scores.tsv`     | Biological plausibility score per multi-start solution |
| `run_config.json`           | Full configuration used for this run (reproducibility) |

## Optimization diagnostics

| File                      | Description                                             |
|---------------------------|---------------------------------------------------------|
| `pareto_stats.tsv`        | Min/mean/median/std of each loss component              |
| `pareto_front_with_J.tsv` | All solutions with scalar total loss `J`                |
| `pareto_points.tsv`       | All solutions with biological scores appended           |
| `pareto_front.npz`        | Raw arrays: `F` (loss components), `X` (parameters), `J` (total losses) |

## mRNA outputs (when `--rna-data` is provided and genes match model proteins)

| File                      | Description                                             |
|---------------------------|---------------------------------------------------------|
| `mrna_fit_timeseries.tsv` | Simulated `R(t)` vs observed mRNA per matched gene/time |
| `mrna_diagnostics.tsv`    | Per-gene R², MSE, MAE for mRNA fit                      |

!!! danger "RNA fit"
    `mrna_fit_timeseries.tsv` is only written when:

    1. `--rna-data` is provided
    2. At least one mRNA gene symbol matches a model protein
    3. The simulated `R(t)` from the ODE is available

    The file contains model-simulated `R(t)`, not a copy of the observed mRNA input.

## Pre-optimization snapshot

A `preopt_snapshot/` subdirectory is written before optimization starts. It
contains all input matrices, labels, and configuration for reproducibility:

| File                     | Description                         |
|--------------------------|-------------------------------------|
| `meta.txt`               | Dimensions and configuration summary |
| `sites.txt`              | Site labels                          |
| `proteins.txt`           | Protein labels                       |
| `kinases.txt`            | Kinase labels                        |
| `Y.tsv`, `P_scaled.tsv`  | Phosphosite data (raw and scaled)    |
| `A_data.tsv`, `A_scaled.tsv` | Protein abundance data            |
| `Cg.tsv`, `Cl.tsv`       | Crosstalk coupling matrices          |
| `K_site_kin.tsv`         | Kinase-site interaction matrix       |
| `xl.tsv`, `xu.tsv`       | Parameter lower/upper bounds         |

## Network export

| File               | Description                                           |
|--------------------|-------------------------------------------------------|
| `network_nodes.tsv`| Node table for Cytoscape (proteins, kinases, sites)   |
| `network_edges.tsv`| Edge table for Cytoscape (kinase→site, site→protein)  |

## Plots

| File                         | Description                                  |
|------------------------------|----------------------------------------------|
| `fit_<Protein>.png`          | Per-protein: mRNA / protein / phosphosite fit |
| `fitted_internal_states.png` | S(t) and Kdyn(t) dynamics                    |
| `goodness_of_fit.png`        | Global observed vs simulated scatter         |
| `pareto_f1_f2.png`           | f1 vs f2 with f3 color coding                |
| `pareto_param_corr.png`      | Parameter correlation heatmap                |
| `biological_scores.png`      | Biological score distribution                |

## Downstream analyses

| Directory        | Contents                                      |
|------------------|-----------------------------------------------|
| `steadystate/`   | Steady-state simulation results               |
| `knockouts/`     | Per-kinase knockout screening results         |
| `sensitivity/`   | Sobol sensitivity indices                     |
| `equations/`     | ODE equation reports for each protein/site    |

## Interpreting outputs

- **`p(t)` in `fit_timeseries.tsv`** = phosphosite occupancy fit. Compare `sim_t*` vs `data_t*` columns.
- **`A(t)` in `fit_timeseries.tsv`** = protein abundance fit.
- **`R(t)` in `mrna_fit_timeseries.tsv`** = mRNA level fit (only when RNA ODE state is active).
- **`S(t)` in `internal_states.tsv`** = protein activation state (not directly observed; derived from model).
- **`Kdyn(t)` in `internal_states.tsv`** = kinase activity (derived; not directly fitted to data).
- **`k_act(t)`, `s_prod(t)` in `derived_rates.npz`** = derived rates from data, not optimized parameters.
