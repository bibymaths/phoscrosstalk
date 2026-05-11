# PhosCrosstalk — Educational Notebook Suite

This directory contains sixteen Jupyter notebooks that walk through the PhosCrosstalk
pipeline step by step, from raw data loading to parameter optimisation, neural ODE
refinement, PINN mode, uncertainty quantification, and sensitivity analysis.

## Setup

Generate the sample data before running any notebook:

```bash
uv run python scripts/create_sample_data.py
```

Then launch JupyterLab (or Jupyter Notebook) from the **repository root**:

```bash
uv run jupyter lab
```

All notebooks resolve paths relative to the repository root, so run them from
`notebooks/` **or** from the root — both work via the `PROJECT_ROOT` detection
pattern at the top of each notebook.

## Notebooks

| # | File | Description |
|---|------|-------------|
| 00 | `00_framework_overview_and_data_flow.ipynb` | Big-picture architecture, data-flow diagram, sample-data shapes, and module summary |
| 01 | `01_config_runtime_and_time_axes.ipynb` | `config.toml` sections, `load_config()`, time axes, interpolation modes, and bounds |
| 02 | `02_input_data_structures_and_loading.ipynb` | All four input files, `load_site_data()` / `load_rna_data()` return values, and raw trajectory plots |
| 03 | `03_scaling_weights_and_observation_matrices.ipynb` | Three scaling modes (`none`/`minmax`/`log-minmax`), `build_weight_matrices()`, and weight heatmaps |
| 04 | `04_network_priors_crosstalk_and_entity_masks.ipynb` | Cg/Cl crosstalk matrices, `K_site_kin`, R matrix, kinase Laplacian, TF weights, and site–protein masks |
| 05 | `05_model_dimensions_bounds_and_theta_layout.ipynb` | `ModelDims`, `create_bounds()`, theta vector layout, `decode_theta()`, and gamma encoding |
| 06 | `06_derived_rates_kact_sprod.ipynb` | `k_act(t)` and `s_prod(t)` closures, softplus transfer function, interpolation modes |
| 07 | `07_ode_state_layout_and_simulation.ipynb` | ODE state vector `y=[R,S,A,Kdyn,p]`, forward `simulate()`, mechanism variants |
| 08 | `08_residual_blocks_losses_and_objectives.ipynb` | Residual vector blocks, loss components f1–f4, weight matrices, loss-type transfer functions |
| 09 | `09_optimizer_backends_and_multistart_outputs.ipynb` | Multi-start strategy, backend options, `NetworkProblem`, selection by minimum total loss |
| 10 | `10_post_fit_outputs_and_visualisation.ipynb` | Output file formats, `decode_theta()` parameter table, fit-timeseries plots, bio_score |
| 11 | `11_uncertainty_bootstrap_and_profile_likelihood.ipynb` | Residual bootstrap CIs, profile likelihood shapes, identifiability interpretation |
| 12 | `12_neuralode_latent_rate_refinement.ipynb` | `NeuralRateGenerator` MLP, frozen-theta vs joint training, neural bundle format |
| 13 | `13_pinn_universal_ode_mode.ipynb` | PINN / Universal ODE mode, physics + data loss, PINN bundle format |
| 14 | `14_sensitivity_steadystate_and_knockouts.ipynb` | Global sensitivity heatmap, steady-state long-horizon simulation, in-silico knockouts |
| 15 | `15_minimal_end_to_end_pipeline_for_user_data.ipynb` | Step-by-step guide for new user data: load → scale → build → simulate → troubleshoot |

## Output files

Each notebook saves figures to `notebooks/_outputs/` (created automatically).

## Key concepts

- **`ModelDims(K, M, N)`** — K proteins, M kinases, N phosphosites
- **theta** `∈ R^(2K+2+3M+N+4)` — the flat optimisation parameter vector
- **ODE state** `y = [R_rna(K), S(K), A(K), Kdyn(M), p(N)]` — total dim `3K+M+N`
- All positive rate parameters are stored and optimised in **log-space**
