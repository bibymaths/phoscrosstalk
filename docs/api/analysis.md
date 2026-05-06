# Analysis

## Biological Role

The `analysis` module exports all post-optimisation results: fitted parameter tables, simulated vs. observed trajectory files, internal ODE state traces, mRNA outputs, and diagnostic plots. It is the final pipeline stage and writes all artefacts to the configured output directory.

## Implementation Overview

The main entry point is `save_run_results`, which orchestrates the full export pipeline. It calls `save_fitted_simulation`, `plot_run_diagnostics`, `print_parameter_summary`, and `save_mrna_outputs` in sequence.

`_save_dense_simulation` is called when `[data_interpolation] enabled = true`. It integrates at a fine time grid and writes long-format TSV files with columns `entity_type, entity, site, protein, time, value, series_type, source, interpolation_method`.

**Output files written to `[paths] output_dir`:**

| File | Content |
|------|---------|
| `pareto_stats.tsv` | Objective statistics across multi-start runs |
| `pareto_front_with_J.tsv` | All solutions with scalarised J value |
| `pareto_points.tsv` | All solutions with `bio_score` |
| `pareto_front.npz` | Binary archive: `F`, `X`, `J` arrays |
| `fitted_params.npz` | Decoded theta + protein/site arrays |
| `fit_timeseries.tsv` | Wide-format sim vs data (`sim_t0`, `data_t0`, …) |
| `internal_states.tsv` | `S_sim` (protein activity) + `Kdyn_sim` (kinase activity) |
| `fit_timeseries_dense.tsv` | Long-format dense output (only when `data_interpolation.enabled`) |
| `mrna_fit_timeseries.tsv` | mRNA simulation output (skipped with warning if no RNA data) |
| `mrna_fit_timeseries_dense.tsv` | Dense mRNA output |

**`fit_timeseries_dense.tsv` column schema:**

| Column | Description |
|--------|-------------|
| `entity_type` | `"phosphosite"`, `"protein"`, or `"mrna"` |
| `entity` | Entity identifier string |
| `site` | Site name (empty for protein/mRNA rows) |
| `protein` | Parent protein name |
| `time` | Time point (float) |
| `value` | Simulated or interpolated observed value |
| `series_type` | `"simulated"` or `"observed"` |
| `source` | Data source label |
| `interpolation_method` | Method used: `"linear"`, `"cubic_hermite"`, or `"none"` |

## Configuration Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `[simulation] t_dense_n` | `int` | `200` | Number of dense time points for `simulate_dense` |
| `[simulation] t_dense_min` | `float` | `0.0` | Start of dense time grid |
| `[simulation] t_dense_max` | `float` | `120.0` | End of dense time grid |
| `[data_interpolation] enabled` | `bool` | `false` | Write `fit_timeseries_dense.tsv` |
| `[data_interpolation] method` | `str` | `"linear"` | Interpolation method for observed data in dense output |
| `[analysis] plot_format` | `str` | `"png"` | Plot output format |
| `[analysis] dpi` | `int` | `150` | Plot resolution |
| `[analysis] plot_top_n` | `int` | `5` | Number of top solutions to plot |
| `[paths] output_dir` | `str` | `"results"` | Directory for all output files |

## API Reference

```python
def save_run_results(
    pareto_F, pareto_X, pareto_J,
    best_theta,
    decoded_params: dict,
    simulate_fn,
    args_static: tuple,
    t_eval: np.ndarray,
    P_data, A_data, rna_data,
    prot_names, site_names, kin_names,
    cfg,
    output_dir: str,
) -> None:
    """Orchestrate full post-optimisation export."""

def save_fitted_simulation(
    best_theta, simulate_fn, args_static,
    t_eval, P_data, A_data,
    prot_names, site_names,
    output_dir: str,
    cfg=None,
) -> None:
    """Write fit_timeseries.tsv and optionally fit_timeseries_dense.tsv."""

def plot_fitted_simulation(
    fit_df: pd.DataFrame,
    output_dir: str,
    cfg=None,
) -> None:
    """Plot simulated vs observed phosphosite trajectories."""

def plot_internal_states(
    best_theta, simulate_fn, args_static,
    t_eval, kin_names, prot_names,
    output_dir: str,
) -> None:
    """Write internal_states.tsv and generate kinase/protein activity plots."""

def plot_run_diagnostics(
    pareto_F, pareto_J,
    output_dir: str,
    cfg=None,
) -> None:
    """Plot Pareto front, loss history, and bio-score diagnostics."""

def print_parameter_summary(
    best_theta,
    prot_names, kin_names, site_names,
    K: int, M: int, N: int,
) -> None:
    """Print decoded parameter table to stdout."""

def print_biological_scores(
    pareto_X,
    K: int, M: int,
) -> None:
    """Print bio_score statistics for the Pareto set to stdout."""

def save_mrna_outputs(
    best_theta, simulate_fn, args_static,
    t_eval, rna_data, prot_names,
    output_dir: str,
    cfg=None,
) -> None:
    """
    Write mrna_fit_timeseries.tsv.
    Skips with a warning if rna_simulated is None.
    """
```

## Known Limitations

- `_save_dense_simulation` calls `simulate_dense` which is not JIT-compiled; dense output for large models is slow.
- `save_mrna_outputs` writes a stub file with a warning message when `rna_simulated=None`; downstream tools should check for the warning flag.
- Plot functions use Matplotlib and are not reproducible across different Matplotlib versions if default rcParams differ.
