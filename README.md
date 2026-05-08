<p align="center">
  <img src="docs/assets/logo.png" width="300" alt="PhosCrosstalk logo">
</p>

# PhosCrosstalk

Global phospho-network ODE modeling with PTM crosstalk, kinase-site priors, and TF/mRNA integration — for computational
biologists inferring time-series mrna + phosphoproteomics data.

## Installation

After cloning in your favourite IDE, and installing `uv` on your machine:

```bash
uv sync --all-extras
```

## Quick start

### 1. Edit config.toml with your data paths and settings.

#### Minimal config.toml:

```

```toml
[paths]
data       = "data_timeseries/input1.csv"
ptm_intra  = "data_curated/processed/ptm_intra.db"
ptm_inter  = "data_curated/processed/ptm_inter.db"
output_dir = "results/run01"

[model]
mechanism = "dist"

[optimisation]
n_starts  = 3
max_steps = 500
```

### 2. Run the pipeline:

```bash
phoscrosstalk
```

### 3. Run the dashboard:

```bash
phoscrosstalk-app
```

The dashboard has two modes, selectable in the sidebar:

#### ⚙️ Configure & Run mode

Use this mode to:

1. **Edit config options** in the Config Editor tab (all `config.toml` sections exposed as widgets).
2. **Save config.toml** to a file path of your choice.
3. **Launch the pipeline** from the Run Pipeline tab — shows a live shell console.
4. **Stop a running pipeline** with the ⏹ Stop button.
5. **Browse run history** in the Run History tab (configs, logs, and result links).

Run artefacts are stored under `runs/<timestamp>/`:

| File | Description |
|------|-------------|
| `config.toml` | Config used for this run |
| `phoscrosstalk.log` | Full stdout + stderr |
| `run_metadata.json` | Status, timing, command |

#### 🔬 Explore Results mode

Use this mode to inspect a completed run directory (existing behaviour):

1. Enter the path to a completed run's output directory in the sidebar.
2. Browse tabs: Fit Explorer, Internal States, Derived Rates, Forward Simulation, Live Knockout, Sensitivity, Steady-State, Network, Neural Refined.

## Documentation

- [Model overview](docs/model.md)
- [Configuration reference](docs/configuration.md)
- [Input data format](docs/input_data.md)
- [Outputs](docs/outputs.md)
- [Running the pipeline](docs/running.md)
- [Dashboard](docs/dashboard.md)
- [Developer guide](docs/developer.md)
- [Troubleshooting](docs/troubleshooting.md)

## License

BSD 3-Clause — see [LICENSE](LICENSE).
