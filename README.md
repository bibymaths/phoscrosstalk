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
