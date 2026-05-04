# Running PhosCrosstalk

## Base command

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

## Mechanism variants

=== "Distributive"

    ```bash
    phoscrosstalk \
      --config config.toml \
      --data data_timeseries/filtered_input1.csv \
      --ptm-intra data_curated/processed/ptm_intra.db \
      --ptm-inter data_curated/processed/ptm_inter.db \
      --kinase-tsv data_interactions/kinase_sites.tsv \
      --outdir results/dist_run \
      --mechanism dist \
      --n-starts 3 \
      --max-steps 20000
    ```

=== "Sequential"

    ```bash
    phoscrosstalk \
      --config config.toml \
      --data data_timeseries/filtered_input1.csv \
      --ptm-intra data_curated/processed/ptm_intra.db \
      --ptm-inter data_curated/processed/ptm_inter.db \
      --kinase-tsv data_interactions/kinase_sites.tsv \
      --outdir results/seq_run \
      --mechanism seq \
      --n-starts 3 \
      --max-steps 20000
    ```

=== "Random/Cooperative"

    ```bash
    phoscrosstalk \
      --config config.toml \
      --data data_timeseries/filtered_input1.csv \
      --ptm-intra data_curated/processed/ptm_intra.db \
      --ptm-inter data_curated/processed/ptm_inter.db \
      --kinase-tsv data_interactions/kinase_sites.tsv \
      --outdir results/rand_run \
      --mechanism rand \
      --n-starts 3 \
      --max-steps 20000
    ```

## With downstream analyses

```bash
phoscrosstalk \
  --config config.toml \
  --data data_timeseries/filtered_input1.csv \
  --ptm-intra data_curated/processed/ptm_intra.db \
  --ptm-inter data_curated/processed/ptm_inter.db \
  --kinase-tsv data_interactions/kinase_sites.tsv \
  --outdir results/full_run \
  --mechanism dist \
  --n-starts 3 \
  --max-steps 20000 \
  --run-steadystate \
  --run-knockouts \
  --run-sensitivity
```

## With KEA kinase-substrate table

Use `--kea-ks-table` instead of `--kinase-tsv` when using KEA3 data:

```bash
phoscrosstalk \
  --config config.toml \
  --data data_timeseries/filtered_input1.csv \
  --ptm-intra data_curated/processed/ptm_intra.db \
  --ptm-inter data_curated/processed/ptm_inter.db \
  --kea-ks-table data_curated/processed/ks_psite_table.tsv \
  --outdir results/kea_run \
  --mechanism dist
```

## Smoke test

!!! tip "Smoke test"
    Run with 1 start and 50 steps to verify the pipeline runs end to end:

    ```bash
    phoscrosstalk \
      --config config.toml \
      --data data_timeseries/filtered_input1.csv \
      --ptm-intra data_curated/processed/ptm_intra.db \
      --ptm-inter data_curated/processed/ptm_inter.db \
      --kinase-tsv data_interactions/kinase_sites.tsv \
      --outdir results/smoke \
      --mechanism dist \
      --n-starts 1 \
      --max-steps 50
    ```

## Dashboard

```bash
streamlit run phoscrosstalk/app.py
```

Point the dashboard at an existing results directory to explore fitted
trajectories, parameter distributions, and kinase activity.

## Help

```bash
phoscrosstalk --help
```
