The pre\-optimization snapshot is written by `phoscrosstalk.analysis._save_preopt_snapshot_txt_csv(...)` and is created
in the folder `args.outdir/preopt_snapshot/`. It saves **metadata**, **labels**, and **all intermediate numeric
arrays/matrices** that exist right before optimization starts.

## 1\) Folder and file types

Everything goes under `preopt_snapshot/` and is saved as:

- `\*.txt` for text metadata / labels
- `\*.tsv` (tab\-separated) for numeric vectors/matrices

No `\*.npz` is used.

## 2\) Metadata files (human readable)

### `meta.txt`

A plain text summary that records:

- **Counts**
    - `n_sites` \= number of phosphosites being fitted
    - `n_proteins` \= number of proteins
    - `n_kinases` \= number of kinases
- **Key configuration copied from CLI args**
    - `scale_mode` (e.g. `minmax`, `none`, `log-minmax`)
    - `mechanism` (e.g. `dist`, `seq`, `rand`)
    - `weight_scheme` (e.g. `uniform`, etc.)
    - `length_scale` (for local PTM decay)
    - `lambda_net` (network/Laplacian regularization strength)
    - `reg_lambda` (L2 regularization strength)
- **Shapes section**
    - Prints the shape of every saved array (so you can quickly see what sizes were actually constructed), including
      `P_scaled`, `Y`, `A_scaled`, `Cg`, `Cl`, `K_site_kin`, `R`, `L_alpha`, weight matrices, masks, and bounds.

This file is meant to let you confirm “what exactly went into the optimizer” without loading any matrices.

### `sites.txt`

One site identifier per line, in the exact order used internally (this order matters because it matches the rows of `Y`,
`P_scaled`, `Cg`, `Cl`, etc.).

### `proteins.txt`

One protein name per line, in the exact order used internally (this order matches indices used by `site_prot_idx`,
masks, etc.).

### `kinases.txt`

One kinase name per line, order used internally (matches indices of `K_site_kin` columns / `R` rows / `L_alpha` axes).

### `A_proteins.txt`

If protein\-level activity data exists, this lists the protein names corresponding to rows of `A_data` / `A_scaled`.
Otherwise it is empty.

## 3\) Numeric snapshot files (TSV)

All these are saved as tab\-separated numeric tables.

### Time and site positions

- `t.tsv`  
  Vector of timepoints used for the time series.
- `positions.tsv`  
  Vector of site positions on the protein (used for local coupling / distance weighting).

### Raw and scaled data

- `Y.tsv`  
  The loaded experimental site time series (after any filtering). Shape is typically `(n_sites, n_timepoints)`.
- `P_scaled.tsv`  
  Scaled/normalized version of `Y` (after applying `scale_mode` and NaN/inf cleanup). Same shape as `Y`.
- `A_data.tsv`  
  Optional raw protein activity time series (if provided). If absent, saved as an empty `0 x 0` matrix.
- `A_scaled.tsv`  
  Scaled/normalized version of `A_data`. If no activity data, this is saved as a `0 x n_timepoints` matrix (as
  constructed in `main.py`).

### Weight matrices (used in objective calculations)

- `W_data.tsv`  
  Weights applied to site time series points (same shape as `Y`/`P_scaled`).
- `W_data_prot.tsv`  
  Weights applied to protein activity time series points (shape matches `A_data`/`A_scaled` when present).

### PTM coupling matrices (graphs encoded as matrices)

- `Cg.tsv`  
  Global coupling matrix built from PTM DB relationships, then row\-normalized. Encodes cross\-influences between
  sites (shape `(n_sites, n_sites)`).
- `Cl.tsv`  
  Local coupling matrix (distance/length scale based) built from PTM DB + positions, then row\-normalized (shape
  `(n_sites, n_sites)`).

### Kinase/site mapping and derived matrices

- `K_site_kin.tsv`  
  Site \(\rightarrow\) kinase mapping matrix (shape `(n_sites, n_kinases)`).
- `R.tsv`  
  Saved as `K_site_kin.T` (shape `(n_kinases, n_sites)`), used by the model internally.

### Regularization structure

- `L_alpha.tsv`  
  Kinase\-kinase Laplacian matrix used for α regularization (shape `(n_kinases, n_kinases)`). If no unified graph or
  `lambda_net` \<= 0, it will be all zeros.

### Index mappings and masks (integers saved as TSV)

- `site_prot_idx.tsv`  
  Integer vector of length `n_sites` mapping each site row to a protein index (0\-based), aligned to `proteins.txt`.
- `kin_to_prot_idx.tsv`  
  Integer vector of length `n_kinases` mapping each kinase to a protein index, or `-1` if not found in `proteins`.
- `receptor_mask_prot.tsv`  
  Integer vector (0/1) marking which proteins are considered receptors (aligned to `proteins.txt`).
- `receptor_mask_kin.tsv`  
  Integer vector (0/1) marking which kinases are considered receptors (aligned to `kinases.txt`).

### Optimization bounds

- `xl.tsv`  
  Lower bound vector for the optimization decision variables.
- `xu.tsv`  
  Upper bound vector for the optimization decision variables.

## 4\) What this snapshot represents

It captures the **exact inputs to `NetworkOptimizationProblem(...)`** (data, scaling, weights, graph/matrix structures,
mappings/masks, and bounds) right before optimization begins, so runs are reproducible/debuggable even if the upstream
databases/files change later.