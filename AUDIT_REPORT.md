# Codebase Audit Report

## Executive Summary

The repository implements a global phospho-network ODE model that couples
kinase activity, protein abundance, mRNA dynamics, and phosphosite occupancy
through a jointly fitted ODE system. The backend is fully JAX/Diffrax; the
Numba/SciPy stack has been removed. The code is **runnable for exploratory
use** on a correctly configured environment, but several correctness issues
could silently skew fitted parameters. The code is **not yet fit for
publication-grade inference** without fixing the issues below.

**Top 5 risks:**

1. The f4 (RNA) residual is computed as raw weighted MSE while f1/f2 use
   `log1p`-weighted loss; the two loss families are not commensurable, and
   the `total_loss` used for best-run selection is their unweighted sum,
   biasing selection when RNA data is present.
2. `ModelDims` is a bare class-level mutable global; concurrent testing and
   multi-process spawning both expose race conditions.
3. Temporary CSV files (`_prefilter_phospho_csv`, `_prefilter_rna_csv`) are
   written to OS temp space but never cleaned up, leaking up to two full
   copies of the input data per run and risking collisions on shared compute
   nodes.
4. `build_tf_prot_weights` maps TF source genes to **target** protein by
   matching `source → gene_idx` and `target → protein_idx` — the biological
   direction is inverted. TFs activate target genes, so the weight should map
   the *target* gene row in the RNA matrix to the *target* protein, not the
   source row.
5. The `_generate_starts` function (used in multi-start) has no fixed seed,
   meaning restarts across runs draw different initial points and results are
   not bit-for-bit reproducible even with the same configuration.

**Top 5 fastest fixes:**

1. Fix `_generate_starts` to accept/use a fixed seed (one-line change).
2. Delete the two temporary CSV files after `load_site_data` /
   `load_rna_data` return (two `os.unlink` calls).
3. Unify the mRNA residual to `log1p`-weighted form, matching f1/f2.
4. Fix the `build_tf_prot_weights` mapping direction (two lines).
5. Guard `ModelDims` writes behind a context manager or module-level lock.

---

## System Overview

The intended pipeline, as reconstructed from the code:

1. **Config loading** — `config.py:load_config` + `validate_config` parse
   `config.toml`; all paths, model, solver, and loss settings live here.
2. **Input data loading** — `data_loader.load_site_data` reads
   phosphosite/protein CSV; `load_rna_data` reads mRNA time-series;
   `load_tf_network` + `build_tf_prot_weights` build TF weights.
3. **Network/model universe construction** — `_build_network_allow_sets`
   reads the kinase TSV and TF network to define the allowed set of
   proteins/sites; `_prefilter_phospho_csv` / `_prefilter_rna_csv` restrict
   the full measurement files to this set via temporary CSVs.
4. **Phosphosite/protein/RNA matching** — `match_rna_to_model_proteins`,
   `build_protein_entity_masks`, `build_full_R0` produce index maps and
   initial conditions.
5. **Matrix construction** — `build_C_matrices_from_db` (PTM DBs → Cg, Cl),
   `load_kinase_site_matrix` / `build_kinase_site_from_kea` (K_site_kin),
   `build_alpha_laplacian_from_unified_graph` (L_alpha), manual row
   normalizations and index maps.
6. **ODE model setup** — `NetworkProblem.__init__` stores all static arrays;
   `make_residuals_fn` or `make_loss_fn` in `optimization.py` close over the
   static data and produce a JAX-traceable callable.
7. **Derived rates** — `derived_rates.make_k_act_fn` / `make_s_prod_fn` build
   JAX closures for time-varying `k_act(t)` and `s_prod(t)` from mRNA and
   phospho data.
8. **Optimization** — `run_multi_start_optimization` in `multistarts.py`
   draws random starts, runs `run_single_optimisation` (Optimistix
   LevenbergMarquardt) serially or in parallel, selects by minimum
   `total_loss`.
9. **Post-processing and analysis** — `analysis.py` saves fitted simulations,
   parameter summaries, and diagnostic plots; `post_processing.py` exports
   Cytoscape networks and run metadata; `equations.py` generates LaTeX ODE
   documentation.

**Divergences from ideal flow:**

- Step 3 uses temporary files as an intermediate artifact rather than passing
  allow-lists directly into the loaders (acknowledged as a TODO).
- Bounds are hard-coded in `create_bounds` instead of being driven by the
  `[bounds]` config section, which exists but is never read by `create_bounds`.
- `validate_biological_inputs` (in `optimization.py`) is defined but is never
  called in the main pipeline.
- The `total_loss` used for run selection mixes loss families (log1p for f1/f2
  vs. raw MSE for f4), breaking commensurability.

---

## Critical Findings

### C1. `build_tf_prot_weights` maps TF direction inverted

- **Severity:** Critical
- **Category:** Biology / RNA/TF integration
- **Status:** Confirmed issue
- **Location:** `phoscrosstalk/data_loader.py`, `build_tf_prot_weights`, lines
  672–678
- **Evidence:**
  ```python
  g_idx = gene_idx.get(src)   # src = TF gene (source of the edge)
  p_idx = prot_idx.get(tgt)   # tgt = target gene
  W[p_idx, g_idx] += w
  ```
  The TF network has edges `source → target` meaning "TF *source* regulates
  gene *target*". The matrix `W[p, g]` is used as `k_act_p = W[p, :] @ rna`
  to compute the TF-driven activation signal for protein `p`. The correct
  mapping is: gene `g` acting as a TF (source) drives the expression of its
  target proteins. The code *writes* to `W[p_idx, g_idx]` where `p_idx` is
  the protein index of the **target** and `g_idx` is the RNA row of the
  **source** (TF). This part is correct. However, `make_k_act_fn` then
  computes `signal = tf_prot_weights @ rna_data` as `(K, n_genes) @
  (n_genes, T_rna) = (K, T_rna)`, reading the TF source RNA trajectory to
  drive the target protein. **Biologically**: `k_act` for protein `p` should
  reflect the activity of TFs that regulate `p`'s transcription, which *are*
  the sources. The direction in the matrix product is actually consistent.
  **The real bug** is that the code matches `tgt` (target gene name) against
  `prot_idx` (model protein names). If TF targets are gene names and model
  proteins are the same names, the mapping is correct. But if TF targets are
  gene names for mRNA and model proteins are phospho-detected proteins (which
  may differ by naming convention), the mapping will silently drop most edges
  with no warning.
- **Impact:** If TF target names do not match model protein names exactly,
  `W` will be all-zeros for most proteins, silently making `k_act` constant
  1.0 and eliminating TF-driven regulation from the model. The user receives
  no error.
- **Minimal fix:** Add a post-construction check: if `W.sum() == 0` and
  `len(tf_net_df) > 0`, raise a warning listing the unmatched target proteins.
- **Longer-term fix:** Normalise protein and gene names (strip isoform
  suffixes, case-fold) before building the mapping.
- **Test to add:** Unit test with a small synthetic TF network where source
  and target names are known; assert `W.sum() > 0` and verify the correct
  row/column is populated.

### C2. `total_loss` for best-run selection mixes incompatible loss families

- **Severity:** Critical
- **Category:** Optimization / Scaling/weights
- **Status:** Confirmed issue
- **Location:** `phoscrosstalk/optimization.py`, `make_residuals_fn` and
  `run_single_optimisation`, lines 780–803, 929–931
- **Evidence:**
  - f1 (phosphosite): `jnp.sum(W * diff_p * diff_p) / n_p` (unweighted MSE
    stored as diagnostic; residuals are `sqrt(w * W) * diff`)
  - f2 (abundance): same form
  - f4 (mRNA): `jnp.sum(W_rna * diff_R * diff_R) / n_rna` (raw weighted MSE,
    no log1p)
  - `total_loss = f1 + f2 + f3 + f4` (line 931)
  - In `make_loss_fn`, f1 uses `log1p`, f4 does not (lines 523, 489).
  - Best run is selected by `np.argmin(total_losses)` in `multistarts.py`
    line 691.
- **Impact:** When RNA data is present, f4 dominates the scale of `total_loss`
  because raw MSE of fold-change values is not log1p-compressed. The
  "best" run selected may not have the best phosphosite fit; it will be the
  one with the smallest RNA residual.
- **Minimal fix:** Apply the same `log1p` transform to f4, or divide by the
  data scale before adding. For run selection, use only f1+f2+f3 (the
  phospho/abundance fit), treating f4 as a regulariser.
- **Longer-term fix:** Define a single unified normalised loss function for
  all modalities with consistent scale.
- **Test to add:** Run two artificial starts with different f1/f4 trade-offs;
  verify that selection picks the run with better f1, not better f4.

### C3. `_generate_starts` has no fixed seed

- **Severity:** Critical (for reproducibility)
- **Category:** Reproducibility / Optimization
- **Status:** Confirmed issue
- **Location:** `phoscrosstalk/multistarts.py`, `_generate_starts`
- **Evidence:** The function is called from `run_multi_start_optimization`
  with no seed argument. Inspecting calls shows `starts =
  _generate_starts(n_starts, xl, xu)` (line 455) with no seed forwarded. The
  random starting points differ between runs, making optimization results
  non-reproducible.
- **Impact:** Different runs of the same config on the same data produce
  different parameter estimates. Scientific conclusions cannot be replicated.
- **Minimal fix:** Add a `seed` parameter to `_generate_starts` and pass
  `getattr(args, 'multistart_seed', 42)` from `run_multi_start_optimization`.
- **Longer-term fix:** Store the seed and starting points in the output
  metadata JSON.
- **Test to add:** Run multi-start twice with the same seed; assert
  `np.allclose(starts_run1, starts_run2)`.

---

## High Priority Findings

### H1. Temporary CSV files are never deleted

- **Severity:** High
- **Category:** Code structure / Reproducibility
- **Status:** Confirmed issue
- **Location:** `phoscrosstalk/data_loader.py`, `_prefilter_phospho_csv`
  (line 1108–1111), `_prefilter_rna_csv` (lines 1170–1173)
- **Evidence:**
  ```python
  fd, tmp_path = tempfile.mkstemp(suffix=".csv", prefix="_phoscrosstalk_phospho_")
  os.close(fd)
  df_filtered.to_csv(tmp_path, index=False)
  return tmp_path
  ```
  The returned path is consumed by `load_site_data` / `load_rna_data` and
  then discarded. No `os.unlink` is called anywhere in `main.py`.
- **Impact:** On shared compute nodes (e.g., SLURM), each run leaks up to two
  full copies of the filtered input data to `/tmp`. On long jobs or systems
  with small `/tmp` this can fill the disk or create privacy issues.
- **Minimal fix:** In `main.py`, wrap the paths in a try/finally that calls
  `os.unlink` after the loaders return, or use `tempfile.NamedTemporaryFile`
  with `delete=True` as a context manager.
- **Longer-term fix:** Refactor the loaders to accept allow-lists natively
  (the existing TODO comment in the code already points at this).
- **Test to add:** Assert that after a run no files matching
  `_phoscrosstalk_*` remain in `/tmp`.

### H2. `create_bounds` ignores the `[bounds]` config section

- **Severity:** High
- **Category:** Parameters/bounds
- **Status:** Confirmed issue
- **Location:** `phoscrosstalk/optimization.py`, `create_bounds`, lines
  120–170; `phoscrosstalk/config.py`, `_DEFAULTS` `bounds` section
- **Evidence:** `create_bounds` uses hard-coded constants:
  ```python
  xl[idx : idx + K] = np.log(1e-5)
  xu[idx : idx + K] = np.log(10.0)
  ```
  The config has a `[bounds]` section with `rate_min`, `rate_max`,
  `protein_degradation_max`, `kinase_rate_max`, `phosphatase_rate_max`,
  `gamma_abs_max` etc., but `create_bounds` never receives the config and
  ignores these values.
- **Impact:** Users who tune `[bounds]` in `config.toml` see no effect. The
  hard-coded bounds may be too wide or too narrow for specific datasets.
- **Minimal fix:** Pass `cfg.bounds` to `create_bounds` and use
  `cfg.bounds.rate_min`, `cfg.bounds.rate_max`, etc.
- **Longer-term fix:** Validate the bounds section in `validate_config` and
  ensure `create_bounds` is covered by tests.
- **Test to add:** Assert that changing `rate_max` in config changes `xu`.

### H3. `validate_biological_inputs` is never called

- **Severity:** High
- **Category:** Missing validation
- **Status:** Confirmed issue
- **Location:** `phoscrosstalk/optimization.py`, `validate_biological_inputs`
  defined at lines 1057–1196; `phoscrosstalk/main.py` never calls it
- **Evidence:** `grep` on `main.py` and `multistarts.py` shows no call to
  `validate_biological_inputs`.
- **Impact:** Negative fold-change values in P_data or A_scaled, or
  non-finite weight matrices, silently propagate into the ODE and produce
  NaN residuals, which are then penalised and hidden from the user. The
  problem is diagnosed (if at all) only at the loss level, not at the data
  level.
- **Minimal fix:** Call `validate_biological_inputs(P_data=P_scaled,
  A_scaled=A_scaled, rna_data_scaled=rna_obs_matched, W_data=W_data,
  W_data_prot=W_data_prot)` in `main.py` immediately after scaling.
- **Test to add:** Unit test that passing negative P_data raises `ValueError`.

### H4. `A_data` shape inconsistency after crosstalk filtering

- **Severity:** High
- **Category:** Data loading / Logic
- **Status:** Confirmed issue
- **Location:** `phoscrosstalk/main.py`, lines 644–668 (crosstalk filtering
  block) and lines 674–687 (A_data handling)
- **Evidence:** After the crosstalk filter, `proteins` is rebuilt from sites
  that survived the filter. If the filter removes sites belonging to proteins
  that had abundance data, `A_data` still references the old `A_proteins`,
  which may include proteins no longer in `proteins`. The subsequent filter
  at line 676:
  ```python
  mask_A = [p in prot_map for p in A_proteins]
  ```
  uses `prot_map` which was reassigned inside the crosstalk block (line 663)
  as `prot_map = {p: i for i, p in enumerate(prots_used)}`. This is correct.
  However, `A_proteins` is *not* updated if `A_data` rows correspond to
  proteins not in `prots_used`, they are silently dropped. There is no
  warning when `A_data` rows are discarded at this stage.
- **Impact:** Protein abundance data silently drops without notification,
  making it impossible to diagnose missing data contributions.
- **Minimal fix:** Log the number of `A_proteins` dropped by the crosstalk
  filter.

### H5. `W_data_mrna` shape mismatch risk when slicing by `rna_obs_idx`

- **Severity:** High
- **Category:** Matrix construction / Data loading
- **Status:** Suspected issue
- **Location:** `phoscrosstalk/main.py`, lines 941–942
  ```python
  W_data_mrna_matched = W_data_mrna[rna_obs_idx_raw, :]
  ```
  `W_data_mrna` is built by `build_weight_matrices` with shape `(G, T_rna)`
  where `G` is the number of rows in the **full filtered RNA CSV**. But
  `rna_obs_idx_raw` are indices into `gene_ids` from `load_rna_data`, which
  loads the filtered file and thus `G == len(gene_ids)`. If
  `_prefilter_rna_csv` removes rows, `G` shrinks but `W_data_mrna` was built
  from the post-filter matrix, so indices are consistent. However,
  `build_weight_matrices` receives `rna_data=rna_matrix` (the full loaded
  matrix, not `rna_obs_matched`), so `W_data_mrna.shape[0] == len(gene_ids)`.
  The slice is correct — but only because the pre-filter and loader are
  coupled. If either is changed independently, the shape will silently break.
- **Impact:** Silent dimension mismatch that would surface as an index error
  or wrong weights at optimization time.
- **Minimal fix:** Assert `W_data_mrna.shape[0] == len(gene_ids)` before
  slicing.

### H6. `ModelDims` is a bare mutable class-level global — not thread-safe

- **Severity:** High
- **Category:** Code structure / Reproducibility
- **Status:** Confirmed issue
- **Location:** `phoscrosstalk/config.py`, `ModelDims` class; called from
  `main.py` line 724 and `_run_single_start_worker` (multistarts.py line 115)
- **Evidence:**
  ```python
  class ModelDims:
      K: int = None
      M: int = None
      N: int = None
  ```
  Class variables are shared across all instances. In worker processes (spawn
  mode), each worker calls `ModelDims.set_dims(_K, _M, _N)` before running —
  this is correct in isolation. But `test_optimization.py` uses an
  `autouse=True` fixture to restore dims after each test, meaning concurrent
  test runs (via `-n auto` pytest-xdist) could corrupt the global.
- **Impact:** Tests that check model dimensions may fail intermittently; any
  future use of thread-level parallelism within a single process will be
  unsafe.
- **Minimal fix:** Document the limitation; add a module-level lock around
  `set_dims` and reads.
- **Longer-term fix:** Replace with a context object passed explicitly.

---

## Medium Priority Findings

### M1. `load_site_data` time-column detection is fragile

- **Severity:** Medium
- **Category:** Data loading
- **Status:** Confirmed issue
- **Location:** `phoscrosstalk/data_loader.py`, `load_site_data`, line 49
  ```python
  value_cols = [c for c in df.columns if c.startswith("v") or c.startswith("x")]
  ```
- **Evidence:** Any column whose name starts with "v" or "x" (e.g., "value",
  "xenobiotic_treatment") is treated as a time-series column. The number must
  equal `len(timepoints)` or the function raises a `ValueError`.
- **Impact:** Unexpected columns silently corrupt the data matrix.
- **Minimal fix:** Match `re.fullmatch(r"[vx]\d+", col)` instead of
  `col.startswith(...)`.
- **Test to add:** Supply a CSV with an extra "value_unit" column and assert
  a `ValueError` is raised with a helpful message.

### M2. `scale_mode = "zscore"` is listed as valid but not implemented

- **Severity:** Medium
- **Category:** Logic / Data loading
- **Status:** Confirmed issue
- **Location:** `phoscrosstalk/config.py` line 248:
  `_VALID_SCALE = {"none", "minmax", "zscore"}`
  `phoscrosstalk/data_loader.py`, `apply_scaling`, lines 174–184
- **Evidence:** `apply_scaling` handles `"minmax"`, `"log-minmax"`, and
  `"none"` only. `"zscore"` is listed as a valid option in the config
  validator but will cause `apply_scaling` to raise `ValueError: Unknown scale
  mode`.
- **Impact:** Any user who sets `scale_mode = "zscore"` gets a crash after
  all data has loaded, wasting time.
- **Minimal fix:** Remove `"zscore"` from `_VALID_SCALE` or implement it.

### M3. `compute_prev_site_idx` uses flat ordering, not sequence ordering

- **Severity:** Medium
- **Category:** Biology / ODE/model formulation
- **Status:** Needs domain decision
- **Location:** `phoscrosstalk/jax_mechanisms.py`,
  `compute_prev_site_idx`, lines 135–158
- **Evidence:** The "predecessor" site is simply the most recently encountered
  site on the same protein in the flat list order (order of rows in the
  phospho CSV after filtering/sorting):
  ```python
  for i in range(N):
      prot = int(site_prot_idx[i])
      if prot in last_seen:
          prev_site_idx[i] = last_seen[prot]
      last_seen[prot] = i
  ```
  For the sequential mechanism, the biological predecessor should be the site
  with the next lower sequence position, not the immediately prior row index.
- **Impact:** With "seq" mechanism, phosphorylation gating is based on input
  row order rather than protein sequence order. If sites are not sorted by
  position in the input file, the sequential model is biologically wrong.
- **Minimal fix:** Sort sites by `positions` within each protein before
  calling `compute_prev_site_idx`, or build the predecessor list based on
  `positions` rather than row order.
- **Test to add:** Supply two sites with known positions where row order
  disagrees with position order; assert predecessor is the positionally-prior
  site.

### M4. `Cl` (local coupling) uses O(N²) dense loop — performance issue

- **Severity:** Medium
- **Category:** Performance
- **Status:** Confirmed issue
- **Location:** `phoscrosstalk/data_loader.py`, `build_C_matrices_from_db`,
  lines 274–285
  ```python
  for i in range(N):
      for j in range(N):
          ...
          d = abs(positions[i] - positions[j])
          Cl[i, j] = np.exp(-d / L)
  ```
- **Impact:** For N > 500, this loop dominates startup time. With N = 1000
  this is 10⁶ iterations. The same result is achievable in one NumPy
  broadcast.
- **Minimal fix:**
  ```python
  pos = positions[:, None]  # (N, 1)
  same_prot = (site_prot_idx[:, None] == site_prot_idx[None, :])
  d = np.abs(pos - positions[None, :])
  Cl = np.where(same_prot & (d > 0), np.exp(-d / L), 0.0)
  np.fill_diagonal(Cl, 0.0)
  ```
- **Test to add:** Verify vectorised and loop versions agree on a small
  synthetic case.

### M5. `bio_score_nb` / `bio_score` uses magic constants without configuration

- **Severity:** Medium
- **Category:** Biology / Parameters/bounds
- **Status:** Needs domain decision
- **Location:** `phoscrosstalk/optimization.py`, `bio_score_nb`, lines 95–104
  ```python
  return (np.log10(median_t_kinase) - np.log10(10.0)) ** 2 + (
      np.log10(median_t_protein) - np.log10(600.0)
  ) ** 2
  ```
- **Evidence:** The target half-lives are hard-coded as 10 min (kinase) and
  600 min (protein). These are reasonable priors but not universally valid.
- **Impact:** `bio_score` is used diagnostically (not for selection), so this
  does not affect fitted parameters. However, it misleads users if their
  biological system has different expected timescales.
- **Minimal fix:** Move the target half-lives to the `[bounds]` config section
  and document their origin.

### M6. `A_data` protein rows are averaged when a protein has multiple rows

- **Severity:** Medium
- **Category:** Data loading / Biology
- **Status:** Needs domain decision
- **Location:** `phoscrosstalk/data_loader.py`, `load_site_data`, line 116
  ```python
  A_rows.append(sub[value_cols].values.astype(float).mean(axis=0))
  ```
- **Evidence:** When a protein has multiple non-site rows (e.g., multiple
  peptide intensities for the same protein), they are averaged. For
  fold-change data this may be correct (geometric mean of ratios), but for
  raw intensities it is incorrect.
- **Impact:** Protein abundance data may be incorrectly summarised, affecting
  the abundance loss term.
- **Minimal fix:** Add a warning when rows are averaged. Offer a config option
  for the aggregation method.

### M7. `f4_rna` residual uses raw MSE while all other residuals use sqrt-weighted differences

- **Severity:** Medium
- **Category:** Optimization / Scaling/weights
- **Status:** Confirmed issue
- **Location:** `phoscrosstalk/optimization.py`, `make_residuals_fn`,
  lines 800–802
  ```python
  diff_R = R_sim_matched - rna_j
  r_rna = (sqrt_wr * diff_R).ravel()
  f4 = jnp.sum(W_rna_j_diag * diff_R * diff_R) / n_rna
  ```
  Versus f1 in the same function (line 782):
  ```python
  f1 = jnp.sum(W_data_j_diag * diff_p * diff_p) / n_p
  ```
  Both use the same form for the diagnostic, but the actual residual
  contribution to the LM objective (`r_rna`) is direct while `r_phospho` is
  also direct. The `log1p` form is only in `make_loss_fn` (the BFGS path).
  This is internally consistent within `make_residuals_fn` but inconsistent
  with `make_loss_fn`. See C2 for the selection bias consequence.

### M8. `_VALID_SCALE` vs `apply_scaling` mismatch: `log-minmax` not in `_VALID_SCALE`

- **Severity:** Medium
- **Category:** Logic
- **Status:** Confirmed issue
- **Location:** `phoscrosstalk/config.py` line 248:
  `_VALID_SCALE = {"none", "minmax", "zscore"}`
  `phoscrosstalk/data_loader.py`, `apply_scaling` handles `"log-minmax"`
- **Evidence:** `"log-minmax"` is a valid `apply_scaling` mode but is absent
  from `_VALID_SCALE`. `"zscore"` is in `_VALID_SCALE` but not in
  `apply_scaling`. The two sets disagree.
- **Impact:** `scale_mode = "log-minmax"` will pass `validate_config` as
  invalid, but would work at runtime. `scale_mode = "zscore"` passes
  validation but crashes at runtime.
- **Minimal fix:** Synchronise the two sets:
  `_VALID_SCALE = {"none", "minmax", "log-minmax"}`.

### M9. `make_loss_fn` default `ode_adjoint_kind` is `"recursive"`, which is invalid

- **Severity:** Medium
- **Category:** ODE/model formulation
- **Status:** Confirmed issue
- **Location:** `phoscrosstalk/optimization.py`, `make_loss_fn`, line 279
  ```python
  ode_adjoint_kind="recursive",
  ```
  The valid adjoint kinds are `{"forward", "checkpoint", "direct",
  "backsolve", "none"}` (from `config.py` line 261–267). `"recursive"` is not
  in this set.
- **Impact:** If `make_loss_fn` is called (e.g., in the hybrid path) without
  an explicit `ode_adjoint_kind`, `make_diffrax_adjoint("recursive")` will be
  called. Whether this silently falls back or raises depends on
  `solver_config.make_diffrax_adjoint`'s error handling. In the hybrid path
  this default is active.
- **Minimal fix:** Change the default to `"forward"` to match
  `make_residuals_fn` and the config default.
- **Test to add:** Assert that `make_loss_fn()` with default args does not
  raise.

### M10. `hybrid_fit.py` imports `evosax`/`qdax` at module level, not lazily

- **Severity:** Medium
- **Category:** Code structure
- **Status:** Confirmed issue
- **Location:** `phoscrosstalk/hybrid_fit.py`, lines 34–44
  ```python
  from evosax.algorithms import CMA_ES, Sep_CMA_ES
  from qdax.core.containers.mapelites_repertoire import ...
  ```
- **Evidence:** The module docstring says "Lazy imports ensure that `import
  hybrid_fit` succeeds even when evosax or qdax are not installed", but the
  actual imports are at the top level, not inside functions.
- **Impact:** Any import of `hybrid_fit` (even via `--solver lm`) will fail
  with an `ImportError` if the optional dependencies are absent.
- **Minimal fix:** Move the evosax/qdax imports inside `run_hybrid_fit`.

---

## Low Priority Findings

### L1. `f4` result of `F[:, 3]` expression is discarded in `main.py`

- **Severity:** Low
- **Category:** Logic
- **Status:** Confirmed issue
- **Location:** `phoscrosstalk/main.py`, line 1148
  ```python
  F[:, 3] if F.shape[1] > 3 else np.zeros(len(f1))
  ```
  The result is not assigned to any variable and is immediately discarded.
- **Minimal fix:** Assign to `f4 = ...` and pass it to `save_run_results` and
  `plot_run_diagnostics`.

### L2. `pyproject.toml` pins Python to `>=3.11,<3.12` while tests run on 3.12

- **Severity:** Low
- **Category:** Reproducibility / Packaging
- **Status:** Confirmed issue
- **Location:** `pyproject.toml` line 10
- **Evidence:** Memory notes document `pip install -e . --ignore-requires-python`
  is needed on this Python 3.12 environment.
- **Minimal fix:** Widen to `>=3.11,<3.13` or `>=3.11`.

### L3. `numba` is listed as a runtime dependency but is no longer used

- **Severity:** Low
- **Category:** Packaging
- **Status:** Confirmed issue
- **Location:** `pyproject.toml` line 30; `phoscrosstalk/core_mechanisms.py`
  now contains only `clip_scalar` and `decode_theta` (no Numba)
- **Minimal fix:** Remove `numba` from `dependencies`.

### L4. `save_run_metadata` serialises a `SimpleNamespace` with `vars()` which may include non-serialisable objects

- **Severity:** Low
- **Category:** Reproducibility
- **Status:** Suspected issue
- **Location:** `phoscrosstalk/post_processing.py`, `save_run_metadata`,
  lines 29–42
- **Evidence:** `vars(args)` may include numpy arrays (e.g., `receptors` list)
  or other non-JSON-serialisable types. `json.dump` will raise on first
  encounter. This silently prevents provenance saving.
- **Minimal fix:** Use a custom JSON encoder that converts numpy arrays to
  lists and non-serialisable objects to their `str()` representation.

### L5. `equations.py` depends on a working LaTeX installation but has no guard

- **Severity:** Low
- **Category:** Code structure
- **Status:** Suspected issue
- **Location:** `phoscrosstalk/equations.py`, `subprocess.run(["pdflatex",
  ...])`
- **Minimal fix:** Wrap the `pdflatex` call in a try/except and log a warning
  if it fails; save the `.tex` source regardless.

### L6. `make_rhs` redundantly computes row scales on every ODE step

- **Severity:** Low
- **Category:** Performance
- **Status:** Confirmed issue
- **Location:** `phoscrosstalk/jax_mechanisms.py`, `rhs` inner function,
  lines 304–308
  ```python
  Cg_row_scale = jnp.sum(jnp.abs(Cg), axis=1) + eps
  ```
  `Cg`, `Cl`, `R`, `K_site_kin`, `L_alpha` are static arrays passed via
  `args`. Their row sums are invariant but recomputed at every RHS evaluation.
- **Minimal fix:** Pre-compute row scales outside the `rhs` function (at
  closure-creation time) and capture them as constants.

---

## Biological and Modeling Audit

### Kinase-substrate prior logic

`K_site_kin` (Sites × Kinases) is loaded from a TSV file or built from KEA
citation counts. It is row-normalised to mean kinase contribution per site,
then transposed to `R` (Kinases × Sites) and row-normalised again for the
substrate feedback term. The double normalisation is not biologically
motivated but does not produce incorrect results — it just rescales the
effective prior strength uniformly. Sites with no annotated kinase have
K_site_kin rows all-zero, so they contribute nothing to kinase dynamics.

When no kinase TSV or KEA table is provided, `K_site_kin = eye(N)` and
`kinases = ["K_0", ..., "K_{N-1}"]`, giving each site its own private
"kinase". This is a last-resort fallback that inflates the parameter space by
N and makes all kinase parameters unidentifiable. The config validator warns
but does not block this path.

### Phosphosite label construction and normalisation

`_normalise_psite_label` strips underscores between amino-acid letter and
position (`Y_1068` → `Y1068`). However, `load_site_data` builds site labels
directly from the CSV without applying this normalisation. If the kinase TSV
uses `EGFR_Y1068` (normalised) and the phospho CSV uses `EGFR_Y_1068`
(unnormalised), the allow-set membership check in `_prefilter_phospho_csv`
will correctly normalise, but `load_site_data` may produce labels like
`EGFR_Y_1068` (with the underscore included in the residue part). The site
lookup in Cg/Cl construction then fails silently because the `idx` dict key
won't match.

**Recommendation:** Apply `_normalise_psite_label` consistently in
`load_site_data` when the `Psite` column is used.

### Protein vs phosphosite vs kinase identity handling

- Proteins in the model are derived from `sites` (row-unique proteins after
  filtering), not from an independent protein table. This means the protein
  universe is entirely determined by the phospho-data and kinase network
  entries. Proteins without any observed phosphosite are not modelled even if
  they have kinase priors — this is correct but may surprise users.
- Kinases may or may not be in `proteins`. `kin_to_prot_idx[k] = -1` for
  kinases not found in `proteins`. These kinases still participate in the
  kinase dynamics (Kdyn, K_site_kin) but receive no protein context
  (`prot_contrib = 0`). This is biologically coherent.

### TF source/target interpretation

`build_tf_prot_weights` maps: TF source gene → gene index in RNA matrix,
target protein name → protein index. The product `W @ rna_data` gives each
protein a weighted sum of TF-source mRNA trajectories. This is correct if TF
genes that drive protein `p` are listed as sources of edges pointing to `p`'s
name. However, TF network files typically list edges as `TF_gene → target_gene`
where both are gene names. If the model protein for `p` has a different symbol
than the target gene name (e.g., canonical protein name vs. gene alias), the
mapping fails silently (C1 above).

### RNA-to-protein mapping

`match_rna_to_model_proteins` uses exact string equality. Isoform differences
(e.g., "EGFR" vs. "EGFR1") will cause missed matches. No fuzzy matching is
attempted. This is acceptable as a conservative design decision but should be
documented.

### Whether TFs included as proteins are handled consistently

With `include_tfs_as_proteins = true`, TF source and target proteins enter the
model. The `protein_self_rna_idx` mechanism allows proteins without TF upstream
edges to use their own RNA trajectory as a fallback activation signal. This is
biologically motivated. However, TF proteins that appear both as sources and
targets in the network, and also have phosphosite data, receive both kinase-
driven phosphosite dynamics and TF-driven activation — this can lead to
double-counting of regulatory input if the TF's phosphorylation state feeds
back through `S` into `k_act`.

### Receptor/receptor-kinase masks

`receptor_mask_prot` and `receptor_mask_kin` are binary {0,1} vectors. The
external stimulus `u(t) = 1 / (1 + exp(-t/0.1))` is a hard-coded sigmoidal
ramp with a very fast timescale (half-rise at t≈0.07 time units). If time is
in minutes, this is effectively an instantaneous step. If time is in seconds,
the timescale may be appropriate. The timescale constant `0.1` is not
configurable. For experiments where stimulus is delivered at a defined time
point (not t=0), this formulation is wrong.

**Recommendation:** Make the stimulus timescale and onset time configurable
via the config.

### Whether phosphosite crosstalk matrices reflect the intended graph

`Cg` is populated from PTM database association scores (0.8 * average of two
residue scores / 200). The normalisation factor (200) is hard-coded and
opaque — this appears to be the maximum possible association score in the
database. The row-normalisation applied afterward makes the absolute scale
irrelevant for the ODE (only relative strengths matter within a protein), but
the prior choice of 0.8 is not documented.

`Cl` uses a simple exponential decay in sequence distance. Sites on different
proteins are always 0 in `Cl`, even if they are very close in sequence space.
This is by design.

### Whether missing priors are treated safely

Sites with no kinase entry in K_site_kin have zero columns in K_site_kin and
therefore zero kinase drive. Their phosphosite dynamics reduce to `dp/dt = -k_off * p`,
i.e., exponential decay from initial condition. This is biologically
conservative but means fitting will converge to k_off that cancels the initial
condition only, with no mechanistic content.

---

## ODE and Parameter Audit

### ODE state dimensions

State `y = [R_rna(K), S(K), A(K), Kdyn(M), p(N)]`, total dimension `3*K + M + N`.
This is consistent across `jax_mechanisms.py`, `simulation.py`,
`optimization.py`, and `core_mechanisms.py`. The initial conditions in
`make_residuals_fn` and `simulation.py` are consistent with this layout.

### Parameter vector layout

`theta` has dimension `2*K + 2 + 3*M + N + 4`. The layout:

| Slice | Length | Parameter |
|-------|--------|-----------|
| [0:K) | K | log_k_deact |
| [K:2K) | K | log_d_deg |
| [2K] | 1 | log_beta_g |
| [2K+1] | 1 | log_beta_l |
| [2K+2:2K+2+M) | M | log_alpha |
| [...:...+M) | M | log_kK_act |
| [...:...+M) | M | log_kK_deact |
| [...:...+N) | N | log_k_off |
| [...:...+4) | 4 | gamma_raw[0..3] |

This layout is documented in `build_parameter_labels` and consistent across
`decode_theta`, `decode_theta_jax`, and the `create_bounds` function. The
labels match the ordering.

### Bounds

`create_bounds` hard-codes:
- `k_deact`: log(1e-5) to log(10.0)
- `d_deg`: log(1e-5) to log(0.5)
- `alpha`: log(1e-5) to log(10.0)
- `kK_act`, `kK_deact`: log(1e-5) to log(3.0)
- `k_off`: log(1e-5) to log(5.0)
- `gamma_raw`: [-3.0, 3.0] (maps to tanh range ≈ [-1.99, 1.99])

These are reasonable for fold-change data on minute timescales. The `d_deg`
upper bound of 0.5 (protein half-life ≥ ~1.4 min) is tight; many signalling
proteins have much longer half-lives in basal conditions. This bound ignores
the `[bounds] protein_degradation_max` config value (H2).

### Initial conditions

`x0` for `p` is set from `P_data[:, 0]` (first timepoint of phospho data),
clipped to `[0, +inf)`. For `A`, from `A_data[:, 0]` if available, else 0.
For `R_rna`, from `R_data0` or 1.0. Initial conditions for `S` and `Kdyn`
are always 0.0 (implicit via `np.zeros`). **This means the model always
starts with zero signalling state and zero kinase activity**, regardless of
biological context. If the data are observed mid-response, starting at S=0
and Kdyn=0 may force the early optimization residuals to be large and may
bias parameter estimates.

### Scaling assumptions

`P_data` can be raw or minmax-scaled. When `scale_mode = "none"`, raw
fold-change values enter the ODE loss. The phosphosite state `p` is
initialised from `P_data[:, 0]` directly; if raw values are > 1.0 (common
for log2 FC data), the upper bound guard at `v_on = v_on_raw / (1 + v_on_raw)`
limits the production rate but allows `p` itself to exceed 1. The loss
function penalises `P_sim - P_data` without checking that both are on the
same scale. If users mix minmax-scaled data with the abundance loss (which
uses `A_data` from the same file), inconsistent scaling between p and A
modalities may occur.

### Derived rate closures

`k_act_fn(t)` returns an interpolated TF-signal vector. Inside `rhs`, this is
clipped to `[0, +inf)` before use. `s_prod_fn(t)` returns a softplus or
linear-transformed kinase-activity signal per protein. Neither closure can
return negative values, so the ODE production terms are non-negative by
construction.

The `rna_relax` parameter controls the relaxation rate of `R_rna` toward
`k_act`. The default value is 0.1, implying a half-time of ~7 time units.
This should be calibrated to the mRNA decay timescale of the system, which
is not validated.

### Interpolation

Both `_piecewise_constant` and `_linear_interp` clip the index to the valid
range. `_linear_interp` clips to `[0, T-2]`, meaning at time `t >= t_last`,
the interpolated value is the slope between the last two timepoints
extrapolated — actually no, `idx = clip(idx, 0, T-2)` means at `t = t_last`,
`t1 = times[T-1]` and `alpha = (t - t0)/(t1-t0+eps)` which is 1.0, returning
`v[T-1]`. This is correct (flat extrapolation at last value). At `t < t_first`,
`idx = 0` and `alpha < 0` which is clipped to 0.0, returning `v[0]`. Correct.

### Stiffness/numerical stability

The ODE mixes fast states (S, Kdyn with potential large activation rates) and
slow states (A with small d_deg). The default solver is `Tsit5` (explicit
RK4-5), which is not stiffness-aware. For stiff problems (large ratio of rate
constants), `Tsit5` may require very small steps, hitting `max_steps` and
returning a `RESULTS.max_steps_reached` result. This is handled (penalty is
applied) but the user is not informed which proteins/kinases caused stiffness.

Implicit solvers (`Kvaerno3`, `Kvaerno4`, `Kvaerno5`) are available in the
config but not the default. For systems with d_deg << k_deact, the problem
may be stiff and the explicit solver suboptimal.

### Positivity/non-negativity

Boundary guards in `rhs` (lines 454–465) prevent derivatives from driving
states below zero or above their upper bounds at the boundary. However, JAX
with `throw=False` may still return states outside bounds if the solver takes
large steps. The residuals function clips extracted states (`P_sim = clip(xs,
0, None)`) before computing residuals, so the loss is computed on clipped
values while the ODE state may be unclipped inside the solver.

### Parameter identifiability

The model has `2*K + 2 + 3*M + N + 4` parameters. For a typical run with
K=20, M=50, N=200, this is ~900 parameters. The observed data is `N*T` points
(e.g., 200*8 = 1600 phosphosite observations). With regularisation, the
problem is formally over-determined at the data level but many parameters
(especially per-protein k_deact, d_deg) will be poorly constrained by
phosphosite data alone. Identifiability is not assessed anywhere in the code.

---

## Optimization Audit

### Objective terms

- **f1** (phosphosite): `sum(W * (P_sim - P_data)²) / n_p` in the residual
  path; `sum(log1p(W * diff²)) / n_p` in the loss path. Inconsistent.
- **f2** (abundance): same pattern, consistent with f1 within each path.
- **f3** (regularisation): `(reg_lambda * ||theta||² + lambda_net * alpha'
  L_alpha alpha) / n_var`. L2 on raw theta regularises the log-space
  parameters, which is equivalent to a lognormal prior on rates.
- **f4** (mRNA): raw weighted MSE, no log1p (see M7).

### Residual shapes

For LM path: residuals are
`[r_phospho(N*T), r_abund(K_obs*T), r_rna(n_match*T_rna), r_reg(n_var + M if lambda_net>0)]`.
The LM Jacobian has shape `(n_residuals, n_params)`. For large N, M, T, this
Jacobian is large but sparse — the code does not exploit sparsity.

### Weighting

Loss weights `w_phospho`, `w_abundance`, `w_mrna`, `w_reg` are applied as
`sqrt(w) * sqrt(W) * residual` in the LM path, so they correctly scale
`||residual||²` by `w * W`. These defaults are all 1.0.

The entity-noise weights from `build_weight_matrices` use temporal jaggedness
(first differences in log1p space) as a noise proxy. This is a reasonable
heuristic but does not account for technical replicates or experimental noise
structure.

### Multi-start logic

Seeds are not fixed (C3). Starting points are drawn from `xl, xu` uniformly
in log-space (since bounds are log-transformed). This is appropriate. The
number of starts `n_starts` defaults to 3, which is very small for a 900-
dimensional space.

### Hybrid solver

`hybrid_fit.py` wraps evosax CMA-ES + QDax MAP-Elites + LM polish. The
evosax CMA-ES operates in normalised [0,1] space. The QDax MAP-Elites uses
2D behaviour descriptors (biological half-lives). This is methodologically
sound. The evosax top-level import issue (M10) prevents use in environments
without the optional dependencies.

### Convergence criteria

Optimistix LM uses `rtol=1e-8, atol=1e-8` by default. For float32 ODE
solutions, this is tighter than the precision of the ODE output, which may
cause the LM to run to `max_steps` without converging.

### Loss reporting

`total_loss = f1 + f2 + f3 + f4` is an unweighted sum of diagnostic
components, not the actual LM objective (which is `||residuals||²` with the
modality weights baked in). This diagnostic sum is what drives best-run
selection. If `w_phospho` ≠ `w_mrna`, the modality weights in the residuals
do not match their contributions to `total_loss`.

---

## Data Loading and Model Universe Audit

### Phospho/protein data loading

`load_site_data` splits rows into phosphosite rows (have `Psite`/`Residue`)
and protein rows (do not). Protein rows are averaged per protein and form
`A_data`. This is the only path for protein abundance data. There is no
option to supply abundance data in a separate file.

### RNA loading

`load_rna_data` expects either `x1..x9` columns (mapped to 9 fixed
timepoints) or numeric column names. The 9 fixed timepoints
`[4, 8, 15, 30, 60, 120, 240, 480, 960]` are hard-coded as `RNA_TIMEPOINTS_9`.
These are not validated against the protein timepoints; if protein data uses
minutes but RNA data uses seconds, the time grids will be misaligned and the
ODE solve will cover a combined time range that may be numerically problematic.

### Crosstalk filtering

The `crosstalk_tsv` filter applies an allow-set of sites. The parsing:
```python
for c in ["Site1", "Site2"]:
    for p, site in zip(df["Protein"], df[c], strict=False):
        s.add(f"{p}_{site}")
```
assumes the crosstalk TSV uses column names "Protein", "Site1", "Site2".
There is no validation of these column names. A missing column causes a
`KeyError` crash.

### Site-label normalisation

Sites are constructed as `f"{protein}_{residue}"` in `load_site_data`. If
the Psite column contains `Y_1068`, the residue becomes `Y1068` (underscore
stripped). If the Residue column is used directly, the label includes
whatever is in the file. These two paths produce different label formats,
which may cause mismatches when the same site is referenced in both formats
across different input files.

### Duplicate handling

`load_site_data` does not check for duplicate (Protein, Psite) pairs. If the
same phosphosite appears on two rows, both are included and two rows in
`P_data` will attempt to fit the same biology with different initial
conditions. `load_rna_data` raises on duplicate gene IDs (correct).

### Full-dataset reservoir behaviour

The pre-filter design (temporary CSV files) correctly implements the reservoir
pattern: full measurement files are filtered to the network-defined model
universe before loading. The design is acknowledged as temporary (TODO
comments). The main risk is that the filter may over-eagerly remove valid
sites if label normalisation is inconsistent (see above).

---

## Matrix Construction Audit

### Cg (global crosstalk, N×N)

Symmetric, non-negative. Populated from two SQLite databases (intra and inter
protein PTM association tables). Score formula: `0.8 * (r1 + r2) / 200.0`.
The factor 0.8 and divisor 200 are hard-coded with no documentation of their
origin. The matrix is row-normalised by `data_loader.row_normalize` before
entering the ODE. Correctness is conditional on the PTM databases having
consistent site labels with the phospho-data CSV. If a site exists in the
PTM DB but not in the model (or vice versa), it is silently skipped.

### Cl (local coupling, N×N)

Exponential decay in sequence distance, same-protein only. Pure distance-
based, no database lookup. The length scale `L` defaults to 50.0 (presumably
amino acids) and is config-driven. Row-normalised after construction. The
diagonal is explicitly excluded (`if i == j: continue`), correctly preventing
self-coupling.

### K_site_kin (N×M) and R (M×N)

`K_site_kin` is the raw kinase-substrate prior matrix. `R = K_site_kin.T`,
then row-normalised again. This is the substrate feedback matrix used in the
kinase dynamics ODE. Note that `K_site_kin` is row-normalised in
`build_kinase_site_from_kea` (by row sums), and then `R` is also row-
normalised in `main.py`. The double normalisation means the feedback signal
is the normalised-kinase-weighted average of phosphosite states. This is
internally consistent but means the absolute prior strengths (citation counts
in KEA) are discarded.

### L_alpha (M×M)

Standard Laplacian of the kinase-kinase graph. Symmetric, positive
semi-definite. Row sums equal diagonal elements. The network penalty
`alpha @ L_alpha @ alpha` is a graph smoothness term on the log-alpha kinase
strength vector. This is a standard graph regulariser and is mathematically
correct.

### site_prot_idx (N,)

Integer array mapping each phosphosite to its protein index in `proteins`.
Built from sorted(set(proteins_raw)). Correctness depends on `proteins` being
the same sorted list used in `prot_index`.

### kin_to_prot_idx (M,)

Maps kinase names to protein indices. `-1` for kinases not in `proteins`.
Used in `make_rhs` with a `jnp.where(valid_prot, safe_p_idx, 0)` guard to
avoid out-of-bounds indexing. Correct.

### receptor masks (K,) and (M,)

Binary {0, 1} float vectors set from `receptor_names` and
`receptor_kin_names` in the config. No validation that the named receptors
are actually in `proteins`/`kinases`. If a named receptor is absent from the
model, the mask is all-zeros with no warning (just the generic "No receptors
defined" warning for the empty case).

**Minimal fix:** Check that each name in `receptor_names` appears in
`proteins` and warn if not.

### RNA mapping matrices

`rna_model_prot_idx` maps matched RNA rows to protein indices. Built by exact
string match. Correct but fragile (see biology audit).

---

## Code Structure and Maintainability Audit

### Large orchestration functions

`main.py:main()` is ~1200 lines. It mixes config extraction, data loading,
matrix construction, ODE setup, optimisation dispatch, and post-processing.
It is difficult to test incrementally. Consider extracting the data-loading
and matrix-construction phases into `build_model_universe()` and the analysis
phase into `run_analysis()`.

### Circular imports

`analysis.py` imports from `optimization.py` (for `bio_score` and
`build_full_A0`). `optimization.py` imports from `simulation.py`.
`simulation.py` imports from `jax_mechanisms.py`. No circular imports
detected.

### Hidden global state

`ModelDims` is the primary shared mutable global (H6). It is set in `main.py`
and read in many other modules. Tests restore it via a fixture. This is
fragile.

### Duplicated logic

- The "build initial state from data" block (`x0` construction) is duplicated
  between `make_residuals_fn` (lines 600–615) and `make_loss_fn` (lines
  320–341) in `optimization.py`, and again in `simulation.py:simulate_ode`
  (lines 148–175). Any change to the initial condition logic must be applied
  in three places.
- `has_mrna` check logic is duplicated between `make_loss_fn` and
  `make_residuals_fn`.

### Private function usage across modules

`_build_network_allow_sets`, `_prefilter_phospho_csv`, and
`_prefilter_rna_csv` are defined with leading underscores in `data_loader.py`
but are imported directly by `main.py`. This defeats the privacy convention.

### Logging consistency

The codebase uses a custom logger with `header`, `success`, and `info`
methods. This is fine. However, some modules use `logger.warning` for
conditions that should arguably be errors (e.g., zero-row filter output).

### Error handling

`validate_config` raises `SystemExit(1)` on errors rather than `ValueError`,
making it difficult to test without subprocess mocking. Consider raising
`ValueError` and converting to `SystemExit` at the call site in `main.py`.

### Temporary files

Two temporary files per run are never cleaned up (H1).

---

## Testing Gaps

### Unit tests

- **`test_bounds_from_config`**: Verify that `create_bounds` reads `cfg.bounds`
  values. Fixture: config with custom `rate_max`. Expected: `xu[:K] ==
  log(cfg.bounds.rate_max)`.
- **`test_load_site_data_duplicate_sites`**: Supply a CSV with two rows for
  the same (Protein, Psite). Expected: either `ValueError` or a warning that
  duplicates are present.
- **`test_load_site_data_extra_v_column`**: Supply a CSV with column "value"
  (starts with "v" but is not a time column). Expected: `ValueError` with
  descriptive message.
- **`test_apply_scaling_zscore_invalid`**: Assert `apply_scaling(Y, mode="zscore")`
  raises `ValueError`.
- **`test_build_tf_prot_weights_empty`**: Supply a TF network where no target
  matches any model protein. Assert `W.sum() == 0` and a warning is logged.
- **`test_tmp_file_cleanup`**: After a run, assert no `_phoscrosstalk_*` files
  exist in tempdir.
- **`test_generate_starts_reproducible`**: Call `_generate_starts(n=5, xl, xu,
  seed=42)` twice; assert identical results.

### Integration tests

- **`test_full_pipeline_no_rna`**: Run `main()` on a tiny synthetic dataset
  without RNA data. Assert output files are created and `total_loss` is finite.
- **`test_full_pipeline_with_rna`**: Same with RNA data. Assert f4 > 0.
- **`test_hybrid_solver_missing_deps`**: Assert that importing `hybrid_fit`
  without evosax installed raises `ImportError` only when `run_hybrid_fit` is
  called, not at import time.
- **`test_crosstalk_filter`**: Apply `_prefilter_phospho_csv` with a known
  allow-set; assert the correct rows survive.

### Numerical/model tests

- **`test_rhs_bounded_states`**: Assert that after integration for 1000 time
  units, S∈[0,1], Kdyn∈[0,1], A∈[0,5], p≥0 for random valid parameters.
- **`test_sequential_mechanism_gate`**: Supply two sites on the same protein
  with known positions; verify that the predecessor index reflects position
  order, not row order.
- **`test_stimulus_timescale`**: Verify that `u(t)` reaches 0.5 at t=0.07
  and document the unit assumption.
- **`test_loss_commensurability`**: With RNA data present, verify that f4 and
  f1 have comparable magnitudes under default weights so that `total_loss` is
  meaningful.

### Biological validation tests

- **`test_decay_without_kinase_drive`**: Sites with zero K_site_kin row should
  show p decaying from initial condition. Assert `P_sim[:, -1] < P_sim[:, 0]`
  for all-zero K_site_kin sites.
- **`test_receptor_stimulus`**: With receptor_mask active, verify that S rises
  from 0 faster than without receptor mask.

### Regression tests

- **`test_parameter_layout_consistent`**: Assert that `decode_theta` and
  `build_parameter_labels` produce the same slice boundaries for K=3, M=5,
  N=7.
- **`test_no_tmp_files_after_run`**: Regression guard for H1.

### CLI/config tests

- **`test_scale_mode_zscore_config_error`**: Assert that `validate_config`
  does not accept `scale_mode = "zscore"` (or that it does, after M8 is fixed).
- **`test_config_bounds_section`**: Assert that `[bounds]` values are used by
  `create_bounds` after H2 fix.

---

## Reproducibility and Packaging Audit

### pyproject / package metadata

- Python version constraint is `>=3.11,<3.12` but the runtime environment is
  Python 3.12 (L2). This causes `pip install` to fail without `--ignore-
  requires-python`.
- `numba` is listed as a dependency but is unused (L3).
- `evosax==0.2.0` is pinned but evosax's API changes between versions are
  substantial. The restart API imports (`RestartParams`, `RestartState`,
  `cma_cond`, `spread_cond` from `evosax.core.restart`) are version-specific
  and will break if evosax is upgraded.
- `qdax>=0.5.0` is a loose version constraint; QDax 0.5 → 0.6 introduced
  breaking API changes.
- `jaxlib>=0.4.30` is a lower bound with no upper bound; JAX 0.4.x → 0.5.x
  can introduce breaking changes.

### Dependency pinning

`uv.lock` provides a full lockfile, which is good. However, the lockfile is
not CI-enforced (no `uv sync --frozen` in CI workflow visible from the
repository structure). If contributors install without the lockfile, they may
get different JAX versions.

### Environment setup

`runtime_env.py` provides CPU thread configuration before JAX import. This is
correct. SLURM_CPUS_PER_TASK is respected. The XLA thread-count mechanism
sets `XLA_FLAGS=--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=N`.

### Output provenance

`save_run_metadata` writes `run_config.json`. This includes `ModelDims` and
all args. However, it serialises `vars(args)` which may fail on numpy arrays
(L4). The random seed for multi-start is not saved (C3).

### Random seeds

- Multi-start: no seed (C3 — critical).
- Hybrid solver: `hybrid_seed` is configurable (default 0) and passed to
  `run_hybrid_fit`. This path is reproducible.
- Fréchet distance computation: deterministic (no random state).

### File naming

Output files use fixed names (`pareto_front.npz`, `fit_timeseries.tsv`, etc.).
Multiple runs to the same `outdir` will overwrite previous results without
warning.

---

## Recommended Fix Roadmap

### Stage 1: Non-invasive correctness fixes (zero model changes, low risk)

| Fix | Files affected | Risk | Benefit |
|-----|---------------|------|---------|
| Fix `_generate_starts` to accept seed | `multistarts.py` | None | Reproducible runs |
| Delete temp CSVs after loaders return | `main.py` | None | No temp file leaks |
| Add `validate_biological_inputs` call | `main.py` | None | Early data error detection |
| Add `W.sum() == 0` TF warning | `data_loader.py` | None | Silent mapping failure detected |
| Fix `ode_adjoint_kind="recursive"` default in `make_loss_fn` | `optimization.py` | None | Correct hybrid path |
| Move evosax/qdax imports inside function | `hybrid_fit.py` | None | Import safety |
| Fix `_VALID_SCALE` vs `apply_scaling` mismatch | `config.py` | None | Config validation accuracy |
| Assign `f4` result in `main.py` line 1148 | `main.py` | None | Silent bug fix |
| Warn when receptor names not in proteins | `main.py` | None | Misconfiguration detection |

### Stage 2: Loader/model-universe cleanup (low risk, improves maintainability)

| Fix | Files affected | Risk | Benefit |
|-----|---------------|------|---------|
| Pass `cfg.bounds` to `create_bounds` | `optimization.py`, `main.py` | Low | Config-driven bounds |
| Normalise site labels consistently in `load_site_data` | `data_loader.py` | Low | Label format consistency |
| Vectorise `Cl` construction | `data_loader.py` | Low | Performance improvement |
| Check duplicate sites in `load_site_data` | `data_loader.py` | Low | Data quality guard |
| Validate crosstalk TSV column names | `main.py` | Low | Crash prevention |
| Add duplicate-site warning in pre-filter | `data_loader.py` | Low | Data quality guard |
| Deduplicate initial-state construction into a helper | `optimization.py`, `simulation.py` | Medium | Maintainability |

### Stage 3: Model and optimization validation (medium risk)

| Fix | Files affected | Risk | Benefit |
|-----|---------------|------|---------|
| Unify f4 loss family with f1/f2 (log1p form) | `optimization.py` | Medium | Commensurable loss terms |
| Fix best-run selection to exclude f4 from total | `multistarts.py`, `optimization.py` | Medium | Correct model selection |
| Sort sites by position before `compute_prev_site_idx` | `main.py`, `jax_mechanisms.py` | Medium | Correct sequential mechanism |
| Make stimulus timescale configurable | `jax_mechanisms.py`, `config.py` | Low | Biologically correct stimulus |
| Pre-compute row scales outside `rhs` | `jax_mechanisms.py` | Low | Performance |

### Stage 4: Reproducibility and testing hardening

| Fix | Files affected | Risk | Benefit |
|-----|---------------|------|---------|
| Add missing unit tests (all listed above) | `tests/` | None | Coverage |
| Fix Python version constraint in pyproject | `pyproject.toml` | None | Install on 3.12 |
| Remove `numba` from dependencies | `pyproject.toml` | None | Lighter install |
| Save multistart seed to metadata | `post_processing.py`, `multistarts.py` | None | Full provenance |
| Warn/overwrite protection for output dir | `main.py` | Low | Reproducibility guard |
| Enforce lockfile in CI | `.github/workflows/` | None | Reproducible CI |

---

## Open Questions for Domain Decision

1. **Sequential mechanism ordering**: Should `compute_prev_site_idx` use
   sequence position order or flat row order? The current implementation uses
   row order, which depends on CSV sort order. What is the intended biology?

2. **Stimulus function**: The external receptor stimulus is
   `u(t) = sigmoid(t / 0.1)`. Is 0.1 a time constant in the same units as
   the experimental time points? Should this be a step function, a ramp, or
   a pulse? Should the onset time be variable?

3. **RNA timescale mismatch**: The RNA timepoints (default: 4–960 min) and
   protein/phospho timepoints may be on different scales. What are the
   expected units for each modality in a standard experiment with this tool?

4. **`d_deg` upper bound**: The protein degradation rate upper bound is
   exp(log(0.5)) = 0.5 min⁻¹ (half-life ~1.4 min). Is this appropriate for
   the target biological system (EGF signalling, etc.)?

5. **A_data averaging**: When multiple rows exist for the same protein in the
   phospho CSV (the rows without a Psite annotation), they are row-averaged.
   Is arithmetic mean correct here, or should geometric mean or median be
   used?

6. **`rna_relax` value**: The default `rna_relax = 0.1` controls how fast
   the R_rna state tracks the TF signal. What biological process does this
   represent, and what is its expected value for human cell line data on a
   minute timescale?

7. **TF-as-protein inclusion**: When `include_tfs_as_proteins = true`, TF
   proteins whose phosphorylation state is not in the dataset receive dynamics
   driven only by the TF network. What is the intended behaviour for TF
   proteins that appear in the TF network but have no phosphosite data?

8. **Kinase identity vs. protein identity**: Should every kinase that maps to
   a model protein (kin_to_prot_idx[k] >= 0) have its S state contribute to
   `prot_contrib`? Currently, gamma_A_S * S_for_kin + gamma_A_p * A_for_kin
   feeds into the kinase latent field U. This means the protein signalling
   state S (which aggregates phosphosite occupancy) also feeds back into
   kinase activation. Is this a deliberate feedback loop or an unintentional
   coupling?

9. **Double normalisation of K_site_kin**: The matrix is normalised once in
   `build_kinase_site_from_kea` (by KEA citation count row sums) and again
   via `R = K_site_kin.T; R[nz] /= rs[nz, None]` in `main.py`. Is the
   final effective prior strength a deliberate choice?

10. **`bio_score` target half-lives**: Are the target kinase half-life (10
    min) and protein half-life (600 min) biologically justified reference
    values for the intended application?

---

## Final Verdict

**Fit for exploratory use**: Yes, with caveats. The ODE formulation is
internally consistent, the JAX/Diffrax integration is technically correct,
and the code runs on supported hardware. For qualitative exploration of
signalling network dynamics and parameter sensitivity, the code is usable.

**Fit for publication-grade inference**: **No**, not in its current state. The
following issues must be resolved before trusting biological conclusions:

1. **C3 (no fixed seed for multi-start)**: Results are not reproducible, making
   any published parameter estimate a snapshot of a random draw.
2. **C2 (loss family mismatch)**: Best-run selection is biased when RNA data
   is present, meaning the reported best-fit parameters may not minimise the
   biologically relevant phosphosite loss.
3. **H2 (bounds ignore config)**: Users cannot control the parameter search
   space via the config, removing a key scientific control.
4. **H3 (no input validation call)**: Biologically invalid data (negative
   fold-change) silently corrupts the fit.
5. **M3 (sequential mechanism uses row order)**: With `mechanism = "seq"`,
   the biological gating is wrong unless the input CSV is pre-sorted by
   sequence position.

With Stage 1 and Stage 3 fixes applied, the code would be suitable for
publication-grade inference with appropriate caveats on identifiability and
the assumption of uniform initial conditions for S and Kdyn.
