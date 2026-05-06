# Derived Rates

## Biological Role

The `derived_rates` module constructs two time-varying rate vectors that are injected into the ODE right-hand side as data-derived closures — they are **not** fitted parameters.

- **`k_act(t)`** — the transcription-factor-driven kinase activation signal. For each model protein `p`, it is computed as a weighted sum of TF mRNA trajectories: `k_act_p(t) = Σ_g tf_prot_weights[p, g] * rna_data[g, t]`. Proteins without TF upstream edges fall back to their own RNA trajectory (if available) or to a neutral constant of 1.0.
- **`s_prod(t)`** — the aggregated phosphorylation drive from kinase/protein signals, representing per-protein protein synthesis rate. It is derived from observed phosphosite data weighted by kinase-to-site connectivity.

Both closures are fixed before optimisation begins. Because they are built from experimental data using JAX-native interpolation, they are fully differentiable through the ODE solve.

`build_data_interpolations` provides a separate set of SciPy/NumPy-backed interpolation callables for diagnostic visualisation only. These must never be passed into JAX-traced code.

## Implementation Overview

**`make_k_act_fn`** uses matrix multiplication `signal = tf_prot_weights @ rna_data` to pre-compute weighted signals for all proteins at all time points. The per-protein fallback (`protein_self_rna_idx`) is applied during construction so the returned closure has no Python branching on traced values.

**`make_s_prod_fn`** computes kinase activity time series as `kin_activity = R_kin_site @ Y_data` and aggregates to proteins by summing over kinases mapped to the same protein index. The `s_prod_fn_type` controls whether the result passes through `softplus` (always positive) or `linear` (identity).

Both factories fall back to safe constant vectors when required inputs are absent:
- `k_act_fn` returns `jnp.ones(K)` when `t_rna`, `rna_data`, or `tf_prot_weights` is `None`.
- `s_prod_fn` returns `jnp.full(K, 0.1)` when `Y_data` is empty or `R_kin_site` has no rows.

**JAX-traceable interpolation backends** (`_piecewise_constant` / `_linear_interp`) are the only backends used inside the ODE RHS. They use `jnp.searchsorted` with no Python branching on traced values and operate on pre-built `(D, T)` value matrices.

**`build_data_interpolations`** uses `numpy.interp` (linear) or `scipy.interpolate.PchipInterpolator` (cubic Hermite). NaN handling (leading fill, trailing forward-fill) is performed on a working copy; original arrays are never modified.

## Configuration Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `[time] interpolation` | `str` | `"piecewise_constant"` | Interpolation mode for `k_act_fn` and `s_prod_fn`: `"piecewise_constant"` or `"linear"` |
| `[derived_rates] s_prod_fn` | `str` | `"softplus"` | Scaling applied to `s_prod` signal: `"softplus"` (always positive) or `"linear"` |
| `[derived_rates] rna_relax` | `float` | `0.1` | Relaxation rate controlling how quickly the `R_rna` ODE state tracks `k_act(t)` |
| `[data_interpolation] enabled` | `bool` | `false` | Enable `build_data_interpolations` for diagnostic dense output |
| `[data_interpolation] method` | `str` | `"linear"` | Method for diagnostic interpolation: `"linear"` or `"cubic_hermite"` |
| `[data_interpolation] fill_forward_nans_at_end` | `bool` | `false` | Forward-fill trailing NaNs in diagnostic interpolation copies |
| `[data_interpolation] replace_nans_at_start` | `str\|null` | `null` | Strategy for leading NaNs: `null`, `"zero"`, or `"first_valid"` |

## API Reference

```python
def make_k_act_fn(
    t_rna: np.ndarray | None,
    rna_data: np.ndarray | None,
    tf_prot_weights: np.ndarray | None,
    K: int,
    interp_mode: str = "piecewise_constant",
    protein_self_rna_idx: np.ndarray | None = None,
) -> callable:
    """
    Build a JAX closure k_act_fn(t) -> jnp.array(shape=(K,)).

    Priority for each protein p:
      1. TF upstream signal:   k_act_p(t) = Σ_g tf_prot_weights[p,g] * rna_data[g,t]
      2. Self-RNA fallback:    k_act_p(t) = rna_data[protein_self_rna_idx[p], t]
      3. Neutral constant:     k_act_p(t) = 1.0

    Returns constant 1.0 vector when any required input is None.

    Parameters
    ----------
    t_rna               : (T_rna,) mRNA time points
    rna_data            : (n_genes, T_rna) mRNA fold-change matrix
    tf_prot_weights     : (K, n_genes) TF-to-protein weight matrix
    K                   : number of model proteins
    interp_mode         : "piecewise_constant" or "linear"
    protein_self_rna_idx: (K,) int array; -1 = no self-RNA available
    """
```

```python
def make_s_prod_fn(
    t_protein: np.ndarray,
    Y_data: np.ndarray,
    R_kin_site: np.ndarray,
    kin_to_prot_idx: np.ndarray,
    K: int,
    M: int,
    s_prod_fn_type: str = "softplus",
    interp_mode: str = "piecewise_constant",
) -> callable:
    """
    Build a JAX closure s_prod_fn(t) -> jnp.array(shape=(K,)).

    Computes:
        activity(k, t) = R_kin_site[k, :] @ Y_data[:, t_idx]
        s_prod(p, t)   = f(Σ_{k: kin_to_prot_idx[k]==p} activity(k, t))

    where f is softplus (default) or linear.
    Returns constant 0.1 when Y_data is empty or R_kin_site has no rows.

    Parameters
    ----------
    t_protein       : (T,) protein/phospho time points
    Y_data          : (N_sites, T) observed phospho data
    R_kin_site      : (M, N_sites) kinase-to-site weight matrix
    kin_to_prot_idx : (M,) integer array mapping kinases to proteins (-1 = no mapping)
    K               : number of model proteins
    M               : number of model kinases
    s_prod_fn_type  : "softplus" or "linear"
    interp_mode     : "piecewise_constant" or "linear"
    """
```

```python
def build_data_interpolations(
    t_obs: np.ndarray,
    P_data: np.ndarray | None = None,
    A_data: np.ndarray | None = None,
    rna_data: np.ndarray | None = None,
    method: str = "linear",
    fill_forward_nans_at_end: bool = False,
    replace_nans_at_start: str | None = None,
) -> dict:
    """
    Build continuous interpolation callables for observed datasets.
    For diagnostic/visualisation only — NOT used in the ODE or the loss.

    Parameters
    ----------
    t_obs                  : (T,) sorted observed time points
    P_data                 : (N_sites, T) phosphosite data; may contain NaN
    A_data                 : (K_obs, T) protein abundance data; may contain NaN
    rna_data               : (n_genes, T) mRNA data; may contain NaN
    method                 : "linear" (numpy interp) or "cubic_hermite" (SciPy PCHIP)
    fill_forward_nans_at_end: forward-fill trailing NaNs in working copy only
    replace_nans_at_start  : None | "zero" | "first_valid"

    Returns
    -------
    dict with keys:
        "t_obs"                     – original time vector (reference)
        "P_interp"                  – callable fn(t_q) -> (N_sites,) or (N_sites, T_q)
        "A_interp"                  – callable fn(t_q) -> (K_obs,) or (K_obs, T_q)
        "rna_interp"                – callable fn(t_q) -> (n_genes,) or (n_genes, T_q)
        "method"                    – interpolation method used
        "nan_fill_log"              – list of NaN handling messages
        "original_arrays_unchanged" – always True
    """
```

## Known Limitations

- `build_data_interpolations` callables use SciPy/NumPy and **must not** be passed into JAX-traced or JIT-compiled code. The JAX-traceable `make_k_act_fn` / `make_s_prod_fn` closures are the correct choice for the ODE RHS.
- `method="cubic_hermite"` requires SciPy. An `ImportError` is raised if SciPy is absent.
- When `t_rna` uses a different time axis than the phospho/protein observations, pass the RNA-specific time vector as `t_obs` when calling `build_data_interpolations` for RNA data. Using the protein time vector with RNA data that has a different grid will produce incorrect interpolants.
- Rows where all values are NaN cannot be interpolated; `build_data_interpolations` returns a NaN-returning callable for those rows and logs a message in `nan_fill_log`.
