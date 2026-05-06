# Mechanisms

## Biological Role

The `mechanisms` module implements kinase–substrate phosphorylation kinetics as a JAX-compatible ODE right-hand side (RHS). Three variants of kinase-driven phosphorylation are supported:

- **Distributive (`"dist"`)**: Each phosphosite can be modified independently. The kinase acts on sites without memory of prior modifications — a gate of 1.0 is applied uniformly.
- **Sequential (`"seq"`)**: Sites are phosphorylated in an ordered, site-by-site fashion. A downstream site's activation is gated by the bounded occupancy proxy of its predecessor site on the same protein (computed by `compute_prev_site_idx`). A small leak constant (`seq_leak = 1e-3`) prevents complete structural blocking when the predecessor is unphosphorylated.
- **Random / crowding-aware (`"rand"`)**: A mean-field occupancy gate suppresses phosphorylation as the protein-level phosphosite occupancy rises, modelling competitive crowding of available substrate.

The five ODE state variables per time point are: mRNA/transcriptional drive (`R_rna`), protein signalling state (`S`), protein abundance (`A`), kinase activity (`Kdyn`), and relative phosphosite signal (`p`).

## Implementation Overview

`make_rhs` returns a single JAX-traceable closure compatible with `diffrax.ODETerm`. The closure captures the mechanism string, derived-rate closures `k_act_fn(t)` and `s_prod_fn(t)`, and the RNA relaxation constant `rna_relax` at construction time. Only the ODE state `y` and the static topology `args` tuple vary at solve time.

**State layout:**

```
y = [R_rna (K,) | S (K,) | A (K,) | Kdyn (M,) | p (N,)]
    total dim = 3*K + M + N
```

**`args` tuple (positional order):**
```
(theta, Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
 kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin, prev_site_idx)
```

**Parameter vector layout** (length `2*K + 2 + 3*M + N + 4`):

| Slice | Length | Content |
|-------|--------|---------|
| `[0 : K)` | K | `log_k_deact` — protein deactivation rates |
| `[K : 2K)` | K | `log_d_deg` — protein degradation rates |
| `[2K : 2K+1)` | 1 | `log_beta_g` — global coupling strength |
| `[2K+1 : 2K+2)` | 1 | `log_beta_l` — local coupling strength |
| `[2K+2 : 2K+2+M)` | M | `log_alpha` — kinase activity amplitudes |
| `[... : ...+M)` | M | `log_kK_act` — kinase activation rates |
| `[... : ...+M)` | M | `log_kK_deact` — kinase deactivation rates |
| `[... : ...+N)` | N | `log_k_off` — phosphatase rates (per site) |
| `[... : end)` | 4 | `gamma_raw[0..3]` — coupling modulation coefficients (tanh-transformed) |

Note: `k_act` and `s_prod` are no longer fitted parameters — they are derived from experimental data via [`derived_rates`](derived_rates.md) and injected as closures.

The `decode_theta` function exponentiates all log-rate slices and applies `2*tanh` to the four raw gamma values, producing biologically meaningful rate constants. The `gamma_*` parameters are bounded to `(-2, +2)` by this transformation.

Degree-normalised network fields are used throughout the RHS to ensure transferability across networks of different sizes. The smooth receptor stimulus `u(t) = sigmoid(t/0.1)` is added to S- and kinase-drive terms via boolean masks.

## Configuration Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `[model] mechanism` | `str` | `"dist"` | Phosphorylation mechanism: `"dist"`, `"seq"`, or `"rand"` |
| `[solver] ode_solver` | `str` | `"tsit5"` | Diffrax solver used during integration (see [Simulation](simulation.md)) |

## API Reference

```python
def decode_theta(theta, K: int, M: int, N: int) -> tuple:
    """
    Decode the flat log-scale parameter vector into biological rate constants.

    Parameters
    ----------
    theta : jax.Array, shape (2*K + 2 + 3*M + N + 4,)
    K, M, N : int  —  proteins, kinases, phosphosites

    Returns
    -------
    tuple of 12 entries:
        k_deact       (K,)   – protein deactivation rates
        d_deg         (K,)   – protein degradation rates
        beta_g        scalar – global coupling strength
        beta_l        scalar – local coupling strength
        alpha         (M,)   – kinase activity amplitudes
        kK_act        (M,)   – kinase activation rates
        kK_deact      (M,)   – kinase deactivation rates
        k_off         (N,)   – phosphatase (dephosphorylation) rates
        gamma_S_p     scalar – phosphosite→signalling coupling
        gamma_A_S     scalar – signalling→abundance coupling
        gamma_A_p     scalar – phosphosite→abundance coupling
        gamma_K_net   scalar – kinase network diffusion coupling
    """
```

```python
def compute_prev_site_idx(site_prot_idx: np.ndarray, N: int) -> np.ndarray:
    """
    Build the predecessor-site index array for the sequential mechanism.

    For each site i, stores the index j < i of the most recently seen
    site on the same protein in the flat site ordering, or -1 if i is
    the first site of that protein.

    Pure NumPy — called once at setup time, not inside the JAX trace.

    Parameters
    ----------
    site_prot_idx : np.ndarray, shape (N,)  – integer protein index per site
    N             : int                     – number of sites

    Returns
    -------
    prev_site_idx : np.ndarray, shape (N,), dtype int32
    """
```

```python
def make_rhs(
    K: int,
    M: int,
    N: int,
    mechanism: str,
    k_act_fn=None,
    s_prod_fn=None,
    rna_relax: float = 0.1,
    abundance_max: float = 5.0,
) -> callable:
    """
    Return a JAX-compatible RHS function for diffrax.ODETerm.

    Parameters
    ----------
    K, M, N     : int    – model dimensions (proteins, kinases, sites)
    mechanism   : str    – "dist", "seq", or "rand"
    k_act_fn    : callable | None  – JAX closure k_act_fn(t) -> (K,)
    s_prod_fn   : callable | None  – JAX closure s_prod_fn(t) -> (K,)
    rna_relax   : float  – relaxation rate for mRNA state toward k_act
    abundance_max : float – upper clip for protein abundance state A

    Returns
    -------
    rhs : callable  – signature rhs(t, y, args) -> dy/dt
    """
```

```python
def compute_objectives_jax(
    theta,
    P_data, P_sim,
    A_scaled, A_sim,
    W_data, W_data_prot,
    prot_idx_for_A,
    L_alpha,
    lambda_net: float,
    reg_lambda: float,
    n_p: int, n_A: int, n_var: int,
    K: int, M: int, N: int,
) -> tuple:
    """
    Compute the three objective components (f1, f2, f3) as JAX scalars.

    f1 : phosphosite weighted MSLE
    f2 : protein abundance weighted MSLE
    f3 : L2 regularisation + Laplacian network term
    """
```

## Known Limitations

- The sequential mechanism requires that `compute_prev_site_idx` has at least one predecessor index ≥ 0; proteins with only one site are treated as having no predecessor (`prev_site_idx[i] = -1`) and their first site uses a gate of 1.0.
- `k_act_fn` and `s_prod_fn` must be JAX-traceable if passed into the RHS; standard Python/NumPy callables will cause tracing errors. Use the factories in [`derived_rates`](derived_rates.md).
- The `abundance_max` parameter (default 5.0) hard-clips `A` in the RHS; values in data should be pre-scaled to the `[0, 5]` range.
