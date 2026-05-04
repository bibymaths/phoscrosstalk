# Model

## Overview

PhosCrosstalk integrates five coupled ODE state variables into a single
network-level model:

```
y = [R, S, A, Kdyn, p]

R(t)    – mRNA levels                    shape (K,)
S(t)    – protein activation state       shape (K,)
A(t)    – protein abundance              shape (K,)
Kdyn(t) – kinase activity                shape (M,)
p(t)    – phosphosite occupancy          shape (N,)

Full dimension: 3K + M + N
```

where **K** = number of model proteins, **M** = number of kinases,
**N** = number of phosphosites.

## ODE backend

The ODE system is solved with [Diffrax](https://github.com/patrick-kidger/diffrax)
using the Tsit5 solver with PID-controller adaptive stepping (JAX backend).

The right-hand side (RHS) is provided by `jax_mechanisms.make_rhs`.

## Derived rates

Two rates are computed from data and **not** optimized:

| Rate         | Source                                    |
|--------------|-------------------------------------------|
| `k_act(t)`   | Derived from TF/mRNA regulatory signal    |
| `s_prod(t)`  | Derived from kinase/protein signals       |

If `--rna-data` and `--tf-net` are absent, `k_act` defaults to a constant
vector of 1.0. If kinase/protein signals are absent, `s_prod` defaults to 0.1.

## Parameter vector

The optimized parameter vector `theta` has dimension `2K + 2 + 3M + N + 4`:

| Parameter group | Symbol         | Size | Description                         |
|-----------------|----------------|------|-------------------------------------|
| Protein deactivation | `k_deact` | K   | Per-protein deactivation rate        |
| Protein degradation  | `d_deg`   | K   | Per-protein degradation rate         |
| Global coupling      | `beta_g`  | 1   | Global PTM crosstalk coupling        |
| Local coupling       | `beta_l`  | 1   | Sequence-proximity coupling          |
| Kinase strength      | `alpha`   | M   | Per-kinase global strength           |
| Kinase activation    | `kK_act`  | M   | Per-kinase activation rate           |
| Kinase deactivation  | `kK_deact`| M   | Per-kinase deactivation rate         |
| Phosphatase rate     | `k_off`   | N   | Per-site dephosphorylation rate      |
| Coupling terms       | `gamma_*` | 4   | State-coupling coefficients          |

!!! note "Derived rates"
    `k_act` and `s_prod` are **not** part of `theta`. They are closures
    computed from input data and fixed before optimization.

## Phosphorylation mechanisms

Three kinase mechanisms are implemented in `jax_mechanisms.py`:

| Mechanism | Flag     | Description                                                |
|-----------|----------|------------------------------------------------------------|
| Distributive | `dist`| Each kinase acts independently on each site               |
| Sequential   | `seq` | Ordered site modification; prior sites affect subsequent  |
| Random/Cooperative | `rand` | Random-order kinase action with cooperative effects |

## Scalar loss

```
total_loss =
    w_phospho   × MSE(p_sim, p_obs)
  + w_abundance × MSE(A_sim, A_obs)
  + w_rna       × MSE(R_sim, R_obs)     # only when --rna-data is provided
  + w_reg       × (L2_reg + Laplacian_reg)
```

The best solution across all multi-starts is selected by minimum `total_loss`.

## PTM crosstalk

The global coupling matrix **Cg** is built from PTMcode2 inter- and
intra-protein site-pair associations loaded from the provided SQLite databases.
The local matrix **Cl** encodes sequence proximity: sites on the same protein
within a distance threshold (controlled by `length_scale`) are coupled
exponentially.

## Kinase-site priors

The kinase-site matrix **K\_site\_kin** links each phosphosite to the kinases
that phosphorylate it. It is loaded from `--kinase-tsv` (Site/Kinase/weight
columns) or `--kea-ks-table` (KEA citation-count weighted).
