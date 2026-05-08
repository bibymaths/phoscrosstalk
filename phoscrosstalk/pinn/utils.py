# SPDX-License-Identifier: MIT
"""
pinn/utils.py
Small utility helpers for the PINN subpackage.

Contains:
  * state_labels(K, M, N, proteins, kinases, sites) – human-readable state labels.
  * safe_to_numpy(x) – JAX array → NumPy conversion with dtype preservation.
  * pinn_param_count(model) – count trainable float64 parameters in a PyTree.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx


def state_labels(
    K: int,
    M: int,
    N: int,
    proteins: list[str] | None = None,
    kinases: list[str] | None = None,
    sites: list[str] | None = None,
) -> list[str]:
    """
    Return human-readable state labels for the full ODE state vector.

    Layout: [R_rna (K), S (K), A (K), Kdyn (M), p (N)]

    Parameters
    ----------
    K, M, N : int
        Model dimensions.
    proteins, kinases, sites : list[str] | None
        Entity names.  If absent, generic indices are used.

    Returns
    -------
    list[str]  length = 3*K + M + N
    """
    _prot = proteins if proteins and len(proteins) == K else [str(k) for k in range(K)]
    _kin  = kinases  if kinases  and len(kinases)  == M else [str(m) for m in range(M)]
    _site = sites    if sites    and len(sites)    == N else [str(n) for n in range(N)]

    labels: list[str] = []
    for p in _prot:
        labels.append(f"R_rna:{p}")
    for p in _prot:
        labels.append(f"S:{p}")
    for p in _prot:
        labels.append(f"A:{p}")
    for k in _kin:
        labels.append(f"Kdyn:{k}")
    for s in _site:
        labels.append(f"p:{s}")
    return labels


def safe_to_numpy(x) -> np.ndarray:
    """Convert a JAX array (or anything array-like) to a NumPy array."""
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        return np.asarray(jnp.asarray(x))


def pinn_param_count(model) -> int:
    """
    Count the number of trainable float64 leaf parameters in an Equinox model.

    Parameters
    ----------
    model : eqx.Module
        Any Equinox module.

    Returns
    -------
    int  total number of float64 scalar parameters.
    """
    leaves, _ = jax.tree_util.tree_flatten(
        eqx.filter(model, eqx.is_array)
    )
    return int(sum(int(np.prod(np.asarray(l.shape))) for l in leaves if l is not None))
