# SPDX-License-Identifier: MIT
"""
PINN-augmented RHS factory.

The mechanistic RHS is imported and reused directly from mechanisms.py.
The PINN correction is added additively:

    dy/dt = f_mechanistic(t, y, args) + pinn_model(y, t)

The PINN model is a JAX PyTree (Equinox module) passed through the ODE args
so that gradients flow through it correctly via eqx.filter_value_and_grad.
"""

from __future__ import annotations

import jax.numpy as jnp

from phoscrosstalk.mechanisms import make_rhs


def make_combined_rhs(
        K: int,
        M: int,
        N: int,
        mechanism: str,
        *,
        k_act_fn=None,
        s_prod_fn=None,
        rna_relax: float = 0.1,
        abundance_max: float = 5.0,
):
    """
    Build a PINN-augmented ODE RHS function compatible with diffrax.ODETerm.

    The returned function has signature::

        combined_rhs(t, y, args) -> dy/dt

    where ``args`` is a tuple whose **last element** must be the
    :class:`~phoscrosstalk.pinn.model.PINNAugmentation` model instance.
    All preceding elements of ``args`` are forwarded unchanged to the
    mechanistic RHS.

    When ``pinn_model`` is ``None`` the combined RHS reduces exactly to the
    mechanistic RHS (disabled / no-op PINN path).

    Parameters
    ----------
    K, M, N : int
        Model dimensions (proteins, kinases, phosphosites).
    mechanism : str
        Phosphorylation mechanism key: ``"dist"``, ``"seq"``, or ``"rand"``.
    k_act_fn, s_prod_fn : callable | None
        Derived-rate closures forwarded to the mechanistic RHS factory.
    rna_relax : float
        RNA relaxation rate forwarded to the mechanistic RHS factory.
    abundance_max : float
        Protein abundance hard clip forwarded to the mechanistic RHS factory.

    Returns
    -------
    combined_rhs : callable
        ``(t, y, args) -> dy/dt`` where the LAST element of args is the
        PINNAugmentation model (or None for no-op).
    """
    # Build the mechanistic RHS once at factory time.
    mech_rhs = make_rhs(
        K, M, N, mechanism,
        k_act_fn=k_act_fn,
        s_prod_fn=s_prod_fn,
        rna_relax=rna_relax,
        abundance_max=abundance_max,
    )

    def combined_rhs(t, y, args):
        # args = (*mechanistic_args, pinn_model)
        mech_args = args[:-1]
        pinn_model = args[-1]

        dy_mech = mech_rhs(t, y, mech_args)

        if pinn_model is None:
            return dy_mech

        dy_pinn = pinn_model(y, t)
        return dy_mech + jnp.asarray(dy_pinn, dtype=jnp.float64)

    return combined_rhs
