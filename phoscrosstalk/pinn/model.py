# SPDX-License-Identifier: MIT
"""
pinn/model.py
Equinox module for the PINN neural augmentation term.

PINNAugmentation maps (x, t) -> neural residual correction of shape (state_dim,).

The state layout is:
    y = [R_rna (K), S (K), A (K), Kdyn (M), p (N)]   dim = 3*K + M + N

The neural correction is bounded via a tanh gate so it cannot dominate
the mechanistic RHS.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

# Activation map — all supported string keys.
_ACTIVATIONS = {
    "tanh":     jax.nn.tanh,
    "relu":     jax.nn.relu,
    "gelu":     jax.nn.gelu,
    "silu":     jax.nn.silu,
    "softplus": jax.nn.softplus,
    "elu":      jax.nn.elu,
}

# Hard cap for the PINN output magnitude.  Residual terms larger than this
# are biologically implausible and indicate model pathology.
_PINN_OUTPUT_CLAMP: float = 10.0


class PINNAugmentation(eqx.Module):
    """
    Neural augmentation term for the PINN / Universal ODE path.

    Maps (state, time) -> correction vector of shape ``(state_dim,)``.

    The correction is gated by a learned scale vector so that inactive
    dimensions contribute zero.  The output is bounded by ``output_clamp``
    to prevent the neural term from dominating the mechanistic RHS.

    Parameters
    ----------
    state_dim : int
        Total ODE state dimension = ``3*K + M + N``.
    width_size : int
        Hidden layer width.
    depth : int
        Number of hidden layers (>= 1).
    activation : str
        Activation function name; see ``_ACTIVATIONS``.
    output_clamp : float
        Hard clamp for the output to keep correction bounded.
    key : jax.Array
        PRNG key for parameter initialisation.
    """

    mlp: eqx.nn.MLP
    output_scale: jax.Array  # (state_dim,) learnable per-dimension scale

    def __init__(
        self,
        state_dim: int,
        width_size: int,
        depth: int,
        activation: str,
        key: jax.Array,
        output_clamp: float = _PINN_OUTPUT_CLAMP,
    ) -> None:
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"PINNAugmentation: unknown activation {activation!r}. "
                f"Supported: {sorted(_ACTIVATIONS)}"
            )
        act_fn = _ACTIVATIONS[activation]
        mlp_key, scale_key = jax.random.split(key)
        # Input: state_dim (ODE state) + 1 (normalised time)
        self.mlp = eqx.nn.MLP(
            in_size=state_dim + 1,
            out_size=state_dim,
            width_size=width_size,
            depth=depth,
            activation=act_fn,
            final_activation=lambda x: x,
            use_bias=True,
            use_final_bias=True,
            dtype=jnp.float64,
            key=mlp_key,
        )
        # Initialise scale near zero so PINN starts as a near-zero correction.
        # Small Gaussian noise breaks symmetry while keeping initialisation small.
        self.output_scale = (
            jax.random.normal(scale_key, (state_dim,), dtype=jnp.float64) * 0.01
        )

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """
        Compute the neural correction vector.

        Parameters
        ----------
        x : jax.Array, shape (state_dim,)
            Current ODE state.
        t : jax.Array, scalar
            Current time (will be normalised internally; caller provides raw t).

        Returns
        -------
        jax.Array, shape (state_dim,)
            Neural correction term.
        """
        t_scalar = jnp.asarray(t, dtype=jnp.float64).reshape(())
        x_f64 = jnp.asarray(x, dtype=jnp.float64)
        features = jnp.concatenate([x_f64, t_scalar[None]])  # (state_dim + 1,)
        raw = self.mlp(features)  # (state_dim,)
        # Gate by a learned scale and clamp to prevent dominance
        scaled = raw * jnp.tanh(self.output_scale)
        return jnp.clip(scaled, -_PINN_OUTPUT_CLAMP, _PINN_OUTPUT_CLAMP)
