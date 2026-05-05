import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def frechet_distance(true_coords, pred_coords):
    """
    Compute the discrete Fréchet distance between two curves using JAX.

    Args:
        true_coords: array of shape (n, d)
        pred_coords: array of shape (m, d)

    Returns:
        Scalar JAX array containing the discrete Fréchet distance.
    """
    true_coords = jnp.asarray(true_coords, dtype=jnp.float64)
    pred_coords = jnp.asarray(pred_coords, dtype=jnp.float64)

    # Pairwise Euclidean distances, shape (n, m)
    dist = jnp.linalg.norm(
        true_coords[:, None, :] - pred_coords[None, :, :],
        axis=-1,
    )

    n, m = dist.shape

    # First cell
    cost00 = dist[0, 0]

    # First column:
    # cost[i, 0] = max(cost[i - 1, 0], dist[i, 0])
    first_col = lax.associative_scan(jnp.maximum, dist[:, 0])
    first_col = first_col.at[0].set(cost00)

    # First row:
    # cost[0, j] = max(cost[0, j - 1], dist[0, j])
    first_row = lax.associative_scan(jnp.maximum, dist[0, :])
    first_row = first_row.at[0].set(cost00)

    # Initialize DP matrix with first row and first column.
    cost = jnp.full((n, m), jnp.inf, dtype=dist.dtype)
    cost = cost.at[:, 0].set(first_col)
    cost = cost.at[0, :].set(first_row)

    def row_step(i, cost):
        def col_step(j, cost):
            prev_min = jnp.minimum(
                jnp.minimum(cost[i - 1, j], cost[i, j - 1]),
                cost[i - 1, j - 1],
            )
            value = jnp.maximum(prev_min, dist[i, j])
            return cost.at[i, j].set(value)

        cost = lax.fori_loop(1, m, col_step, cost)
        return cost

    cost = lax.fori_loop(1, n, row_step, cost)

    return cost[-1, -1]