import numpy as np
from numba import njit, prange
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@njit("f8(f8[:, ::1], f8[:, ::1])", fastmath=True, parallel=True)
def frechet_distance(true_coords: FloatArray, pred_coords: FloatArray) -> float:
    """
    Compute the discrete Fréchet distance between two curves.

    Args:
        true_coords (FloatArray): The coordinates of the true curve.
        pred_coords (FloatArray): The coordinates of the predicted curve.

    Returns:
        float: The discrete Fréchet distance between the two curves.
    """

    # Get dimensions
    n, m = len(true_coords), len(pred_coords)

    # Compute pairwise euclidean distance
    p = 2
    dist = np.zeros((n, m))
    for i in prange(n):
        for j in prange(m):
            # Compute Euclidean distance
            # Formula: (sum(|x_i - y_j|^p))^(1/p)
            dist[i, j] = np.sum(np.abs(true_coords[i] - pred_coords[j]) ** p) ** (
                1.0 / p
            )

    cost = np.full((n, m), np.inf)

    cost[0, 0] = dist[0, 0]

    # Compute dynamic programming matrix
    for i in range(1, n):
        # First column
        cost[i, 0] = max(cost[i - 1, 0], dist[i, 0])
    for j in range(1, m):
        # First row
        cost[0, j] = max(cost[0, j - 1], dist[0, j])
    for i in range(1, n):
        # Rest of the matrix
        for j in range(1, m):
            # Minimum of three values
            # (left, above, diagonal)
            # Formula: min(left, above, diagonal)
            cost[i, j] = max(
                min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]), dist[i, j]
            )

    return cost[-1, -1]