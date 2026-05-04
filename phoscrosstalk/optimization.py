"""
optimization.py
Optimistix-based objective functions and parameter fitting for the phospho-network.

Replaces the previous pymoo / ElementwiseProblem backend with:
  - A JAX-differentiable scalarized loss (w1*f1 + w2*f2 + w3*f3 + w4*f4)
  - Optimistix BFGS minimiser for gradient-based parameter fitting
  - A thin NetworkProblem wrapper that preserves the .simulate() interface
    used by steadystate, knockouts, sensitivity, and app modules.

Loss components:
  f1 : phosphosite occupancy loss
  f2 : protein abundance loss
  f3 : regularisation (L2 + Laplacian)
  f4 : mRNA / R_rna state loss (zero when no RNA data provided)
"""

import diffrax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from numba import njit

from phoscrosstalk.config import ModelDims
from phoscrosstalk.core_mechanisms import decode_theta
from phoscrosstalk.jax_mechanisms import (
    compute_objectives_jax,
    compute_prev_site_idx,
    make_rhs,
)
from phoscrosstalk.simulation import build_full_A0, simulate_ode

# ---------------------------------------------------------------------------
# Numba helper kept for non-differentiable analysis paths
# (bio_score used for post-fit reporting in analysis.py)
# ---------------------------------------------------------------------------


@njit(cache=True)
def bio_score_nb(theta, K, M, N):
    """
    Numba-compiled kernel to calculate the Biological Plausibility Score.

    Derives the half-lives (t_half = ln(2)/k) for kinases and proteins from the parameter
    vector theta and penalizes deviations from expected biological time scales.

    Args:
        theta (np.ndarray): Parameter vector (length 2*K+2+3*M+N+4).
        K, M, N (int): Model dimensions.

    Returns:
        float: The calculated biological score (lower is better/more plausible).
    """
    (k_deact, d_deg, _, _, _, kK_act, kK_deact, _, _, _, _, _) = decode_theta(
        theta, K, M, N
    )
    t_half_kinase = np.log(2.0) / kK_deact
    t_half_protein = np.log(2.0) / d_deg

    median_t_kinase = np.sort(t_half_kinase)[len(t_half_kinase) // 2]
    median_t_protein = np.sort(t_half_protein)[len(t_half_protein) // 2]

    return (np.log10(median_t_kinase) - np.log10(10.0)) ** 2 + (
        np.log10(median_t_protein) - np.log10(600.0)
    ) ** 2


def bio_score(theta):
    """
    Wrapper function to calculate the biological plausibility score for a parameter set.

    Args:
        theta (np.ndarray): Parameter vector.

    Returns:
        float: Biological score.
    """
    return float(bio_score_nb(theta, ModelDims.K, ModelDims.M, ModelDims.N))


def create_bounds(K, M, N):
    """
    Generates the lower (xl) and upper (xu) bound vectors for the optimization search space.

    ``k_act`` and ``s_prod`` are no longer optimisation variables.
    The dimension is now ``2*K + 2 + 3*M + N + 4``.

    Args:
        K, M, N (int): Model dimensions.

    Returns:
        tuple: (xl, xu, dim)
    """
    dim = 2 * K + 2 + 3 * M + N + 4
    xl, xu = np.zeros(dim), np.zeros(dim)
    idx = 0
    # Protein: k_deact, d_deg (k_act and s_prod removed – derived from data)
    # k_deact
    xl[idx : idx + K] = np.log(1e-5)
    xu[idx : idx + K] = np.log(10.0)
    idx += K
    # d_deg (restricted upper bound for biological plausibility)
    xl[idx : idx + K] = np.log(1e-5)
    xu[idx : idx + K] = np.log(0.5)
    idx += K
    # Coupling
    xl[idx] = np.log(1e-5)
    xu[idx] = np.log(10.0)
    idx += 1
    xl[idx] = np.log(1e-5)
    xu[idx] = np.log(10.0)
    idx += 1
    # Kinase: alpha, kK_act, kK_deact
    xl[idx : idx + M] = np.log(1e-5)
    xu[idx : idx + M] = np.log(10.0)
    idx += M
    xl[idx : idx + M] = np.log(1e-5)
    xu[idx : idx + M] = np.log(3.0)
    idx += M
    xl[idx : idx + M] = np.log(1e-5)
    xu[idx : idx + M] = np.log(3.0)
    idx += M
    # Site: k_off
    xl[idx : idx + N] = np.log(1e-5)
    xu[idx : idx + N] = np.log(5.0)
    idx += N
    # Gammas (tanh raw)
    xl[idx : idx + 4] = -3.0
    xu[idx : idx + 4] = 3.0
    idx += 4
    return xl, xu, dim


# ---------------------------------------------------------------------------
# Scalarized JAX loss for Optimistix
# ---------------------------------------------------------------------------


def make_loss_fn(
    t,
    P_data,
    A_scaled,
    prot_idx_for_A,
    W_data,
    W_data_prot,
    Cg,
    Cl,
    site_prot_idx,
    K_site_kin,
    R,
    L_alpha,
    kin_to_prot_idx,
    receptor_mask_prot,
    receptor_mask_kin,
    mechanism,
    lambda_net,
    reg_lambda,
    w_phospho=1.0,
    w_abundance=1.0,
    w_reg=1.0,
    rtol=1e-6,
    atol=1e-9,
    max_steps=16384,
    k_act_fn=None,
    s_prod_fn=None,
    t_mrna=None,
    rna_data_scaled=None,
    w_mrna=1.0,
    rna_model_prot_idx=None,
    R_data0=None,
):
    """
    Build a JAX-differentiable scalarized loss function for Optimistix.

    The returned function ``loss_fn(theta, args)`` is compatible with
    ``optimistix.minimise``. It:
      1. Runs diffrax.diffeqsolve inside the loss (over a unified time grid).
      2. Computes f1 (phosphosite), f2 (abundance), f3 (regularisation).
      3. Optionally computes f4 (mRNA/R_rna) when *t_mrna* and *rna_data_scaled*
         are provided.
      4. Returns the weighted total loss plus (f1, f2, f3, f4) as auxiliary output.

    Parameters are frozen at creation time; only ``theta`` varies.

    State layout: y = [R_rna, S, A, Kdyn, p]  (dim = 3*K + M + N)
    """
    K, M, N = ModelDims.K, ModelDims.M, ModelDims.N

    n_p = max(1, P_data.size)
    n_A = max(1, A_scaled.size)
    n_var = 2 * K + 2 + 3 * M + N + 4

    prev_site_idx = compute_prev_site_idx(site_prot_idx.astype(np.int32), N)

    # Build unified time grid (protein ∪ mRNA)
    if t_mrna is not None and len(t_mrna) > 0:
        all_times = np.union1d(t, t_mrna)
    else:
        all_times = np.unique(t)
    all_times = np.sort(all_times).astype(np.float64)

    # Index maps: where in solver output do the protein / mRNA times land?
    prot_time_idx = np.searchsorted(all_times, t)
    if t_mrna is not None and len(t_mrna) > 0:
        mrna_time_idx = np.searchsorted(all_times, t_mrna)
    else:
        mrna_time_idx = None

    # Build initial state from data (new layout: [R_rna, S, A, Kdyn, p])
    T_prot = P_data.shape[1]
    A0_full = build_full_A0(K, T_prot, A_scaled, prot_idx_for_A)

    x0 = np.zeros(3 * K + M + N, dtype=np.float64)
    # R_rna initial condition
    if R_data0 is not None:
        r_data = np.asarray(R_data0, dtype=np.float64)
        if r_data.ndim == 1:
            r0 = r_data.copy()
        else:
            r0 = r_data[:, 0].copy()
        r0 = np.nan_to_num(r0, nan=1.0, posinf=5.0, neginf=0.0)
        x0[:K] = np.clip(r0, 0.0, 10.0)
    else:
        x0[:K] = 1.0  # default fold-change = 1.0
    # A initial condition
    a0 = np.nan_to_num(A0_full[:, 0], nan=1.0, posinf=5.0, neginf=0.0)
    x0[2 * K : 3 * K] = np.clip(a0, 0.0, 5.0)
    # p initial condition
    p0 = np.nan_to_num(P_data[:, 0], nan=0.0, posinf=1.0, neginf=0.0)
    x0[3 * K + M :] = np.clip(p0, 0.0, 1.0)

    # JAX static arrays
    Cg_j = jnp.asarray(Cg, dtype=jnp.float32)
    Cl_j = jnp.asarray(Cl, dtype=jnp.float32)
    K_sk_j = jnp.asarray(K_site_kin, dtype=jnp.float32)
    R_j = jnp.asarray(R, dtype=jnp.float32)
    La_j = jnp.asarray(L_alpha, dtype=jnp.float32)
    spi_j = jnp.asarray(site_prot_idx, dtype=jnp.int32)
    k2p_j = jnp.asarray(kin_to_prot_idx, dtype=jnp.int32)
    rmp_j = jnp.asarray(receptor_mask_prot, dtype=jnp.float32)
    rmk_j = jnp.asarray(receptor_mask_kin, dtype=jnp.float32)
    psi_j = jnp.asarray(prev_site_idx, dtype=jnp.int32)

    y0_j = jnp.asarray(x0, dtype=jnp.float32)
    t_eval = jnp.asarray(all_times, dtype=jnp.float32)

    P_data_j = jnp.asarray(P_data, dtype=jnp.float32)
    A_scaled_j = jnp.asarray(A_scaled, dtype=jnp.float32)
    W_data_j = jnp.asarray(W_data, dtype=jnp.float32)
    W_prot_j = jnp.asarray(W_data_prot, dtype=jnp.float32)
    prot_idx_j = jnp.asarray(prot_idx_for_A, dtype=jnp.int32)
    La_loss_j = jnp.asarray(L_alpha, dtype=jnp.float32)

    prot_idx_solver = jnp.asarray(prot_time_idx, dtype=jnp.int32)

    # mRNA arrays (if available)
    has_mrna = (
        t_mrna is not None
        and rna_data_scaled is not None
        and len(t_mrna) > 0
        and mrna_time_idx is not None
        and rna_model_prot_idx is not None
        and len(rna_model_prot_idx) > 0
    )
    if has_mrna:
        rna_j = jnp.asarray(rna_data_scaled, dtype=jnp.float32)  # (n_match, T_rna)
        mrna_idx_j = jnp.asarray(mrna_time_idx, dtype=jnp.int32)
        rna_prot_idx_j = jnp.asarray(rna_model_prot_idx, dtype=jnp.int32)
        n_rna = max(1, rna_data_scaled.size)
    else:
        rna_j = None
        mrna_idx_j = None
        rna_prot_idx_j = None
        n_rna = 1

    rhs_fn = make_rhs(K, M, N, mechanism, k_act_fn=k_act_fn, s_prod_fn=s_prod_fn)
    term = diffrax.ODETerm(rhs_fn)
    solver = diffrax.Tsit5()
    sctrl = diffrax.PIDController(rtol=rtol, atol=atol)
    saveat = diffrax.SaveAt(ts=t_eval)

    t0_val = float(all_times[0])
    t1_val = float(all_times[-1])

    FAILED_SOLVE_PENALTY = jnp.float32(1e6)

    def loss_fn(theta, _args):
        theta_j = jnp.asarray(theta, dtype=jnp.float32)

        ode_args = (
            theta_j,
            Cg_j,
            Cl_j,
            spi_j,
            K_sk_j,
            R_j,
            La_j,
            k2p_j,
            rmp_j,
            rmk_j,
            psi_j,
        )

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0_val,
            t1=t1_val,
            dt0=0.01,
            y0=y0_j,
            args=ode_args,
            saveat=saveat,
            stepsize_controller=sctrl,
            max_steps=max_steps,
            throw=False,
        )

        xs = sol.ys  # (T_unified, 3K+M+N) – new state layout

        # Sample at protein time indices
        xs_prot = xs[prot_idx_solver, :]
        # New slicing: [R_rna, S, A, Kdyn, p]
        P_sim = jnp.clip(xs_prot[:, 3 * K + M :], 0.0, 1.0).T  # (N, T_prot)
        A_sim = jnp.clip(xs_prot[:, 2 * K : 3 * K], 0.0, 5.0).T  # (K, T_prot)

        f1, f2, f3 = compute_objectives_jax(
            theta_j,
            P_data_j,
            P_sim,
            A_scaled_j,
            A_sim,
            W_data_j,
            W_prot_j,
            prot_idx_j,
            La_loss_j,
            lambda_net,
            reg_lambda,
            n_p,
            n_A,
            n_var,
            K,
            M,
            N,
        )

        # f4: mRNA / R_rna loss (only when RNA data is available)
        if has_mrna:
            xs_rna = xs[mrna_idx_j, :]
            R_sim_rna = jnp.clip(xs_rna[:, :K], 0.0, None).T  # (K, T_rna)
            R_sim_matched = R_sim_rna[rna_prot_idx_j, :]  # (n_match, T_rna)
            diff_R = rna_j - R_sim_matched
            f4 = jnp.sum(jnp.log1p(diff_R * diff_R)) / n_rna
        else:
            f4 = jnp.float32(0.0)

        total = (
            jnp.float32(w_phospho) * f1
            + jnp.float32(w_abundance) * f2
            + jnp.float32(w_reg) * f3
            + jnp.float32(w_mrna) * f4
        )

        # Penalise non-finite results without crashing
        total = jnp.where(jnp.isfinite(total), total, FAILED_SOLVE_PENALTY)
        return total, (f1, f2, f3, f4)

    return loss_fn


def run_single_optimisation(
    loss_fn,
    theta0,
    max_steps=256,
):
    """
    Run a single Optimistix BFGS minimisation from starting point theta0.

    Parameters
    ----------
    loss_fn  : callable (theta, args) -> (scalar, (f1, f2, f3, f4))
    theta0   : np.ndarray
    max_steps: int

    Returns
    -------
    theta_opt : np.ndarray
    total_loss: float
    f1, f2, f3, f4: float
    """
    solver = optx.BFGS(rtol=1e-5, atol=1e-7)
    sol = optx.minimise(
        loss_fn,
        solver,
        jnp.asarray(theta0, dtype=jnp.float32),
        args=None,
        has_aux=True,
        max_steps=max_steps,
        throw=False,
    )
    theta_opt = np.asarray(sol.value, dtype=np.float64)
    total_loss, (f1, f2, f3, f4) = loss_fn(sol.value, None)
    return theta_opt, float(total_loss), float(f1), float(f2), float(f3), float(f4)


# ---------------------------------------------------------------------------
# Thin problem wrapper (keeps .simulate() for downstream modules)
# ---------------------------------------------------------------------------


class NetworkProblem:
    """
    Minimal problem wrapper that preserves the .simulate() interface used by
    steadystate, knockouts, sensitivity, and app modules.

    Does NOT inherit from pymoo.  The _evaluate / optimisation logic has moved
    to make_loss_fn + run_single_optimisation.

    RNA-related attributes (optional):
        t_rna             : np.ndarray | None – RNA time points
        rna_obs_matched   : np.ndarray | None – observed RNA for matched proteins (n_match, T_rna)
        rna_model_prot_idx: np.ndarray | None – protein indices for matched RNA rows
        rna_obs_idx       : np.ndarray | None – gene indices in gene_ids for matched rows
        rna_fit_genes     : list | None       – matched gene/protein names
        loss_weight_rna   : float             – RNA loss weight
        R_data0           : np.ndarray | None – RNA initial condition (K, T_rna or K,)
    """

    def __init__(
        self,
        t,
        P_data,
        Cg,
        Cl,
        site_prot_idx,
        K_site_kin,
        R,
        A_scaled,
        prot_idx_for_A,
        W_data,
        W_data_prot,
        L_alpha,
        kin_to_prot_idx,
        lambda_net,
        reg_lambda,
        receptor_mask_prot,
        receptor_mask_kin,
        mechanism,
        xl,
        xu,
        k_act_fn=None,
        s_prod_fn=None,
        t_rna=None,
        rna_obs_matched=None,
        rna_model_prot_idx=None,
        rna_obs_idx=None,
        rna_fit_genes=None,
        loss_weight_rna=1.0,
        R_data0=None,
        **kwargs,  # absorb legacy keyword args (elementwise_runner, etc.)
    ):
        self.t = t
        self.P_data = P_data
        self.Cg = Cg
        self.Cl = Cl
        self.site_prot_idx = site_prot_idx
        self.K_site_kin = K_site_kin
        self.R = R
        self.A_scaled = A_scaled
        self.prot_idx_for_A = prot_idx_for_A
        self.W_data = W_data
        self.W_data_prot = W_data_prot
        self.L_alpha = L_alpha
        self.kin_to_prot_idx = kin_to_prot_idx
        self.lambda_net = lambda_net
        self.reg_lambda = reg_lambda
        self.receptor_mask_prot = receptor_mask_prot
        self.receptor_mask_kin = receptor_mask_kin
        self.mechanism = mechanism
        self.xl = xl
        self.xu = xu
        self.k_act_fn = k_act_fn
        self.s_prod_fn = s_prod_fn
        # RNA-specific
        self.t_rna = t_rna
        self.rna_obs_matched = rna_obs_matched
        self.rna_model_prot_idx = rna_model_prot_idx
        self.rna_obs_idx = rna_obs_idx
        self.rna_fit_genes = rna_fit_genes
        self.loss_weight_rna = loss_weight_rna
        self.R_data0 = R_data0

    def simulate(self, x):
        """
        Run a simulation for parameter vector x and return phosphosite trajectories.

        Args:
            x (np.ndarray): Parameter vector.

        Returns:
            np.ndarray: P_sim (N_sites x T).
        """
        theta = np.asarray(x, dtype=np.float64)
        K, T = ModelDims.K, self.P_data.shape[1]
        A0 = build_full_A0(K, T, self.A_scaled, self.prot_idx_for_A)

        P_sim, _A_sim = simulate_ode(
            self.t,
            self.P_data,
            A0,
            theta,
            self.Cg,
            self.Cl,
            self.site_prot_idx,
            self.K_site_kin,
            self.R,
            self.L_alpha,
            self.kin_to_prot_idx,
            self.receptor_mask_prot,
            self.receptor_mask_kin,
            self.mechanism,
            k_act_fn=self.k_act_fn,
            s_prod_fn=self.s_prod_fn,
            R_data0=self.R_data0,
        )
        return P_sim

    def simulate_full(self, x):
        """
        Run a full simulation returning all state components including R_sim_rna.

        Args:
            x (np.ndarray): Parameter vector.

        Returns:
            dict: Keys: P_sim, A_sim, S_sim, Kdyn_sim, R_sim, R_sim_rna, t, t_rna,
                solver_times.
        """
        theta = np.asarray(x, dtype=np.float64)
        K, T = ModelDims.K, self.P_data.shape[1]
        A0 = build_full_A0(K, T, self.A_scaled, self.prot_idx_for_A)

        return simulate_ode(
            self.t,
            self.P_data,
            A0,
            theta,
            self.Cg,
            self.Cl,
            self.site_prot_idx,
            self.K_site_kin,
            self.R,
            self.L_alpha,
            self.kin_to_prot_idx,
            self.receptor_mask_prot,
            self.receptor_mask_kin,
            self.mechanism,
            return_full=True,
            k_act_fn=self.k_act_fn,
            s_prod_fn=self.s_prod_fn,
            t_rna=self.t_rna,
            R_data0=self.R_data0,
        )


# Legacy alias so that any remaining code that imports NetworkOptimizationProblem
# still works without crashing.
NetworkOptimizationProblem = NetworkProblem
