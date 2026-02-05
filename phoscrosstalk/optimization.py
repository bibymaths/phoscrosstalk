"""
optimization.py
Pymoo Problem definition, objective functions, and parameter bounds.
"""
import numpy as np
from numba import njit
from pymoo.core.problem import ElementwiseProblem
from phoscrosstalk.config import ModelDims
from phoscrosstalk.simulation import simulate_p_scipy
from phoscrosstalk.core_mechanisms import decode_theta


@njit(cache=True)
def compute_objectives_nb(theta, P_data, P_sim, A_scaled, A_sim, W_data, W_data_prot,
                          prot_idx_for_A, L_alpha, lambda_net, reg_lambda, n_p, n_A, n_var, K, M, N):
    (k_act, k_deact, s_prod, d_deg, beta_g, beta_l, alpha, kK_act, kK_deact, k_off,
     gamma_S_p, gamma_A_S, gamma_A_p, gamma_K_net) = decode_theta(theta, K, M, N)

    # 1. Phosphosite loss (Shape/Kinetic)
    N_sites, T = P_data.shape
    loss_p = 0.0
    for i in range(N_sites):
        # offset = P_sim[i, 0] - P_data[i, 0] # Option to shift
        for j in range(T):
            diff = (P_data[i, j] - P_sim[i, j])
            loss_p += np.log1p(W_data[i, j] * (diff * diff))
    f1 = loss_p / n_p

    # 2. Protein loss
    loss_A = 0.0
    if A_scaled.size > 0:
        K_prot_obs, T_A = A_scaled.shape
        for k in range(K_prot_obs):
            p_idx = prot_idx_for_A[k]
            for j in range(T_A):
                diffA = (A_scaled[k, j] - A_sim[p_idx, j])
                loss_A += np.log1p(W_data_prot[k, j] * (diffA * diffA))
        f2 = loss_A / n_A
    else:
        f2 = 0.0

    # 3. Regularization
    reg = reg_lambda * np.dot(theta, theta)
    reg_net = 0.0
    M_kin = alpha.shape[0]
    if lambda_net > 0.0:
        for i in range(M_kin):
            row_sum = 0.0
            for j in range(M_kin):
                row_sum += L_alpha[i, j] * alpha[j]
            reg_net += alpha[i] * row_sum
        reg_net *= lambda_net

    f3 = (reg + reg_net) / n_var
    return f1, f2, f3


@njit(cache=True)
def bio_score_nb(theta, K, M, N):
    (k_act, k_deact, s_prod, d_deg, _, _, _, kK_act, kK_deact, _, _, _, _, _) = decode_theta(theta, K, M, N)
    t_half_kinase = np.log(2.0) / kK_deact
    t_half_protein = np.log(2.0) / d_deg

    median_t_kinase = np.sort(t_half_kinase)[len(t_half_kinase) // 2]
    median_t_protein = np.sort(t_half_protein)[len(t_half_protein) // 2]

    return ((np.log10(median_t_kinase) - np.log10(10.0)) ** 2 +
            (np.log10(median_t_protein) - np.log10(600.0)) ** 2)


def bio_score(theta):
    return float(bio_score_nb(theta, ModelDims.K, ModelDims.M, ModelDims.N))


def build_full_A0(K, T, A_scaled, prot_idx_for_A):
    A0_full = np.zeros((K, T), dtype=float)
    if A_scaled.size > 0:
        for k, p_idx in enumerate(prot_idx_for_A):
            A0_full[p_idx, :] = A_scaled[k, :]
    return A0_full


def create_bounds(K, M, N):
    dim = 4 * K + 2 + 3 * M + N + 4
    xl, xu = np.zeros(dim), np.zeros(dim)
    idx = 0
    # Protein: k_act, k_deact, s_prod
    for _ in range(3):
        xl[idx:idx + K] = np.log(1e-5);
        xu[idx:idx + K] = np.log(10.0);
        idx += K
    # Protein: d_deg (restricted)
    xl[idx:idx + K] = np.log(1e-5);
    xu[idx:idx + K] = np.log(0.5);
    idx += K
    # Coupling
    xl[idx] = np.log(1e-5);
    xu[idx] = np.log(10.0);
    idx += 1
    xl[idx] = np.log(1e-5);
    xu[idx] = np.log(10.0);
    idx += 1
    # Kinase: alpha, kK_act, kK_deact
    xl[idx:idx + M] = np.log(1e-5);
    xu[idx:idx + M] = np.log(10.0);
    idx += M
    xl[idx:idx + M] = np.log(1e-5);
    xu[idx:idx + M] = np.log(3.0);
    idx += M
    xl[idx:idx + M] = np.log(1e-5);
    xu[idx:idx + M] = np.log(3.0);
    idx += M
    # Site: k_off
    xl[idx:idx + N] = np.log(1e-5);
    xu[idx:idx + N] = np.log(5.0);
    idx += N
    # Gammas (tanh raw)
    xl[idx:idx + 4] = -3.0;
    xu[idx:idx + 4] = 3.0;
    idx += 4
    return xl, xu, dim


class NetworkOptimizationProblem(ElementwiseProblem):
    def __init__(self, t, P_data, Cg, Cl, site_prot_idx, K_site_kin, R,
                 A_scaled, prot_idx_for_A, W_data, W_data_prot, L_alpha, kin_to_prot_idx,
                 lambda_net, reg_lambda, receptor_mask_prot, receptor_mask_kin, mechanism,
                 xl, xu, **kwargs):
        super().__init__(n_var=len(xl), n_obj=3, n_ieq_constr=0, xl=xl, xu=xu, **kwargs)
        self.t = t;
        self.P_data = P_data;
        self.Cg = Cg;
        self.Cl = Cl
        self.site_prot_idx = site_prot_idx;
        self.K_site_kin = K_site_kin;
        self.R = R
        self.A_scaled = A_scaled;
        self.prot_idx_for_A = prot_idx_for_A
        self.W_data = W_data;
        self.W_data_prot = W_data_prot
        self.L_alpha = L_alpha;
        self.kin_to_prot_idx = kin_to_prot_idx
        self.lambda_net = lambda_net;
        self.reg_lambda = reg_lambda
        self.receptor_mask_prot = receptor_mask_prot
        self.receptor_mask_kin = receptor_mask_kin;
        self.mechanism = mechanism
        self.n_p = max(1, self.P_data.size);
        self.n_A = max(1, self.A_scaled.size);
        self.n_var = len(xl)

    def _evaluate(self, x, out, *args, **kwargs):

        theta = x
        K, T = ModelDims.K, self.P_data.shape[1]

        self._A0_full = build_full_A0(K, T, self.A_scaled, self.prot_idx_for_A)
        A0_full = self._A0_full

        P_sim, A_sim = simulate_p_scipy(
            self.t, self.P_data, A0_full, theta, self.Cg, self.Cl, self.site_prot_idx,
            self.K_site_kin, self.R, self.L_alpha, self.kin_to_prot_idx,
            self.receptor_mask_prot, self.receptor_mask_kin, self.mechanism
        )

        if (not np.all(np.isfinite(P_sim)) or not np.all(np.isfinite(A_sim)) or
                np.max(P_sim) > 5.0 or np.min(P_sim) < -1e-6):
            out["F"] = np.array([1e12, 1e12, 1e12])
            return

        f1, f2, f3 = compute_objectives_nb(
            theta, self.P_data, P_sim, self.A_scaled, A_sim, self.W_data, self.W_data_prot,
            self.prot_idx_for_A, self.L_alpha, self.lambda_net, self.reg_lambda,
            self.n_p, self.n_A, self.n_var, ModelDims.K, ModelDims.M, ModelDims.N
        )
        out["F"] = np.array([f1, f2, f3], dtype=float)

    def simulate(self, x):
        """
        Simulate site trajectories P(t) for a decision vector x.

        Returns:
            P_sim: ndarray, shape (N_sites, T), aligned to the rows of the input data.
        """
        theta = np.asarray(x, dtype=np.float64)

        # Use the actually-stored site data matrix.
        # In this codebase it's typically stored as `self.P_data` (scaled data passed from main).
        P_mat = np.asarray(self.P_data, dtype=np.float64)

        # Initial condition is the first column (t0) as a vector (N_sites,)
        P0 = np.ascontiguousarray(P_mat[:, 0], dtype=np.float64)

        K = int(self.K)
        T = int(self.t.shape[0])

        # Build full protein activity/abundance matrix (K x T)
        A0_full = build_full_A0(
            K,
            T,
            np.asarray(self.A_scaled, dtype=np.float64),
            np.asarray(self.prot_idx_for_A, dtype=np.int64),
        )

        P_sim, _A_sim = simulate_p_scipy(
            np.asarray(self.t, dtype=np.float64),
            P0,
            A0_full,
            theta,
            np.asarray(self.Cg, dtype=np.float64),
            np.asarray(self.Cl, dtype=np.float64),
            np.asarray(self.site_prot_idx, dtype=np.int64),
            np.asarray(self.K_site_kin, dtype=np.float64),
            np.asarray(self.R, dtype=np.float64),
            np.asarray(self.L_alpha, dtype=np.float64),
            np.asarray(self.kin_to_prot_idx, dtype=np.int64),
            np.asarray(self.receptor_mask_prot, dtype=np.int64),
            np.asarray(self.receptor_mask_kin, dtype=np.int64),
            self.mechanism,
        )

        return P_sim