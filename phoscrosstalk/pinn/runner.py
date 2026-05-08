# SPDX-License-Identifier: MIT
"""
pinn/runner.py
Main PINN execution entry point.

run_pinn_pipeline(...)
    Called by main.py when pinn.enabled = true.
    Performs single-start PINN joint optimisation and saves all outputs.
    Does NOT use multistart.  Does NOT call the post-fit neuralODE workflow.
"""

from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from phoscrosstalk.config import ModelDims
from phoscrosstalk.logger import get_logger
from phoscrosstalk.optimization import create_bounds
from phoscrosstalk.pinn.config import get_pinn_cfg
from phoscrosstalk.pinn.loss import make_pinn_loss_fn
from phoscrosstalk.pinn.model import PINNAugmentation
from phoscrosstalk.pinn.outputs import save_pinn_outputs, save_pinn_residuals
from phoscrosstalk.pinn.plotting import save_pinn_plots
from phoscrosstalk.pinn.utils import pinn_param_count

logger = get_logger()
_debug_logger = logging.getLogger("phoscrosstalk.pinn.runner")

# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------

_pinn_loss_history: list[dict] = []


def _run_pinn_optax(
    *,
    loss_fn,
    trainable,
    pinn_cfg: SimpleNamespace,
) -> tuple:
    """
    Train (theta, pinn_model) with an Optax Adam loop.

    Returns
    -------
    trainable : (theta, pinn_model) – final trained values
    loss_history : list[dict]
    """
    lr          = float(getattr(pinn_cfg, "learning_rate", 1e-3))
    max_steps   = int(getattr(pinn_cfg, "max_steps",    500))
    print_every = max(1, int(getattr(pinn_cfg, "print_every", 50)))
    grad_clip   = float(getattr(pinn_cfg, "grad_clip",  1.0))

    transforms = []
    if grad_clip > 0:
        transforms.append(optax.clip_by_global_norm(grad_clip))
    transforms.append(optax.adam(learning_rate=lr))
    tx = optax.chain(*transforms)

    # Filter to differentiable array leaves only for optax state initialisation.
    # eqx.filter replaces non-array leaves (e.g. activation functions) with None
    # so that optax does not try to apply zeros_like to Python callables.
    opt_state = tx.init(eqx.filter(trainable, eqx.is_array))

    @eqx.filter_jit
    def step(params, opt_state):
        (loss, aux), grads = eqx.filter_value_and_grad(
            lambda p: loss_fn(p, None), has_aux=True
        )(params)
        # Filter grads to array leaves only (same structure as opt_state).
        grads_arrays = eqx.filter(grads, eqx.is_array)
        # Zero out non-finite gradients to prevent parameter corruption.
        grads_arrays = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g)), grads_arrays
        )
        updates, new_opt_state = tx.update(grads_arrays, opt_state)
        # eqx.apply_updates handles None updates by leaving those leaves unchanged.
        new_params = eqx.apply_updates(params, updates)
        return new_params, new_opt_state, loss, aux

    history: list[dict] = []
    t0 = time.perf_counter()

    for s in range(max_steps):
        trainable, opt_state, loss_val, aux = step(trainable, opt_state)

        if s % print_every == 0 or s == max_steps - 1:
            loss_val.block_until_ready()
            f1, f2, f3, f4, fp = tuple(float(a) for a in aux)
            row = {
                "step": s,
                "total_loss": float(loss_val),
                "f1": f1, "f2": f2, "f3": f3, "f4": f4,
                "f_pinn_reg": fp,
                "elapsed_s": time.perf_counter() - t0,
            }
            history.append(row)
            _debug_logger.info(
                "[pinn] step=%04d  total=%.4e  f1=%.4e  f2=%.4e  "
                "f3=%.4e  f4=%.4e  f_pinn=%.4e",
                s, float(loss_val), f1, f2, f3, f4, fp,
            )

    return trainable, history


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_pinn_pipeline(
    cfg: SimpleNamespace,
    *,
    dims: ModelDims,
    t: np.ndarray,
    P_scaled: np.ndarray,
    A_scaled: np.ndarray,
    prot_idx_for_A: np.ndarray,
    W_data: np.ndarray,
    W_data_prot: np.ndarray,
    Cg: np.ndarray,
    Cl: np.ndarray,
    site_prot_idx: np.ndarray,
    K_site_kin: np.ndarray,
    R: np.ndarray,
    L_alpha: np.ndarray,
    kin_to_prot_idx: np.ndarray,
    receptor_mask_prot: np.ndarray,
    receptor_mask_kin: np.ndarray,
    mechanism: str,
    k_act_fn=None,
    s_prod_fn=None,
    t_rna=None,
    rna_obs_matched=None,
    rna_model_prot_idx=None,
    rna_fit_genes=None,
    R_data0=None,
    W_data_mrna=None,
    outdir: str = "results",
    proteins: list | None = None,
    kinases: list | None = None,
    sites: list | None = None,
) -> dict:
    """
    Run the single-start PINN / Universal ODE optimisation.

    This function:
    1. Builds PINN model and initial theta.
    2. Constructs the PINN loss function (reusing residual/loss infrastructure).
    3. Runs a single Optax training loop — no multistart.
    4. Saves all required outputs.
    5. Returns a result dict for further pipeline use.

    Parameters
    ----------
    cfg : SimpleNamespace
        Fully loaded config from load_config().
    dims : ModelDims
        Model dimensions (K, M, N).
    t, P_scaled, A_scaled, ...
        All the same inputs received by the normal mechanistic pipeline.
    outdir : str
        Top-level output directory.

    Returns
    -------
    dict with keys:
        theta_opt, pinn_model, loss_components, ts, ys, loss_history
    """
    logger.info("[pinn] PINN mode enabled: bypassing mechanistic multistart and post-fit neuralODE.")
    logger.info("[pinn] Running single-start PINN optimization.")
    logger.info("[pinn] Using existing residual/loss infrastructure from phoscrosstalk.optimization.")

    pinn_cfg = get_pinn_cfg(cfg)
    K, M, N = dims.K, dims.M, dims.N
    state_dim = 3 * K + M + N

    proteins_l = list(proteins) if proteins else [str(k) for k in range(K)]
    kinases_l  = list(kinases)  if kinases  else [str(m) for m in range(M)]
    sites_l    = list(sites)    if sites    else [str(n) for n in range(N)]

    # ------------------------------------------------------------------
    # 1. Theta initialisation (centre of the parameter bounds)
    # ------------------------------------------------------------------
    bounds_ns  = getattr(cfg, "bounds", None)
    xl, xu, _  = create_bounds(K, M, N, bounds=bounds_ns)
    theta0 = np.asarray((xl + xu) / 2.0, dtype=np.float64)
    logger.info("[pinn] Initial theta created (dim=%d)", theta0.shape[0])

    # ------------------------------------------------------------------
    # 2. PINN model initialisation
    # ------------------------------------------------------------------
    seed = int(getattr(pinn_cfg, "seed", 0))
    key  = jax.random.PRNGKey(seed)
    pinn_model = PINNAugmentation(
        state_dim=state_dim,
        width_size=int(getattr(pinn_cfg, "width_size", 64)),
        depth=int(getattr(pinn_cfg, "depth", 2)),
        activation=str(getattr(pinn_cfg, "activation", "tanh")),
        key=key,
    )
    n_pinn_params = pinn_param_count(pinn_model)
    logger.info("[pinn] PINNAugmentation initialised — %d trainable parameters.", n_pinn_params)

    # ------------------------------------------------------------------
    # 3. Solver / ODE settings
    # ------------------------------------------------------------------
    solver_cfg = getattr(cfg, "solver", None)
    opt_cfg    = getattr(cfg, "optimisation", None)
    dr_cfg     = getattr(cfg, "derived_rates", None)
    bounds_ns2 = getattr(cfg, "bounds", None)

    ode_solver_kind      = getattr(solver_cfg,  "ode_solver",           "tsit5")
    ode_adjoint_kind     = getattr(solver_cfg,  "ode_adjoint",          "recursive")
    rtol                 = float(getattr(pinn_cfg, "rtol", getattr(solver_cfg, "rtol", 1e-6)))
    atol                 = float(getattr(pinn_cfg, "atol", getattr(solver_cfg, "atol", 1e-9)))
    solver_max_steps     = int(getattr(solver_cfg, "max_steps", 16384))
    dt0                  = float(getattr(solver_cfg, "dt0", 0.01))
    root_find_max_steps  = int(getattr(solver_cfg, "root_find_max_steps", 10))
    rna_relax            = float(getattr(dr_cfg, "rna_relax", 0.1))
    abundance_max        = float(getattr(bounds_ns2, "abundance_max", 5.0)) if bounds_ns2 else 5.0
    lambda_net           = float(getattr(opt_cfg, "lambda_net", 1e-4)) if opt_cfg else 1e-4
    reg_lambda           = float(getattr(opt_cfg, "reg_lambda", 1e-4)) if opt_cfg else 1e-4
    lambda_pinn          = float(getattr(pinn_cfg, "lambda_pinn", 0.1))
    regularize           = str(getattr(pinn_cfg, "regularize", "residual_l2"))

    lw_cfg = getattr(cfg, "loss_weights", None)
    w_phospho  = float(getattr(lw_cfg, "phospho",   1.0)) if lw_cfg else 1.0
    w_abundance= float(getattr(lw_cfg, "abundance", 1.0)) if lw_cfg else 1.0
    w_reg      = float(getattr(lw_cfg, "reg",       1.0)) if lw_cfg else 1.0
    w_mrna     = float(getattr(lw_cfg, "mrna",      1.0)) if lw_cfg else 1.0

    has_rna = (
        t_rna is not None
        and rna_obs_matched is not None
        and rna_model_prot_idx is not None
        and rna_fit_genes is not None
        and len(rna_fit_genes) > 0
    )

    # ------------------------------------------------------------------
    # 4. Build PINN loss function
    # ------------------------------------------------------------------
    logger.info("[pinn] Building PINN loss function …")
    loss_fn = make_pinn_loss_fn(
        dims=dims,
        t=t,
        P_data=P_scaled,
        A_scaled=A_scaled,
        prot_idx_for_A=prot_idx_for_A,
        W_data=W_data,
        W_data_prot=W_data_prot,
        Cg=Cg,
        Cl=Cl,
        site_prot_idx=site_prot_idx,
        K_site_kin=K_site_kin,
        R=R,
        L_alpha=L_alpha,
        kin_to_prot_idx=kin_to_prot_idx,
        receptor_mask_prot=receptor_mask_prot,
        receptor_mask_kin=receptor_mask_kin,
        mechanism=mechanism,
        lambda_net=lambda_net,
        reg_lambda=reg_lambda,
        lambda_pinn=lambda_pinn,
        regularize=regularize,
        w_phospho=w_phospho,
        w_abundance=w_abundance,
        w_reg=w_reg,
        w_mrna=w_mrna,
        rtol=rtol,
        atol=atol,
        max_steps=solver_max_steps,
        dt0=dt0,
        k_act_fn=k_act_fn,
        s_prod_fn=s_prod_fn,
        t_mrna=t_rna if has_rna else None,
        rna_data_scaled=rna_obs_matched if has_rna else None,
        rna_model_prot_idx=rna_model_prot_idx if has_rna else None,
        R_data0=R_data0,
        W_data_mrna=W_data_mrna if has_rna else None,
        rna_relax=rna_relax,
        abundance_max=abundance_max,
        ode_solver_kind=ode_solver_kind,
        ode_adjoint_kind=ode_adjoint_kind,
        root_find_max_steps=root_find_max_steps,
        xl=xl,
        xu=xu,
    )
    logger.info("[pinn] PINN loss function built.")

    # ------------------------------------------------------------------
    # 5. Run single-start training (no multistart)
    # ------------------------------------------------------------------
    theta_j     = jnp.asarray(theta0, dtype=jnp.float64)
    trainable   = (theta_j, pinn_model)

    logger.info(
        "[pinn] Starting Optax training: max_steps=%d  lr=%.2e  lambda_pinn=%.2e",
        int(getattr(pinn_cfg, "max_steps", 500)),
        float(getattr(pinn_cfg, "learning_rate", 1e-3)),
        lambda_pinn,
    )

    t_train_start = time.perf_counter()
    (theta_final, pinn_final), loss_history = _run_pinn_optax(
        loss_fn=loss_fn,
        trainable=trainable,
        pinn_cfg=pinn_cfg,
    )
    t_train_elapsed = time.perf_counter() - t_train_start
    logger.info("[pinn] Training finished in %.1f s (%d steps).", t_train_elapsed, len(loss_history))

    # ------------------------------------------------------------------
    # 6. Recompute final loss diagnostics
    # ------------------------------------------------------------------
    try:
        total_final, aux_final = loss_fn((theta_final, pinn_final), None)
        f1, f2, f3, f4, fp = tuple(float(a) for a in aux_final)
    except Exception as exc:
        _debug_logger.warning("[pinn] Could not recompute final diagnostics: %s", exc)
        f1 = f2 = f3 = f4 = fp = float("nan")

    loss_components = {
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "f_pinn_reg": fp,
        "total_loss": f1 + f2 + f3 + f4 + fp,
    }
    logger.info(
        "[pinn] Final losses: f1=%.4e  f2=%.4e  f3=%.4e  f4=%.4e  f_pinn=%.4e",
        f1, f2, f3, f4, fp,
    )

    theta_opt = np.asarray(theta_final, dtype=np.float64)

    # ------------------------------------------------------------------
    # 7. Forward simulation for dense output
    # ------------------------------------------------------------------
    ts_sim: np.ndarray | None = None
    ys_sim: np.ndarray | None = None
    try:
        from phoscrosstalk.pinn.rhs import make_combined_rhs
        import diffrax
        from phoscrosstalk.mechanisms import compute_prev_site_idx
        from phoscrosstalk.simulation import build_full_A0
        from phoscrosstalk.solver_config import (
            make_diffrax_adjoint,
            make_diffrax_solver,
            make_stepsize_controller,
        )

        sim_cfg    = getattr(cfg, "simulation", None)
        dense_n    = int(getattr(sim_cfg, "dense_n_points", 200)) if sim_cfg else 200
        t_dense    = np.linspace(float(t[0]), float(t[-1]), dense_n)

        combined_rhs = make_combined_rhs(
            K, M, N, mechanism,
            k_act_fn=k_act_fn,
            s_prod_fn=s_prod_fn,
            rna_relax=rna_relax,
            abundance_max=abundance_max,
        )

        prev_idx = compute_prev_site_idx(site_prot_idx.astype(np.int32), N)
        T_prot   = P_scaled.shape[1]
        A0_full  = build_full_A0(K, T_prot, A_scaled, prot_idx_for_A)
        x0 = np.zeros(3 * K + M + N, dtype=np.float64)
        if R_data0 is not None:
            r0 = np.asarray(R_data0, dtype=np.float64)
            r0 = r0[:, 0] if r0.ndim > 1 else r0
            x0[:K] = np.clip(np.nan_to_num(r0, nan=1.0), 0.0, 10.0)
        else:
            x0[:K] = 1.0
        a0 = np.nan_to_num(A0_full[:, 0], nan=1.0)
        x0[2 * K : 3 * K] = np.clip(a0, 0.0, 5.0)
        x0[3 * K + M :]   = np.clip(np.nan_to_num(P_scaled[:, 0], nan=0.0), 0.0, None)

        y0_j   = jnp.asarray(x0, dtype=jnp.float64)
        t_eval = jnp.asarray(t_dense, dtype=jnp.float64)
        sctrl  = make_stepsize_controller(rtol=rtol, atol=atol)

        Cg_j   = jnp.asarray(Cg,              dtype=jnp.float64)
        Cl_j   = jnp.asarray(Cl,              dtype=jnp.float64)
        K_sk_j = jnp.asarray(K_site_kin,      dtype=jnp.float64)
        R_j    = jnp.asarray(R,               dtype=jnp.float64)
        La_j   = jnp.asarray(L_alpha,         dtype=jnp.float64)
        spi_j  = jnp.asarray(site_prot_idx,   dtype=jnp.int32)
        k2p_j  = jnp.asarray(kin_to_prot_idx, dtype=jnp.int32)
        rmp_j  = jnp.asarray(receptor_mask_prot, dtype=jnp.float64)
        rmk_j  = jnp.asarray(receptor_mask_kin,  dtype=jnp.float64)
        psi_j  = jnp.asarray(prev_idx,        dtype=jnp.int32)

        theta_dense_j = jnp.asarray(theta_opt, dtype=jnp.float64)

        ode_args_sim = (
            theta_dense_j,
            Cg_j, Cl_j, spi_j, K_sk_j, R_j, La_j,
            k2p_j, rmp_j, rmk_j, psi_j,
            pinn_final,
        )

        term_sim = diffrax.ODETerm(combined_rhs)
        solver_sim = make_diffrax_solver(ode_solver_kind, root_find_max_steps=root_find_max_steps)

        sol_sim = diffrax.diffeqsolve(
            term_sim,
            solver_sim,
            t0=float(t_dense[0]),
            t1=float(t_dense[-1]),
            dt0=dt0,
            y0=y0_j,
            args=ode_args_sim,
            saveat=diffrax.SaveAt(ts=t_eval),
            stepsize_controller=sctrl,
            max_steps=solver_max_steps,
            throw=False,
        )
        ts_sim = np.asarray(t_dense, dtype=np.float64)
        ys_sim = np.asarray(sol_sim.ys, dtype=np.float64)
        logger.info("[pinn] Dense simulation complete: shape=%s", ys_sim.shape)

    except Exception as exc:
        _debug_logger.warning("[pinn] Dense forward simulation failed: %s", exc)

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    os.makedirs(outdir, exist_ok=True)
    logger.info("[pinn] Saving outputs to %s …", outdir)

    save_pinn_outputs(
        outdir=outdir,
        theta_opt=theta_opt,
        pinn_model=pinn_final,
        loss_components=loss_components,
        ts=ts_sim,
        ys=ys_sim,
        K=K, M=M, N=N,
        sites=sites_l,
        proteins=proteins_l,
        kinases=kinases_l,
        P_data=P_scaled,
        A_scaled=A_scaled,
        prot_idx_for_A=prot_idx_for_A,
        t=t,
        pinn_cfg=pinn_cfg,
        dims=dims,
    )

    pinn_residuals = save_pinn_residuals(
        outdir=outdir,
        ts=ts_sim,
        ys=ys_sim,
        pinn_model=pinn_final,
        K=K, M=M, N=N,
        proteins=proteins_l,
        kinases=kinases_l,
        sites=sites_l,
    )

    save_pinn_plots(
        outdir=outdir,
        pinn_residuals=pinn_residuals,
        ts=ts_sim,
        ys=ys_sim,
        t_obs=t,
        P_data=P_scaled,
        loss_history=loss_history,
        K=K, M=M, N=N,
        proteins=proteins_l,
        kinases=kinases_l,
        sites=sites_l,
    )

    logger.info("[pinn] All PINN outputs saved.")

    return {
        "theta_opt":        theta_opt,
        "pinn_model":       pinn_final,
        "loss_components":  loss_components,
        "ts":               ts_sim,
        "ys":               ys_sim,
        "loss_history":     loss_history,
    }
