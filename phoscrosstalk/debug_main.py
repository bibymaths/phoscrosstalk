#!/usr/bin/env python3
"""
debug_main.py (drop-in edits for your main.py)

Goal: make “flat trajectories” diagnosable in <60 seconds by printing:
- mapping coverage (K_site_kin) and whether IDs matched at all
- R normalization sanity
- PTM matrices sanity
- scaling/weights sanity
- a one-shot simulation sanity check BEFORE running pymoo
- automatic “early fail” when a matrix is effectively empty or NaN/Inf

You can copy these helpers into main.py and call them at the marked points.
"""

import numpy as np
import pandas as pd


# -------------------------
# low-level helpers
# -------------------------

def _fmt_stats(x, name="", axis=None):
    x = np.asarray(x)
    if axis is None:
        return (f"{name} shape={x.shape} dtype={x.dtype} "
                f"min={np.nanmin(x):.3g} max={np.nanmax(x):.3g} "
                f"mean={np.nanmean(x):.3g} std={np.nanstd(x):.3g} "
                f"nan={np.isnan(x).sum()} inf={np.isinf(x).sum()}")
    else:
        s = np.sum(x, axis=axis)
        return (f"{name} sum(axis={axis}) min={np.nanmin(s):.3g} "
                f"max={np.nanmax(s):.3g} mean={np.nanmean(s):.3g} "
                f"nan={np.isnan(s).sum()} inf={np.isinf(s).sum()}")


def _assert_finite(mat, name, hard=True):
    mat = np.asarray(mat)
    ok = np.isfinite(mat).all()
    if not ok:
        msg = (f"[FAIL] {name} contains non-finite values: "
               f"nan={np.isnan(mat).sum()} inf={np.isinf(mat).sum()}")
        if hard:
            raise ValueError(msg)
        print(msg)
    return ok


def _assert_shape(mat, shape, name):
    if tuple(mat.shape) != tuple(shape):
        raise ValueError(f"[FAIL] {name} shape mismatch: got {mat.shape}, expected {shape}")


def _nnz(mat):
    mat = np.asarray(mat)
    return int(np.count_nonzero(mat))


def _row_sums(mat):
    return np.asarray(mat).sum(axis=1)


def _col_sums(mat):
    return np.asarray(mat).sum(axis=0)


def _safe_row_normalize(M, eps=0.0):
    """Rows with sum==0 remain all-zero. No NaNs."""
    M = np.asarray(M, dtype=np.float64)
    rs = M.sum(axis=1)
    out = M.copy()
    nz = rs > eps
    out[nz] /= rs[nz, None]
    return out


def _coverage_report_K_site_kin(K_site_kin, sites, kinases, tag="K_site_kin"):
    K_site_kin = np.asarray(K_site_kin)
    N = len(sites)
    M = len(kinases)

    print(f"\n=== {tag} coverage ===")
    print(_fmt_stats(K_site_kin, f"{tag}"))
    print(f"{tag} nnz={_nnz(K_site_kin)}  density={_nnz(K_site_kin)/(N*M + 1e-12):.6g}")

    deg_sites = (K_site_kin != 0).sum(axis=1)   # per-site #kinases
    deg_kin   = (K_site_kin != 0).sum(axis=0)   # per-kinase #sites

    print(f"{tag} per-site degree:  min={deg_sites.min()}  median={np.median(deg_sites)}  max={deg_sites.max()}")
    print(f"{tag} per-kin degree:   min={deg_kin.min()}  median={np.median(deg_kin)}  max={deg_kin.max()}")

    zero_sites = np.where(deg_sites == 0)[0]
    zero_kin   = np.where(deg_kin == 0)[0]
    print(f"{tag} zero-degree sites:  {len(zero_sites)}/{N}")
    print(f"{tag} zero-degree kinases:{len(zero_kin)}/{M}")

    # If your mapping silently failed (ID mismatch), you will often see:
    # nnz ~ 0, and almost all sites/kinases zero-degree.
    if _nnz(K_site_kin) == 0:
        raise ValueError(
            f"[FAIL] {tag} is all zeros. Mapping likely failed (site IDs mismatch). "
            f"Check that TSV/KSEA site strings exactly match `sites` from load_site_data()."
        )

    # show a few examples of unmapped sites/kinases
    if len(zero_sites) > 0:
        ex = zero_sites[:10]
        print("Examples unmapped sites:", [sites[i] for i in ex])

    if len(zero_kin) > 0:
        ex = zero_kin[:10]
        print("Examples unmapped kinases:", [kinases[j] for j in ex])


def _sanity_report_R(R, N, M, tag="R"):
    R = np.asarray(R)
    print(f"\n=== {tag} sanity ===")
    print(_fmt_stats(R, f"{tag}"))
    _assert_shape(R, (M, N), tag)
    _assert_finite(R, tag, hard=True)

    rs = R.sum(axis=1)
    print(f"{tag} row-sum stats: min={rs.min():.6g} median={np.median(rs):.6g} max={rs.max():.6g}")
    zeros = np.where(rs == 0)[0]
    if len(zeros) > 0:
        print(f"{tag} rows with sum==0: {len(zeros)}/{M} (kinases with no mapped sites)")
    # Not always fatal, but if most rows are zero => Kdyn gets no substrate input -> flat dynamics.
    if (rs == 0).mean() > 0.5:
        print(f"[WARN] >50% of {tag} rows are all-zero. Expect many dead kinases / flat Kdyn drive.")


def _sanity_report_C(Cg, Cl, N, tag="C"):
    Cg = np.asarray(Cg); Cl = np.asarray(Cl)
    print(f"\n=== {tag} sanity ===")
    _assert_shape(Cg, (N, N), "Cg")
    _assert_shape(Cl, (N, N), "Cl")
    _assert_finite(Cg, "Cg", hard=True)
    _assert_finite(Cl, "Cl", hard=True)
    print(_fmt_stats(Cg, "Cg"))
    print(_fmt_stats(Cl, "Cl"))
    print(_fmt_stats(_row_sums(Cg), "Cg row_sums"))
    print(_fmt_stats(_row_sums(Cl), "Cl row_sums"))
    # If both are almost zero, coup ~ tanh(0)=0 and crosstalk contributes nothing (fine, but know it).


def _sanity_report_data(P_scaled, Y, t, tag="data"):
    P_scaled = np.asarray(P_scaled)
    Y = np.asarray(Y)
    t = np.asarray(t)
    print(f"\n=== {tag} sanity ===")
    print(f"t len={len(t)}  t[0:5]={t[:5]}  t[-1]={t[-1]}")
    print(_fmt_stats(Y, "Y raw"))
    print(_fmt_stats(P_scaled, "P_scaled"))
    # flat input data after scaling is a real possibility
    dyn = np.nanmax(P_scaled, axis=1) - np.nanmin(P_scaled, axis=1)
    print(f"P_scaled per-site dynamic range: min={dyn.min():.3g} median={np.median(dyn):.3g} max={dyn.max():.3g}")
    if np.median(dyn) < 1e-6:
        print("[WARN] P_scaled is almost flat for most sites after scaling. "
              "You may be scaling away signal (or Y is flat).")


def _sanity_report_weights(W_data, W_data_prot, tag="weights"):
    W_data = np.asarray(W_data)
    W_data_prot = np.asarray(W_data_prot)
    print(f"\n=== {tag} sanity ===")
    print(_fmt_stats(W_data, "W_data"))
    print(_fmt_stats(W_data_prot, "W_data_prot"))
    if np.any(W_data < 0) or np.any(W_data_prot < 0):
        raise ValueError("[FAIL] Negative weights found. That will break objectives or bias optimization.")


def _one_shot_sim_check(problem, xl, xu, P_scaled, label="preopt simulation"):
    """
    Run one simulation before pymoo and check:
    - output shape matches P_scaled
    - output not all-constant across time
    - no NaN/Inf
    """
    print(f"\n=== {label} ===")
    x0 = 0.5 * (xl + xu)  # neutral point in log-parameter space
    P_pred = problem.simulate(x0)

    P_pred = np.asarray(P_pred)
    print(_fmt_stats(P_pred, "P_pred"))

    if P_pred.shape != P_scaled.shape:
        raise ValueError(f"[FAIL] simulate() shape mismatch: got {P_pred.shape}, expected {P_scaled.shape}")

    _assert_finite(P_pred, "P_pred", hard=True)

    # dynamic range
    dyn = np.max(P_pred, axis=1) - np.min(P_pred, axis=1)
    print(f"P_pred per-site dynamic range: min={dyn.min():.3g} median={np.median(dyn):.3g} max={dyn.max():.3g}")
    if np.median(dyn) < 1e-8:
        print("[WARN] P_pred is essentially flat for most sites at x0. "
              "This is a MODEL-LEVEL issue (matrix drive, gating, or saturation), not a pymoo issue.")

    # compare quickly to data scale
    d_dyn = np.max(P_scaled, axis=1) - np.min(P_scaled, axis=1)
    print(f"Compare median dynamic range: pred={np.median(dyn):.3g} data={np.median(d_dyn):.3g}")


def _filter_dead_kinases(K_site_kin, kinases, kin_to_prot_idx, receptor_mask_kin, L_alpha=None):
    """
    Optional but highly effective:
    remove kinase columns with zero-degree (no mapped sites).
    This is NOT "inventing edges". It's removing dead variables.
    """
    K_site_kin = np.asarray(K_site_kin)
    deg_kin = (K_site_kin != 0).sum(axis=0)
    keep = deg_kin > 0
    if keep.all():
        return K_site_kin, kinases, kin_to_prot_idx, receptor_mask_kin, L_alpha

    kept_idx = np.where(keep)[0]
    dropped_idx = np.where(~keep)[0]
    print(f"[INFO] Filtering dead kinases: keeping {keep.sum()}/{len(kinases)}, dropping {len(dropped_idx)} zero-degree kinases.")
    print("[INFO] Example dropped kinases:", [kinases[i] for i in dropped_idx[:10]])

    K2 = K_site_kin[:, keep]
    kin2 = [kinases[i] for i in kept_idx]
    kin_to_prot_idx2 = kin_to_prot_idx[keep]
    receptor_mask_kin2 = receptor_mask_kin[keep]
    if L_alpha is not None:
        L2 = L_alpha[np.ix_(keep, keep)]
    else:
        L2 = None
    return K2, kin2, kin_to_prot_idx2, receptor_mask_kin2, L2

def sim_summary(problem, tag, x):
    P = problem.simulate(x)
    dr = (P.max(axis=1) - P.min(axis=1))
    print(f"[{tag}] P range stats: min={dr.min():.4g} med={np.median(dr):.4g} max={dr.max():.4g}")
    print(f"[{tag}] P abs stats: min={P.min():.4g} mean={P.mean():.4g} max={P.max():.4g}")
    return P

# -------------------------
# WHERE TO INSERT
# -------------------------
"""
In your main.py, after each stage, add these calls:

1) after load_site_data (+ optional crosstalk filtering):
   _sanity_report_data(P_scaled, Y, t) must be called AFTER scaling, so put it after scaling.

2) after building Cg,Cl:
   _sanity_report_C(Cg, Cl, N=len(sites))

3) after building K_site_kin, kinases:
   _coverage_report_K_site_kin(K_site_kin, sites, kinases)

   Optional:
   K_site_kin, kinases, kin_to_prot_idx, receptor_mask_kin, L_alpha = _filter_dead_kinases(...)

   Then rebuild R from filtered K_site_kin.

4) after building R:
   _sanity_report_R(R, N=len(sites), M=len(kinases))

5) after building weights:
   _sanity_report_weights(W_data, W_data_prot)

6) after constructing problem (before minimize):
   _one_shot_sim_check(problem, xl, xu, P_scaled)

This will tell you exactly why “run remains flat”:
- K_site_kin all zeros => TSV/KSEA site IDs mismatch
- R row sums mostly zero => many dead kinases (filter needed)
- P_scaled almost flat => scaling/data issue
- P_pred flat at neutral x0 => model saturation or missing drive (matrix issue), not optimizer.
"""
