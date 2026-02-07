from __future__ import annotations
from pathlib import Path
import numpy as np
import sys


def _julia():
    # lazy import so normal users don't need julia installed unless they select it
    print("[*] Initializing Julia Runtime...", flush=True)
    from juliacall import Main as jl
    import juliacall

    # 1. Define Dependencies
    required_pkgs = [
        "DifferentialEquations",
        "SciMLSensitivity",
        "ForwardDiff",
        "LinearAlgebra",
        "Metaheuristics",
        "Random"
    ]

    # 2. Check and Install Dependencies
    jl.seval("import Pkg")

    print("[*] Checking Julia dependencies...", flush=True)
    try:
        for pkg in required_pkgs:
            # Try loading. If this hangs, it's precompiling.
            jl.seval(f"using {pkg}")
    except Exception:
        print("[!] Missing packages detected. Installing...", flush=True)
        jl.Pkg.Registry.update()
        for pkg in required_pkgs:
            print(f"    - Installing {pkg}...", flush=True)
            jl.Pkg.add(pkg)
        jl.Pkg.resolve()

    # 3. Load Your Source Code
    here = Path(__file__).resolve().parent / "julia"
    # Assuming PhosBackend.jl is inside the 'src' folder based on your initial tree
    # If it is in 'engines/julia/', remove the 'src' part below.
    src_file = here / "src" / "PhosBackend.jl"

    if not src_file.exists():
        # Fallback if user moved it to engines/julia/PhosBackend.jl
        src_file = here / "PhosBackend.jl"

    if not jl.seval("isdefined(Main, :PhosBackend)"):
        print(f"[*] Compiling PhosBackend from {src_file} (this may take 1-2 mins)...", flush=True)
        jl.include(str(src_file))
        jl.seval("using .PhosBackend")
        print("[*] Compilation complete.", flush=True)

    return jl


def build_ctx(Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
              kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin):
    jl = _julia()

    # Data conversion
    Cg = np.asarray(Cg, dtype=np.float64, order="F")
    Cl = np.asarray(Cl, dtype=np.float64, order="F")
    K_site_kin = np.asarray(K_site_kin, dtype=np.float64, order="F")
    R = np.asarray(R, dtype=np.float64, order="F")
    L_alpha = np.asarray(L_alpha, dtype=np.float64, order="F")

    site_prot_idx = np.asarray(site_prot_idx, dtype=np.int64)
    kin_to_prot_idx = np.asarray(kin_to_prot_idx, dtype=np.int64)
    receptor_mask_prot = np.asarray(receptor_mask_prot, dtype=np.int64)
    receptor_mask_kin = np.asarray(receptor_mask_kin, dtype=np.int64)

    ctx = jl.PhosBackend.SimContext(Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                                    kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin)
    return ctx


def simulate(t, P_scaled, A0_full, theta,
             Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha, kin_to_prot_idx,
             receptor_mask_prot, receptor_mask_kin,
             mechanism="dist", full_output=False):
    jl = _julia()
    ctx = build_ctx(Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                    kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin)

    t = np.asarray(t, dtype=np.float64)
    P_scaled = np.asarray(P_scaled, dtype=np.float64, order="F")
    A0_full = np.asarray(A0_full, dtype=np.float64, order="F")
    theta = np.asarray(theta, dtype=np.float64)

    P_sim, A_sim, S_sim, Kdyn_sim = jl.PhosBackend.simulate(
        ctx, t, P_scaled, A0_full, theta, mechanism=mechanism
    )

    P_sim = np.asarray(P_sim)
    A_sim = np.asarray(A_sim)
    if not full_output:
        return P_sim, A_sim
    return P_sim, A_sim, np.asarray(S_sim), np.asarray(Kdyn_sim)


def nsga3_optimize(t, P_scaled, A0_full, xl, xu,
                   Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha, kin_to_prot_idx,
                   receptor_mask_prot, receptor_mask_kin,
                   mechanism="dist", pop=128, gen=200, multistart=4, seed=1):
    jl = _julia()
    ctx = build_ctx(Cg, Cl, site_prot_idx, K_site_kin, R, L_alpha,
                    kin_to_prot_idx, receptor_mask_prot, receptor_mask_kin)

    t = np.asarray(t, dtype=np.float64)
    P_scaled = np.asarray(P_scaled, dtype=np.float64, order="F")
    A0_full = np.asarray(A0_full, dtype=np.float64, order="F")
    xl = np.asarray(xl, dtype=np.float64)
    xu = np.asarray(xu, dtype=np.float64)

    print(f"[*] Starting Julia Optimization (Pop={pop}, Gen={gen}). Please wait...", flush=True)

    F, X = jl.PhosBackend.nsga3_optimize(
        ctx, t, P_scaled, A0_full, xl, xu,
        mechanism=mechanism, pop=pop, gen=gen, multistart=multistart, seed=seed
    )

    print("[*] Optimization Finished!", flush=True)
    return np.asarray(F), np.asarray(X)