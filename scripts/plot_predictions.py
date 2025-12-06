#!/usr/bin/env python3
"""
Create one figure per protein:
- Protein abundance (model + data)
- All phosphosites for that protein (model + data)

Input: fit_timeseries.tsv from network fitting.
Output: folder of PNGs, one per protein.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Plot individual proteins (model vs data) + their phosphosites")
    parser.add_argument("--sim", default= "../network_fit/fit_timeseries.tsv", help="Path to fit_timeseries.tsv")
    parser.add_argument("--outdir", default="protein_plots", help="Output directory for PNGs")
    args = parser.parse_args()

    # --------------------------
    # Load
    # --------------------------
    if not os.path.exists(args.sim):
        print(f"[!] File not found: {args.sim}")
        return

    df = pd.read_csv(args.sim, sep="\t")
    print(f"[*] Loaded {len(df)} rows (proteins + sites)")

    # Make folder
    os.makedirs(args.outdir, exist_ok=True)

    # --------------------------
    # Identify columns
    # --------------------------
    sim_cols = sorted([c for c in df.columns if c.startswith("sim_t")], key=lambda s: int(s.split("sim_t")[1]))
    data_cols = sorted([c for c in df.columns if c.startswith("data_t")], key=lambda s: int(s.split("data_t")[1]))
    n_points = len(sim_cols)

    # Time axis from your model
    t_vals_default = np.array([0, 0.5, 0.75, 1, 2, 4, 8, 16, 30, 60, 120, 240, 480, 960], dtype=float)
    t_vals = t_vals_default if len(t_vals_default) == n_points else np.arange(n_points)

    # Clean types
    df["Type"] = df["Type"].fillna("")
    df["Residue"] = df["Residue"].fillna("")
    df["Protein"] = df["Protein"].astype(str)

    df_sites = df[df["Type"] == "Phosphosite"].copy()
    df_prots = df[df["Type"] == "ProteinAbundance"].copy()

    proteins = sorted(df["Protein"].unique())
    print(f"[*] Found {len(proteins)} proteins")

    # --------------------------
    # Per-Protein plotting
    # --------------------------
    for prot in proteins:
        print(f"   → Plotting {prot}")

        plt.figure(figsize=(12, 8))
        ax = plt.gca()

        # ---- Protein abundance (if exists)
        row_prot = df_prots[df_prots["Protein"] == prot]
        if not row_prot.empty:
            row_prot = row_prot.iloc[0]

            y_sim = row_prot[sim_cols].values.astype(float)
            y_dat = row_prot[data_cols].values.astype(float)
            has_data = np.any(np.isfinite(y_dat))

            ax.plot(t_vals, y_sim, "-", lw=3, color="blue", label="Protein (model)")
            ax.scatter(t_vals, y_sim, color="blue", s=30)

            if has_data:
                ax.plot(t_vals, y_dat, "k--", lw=2, label="Protein (data)")
                ax.scatter(t_vals, y_dat, color="black", s=35)

        # ---- Phosphosites
        sub = df_sites[df_sites["Protein"] == prot]

        for _, row in sub.iterrows():
            res = row["Residue"]
            y_sim = row[sim_cols].values.astype(float)
            y_dat = row[data_cols].values.astype(float)
            has_data = np.any(np.isfinite(y_dat))

            ax.plot(t_vals, y_sim, "-", alpha=0.7, lw=1.7, label=f"{res} (model)")
            ax.scatter(t_vals, y_sim, s=20, alpha=0.7)

            if has_data:
                ax.plot(t_vals, y_dat, "o--", ms=4, alpha=0.7, label=f"{res} (data)")

        # ---- Format plot
        ax.set_title(f"{prot}", fontsize=14, weight="bold")
        # ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("FC / Scaled abundance")

        # Show only unique legend entries
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="best", ncol=1)

        # ---- Save
        out_path = os.path.join(args.outdir, f"{prot}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    print(f"[*] Saved {len(proteins)} plots to: {args.outdir}")


if __name__ == "__main__":
    main()
