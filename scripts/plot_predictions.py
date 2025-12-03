#!/usr/bin/env python3
"""
Visualizes the time-series predictions from fit_timeseries.tsv.
Can optionally overlay the original data if provided.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser(description="Plot model predictions vs data")
    parser.add_argument("--sim", default="network_fit_pymoo/fit_timeseries.tsv", help="Path to simulation TSV")
    parser.add_argument("--data", help="Optional: Path to original data CSV (e.g., filtered_input1.csv) for overlay")
    parser.add_argument("--top", type=int, default=16, help="Number of sites to plot")
    args = parser.parse_args()

    # Load Simulation
    if not os.path.exists(args.sim):
        print(f"[!] Simulation file {args.sim} not found.")
        return

    df_sim = pd.read_csv(args.sim, sep="\t")
    print(f"[*] Loaded simulation for {len(df_sim)} sites.")

    # Extract time columns
    sim_cols = [c for c in df_sim.columns if c.startswith("sim_t")]
    n_points = len(sim_cols)

    # Default timepoints (matching the fitting script)
    t_vals = [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0]
    if len(t_vals) != n_points:
        # Fallback if dimensions don't match
        t_vals = np.arange(n_points)

    # Load Data if provided
    df_data = None
    if args.data and os.path.exists(args.data):
        df_data = pd.read_csv(args.data, sep=None, engine="python")
        print(f"[*] Loaded original data for overlay.")

    # Create Site ID for matching
    df_sim["id"] = df_sim["Protein"] + "_" + df_sim["Residue"]

    # Select sites to plot (e.g., first N)
    sites_to_plot = df_sim["id"].head(args.top).tolist()

    # Plotting
    n_cols = 4
    n_rows = int(np.ceil(len(sites_to_plot) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), sharex=False)
    axes = axes.flatten()

    for i, site_id in enumerate(sites_to_plot):
        ax = axes[i]

        # Plot Simulation
        row_sim = df_sim[df_sim["id"] == site_id]
        if row_sim.empty: continue

        y_sim = row_sim[sim_cols].values.flatten()
        ax.plot(t_vals, y_sim, 'r-', linewidth=2, label='Model')
        ax.scatter(t_vals, y_sim, color='red', s=10)

        # Plot Data (if available)
        if df_data is not None:
            # Try to find matching row in data
            # Assuming data has Protein/Residue or GeneID/Psite columns
            # We need a robust matcher similar to the loading script

            # Simple heuristic matcher
            match = None
            prot, res = site_id.split("_")

            # Filter by protein first
            if "Protein" in df_data.columns:
                mask = df_data["Protein"] == prot
            elif "GeneID" in df_data.columns:
                mask = df_data["GeneID"] == prot
            else:
                mask = pd.Series([False] * len(df_data))

            temp = df_data[mask]

            # Filter by residue
            # This is tricky because data might be "S620" or "S_620"
            # We check if the residue string exists in Psite or Residue col
            if not temp.empty:
                for idx, row in temp.iterrows():
                    r_val = str(row.get("Residue") or row.get("Psite"))
                    if res in r_val:  # crude match "620" in "S620"
                        # Extract data columns (v1..vN or x1..xN)
                        val_cols = [c for c in df_data.columns if c.startswith("v") or c.startswith("x")]
                        if len(val_cols) == n_points:
                            y_dat = row[val_cols].values.astype(float)
                            ax.plot(t_vals, y_dat, 'k--', alpha=0.6, label='Data')
                            ax.scatter(t_vals, y_dat, color='black', alpha=0.6, s=15)
                            break

        ax.set_title(site_id)
        ax.set_xscale("log")  # Log scale often helps with this time range
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig("prediction_summary.png", dpi = 300)
    print(f"[*] Saved plot to prediction_summary.png")


if __name__ == "__main__":
    main()