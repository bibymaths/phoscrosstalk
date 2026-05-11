#!/usr/bin/env python3
"""
Synthetic JAX time-series generator for PhosCrosstalk-style input data.

Outputs
-------
1. protephospho.csv
   GeneID,Psite,x1,...,x14

2. mrna.csv
   GeneID,x1,...,x9

3. kinase_sites.tsv
   Site,Kinase,weight
   where Site = Gene_Psite

4. tf_mrna.csv
   Source,Target,Weight

5. time_series_samples.png
   Diagnostic plot with one subplot per generator.

Python 3.10+
JAX 0.4.x+
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import lax, random, vmap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 42
    n_genes: int = 3
    sites_per_gene: int = 3
    n_kinases: int = 2
    n_tfs: int = 2
    n_phospho_timepoints: int = 14
    n_mrna_timepoints: int = 14
    output_dir: str = "sample_data"


# =============================================================================
# Core JAX generators
# =============================================================================


@partial(jax.jit, static_argnames=("n_steps",))
def generate_fourier_signal(
    key: jax.Array,
    n_steps: int,
    frequencies: jax.Array,
    amplitudes: jax.Array,
    phases: jax.Array,
    noise_scale: float = 0.03,
) -> jax.Array:
    """Generate a Fourier/sinusoidal signal as a sum of sine waves."""
    t = jnp.linspace(0.0, 1.0, n_steps)

    # Shape: (n_waves, n_steps). Broadcasting avoids Python loops.
    waves = amplitudes[:, None] * jnp.sin(
        2.0 * jnp.pi * frequencies[:, None] * t[None, :] + phases[:, None]
    )

    signal = jnp.sum(waves, axis=0)
    noise = noise_scale * random.normal(key, shape=(n_steps,))

    return signal + noise


@partial(jax.jit, static_argnames=("n_steps",))
def generate_random_walk(
    key: jax.Array,
    n_steps: int,
    drift: float = 0.0,
    step_scale: float = 0.08,
) -> jax.Array:
    """Generate Gaussian random walk / Brownian-motion-like trajectory."""
    increments = drift + step_scale * random.normal(key, shape=(n_steps,))
    return jnp.cumsum(increments)


@partial(jax.jit, static_argnames=("n_steps",))
def generate_ar_process(
    key: jax.Array,
    n_steps: int,
    coefficients: jax.Array,
    noise_scale: float = 0.05,
) -> jax.Array:
    """Generate an AR(p) process using lax.scan."""
    p = coefficients.shape[0]
    eps = noise_scale * random.normal(key, shape=(n_steps,))

    init_history = jnp.zeros((p,), dtype=jnp.float32)

    def step(history: jax.Array, innovation: jax.Array) -> tuple[jax.Array, jax.Array]:
        # history stores previous p values. scan is required here because AR(p)
        # is recurrent and each state depends on previous generated values.
        x_t = jnp.dot(coefficients, history[::-1]) + innovation
        new_history = jnp.concatenate([history[1:], jnp.asarray([x_t])])
        return new_history, x_t

    _, series = lax.scan(step, init_history, eps)
    return series


@partial(jax.jit, static_argnames=("n_steps",))
def generate_structured_series(
    key: jax.Array,
    n_steps: int,
    intercept: float = 0.5,
    slope: float = 0.4,
    seasonal_amplitude: float = 0.25,
    seasonal_frequency: float = 2.0,
    seasonal_phase: float = 0.0,
    noise_scale: float = 0.04,
) -> jax.Array:
    """Generate trend + seasonality + Gaussian noise."""
    t = jnp.linspace(0.0, 1.0, n_steps)

    trend = intercept + slope * t
    seasonality = seasonal_amplitude * jnp.sin(
        2.0 * jnp.pi * seasonal_frequency * t + seasonal_phase
    )
    noise = noise_scale * random.normal(key, shape=(n_steps,))

    return trend + seasonality + noise


# =============================================================================
# Batched JAX generation
# =============================================================================


@partial(jax.jit, static_argnames=("n_series", "n_steps"))
def generate_batched_fourier(
    key: jax.Array,
    n_series: int,
    n_steps: int,
) -> jax.Array:
    """Generate multiple independent Fourier series with vmap over PRNG keys."""
    keys = random.split(key, n_series)

    frequencies = jnp.asarray([1.0, 2.0, 4.0], dtype=jnp.float32)
    amplitudes = jnp.asarray([0.6, 0.25, 0.12], dtype=jnp.float32)
    phases = jnp.asarray([0.0, 0.8, 1.7], dtype=jnp.float32)

    return vmap(
        lambda k: generate_fourier_signal(
            k,
            n_steps,
            frequencies,
            amplitudes,
            phases,
            noise_scale=0.03,
        )
    )(keys)


@partial(jax.jit, static_argnames=("n_series", "n_steps"))
def generate_batched_random_walk(
    key: jax.Array,
    n_series: int,
    n_steps: int,
) -> jax.Array:
    """Generate multiple independent random walks using vmap."""
    keys = random.split(key, n_series)

    return vmap(
        lambda k: generate_random_walk(
            k,
            n_steps,
            drift=0.01,
            step_scale=0.07,
        )
    )(keys)


@partial(jax.jit, static_argnames=("n_series", "n_steps"))
def generate_batched_ar(
    key: jax.Array,
    n_series: int,
    n_steps: int,
) -> jax.Array:
    """Generate multiple independent AR(p) trajectories using vmap."""
    keys = random.split(key, n_series)
    coefficients = jnp.asarray([0.65, -0.25, 0.12], dtype=jnp.float32)

    return vmap(
        lambda k: generate_ar_process(
            k,
            n_steps,
            coefficients,
            noise_scale=0.05,
        )
    )(keys)


@partial(jax.jit, static_argnames=("n_series", "n_steps"))
def generate_batched_structured(
    key: jax.Array,
    n_series: int,
    n_steps: int,
) -> jax.Array:
    """Generate multiple independent trend + seasonal + noise trajectories."""
    keys = random.split(key, n_series)

    # Series-specific parameters are vectorized; no Python loop over samples.
    slopes = jnp.linspace(0.15, 0.65, n_series)
    phases = jnp.linspace(0.0, jnp.pi, n_series)
    amplitudes = jnp.linspace(0.15, 0.35, n_series)

    def one_series(k: jax.Array, slope: float, phase: float, amp: float) -> jax.Array:
        return generate_structured_series(
            k,
            n_steps,
            intercept=0.4,
            slope=slope,
            seasonal_amplitude=amp,
            seasonal_frequency=2.0,
            seasonal_phase=phase,
            noise_scale=0.035,
        )

    return vmap(one_series)(keys, slopes, phases, amplitudes)


# =============================================================================
# Data shaping utilities
# =============================================================================


@jax.jit
def minmax_scale_rows(x: jax.Array, floor: float = 1e-3) -> jax.Array:
    """Min-max scale each row independently to [floor, 1]."""
    row_min = jnp.min(x, axis=1, keepdims=True)
    row_max = jnp.max(x, axis=1, keepdims=True)

    scaled = (x - row_min) / (row_max - row_min + 1e-8)
    return floor + (1.0 - floor) * scaled


def make_gene_names(n_genes: int) -> list[str]:
    """Create simple synthetic gene names."""
    base = ["EGFR", "MET", "ERBB2", "MAPK1", "AKT1", "STAT3", "JUN", "FOS"]
    if n_genes <= len(base):
        return base[:n_genes]
    return base + [f"GENE{i}" for i in range(len(base) + 1, n_genes + 1)]


def make_site_names(sites_per_gene: int) -> list[str]:
    """Create synthetic phosphosite labels."""
    residues = ["Y", "S", "T"]
    positions = [1068, 1173, 992, 202, 204, 473, 705, 727]

    names = []
    for i in range(sites_per_gene):
        names.append(f"{residues[i % len(residues)]}{positions[i % len(positions)]}")
    return names


def dataframe_from_matrix(
    ids: Sequence[str],
    matrix: np.ndarray,
    id_col: str,
) -> pd.DataFrame:
    """Convert matrix rows to x1...xT wide-format DataFrame."""
    n_time = matrix.shape[1]
    cols = [f"x{i + 1}" for i in range(n_time)]
    df = pd.DataFrame(matrix, columns=cols)
    df.insert(0, id_col, list(ids))
    return df


# =============================================================================
# Dataset builders
# =============================================================================


def build_protephospho_dataframe(
    genes: list[str],
    site_names: list[str],
    phospho_matrix: np.ndarray,
    protein_matrix: np.ndarray,
) -> pd.DataFrame:
    """
    Build combined protein + phosphosite wide table.

    Format:
        GeneID,Psite,x1,x2,...,xT

    Rows:
        - Protein abundance row: Psite is empty string.
        - Phosphosite rows: Psite contains site label, e.g. Y1068.
    """
    rows = []
    n_time = phospho_matrix.shape[1]

    if protein_matrix.shape != (len(genes), n_time):
        raise ValueError(
            "protein_matrix must have shape (n_genes, n_time). "
            f"Got {protein_matrix.shape}, expected {(len(genes), n_time)}."
        )

    expected_phospho_rows = len(genes) * len(site_names)
    if phospho_matrix.shape != (expected_phospho_rows, n_time):
        raise ValueError(
            "phospho_matrix must have shape (n_genes * sites_per_gene, n_time). "
            f"Got {phospho_matrix.shape}, expected {(expected_phospho_rows, n_time)}."
        )

    phospho_idx = 0

    for gene_idx, gene in enumerate(genes):
        # Protein abundance row: Psite empty
        protein_row = {
            "GeneID": gene,
            "Psite": "",
        }
        for t_idx, value in enumerate(protein_matrix[gene_idx], start=1):
            protein_row[f"x{t_idx}"] = float(value)
        rows.append(protein_row)

        # Phosphosite rows
        for site_idx, site in enumerate(site_names, start=1):
            site_row = {
                "GeneID": gene,
                "Psite": site,
            }
            for t_idx, value in enumerate(phospho_matrix[phospho_idx], start=1):
                site_row[f"x{t_idx}"] = float(value)
            rows.append(site_row)

            phospho_idx += 1

    return pd.DataFrame(rows)


def build_kinase_sites_dataframe(
    genes: list[str],
    site_names: list[str],
    kinases: list[str],
    seed: int,
) -> pd.DataFrame:
    """Build Site/Kinase/weight table where Site is Gene_Psite."""
    rng = np.random.default_rng(seed)
    rows = []

    for gene in genes:
        for site in site_names:
            n_edges = int(rng.integers(1, min(3, len(kinases)) + 1))
            chosen = rng.choice(kinases, size=n_edges, replace=False)
            raw_weights = rng.uniform(0.1, 1.0, size=n_edges)
            weights = raw_weights / np.maximum(raw_weights.sum(), 1e-12)

            for kinase, weight in zip(chosen, weights, strict=False):
                rows.append(
                    {
                        "Site": f"{gene}_{site}",
                        "Kinase": str(kinase),
                        "weight": float(weight),
                    }
                )

    return pd.DataFrame(rows)


def build_tf_mrna_dataframe(
    tfs: list[str],
    targets: list[str],
    seed: int,
) -> pd.DataFrame:
    """Build TF-to-target mRNA regulatory edge table."""
    rng = np.random.default_rng(seed)
    rows = []

    for target in targets:
        n_edges = int(rng.integers(1, min(3, len(tfs)) + 1))
        chosen = rng.choice(tfs, size=n_edges, replace=False)

        for source in chosen:
            rows.append(
                {
                    "Source": str(source),
                    "Target": str(target),
                    "Weight": float(rng.uniform(0.0, 1.0)),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Plotting and reporting
# =============================================================================


def summarize(name: str, x: np.ndarray) -> None:
    """Print shape and simple summary statistics."""
    print(
        f"{name:>20s} | shape={x.shape} | "
        f"mean={np.mean(x): .4f} | std={np.std(x): .4f}"
    )


def save_diagnostic_plot(
    fourier: np.ndarray,
    random_walk: np.ndarray,
    ar: np.ndarray,
    structured: np.ndarray,
    output_path: Path,
) -> None:
    """Save one subplot per synthetic generator."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=False)

    panels = [
        ("Fourier / sinusoidal", fourier),
        ("Gaussian random walk", random_walk),
        ("AR(p)", ar),
        ("Trend + seasonality + noise", structured),
    ]

    for ax, (title, matrix) in zip(axes, panels, strict=False):
        # Plot only a few rows for readability; generation itself was batched by vmap.
        for row in matrix[: min(5, matrix.shape[0])]:
            ax.plot(row, lw=1.8, alpha=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("time index")
        ax.set_ylabel("signal")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    cfg = SyntheticConfig()
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    key = random.PRNGKey(cfg.seed)
    key_fourier, key_rw, key_ar, key_structured, key_protein, key_mrna = random.split(key, 6)

    genes = make_gene_names(cfg.n_genes)
    site_names = make_site_names(cfg.sites_per_gene)
    n_sites_total = cfg.n_genes * cfg.sites_per_gene

    # -------------------------------------------------------------------------
    # Generate phosphosite/protein-like trajectories
    # -------------------------------------------------------------------------

    fourier = generate_batched_fourier(
        key_fourier,
        n_sites_total,
        cfg.n_phospho_timepoints,
    )
    random_walk = generate_batched_random_walk(
        key_rw,
        n_sites_total,
        cfg.n_phospho_timepoints,
    )
    ar = generate_batched_ar(
        key_ar,
        n_sites_total,
        cfg.n_phospho_timepoints,
    )
    structured = generate_batched_structured(
        key_structured,
        n_sites_total,
        cfg.n_phospho_timepoints,
    )

    protein_raw = generate_batched_structured(
        key_protein,
        cfg.n_genes,
        cfg.n_phospho_timepoints,
    )

    protein = minmax_scale_rows(protein_raw)

    # Combine methods into one realistic positive phosphosite signal matrix.
    # Scaling is JIT-compiled and row-wise.
    phospho = minmax_scale_rows(
        0.35 * fourier + 0.15 * random_walk + 0.20 * ar + 0.30 * structured
    )

    # -------------------------------------------------------------------------
    # Generate mRNA trajectories
    # -------------------------------------------------------------------------

    mrna_structured = generate_batched_structured(
        key_mrna,
        cfg.n_genes,
        cfg.n_mrna_timepoints,
    )
    mrna = minmax_scale_rows(mrna_structured)

    # Move from device to host only at file-output boundary.
    fourier_np = np.asarray(fourier)
    random_walk_np = np.asarray(random_walk)
    ar_np = np.asarray(ar)
    structured_np = np.asarray(structured)
    protein_np = np.round(np.asarray(protein), 6)
    phospho_np = np.asarray(phospho)
    mrna_np = np.asarray(mrna)

    # -------------------------------------------------------------------------
    # Build output tables
    # -------------------------------------------------------------------------

    protephospho_df = build_protephospho_dataframe(
        genes=genes,
        site_names=site_names,
        phospho_matrix=phospho_np,
        protein_matrix=protein_np,
    )

    mrna_df = dataframe_from_matrix(
        ids=genes,
        matrix=mrna_np,
        id_col="GeneID",
    )

    kinases = genes[: cfg.n_kinases]
    tfs = genes[-cfg.n_tfs :]

    kinase_sites_df = build_kinase_sites_dataframe(
        genes=genes,
        site_names=site_names,
        kinases=kinases,
        seed=cfg.seed + 10,
    )

    tf_mrna_df = build_tf_mrna_dataframe(
        tfs=tfs,
        targets=genes,
        seed=cfg.seed + 20,
    )

    # -------------------------------------------------------------------------
    # Save files
    # -------------------------------------------------------------------------

    protephospho_path = outdir / "protephospho.csv"
    mrna_path = outdir / "mrna.csv"
    kinase_sites_path = outdir / "kinase_sites.tsv"
    tf_mrna_path = outdir / "tf_mrna.csv"
    plot_path = outdir / "time_series_samples.png"

    protephospho_df.to_csv(protephospho_path, index=False)
    mrna_df.to_csv(mrna_path, index=False)
    kinase_sites_df.to_csv(kinase_sites_path, sep="\t", index=False)
    tf_mrna_df.to_csv(tf_mrna_path, index=False)

    save_diagnostic_plot(
        fourier=fourier_np,
        random_walk=random_walk_np,
        ar=ar_np,
        structured=structured_np,
        output_path=plot_path,
    )

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    print("\nSynthetic JAX time-series generation complete.\n")
    summarize("fourier", fourier_np)
    summarize("random_walk", random_walk_np)
    summarize("ar_process", ar_np)
    summarize("structured", structured_np)
    summarize("protephospho", phospho_np)
    summarize("mrna", mrna_np)

    print("\nSaved files:")
    print(f"  {protephospho_path}")
    print(f"  {mrna_path}")
    print(f"  {kinase_sites_path}")
    print(f"  {tf_mrna_path}")
    print(f"  {plot_path}")


if __name__ == "__main__":
    main()