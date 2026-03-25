# Codebase Audit Report

**Repository:** `bibymaths/phoscrosstalk`  
**Branch:** `develop` (as available)  
**Audit Date:** 2026-03-25  
**Scope:** Architecture, Documentation, Core Engines, Scripts, Utilities

---

## 1. Missing Elements

### 1.1 Undocumented Modules in Repository Structure

The `README.md` repository structure tree lists only 9 files under `phoscrosstalk/`, but the actual package contains **15 Python modules**. The following are completely absent from the README structure:

| File | Purpose |
|---|---|
| `logger.py` | Singleton `RichLogger` — primary logging abstraction used by all modules |
| `config.py` | `ModelDims` global state container and `DEFAULT_TIMEPOINTS` constant |
| `weighting.py` | Weight-matrix construction for time-series and protein data |
| `fretchet.py` | Numba-parallelized discrete Fréchet distance used for best-solution selection |
| `hyperparam.py` | `BOUNDS_CONFIG`, `create_bounds`, and hyperparameter scan logic |
| `post_processing.py` | Cytoscape export, residual heatmaps, parameter clustermaps, run-metadata provenance |
| `debug_main.py` | Drop-in diagnostic helpers for matrix sanity checks before optimization |

### 1.2 Missing `scripts/fit_hpc.py`

`run_phoscrosstalk.slurm` invokes `scripts/fit_hpc.py` via `mpiexec`/`mpi4py.futures`, but this file does not exist in the repository. There is no documentation explaining it was removed, renamed, or is user-supplied. The SLURM script is therefore non-functional as committed.

### 1.3 `--receptors` / `--receptor-kinases` Flags Undocumented

`main.py` exposes `--receptors` and `--receptor-kinases` CLI flags that govern which proteins/kinases receive the external stimulus `u(t)`. None of the documentation files (`README.md`, `docs/phoscrosstalk.md`, `docs/input_data.md`, `docs/curated_data.md`) mention these flags, their semantics, or how the external stimulus is defined in the ODE system.

### 1.4 No Documentation for `--crosstalk-tsv` Flag

`main.py` accepts an optional `--crosstalk-tsv` argument that filters the site list to a user-supplied whitelist of PTM pairs. This flag is never mentioned in any documentation or usage example, making it undiscoverable to users.

### 1.5 `DEFAULT_TIMEPOINTS` Assumption Undocumented

`config.py` defines 14 hard-coded timepoints (`DEFAULT_TIMEPOINTS`) that are used by `data_loader.load_site_data` when parsing the input CSV. If the user's data has a different number or order of time columns the pipeline raises a `ValueError`. This critical constraint is not stated anywhere in the documentation.

### 1.6 No Documentation for `scripts/uniprot.sh`

The `scripts/uniprot.sh` Bash utility (used for gene→UniProt accession mapping via the UniProt REST API) has no corresponding documentation entry. Its required input format (`proteins.unique.txt`) and output format are not described in any doc file.

### 1.7 Biological Score (`bio_score`) Selection Logic Undocumented

`optimization.py` exposes a `bio_score` function that penalises solutions whose kinase/protein half-lives deviate from expected biological values (10 min / 600 min targets). The README and all other docs describe Fréchet distance as the sole best-solution selection criterion; the biological score's role in the pipeline is never mentioned.

### 1.8 `--scale-mode` Modes Not Described

The `--scale-mode` flag accepts `minmax`, `none`, and `log-minmax`, but no documentation explains the mathematical difference between these modes or advises when each is appropriate. Users cannot make an informed choice.

### 1.9 `run_phoscrosstalk.slurm` Has No Documentation

The SLURM script is undocumented: there is no README section, HPC guide, or doc file explaining how to adapt it to a different cluster, what `--partition=BigMem` implies, or how the MPI parallelism strategy differs from the `multiprocessing.Pool` used in the standard `main.py` entry point.

### 1.10 `debug_main.py` Role and Status Undocumented

`debug_main.py` is shipped as part of the package but its purpose is not described anywhere. All calls to its helpers inside `main.py` are commented out, leaving it unclear to contributors whether it is production code, a development tool, or dead code.

### 1.11 No Documentation for the `equations/` Output Sub-directory

`main.py` calls `generate_equations_report(...)` which writes LaTeX sources and (optionally) compiles them. The `README.md` lists `equations/` in the output files section only with the description "Automatically generated LaTeX report of the specific ODE system fitted" — with no detail about what the LaTeX file contains, how to compile it, or what dependencies (`pdflatex`, `extarticle` package, etc.) are needed.

### 1.12 `post_processing.py` Outputs Absent from README

The pipeline generates `run_config.json`, `cytoscape_network.csv`, `residual_heatmap.png`, and `parameter_clustermap.png` via `post_processing.py`. None of these outputs appear in the README's **Output Files** table.

### 1.13 `preopt_snapshot/` Folder Not Listed in README Output Files

The snapshot directory — fully documented in `docs/snapshot_preoptimization.md` — is absent from the README's **Output Files** section, creating a discoverability gap.

---

## 2. Areas Requiring Fixes (Documentation/Sync Issues)

### 2.1 README Repository Structure Is Stale

The README lists only these modules:

```
main.py, data_curator.py, core_mechanisms.py, optimization.py,
simulation.py, analysis.py, sensitivity.py, knockouts.py, app.py
```

Seven additional modules (`logger.py`, `config.py`, `weighting.py`, `fretchet.py`, `hyperparam.py`, `post_processing.py`, `debug_main.py`) exist in the package but are omitted. The structure ends with `└── README.md`, implying `README.md` is inside the package directory — which is incorrect; the real `README.md` is at the repository root.

### 2.2 Installation Instructions Reference Non-Existent `requirements.txt`

The README instructs users to run `pip install -r requirements.txt`, but the repository uses `pyproject.toml` with `hatchling` as the build backend and provides `uv.lock` for dependency locking. There is no `requirements.txt`. The correct install command is `pip install .` or `uv sync`.

### 2.3 README States Python ≥ 3.10, but `pyproject.toml` Requires `>=3.11,<3.12`

The README badge and **Installation** section say "Python ≥ 3.10". The `pyproject.toml` specifies `requires-python = ">=3.11,<3.12"`. The actual constraint is both more restrictive (minimum 3.11) and narrower (maximum below 3.12) than documented.

### 2.4 `docs/input_data.md` References Non-Existent Script Files

The document names five helper scripts as independent files:

- `build_kinase_networks.py`
- `build_site_level_ks_map.py`
- `build_db_from_ptmcode2.py`
- `merge_datasets.py`
- `harmonizomedownloader.py`
- `make_kinase.py`
- `filter_data.py`

All of this logic is consolidated inside **`data_curator.py`** as methods of the `DataCurator` class. None of the individual script files exist in the repository, making `docs/input_data.md` a description of a prior (or hypothetical) architecture rather than the current one.

### 2.5 `docs/input_data.md` Describes `.graphml` Output That Is Not Produced

The document states that `build_kinase_networks.py` generates both `unified_kinase_graph.gpickle` **and** `unified_kinase_graph.graphml`. Examining `data_curator.py`, the `graphml` export path exists in code but the `.graphml` file is not mentioned in `docs/curated_data.md`'s output artifact table, and the README never references it. Users do not know this portable format exists.

### 2.6 README Describes "NSGA-II" but Multi-Start Uses "UNSGA-III + NSGA-2"

The README's **Multi-Objective Evolutionary Optimization** section states strategies include "NSGA-II (diversity-focused) and UNSGA-III (convergence-focused)." The actual `multistarts.py` runs three named strategies:

1. `Balanced (UNSGA3, p=12)` 
2. `High-Res (UNSGA3, p=15)`
3. `Diversity (NSGA2)`

The README omits the fact that two separate UNSGA-III runs are used (with different partition densities), describing the ensemble as simply two algorithms rather than three.

### 2.7 README Refers to "NSGA-II" as Optimization but Badge Shows "NSGA3 | UNSGA3"

The shield badge in the README reads `Optimization-NSGA3 | UNSGA3` but the body text says "NSGA-II (diversity-focused)". There is a direct contradiction between the badge and the prose.

### 2.8 `docs/snapshot_preoptimization.md`: States No `*.npz` Is Used, Contradicting `main.py`

The snapshot doc explicitly states: *"No `*.npz` is used."* However, `analysis.py`'s `save_fitted_simulation()` writes `fitted_params.npz`, and `app.py`'s `load_snapshot_data()` reads `fitted_params.npz` from the output directory. The snapshot folder itself does not use `.npz`, but the overall pipeline absolutely does. This phrasing creates a misleading impression.

### 2.9 `docs/curated_data.md` Option C (`--convert-csv`) Undocumented in Main README

`curated_data.md` documents the `--convert-csv` flag for converting custom kinase CSVs, but the README's **Data Curation** section only shows the `--all` command. Users unaware of this option would manually reformat their data.

### 2.10 `docs/phoscrosstalk.md` Variant 4 (`--tune`) Description Is Misleading

The doc states: *"Note that `--tune` usually runs before the main fitting loop."* This is vague. In `main.py`, when `--tune` is specified, the hyperparameter scan executes **and then the Cl matrix is rebuilt** with the tuned `length_scale` before optimization begins — the main fitting loop still runs afterwards. The documentation does not clarify this, making users think `--tune` is a standalone pre-step.

### 2.11 `core_mechanisms.py` Module Docstring References Wrong Filename

The module-level docstring at the top of `core_mechanisms.py` starts with `core_mechanisms_fast.py` — a prior filename. The actual file is named `core_mechanisms.py`. This is a stale artifact from a refactor.

### 2.12 `steadystate.py` Docstring Has a Blank Line Where Text Should Be

The `run_steadystate_analysis` docstring contains an empty second line (`\n\n`) before the description, which appears to be a leftover from an incomplete edit. While minor, it affects documentation quality.

### 2.13 README Optimization Step Description Omits Fréchet-Based Selection

The README's pipeline description ends at "Pareto-optimal solutions" but never mentions that the best single solution is selected from the Pareto front using the discrete Fréchet distance — a non-trivial step that directly determines all downstream analysis outputs.

---

## 3. Completeness Gaps

### 3.1 Input CSV Format Underspecified

`docs/input_data.md` does not describe the primary time-series CSV format consumed by `main.py --data`. `data_loader.load_site_data` requires:

- A `Protein` **or** `GeneID` column.
- A `Psite` **or** `Residue` column.
- Value columns prefixed with `v` or `x` whose count must exactly match `DEFAULT_TIMEPOINTS` (14 columns).
- Optional `Type` column (`"site"` vs `"protein"`) to separate phosphosite rows from protein abundance rows.

None of this is documented. Users must read source code to understand their data format requirements.

### 3.2 `ModelDims` Global State Pattern Not Documented

`config.py` uses a class-level mutable singleton (`ModelDims`) to store `K`, `M`, `N` globally. This pattern has important implications: modules called before `ModelDims.set_dims()` will see `None` values and produce misleading errors. This architectural decision is not explained anywhere, including the architectural overview in the README.

### 3.3 Parameter Vector (`theta`) Layout Not Summarised in Any Doc

The flat parameter vector `theta` has a layout of `4K + 2 + 3M + N + 4` elements. The `sensitivity.py` module documents this order in `_generate_param_labels`, and `core_mechanisms.decode_theta` implements the unpacking, but no document provides a human-readable table of what each block represents. This makes it difficult to interpret `fitted_params.npz` or `xl.tsv`/`xu.tsv` outputs without reading source code.

### 3.4 Weighting Schemes Not Described

`main.py` exposes `--weight-scheme` with four choices (`uniform`, `early_emphasis`, `early_emphasis_moderate`, `flat_no_noise`), but no documentation explains what temporal weighting profile each scheme applies or when to choose one over another. Users have no basis for selecting a non-default scheme.

### 3.5 Fréchet Distance Implementation Nuances Undocumented

`fretchet.py` implements the discrete Fréchet distance using Numba with `parallel=True`. The function is used in `multistarts.py` to score entire Pareto-front populations against all phosphosite trajectories simultaneously. The high-level docs only mention "Fréchet Distance Selection" without explaining what curves are being compared, what the score represents, or why it was chosen over simple L2 distance.

### 3.6 Knockout Screen Perturbation Logic Not Fully Described

The README describes knockouts as "systematically perturbs kinases, proteins, or sites to predict network-wide impacts." `knockouts.py` implements three distinct perturbation types:

- **Kinase KO**: sets `alpha` to near zero.
- **Protein KO**: sets `s_prod` (synthesis rate) to near zero.
- **Site KO**: zeros out the corresponding column of `K_site_kin` (simulating an alanine substitution).

The parameter-level mechanism of each perturbation is never documented, only the high-level biological interpretation.

### 3.7 Steady-State Time Vector Construction Not Documented

`steadystate.py` builds a piecewise time vector: 100 linearly-spaced points from 0–100, concatenated with 50 linearly-spaced points from 101–10,000 (note: not log-spaced, despite the comment in the source saying "Log-spaced"). The `docs/snapshot_preoptimization.md` does not cover this module, and the README mentions only `T=10,000` without explaining the rationale for the piecewise grid or the discrepancy with the source comment.

### 3.8 Global Sensitivity Analysis Sampling Strategy Not Documented

`sensitivity.py` uses SALib's `saltelli` sampler with a default sample size defined internally. The number of model evaluations this implies (typically `N * (2D + 2)` where `D` is the parameter dimension) is not disclosed to users, making it impossible to estimate compute time for large models.

### 3.9 `app.py` Missing Module Docstring and No Dashboard User Guide

`app.py` has no module-level docstring. While the README mentions a Streamlit dashboard with high-level capabilities, there is no dedicated doc page, tutorial, or screenshot describing what each dashboard panel does, what files it expects, or how to navigate it.

### 3.10 `pyproject.toml` Lists `pynetphorest==0.1.3` but It Is Never Imported

A search of all `.py` files reveals no `import pynetphorest` statement. This dependency appears to be vestigial — either a planned but unimplemented feature or a remnant of a prior version. It adds installation overhead and a potential point of failure without providing any functionality.

### 3.11 No Error-Handling Documentation

There is no documentation on expected failure modes, such as:
- What happens when `--kinase-tsv` and `--kea-ks-table` are both omitted (the code silently falls back to an identity matrix with auto-generated kinase names — a completely undocumented behavior).
- What happens if the SQLite PTM databases contain no matching proteins for the input sites.
- How the optimizer behaves with a very small site count (potentially fewer sites than Pareto reference directions).

---

## 4. Duplication

### 4.1 `create_bounds` Defined in Both `optimization.py` and `hyperparam.py`

`optimization.py` contains a `create_bounds(K, M, N)` function. `hyperparam.py` also contains an identically named `create_bounds(K=None, M=None, N=None)` function with the same logic but different default handling. `main.py` imports `create_bounds` from `optimization.py`, while `hyperparam.py` uses its own local version internally. This creates two sources of truth for parameter bounds, and any change to bound values must be synchronised manually in both places.

### 4.2 `build_full_A0` Defined in Both `optimization.py` and Imported from `simulation.py`

`optimization.py` defines `build_full_A0`. `knockouts.py`, `steadystate.py`, `sensitivity.py`, and `app.py` all import it from `simulation.py` (e.g. `from phoscrosstalk.simulation import simulate_p_scipy, build_full_A0`). However, `simulation.py` itself re-imports or re-exports this function from `optimization.py`. This indirect dependency chain is opaque and creates ambiguity about the canonical source of this utility.

### 4.3 `docs/phoscrosstalk.md` and `docs/curated_data.md` Overlap on Curator Usage

`docs/curated_data.md` is the authoritative page for the data curation pipeline, but `docs/phoscrosstalk.md` (the CLI usage guide) also re-documents several curator command variants inline (e.g., the `--all`, `--kea`, `--ptmcode` flags). Users reading only one doc may miss options documented only in the other; users reading both will encounter redundant content.

### 4.4 `docs/useful_cli.md` Contains Only Developer Scratch Commands

`docs/useful_cli.md` currently contains two shell one-liners (concatenating `.py` files and cleaning `__pycache__`). These are developer housekeeping snippets, not user-facing CLI documentation. The filename implies a more substantive reference, creating confusion about the document's intended scope.

### 4.5 README Duplicated Within Repository

There is a `README.md` at the repository root and a second `README.md` at `phoscrosstalk/README.md` (inside the package directory). Both are identical. This duplication can lead to documentation drift if one is updated and the other is not.

### 4.6 Commented-Out Debug Calls Scattered Throughout `main.py`

`main.py` contains nine commented-out calls to `debug_main.py` helpers (e.g., `# _sanity_report_data(...)`, `# _one_shot_sim_check(...)`). These create visual noise, imply the debug module is in an intermediate state between "active" and "removed," and duplicate the problem of having undocumented code in the package.

### 4.7 `hyperparam.py` Re-implements Optimization Loop Locally

`hyperparam.py::run_hyperparameter_scan` constructs its own `NetworkOptimizationProblem`, its own `multiprocessing.Pool`, and its own `UNSGA3` run — effectively duplicating the optimization logic already in `main.py` and `multistarts.py`. There is no shared abstraction. Any change to optimization defaults (e.g., termination criteria, reference direction partitions) must be replicated in both `multistarts.py` and `hyperparam.py`.
