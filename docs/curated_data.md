# PhosCrosstalk Data Curation

This module (`phoscrosstalk.data_curator`) automates the acquisition, cleaning, and formatting of biological prior
knowledge required for the PhosCrosstalk modeling pipeline. It consolidates data from **KEA**, **PhosphoSitePlus**, and
**PTMcode2** into optimized formats (SQLite databases, NetworkX graphs, and lookup tables).

## 1. Prerequisites

Ensure your Python environment has the necessary dependencies installed:

```bash
pip install pandas networkx requests numpy

```

## 2. Data Sources & Input Files

The pipeline uses three primary datasets. Some can be downloaded automatically, while others must be obtained manually
due to licensing or hosting restrictions.

### A. Automatically Downloaded (via Harmonizome)

The script can automatically fetch these from the [Harmonizome repository](https://maayanlab.cloud/Harmonizome/).

* **Kinase Enrichment Analysis (KEA3):** Used to build the kinase-kinase interaction network and kinase-substrate
  mapping.
* *File:* `kinase_networks.zip`
* *File:* `gsl_database.zip`


* **PhosphoSitePlus (PSP):** Used for additional disease and substrate annotations.

### B. Manually Downloaded (Required)

You must manually download the **PTMcode2** dataset, as it is not hosted on Harmonizome.

* **Source:** [PTMcode2 Downloads](https://www.google.com/search?q=https://ptmcode.embl.de/downloads.cgi)
* **Files Needed:**

1. `ptmcode2_within_homosapiens.txt.gz` (Rename to `within.gz`)
2. `ptmcode2_between_homosapiens.txt.gz` (Rename to `between.gz`)

---

## 3. Directory Setup

Before running the full pipeline, organize your manually downloaded files into the following structure. The script will
create the `processed/` folder automatically.

```text
project_root/
├── data_curated/
│   └── raw/
│       ├── ptmcode/
│       │   ├── within.gz       <-- (Manual Download)
│       │   └── between.gz      <-- (Manual Download)
│       │
│       └── kea/                <-- (Optional: Script downloads these if missing)
│           ├── kinase_networks.zip
│           └── gsl_database.zip

```

---

## 4. Usage

Run the curator as a module from your project root.

### Option A: The "All-in-One" Command (Recommended)

This command attempts to download missing KEA/PSP data, process the PTMcode files you provided, and generate all final
artifacts.

```bash
python3 -m phoscrosstalk.data_curator \
  --all \
  --ptmcode data_curated/raw/ptmcode/within.gz data_curated/raw/ptmcode/between.gz

```

### Option B: Step-by-Step Execution

**1. Download only (Harmonizome data):**

```bash
python3 -m phoscrosstalk.data_curator --download

```

**2. Build Kinase Networks & Maps (requires KEA data):**

```bash
python3 -m phoscrosstalk.data_curator --kea

```

**3. Build PTM Databases (requires PTMcode files):**

```bash
python3 -m phoscrosstalk.data_curator \
  --ptmcode data_curated/raw/ptmcode/within.gz data_curated/raw/ptmcode/between.gz

```

### Option C: Convert Custom Data

If you have a custom CSV file (e.g., from an experiment) mapping kinases to sites, you can convert it to the model's
standard TSV format.

* **Input CSV Columns:** `GeneID`, `Psite`, `Kinase` (e.g., `{CDK1, ERK2}`)
* **Command:**

```bash
python3 -m phoscrosstalk.data_curator --convert-csv path/to/my_input.csv

```

---

## 5. Output Artifacts

After running the script, the `data_curated/processed/` directory will contain the files needed for `main.py`.

| File                               | Format | Description                                              | Used By `main.py` Flag |
|------------------------------------|--------|----------------------------------------------------------|------------------------|
| **`ptm_intra.db`**                 | SQLite | Database of co-evolving sites on the *same* protein.     | `--ptm-intra`          |
| **`ptm_inter.db`**                 | SQLite | Database of co-evolving sites on *interacting* proteins. | `--ptm-inter`          |
| **`unified_kinase_graph.gpickle`** | Pickle | NetworkX graph of kinase-kinase hierarchy/interactions.  | `--unified-graph-pkl`  |
| **`ks_psite_table.tsv`**           | TSV    | Flat table of Kinase-Substrate interactions.             | `--kea-ks-table`       |
| **`ks_psite_index.pkl`**           | Pickle | Fast lookup dictionary for KS interactions.              | *(Internal use)*       |

---
