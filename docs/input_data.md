The data generation process is designed to build a bioinformatics pipeline focused on **phosphorylation networks**, *
*kinase-substrate interactions**, and **protein translation modification (PTM) co-evolution**.

The pipeline fetches raw data (mostly from Harmonizome, PTMcode, and KEA), cleans it, and structures it into three main
formats: **Graph Networks (NetworkX)**, **Relational Databases (SQLite)**, and **Lookup Indices (Pickle/TSV)**.

Here is the breakdown of how the data is generated, categorized by the specific data domain.

---

### 1. Kinase-Kinase Interaction Networks

**Script:** `build_kinase_networks.py`
This module generates a multiplex graph representing how kinases interact with one another across different biological
layers (e.g., co-expression, protein-protein interaction).

* **Input:** A ZIP file (`kinase_networks.zip`) containing TSV files from KEA (Kinase Enrichment Analysis).
* **Process:**

1. **Layer Extraction:** It reads individual TSVs (e.g., `kk_degene_net.tsv`), treating each as a specific "layer" of
   evidence.
2. **Graph Construction:** It builds a NetworkX graph for each layer.
3. **Unification:** It merges all layers into a `Unified Graph`. If an edge exists in multiple layers, it calculates
   statistics (sum of weights, mean weight, support count).


* **Output:**
* `unified_kinase_graph.gpickle`: A Python-native NetworkX object containing full metadata.
* `unified_kinase_graph.graphml`: A standard XML-based graph format for portability.

### 2. Kinase-Substrate (Site-Level) Mapping

**Script:** `build_site_level_ks_map.py`
This module creates a mapping connecting specific Kinases to the specific residues (Phosphosites) they phosphorylate on
Substrate proteins.

* **Input:** `gsl_database.zip` (specifically `kinase_substrate_phospho-site_level_pmid_resource_database.txt` from
  KEA).
* **Process:**

1. **Parsing:** Extracts Kinase names and Substrate Sites (formatted as `GENE_SITE`, e.g., `EGFR_Y1068`).
2. **Normalization:** Converts gene names and residues to uppercase to ensure consistency.
3. **Indexing:** Builds a dictionary where the key is the substrate site, and values are sets of Kinases, PMIDs (
   references), and data sources.


* **Output:**
* `ks_psite_table.tsv`: A flat table for easy reading.
* `ks_psite_index.pkl`: A Python dictionary (Pickle) optimized for fast lookups in your module (O(1) access).

### 3. Co-Phosphorylation (intra- and inter-protein)

**Script:** `build_db_from_ptmcode2.py`
This module processes data from **PTMcode2** to find pairs of phosphorylation sites that are functionally linked (
co-evolving or structurally coupled), either within the same protein or between two interacting proteins.

* **Input:** Gzipped "within" (intra) and "between" (inter) files from PTMcode2.
* **Process:**

1. **Filtering:** It strictly filters for `Homo sapiens` and ensures both PTMs in the pair are of type
   `phosphorylation`.
2. **Extraction:** Extracts Protein ID, Residue 1, Score 1, Protein ID 2, Residue 2, Score 2.
3. **Database Injection:** Instead of a flat file, it uses `sqlite3` to insert these millions of rows into a structured
   SQL database for efficient querying.


* **Output:**
* `ptm_intra.db`: SQLite DB for sites on the *same* protein.
* `ptm_inter.db`: SQLite DB for sites on *interacting* proteins.

### 4. Data Integration & Enrichment

**Script:** `merge_datasets.py`
This script attempts to combine the functional links from PTMcode2 with biological annotations from **dbPTM**.

* **Input:** `dbPTM phospho.gz`, PTMcode files, and a `mapped_ids` file (to translate Protein IDs to UniProt IDs).
* **Process:**

1. **ID Mapping:** Translates PTMcode protein IDs to UniProt IDs so they can talk to dbPTM.
2. **Merging:** Joins the tables based on the UniProt ID + Residue Position.
3. **Enrichment:** Adds literature references (PMIDs) and peptide sequences from dbPTM to the PTMcode data.


* **Output:** A highly compressed TSV (`.tar.xz`) containing the merged, enriched dataset.

### 5. Utilities & Raw Data Acquisition

* **`harmonizomedownloader.py`**: This is the entry point for raw data. It downloads datasets from the **Harmonizome**
  repository (including KEA, PhosphoSitePlus, etc.).
* **`make_kinase.py`**: A helper script that converts a specific custom CSV format (GeneID, Psite, Kinase List) into a
  standardized "Site-Kinase-Weight" TSV.
* **`filter_data.py`**: A helper script to filter datasets based on a "valid" list of GeneIDs or Gene-Site pairs.

### Summary of Data Flow

1. **Download:** `harmonizomedownloader.py` fetches KEA/PhosphoSitePlus data.
2. **Process Networks:** `build_kinase_networks.py` turns KEA data into a Kinase Graph (`.graphml`).
3. **Process Interactions:** `build_site_level_ks_map.py` turns KEA data into a Kinase-Substrate Lookup Index (`.pkl`).
4. **Process Co-evolution:** `build_db_from_ptmcode2.py` turns PTMcode data into SQL Databases (`.db`).
5. **Merge:** `merge_datasets.py` combines these sources into a master annotation file.

Your "module" likely consumes the **`.pkl` index** for fast lookups, the **`.db` files** for querying co-phosphorylation
scores, and the **`.graphml`** file to understand kinase hierarchy.