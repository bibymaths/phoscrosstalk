<img src="images/phoscrosstalk_logo.svg" width="450">

<p align="left">

<img src="[https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white](https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white)">

<img src="[https://img.shields.io/badge/License-BSD--3--Clause-2ea043](https://img.shields.io/badge/License-BSD--3--Clause-2ea043)">

<img src="[https://img.shields.io/badge/ODE--Model-Systems%20Biology-blueviolet](https://img.shields.io/badge/ODE--Model-Systems%20Biology-blueviolet)">

<img src="[https://img.shields.io/badge/Optimization-NSGA3%20%7C%20UNSGA3-ff69b4](https://www.google.com/search?q=https://img.shields.io/badge/Optimization-NSGA3%2520%257C%2520UNSGA3-ff69b4)">

</p>

# **PhosCrosstalk**

**Global phospho-network ODE modeling with PTM-based crosstalk integration and multi-objective evolutionary optimization
**

PhosCrosstalk is a systems-level phosphorylation modeling framework that integrates **PTMcode2-derived inter/intra
crosstalk**, **KEA3 kinase-substrate networks**, and **experimental phosphosite time-series** into a single unified *
*global ODE model**.
It reconstructs protein activation, kinase activity, and phosphosite kinetics across an entire network, using **parallel
Multi-Objective Evolutionary Algorithms (MOEAs)** via `pymoo` to fit large parameter sets efficiently and robustly.

PhosCrosstalk provides a full end-to-end pipeline:

* **Automated Data Curation**: Downloads and standardizes KEA, PhosphoSitePlus, and PTMcode2 data.
* **Network Construction**: Builds multiplex kinase graphs and functional crosstalk matrices.
* **Global Optimization**: Fits kinetic parameters using advanced MOEAs (NSGA-II, UNSGA-III).
* **Simulation & Analysis**: Runs steady-state convergence, in-silico knockouts, and global sensitivity analysis (
  Sobol).
* **Interactive Visualization**: Includes a Streamlit dashboard for exploring trajectories and dynamic network
  animations.

---

---

# **Key Features**

### **1. Global ODE Phospho-Network Model**

The model captures the dynamics of three interconnected biological layers:

* **Protein Activation (`S`)**: Fraction of active protein.
* **Kinase Activity (`K_dyn`)**: Dynamic activity of kinases, regulated by upstream inputs and network topology.
* **Phosphosite Occupancy (`p`)**: Fractional phosphorylation of specific residues.

The coupled ODE system integrates:

* **Kinase-Substrate Interactions**: Directional phosphorylation driven by kinase activity (`K_dyn`).
* **Global Crosstalk**: Functional coupling from PTMcode2 inter/intra-protein associations (`β_g * Cg`).
* **Local Proximity**: Sequence-based influence between nearby residues (`β_l * Cl`).
* **Mechanistic Flexibility**: Supports **Distributive**, **Sequential**, and **Random/Cooperative** kinetic mechanisms.

---

### **2. Automated Data Curation Pipeline**

A built-in curator module (`data_curator.py`) handles the heavy lifting of data acquisition:

* **Downloads** raw datasets from Harmonizome (KEA, PhosphoSitePlus).
* **Processes** PTMcode2 files into optimized SQLite databases.
* **Constructs** a unified Kinase-Kinase interaction graph (NetworkX/Pickle).
* **Maps** Kinase-Substrate relationships into fast lookup indices.

---

### **3. Multi-Objective Evolutionary Optimization**

PhosCrosstalk uses `pymoo` to solve a multi-objective problem, simultaneously minimizing:

1. **Phosphosite Error**: Difference between simulated and observed phosphorylation profiles.
2. **Protein Abundance Error**: Difference between simulated and observed protein levels.
3. **Model Complexity**: Regularization terms (L2 and Laplacian network smoothing).

Strategies include **NSGA-II** (diversity-focused) and **UNSGA-III** (convergence-focused), run in parallel to find
robust Pareto-optimal solutions.

---

### **4. Advanced Post-Optimization Analysis**

Beyond simple fitting, the framework offers deep analytical tools:

* **Steady-State Analysis**: Simulates long-term convergence ().
* **In-Silico Knockouts**: Systematically perturbs kinases, proteins, or sites to predict network-wide impacts (Fold
  Change analysis).
* **Global Sensitivity Analysis (GSA)**: Computes Sobol indices to identify high-impact parameters.
* **Fréchet Distance Selection**: Selects the biologically "best" trajectory from the Pareto front.

---

### **5. Interactive Dashboard**

A comprehensive **Streamlit** app allows you to:

* Visualize fitted trajectories vs. experimental data.
* Explore sensitivity rankings and parameter distributions.
* Run real-time knockout simulations.
* **Animate** the flow of kinase activity through the network over time.

---

---

# **Repository Structure**

```
phoscrosstalk/
│
├── __init__.py
├── main.py                     # Entry point for modeling & optimization
├── data_curator.py             # Pipeline for downloading & processing raw data
├── core_mechanisms.py          # Numba-accelerated ODE kernels
├── optimization.py             # Pymoo Problem definitions & objectives
├── simulation.py               # Scipy odeint wrappers
├── analysis.py                 # Post-processing & static plotting
├── sensitivity.py              # SALib Global Sensitivity Analysis
├── knockouts.py                # Systematic in-silico perturbation screens
├── app.py                      # Interactive Streamlit Dashboard
│
└── README.md

```

---

# **Installation**

PhosCrosstalk requires Python ≥ 3.10.

```bash
git clone https://github.com/<yourname>/phoscrosstalk.git
cd phoscrosstalk

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

```

**Key Dependencies:** `numpy`, `scipy`, `pandas`, `numba`, `pymoo`, `networkx`, `salib`, `streamlit`, `rich`.

---

# **Data Curation**

Before modeling, you must curate the biological prior knowledge.

### **1. Download Manual Files**

Download the **PTMcode2** within/between files from
the [PTMcode website](https://www.google.com/search?q=https://ptmcode.embl.de/downloads.cgi) and place them in a
folder (e.g., `data/ptmcode2/`).

### **2. Run the Curator**

This command downloads KEA/PSP data automatically and processes your PTMcode files:

```bash
python3 -m phoscrosstalk.data_curator \
  --all \
  --ptmcode data/ptmcode2/within.gz data/ptmcode2/between.gz

```

*Outputs are saved to `data_curated/processed/`.*

---

# **Usage**

### **Run the Modeling Pipeline**

Execute the main optimization routine using your time-series data and the curated artifacts:

```bash
phoscrosstalk \
  --data data_timeseries/filtered_input1.csv \
  --ptm-intra data_curated/processed/ptm_intra.db \
  --ptm-inter data_curated/processed/ptm_inter.db \
  --kea-ks-table data_curated/processed/ks_psite_table.tsv \
  --unified-graph-pkl data_curated/processed/unified_kinase_graph.gpickle \
  --outdir results/experiment_01 \
  --cores 16 \
  --mechanism rand \
  --gen 300 \
  --run-steadystate \
  --run-knockouts \
  --run-sensitivity

```

### **Run the Dashboard**

Explore the results interactively:

```bash
streamlit run phoscrosstalk/app.py

```

*(Point the sidebar to your `results/experiment_01` directory)*

---

# **Output Files**

The pipeline generates a rich set of results in the output directory:

* **`fit_timeseries.tsv`**: Long-format table of Observed vs. Simulated values for all sites.
* **`fitted_params.npz`**: Complete archive of optimized parameters and model state.
* **`pareto_front_with_J.tsv`**: Objective values for all solutions on the Pareto front.
* **`knockouts/`**: Tables and heatmaps of Fold Changes for every in-silico knockout.
* **`sensitivity/`**: Sobol indices (`sobol_indices_labeled.tsv`) and perturbation trajectories.
* **`equations/`**: Automatically generated LaTeX report of the specific ODE system fitted.

---

# **Why PhosCrosstalk Exists**

Phosphorylation is not isolated. Sites influence each other across:

* protein domains
* protein complexes
* signaling cascades
* PTM interaction networks

Most modeling approaches treat sites independently or only use kinase–substrate data.
PhosCrosstalk closes the gap: it integrates **global PTM relationships**, **local sequence context**, and **experimental
time-series**, giving a mechanistic, quantitative reconstruction of network-level phosphorylation dynamics.

This creates a bridge between:

✔ dynamic ODE modeling
✔ phosphoproteomics
✔ PTM curation databases
✔ machine-learning residue prediction tools

---

# **Citation**

1. **Casado, P.,** Rodriguez-Prados, J.-C., Cosulich, S. C., Guichard, S., Vanhaesebroeck, B., & Cutillas, P. R. (2013).
   Kinase-Substrate Enrichment Analysis provides insights into the heterogeneity of signaling pathway activation in
   leukemia cells. *Science Signaling*, *6*(264),
   rs6. [https://doi.org/10.1126/scisignal.2003573](https://doi.org/10.1126/scisignal.2003573)
2. **Hornbeck, P. V.,** Zhang, B., Murray, B., Kornhauser, J. M., Latham, V., & Skrzypek, E. (2015). PhosphoSitePlus,
   2014: mutations, PTMs and recalibrations. *Nucleic Acids Research*, *43*(D1),
   D512–D520. [https://doi.org/10.1093/nar/gku1267](https://doi.org/10.1093/nar/gku1267)
3. **Horn, H.,** Schoof, E., Kim, J., Robin, X., Miller, M. L., Diella, F., Palma, A., Cesareni, G., Jensen, L. J., &
   Linding, R. (2014). KinomeXplorer: an integrated platform for kinome biology studies. *Nature Methods*, *11*(6),
   603–604. [https://doi.org/10.1038/nmeth.2968](https://doi.org/10.1038/nmeth.2968)
4. **Minguez, P.,** Letunic, I., Parca, L., & Bork, P. (2013). PTMcode: a database of known and predicted functional
   associations between post-translational modifications in proteins. *Nucleic Acids Research*, *41*(D1),
   D306–D311. [https://doi.org/10.1093/nar/gks1230](https://doi.org/10.1093/nar/gks1230)
5. **Linding, R.,** Jensen, L. J., Pasculescu, A., Olhovsky, M., Colwill, K., Bork, P., Yaffe, M. B., & Pawson, T. (
   2008). NetworKIN: a resource for exploring cellular phosphorylation networks. *Nucleic Acids Research*, *36*(Database
   issue),
   D695–D699. [https://doi.org/10.1093/nar/gkm902](https://www.google.com/search?q=https://doi.org/10.1093/nar/gkm902)