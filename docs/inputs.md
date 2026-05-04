# Inputs

PhosCrosstalk requires several input files. All paths can be passed on the CLI
or set via `config.toml`.

## Phosphosite time-series (`--data`)

Primary data file containing protein/phosphosite fold-change or intensity
values across time.

**Expected format:**

| Column      | Description                                      |
|-------------|--------------------------------------------------|
| `Protein`   | Gene/protein symbol (or `GeneID`)                |
| `Psite` / `Residue` | Phosphosite identifier (e.g. `S473`)   |
| `x1`…`xN`  | Time-series values at each time point            |

- Rows **with** a valid `Psite`/`Residue` entry = phosphosite data
- Rows **without** a site entry = protein abundance data

```bash
head data_timeseries/filtered_input1.csv
```

!!! warning "Column names"
    The file must contain either `Protein` or `GeneID`, and either
    `Psite` or `Residue`. Any other column name will raise an error.

## mRNA time-series (`--rna-data`)

mRNA fold-change data used to fit the `R(t)` ODE state.

**Expected format:**

```
GeneID,x1,x2,x3,x4,x5,x6,x7,x8,x9
SMAD7,1.02,1.15,...
```

Columns `x1`–`x9` are mapped to time points:
`[4, 8, 15, 30, 60, 120, 240, 480, 960]` minutes.

```bash
head data_timeseries/filtered_input3.csv
```

!!! danger "Critical"
    mRNA fit outputs (`mrna_fit_timeseries.tsv`) are only written when `--rna-data`
    is provided **and** at least one mRNA gene matches a model protein.
    Do not interpret the absence of this file as a model failure.

## TF→mRNA network (`--tf-net`)

Regulatory network describing which transcription factors regulate which genes.

**Required columns (case-insensitive):** `Source`, `Target`, `Weight`

- `Source` = TF gene symbol
- `Target` = regulated gene symbol
- `Weight` = regulatory weight (e.g. correlation coefficient)

```bash
head data_interactions/tf_mrna.csv
```

!!! warning "Direction matters"
    `Source,Target,Weight` means TF **Source** regulates gene **Target**.
    Swapping source and target will invert the regulatory signal.

**Check TF symbols against RNA data:**

```bash
awk -F',' '
  NR==FNR { if (FNR > 1) rna[$1]; next }
  FNR > 1 {
    if (!($1 in rna)) missing[$1]
    if (!($2 in rna)) missing[$2]
  }
  END { for (g in missing) print g }
' data_timeseries/filtered_input3.csv data_interactions/tf_mrna.csv | sort
```

## PTMcode2 databases (`--ptm-intra`, `--ptm-inter`)

SQLite databases of PTM functional associations derived from PTMcode2.

- `--ptm-intra`: intra-protein site pairs (table `intra_pairs`)
- `--ptm-inter`: inter-protein site pairs (table `inter_pairs`)

These are used to build the global crosstalk coupling matrix `Cg`.

## Kinase-site prior (`--kinase-tsv` or `--kea-ks-table`)

Maps phosphosites to their kinases. At least one must be provided.

**`--kinase-tsv` format:**

| Column  | Description            |
|---------|------------------------|
| `Site`  | Site label (Protein_Residue) |
| `Kinase`| Kinase gene symbol     |
| `weight`| Interaction weight (optional) |

```bash
head data_interactions/kinase_sites.tsv
```

**`--kea-ks-table` format (KEA/KS table):**

| Column           | Description              |
|------------------|--------------------------|
| `substrate_site` | Phosphosite label        |
| `kinase`         | Kinase name              |
| `pmid`           | PubMed ID (used for count-based weighting) |

!!! note "Fallback"
    If neither `--kinase-tsv` nor `--kea-ks-table` is provided, the model
    uses an identity kinase-site mapping (each site has one implicit kinase).

## Kinase-kinase graph (`--unified-graph-pkl`)

Optional pickled NetworkX graph describing kinase-kinase regulatory
relationships. Used to build the Laplacian regularizer `L_alpha`.

If absent, `L_alpha` is set to zero (no kinase network regularization).
