# Troubleshooting

## Data loading errors

### Wrong `tf_mrna.csv` header direction

**Symptom:** TF regulatory signal is inverted; genes that should be upregulated
appear suppressed.

**Cause:** `Source` and `Target` columns are swapped. The convention is:
`Source` = TF (transcription factor), `Target` = regulated gene.

**Fix:** Verify and correct the column order:
```bash
head data_interactions/tf_mrna.csv
# Expected: Source,Target,Weight
```

---

### mRNA gene symbols not found in model

**Symptom:** Warning `[!] No RNA genes matched model proteins. RNA loss disabled.`

**Cause:** The gene symbols in `--rna-data` do not match the protein symbols
in `--data`.

**Fix:**
```bash
# List unmatched TF symbols
awk -F',' '
  NR==FNR { if (FNR > 1) rna[$1]; next }
  FNR > 1 { if (!($1 in rna)) missing[$1]; if (!($2 in rna)) missing[$2] }
  END { for (g in missing) print g }
' data_timeseries/filtered_input3.csv data_interactions/tf_mrna.csv | sort
```

---

### No kinase-site overlap

**Symptom:** Warning or error about empty kinase-site matrix.

**Cause:** No sites in `--kinase-tsv` or `--kea-ks-table` match the model sites.

**Fix:** Check that site labels use the same format in both files (e.g. `PROT_S473`):
```bash
head data_interactions/kinase_sites.tsv
```

---

## RNA fit outputs missing

**Symptom:** `mrna_fit_timeseries.tsv` is not written after a run.

**Cause:** One of:
1. `--rna-data` was not provided
2. No mRNA gene symbols matched model proteins
3. Simulated `R(t)` was not available from the solver

**This is not a bug.** RNA fit outputs are only written when all three conditions
are satisfied. Check the log for `[*] RNA-to-model mapping: N matched genes`.

!!! danger "Do not interpret input mRNA as fitted mRNA"
    Never copy `--rna-data` values into `mrna_fit_timeseries.tsv` manually.
    The file must contain simulated `R(t)` values from the ODE.

---

## Diffrax non-finite solve

**Symptom:** Loss is `NaN` or `inf`; optimization fails immediately.

**Cause:** The ODE solver produced non-finite output, usually due to:
- Initial parameter values outside plausible range
- Very stiff dynamics (try tighter `rtol`/`atol` in `config.toml`)
- Extremely large loss weights

**Fix:** Start with a smoke test to verify the ODE runs:
```bash
phoscrosstalk ... --n-starts 1 --max-steps 50
```
Check `pareto_front.npz` for `NaN` losses.

---

## Parameter label mismatch

**Symptom:** Errors like `IndexError` or `ValueError` when decoding parameters,
or wrong proteins/sites in output files.

**Cause:** The parameter vector dimension assumes `2K + 2 + 3M + N + 4`. If
model dimensions changed between a previous run and the current one, a stale
`fitted_params.npz` may have an incompatible length.

**Fix:** Always rerun from scratch when the input data changes. Do not reuse
`fitted_params.npz` from a run with different protein/kinase/site counts.

---

## Missing `data_t*`/`sim_t*` columns in `protein_fit_timeseries.tsv`

**Symptom:** `plot_goodness_of_fit` raises `ValueError: No sim_t*/data_t* columns found`.

**Cause:** The TSV was written by an older pipeline version that used different
column naming.

**Fix:** Re-run the full pipeline to regenerate outputs with the current format.

---

## All multi-start runs fail

**Symptom:** `RuntimeError` at the end of optimization.

**Cause:** Every optimization start returned a non-finite loss.

**Fix:**
1. Verify data files are correctly formatted
2. Check that kinase-site matrix is not all zeros
3. Run a smoke test with `--n-starts 1 --max-steps 50`
4. Inspect solver `rtol`/`atol` in `config.toml`

---

## Streamlit dashboard crashes

**Symptom:** Dashboard errors with `KeyError` or `FileNotFoundError`.

**Cause:** A required result file is missing, or the file was written by an
older pipeline version.

**Fix:** Re-run the full optimization pipeline and then relaunch the dashboard:
```bash
uv run streamlit run phoscrosstalk/app.py
```
