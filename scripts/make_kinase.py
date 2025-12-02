#!/usr/bin/env python3
import pandas as pd

# 1) Load your original file
df = pd.read_csv("../data/input2.csv")

rows = []

for _, row in df.iterrows():
    gene = str(row["GeneID"])
    psite = str(row["Psite"])

    # Psite like "S_620" → "S620", then Site = "ABL2_S620"
    psite_clean = psite.replace("_", "")
    site_id = f"{gene}_{psite_clean}"

    # Kinase column like "{CDK1,CDK2,CDK5}" → ["CDK1","CDK2","CDK5"]
    kin_str = str(row["Kinase"]).strip("{} ")
    kin_list = [k.strip() for k in kin_str.split(",") if k.strip()]

    for k in kin_list:
        rows.append({
            "Site": site_id,
            "Kinase": k,
            "weight": 1.0
        })

out = pd.DataFrame(rows, columns=["Site", "Kinase", "weight"])

# 2) Save as TSV for the model
out.to_csv("kinase_sites.tsv", sep="\t", index=False)

print("[*] Wrote kinase_sites.tsv with shape", out.shape)