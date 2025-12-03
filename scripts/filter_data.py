import pandas as pd

# Load both
df1 = pd.read_csv("../data/input1.csv")
df2 = pd.read_csv("../data/input2.csv")

# Allowed GeneIDs and (GeneID, Psite) pairs from input2
allowed_geneids = set(df2["GeneID"].astype(str).unique())
allowed_pairs   = set(map(tuple, df2[["GeneID", "Psite"]].astype(str).values))

def keep_row(row):
    gene  = str(row["GeneID"])
    psite = row["Psite"]

    # CASE B: Psite empty → keep only if GeneID exists in both files
    if pd.isna(psite) or str(psite).strip() == "":
        return gene in allowed_geneids

    # CASE A: Psite not empty → keep only if (GeneID, Psite) pair matches
    return (gene, str(psite)) in allowed_pairs

filtered = df1[df1.apply(keep_row, axis=1)]

filtered.to_csv("filtered_input1.csv", index=False)