#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:-proteins.unique.txt}
OUT=${2:-idmap_ptmcode_uniprot.tsv}

> "$OUT"

while read -r gene; do
  [[ -z "$gene" ]] && continue

  # Query UniProtKB: human only, by gene name
  res=$(curl -s \
    "https://rest.uniprot.org/uniprotkb/search?query=gene:${gene}+AND+organism_id:9606&fields=accession,reviewed,gene_names&format=tsv")

  # If nothing returned (or just header), mark NA
  n_lines=$(echo "$res" | wc -l)
  if [[ "$n_lines" -le 1 ]]; then
    echo -e "${gene}\tNA" >> "$OUT"
    continue
  fi

  # Process the TSV (skip header), pick best row:
  # 1) reviewed == reviewed
  # 2) gene name list contains the symbol as a separate token
  # 3) fallback: first data line
  acc=$(
    echo "$res" \
    | awk -v gene="$gene" '
        BEGIN {
          FS = "\t";
          best_acc = "";
        }
        NR == 1 { next }  # skip header
        {
          entry    = $1;
          reviewed = $2;
          genes    = $4;

          # keep the very first line as fallback
          if (best_acc == "") {
            best_acc = entry;
          }

          # Prefer reviewed
          if (reviewed == "reviewed") {
            # check if gene symbol appears as a separate token
            n = split(genes, g, /[ ,;]/);
            for (i = 1; i <= n; i++) {
              if (g[i] == gene) {
                best_acc = entry;
                # found best possible – we can stop early
                print best_acc;
                exit 0;
              }
            }
            # if reviewed but no exact symbol match, still better than unreviewed
            best_acc = entry;
          }
        }
        END {
          if (best_acc != "") {
            print best_acc;
          }
        }
    '
  )

  if [[ -z "$acc" ]]; then
    echo -e "${gene}\tNA" >> "$OUT"
  else
    echo -e "${gene}\t${acc}" >> "$OUT"
  fi

  # optional: be nice to UniProt
  # sleep 0.1

done < "$INPUT"
