#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import tomllib


def toml_list(values: list[str]) -> str:
    return "[" + ", ".join(f'"{v}"' for v in values) + "]"


def extract_site_protein(site: str) -> str:
    """
    EPHA2_Y575 -> EPHA2
    MAPK1_T185_Y187 -> MAPK1
    """
    return site.split("_", 1)[0].strip()


def replace_or_insert_model_key(text: str, key: str, value: str) -> str:
    pattern = rf"(?m)^({re.escape(key)}\s*=\s*).*$"

    if re.search(pattern, text):
        return re.sub(pattern, rf"\1{value}", text)

    model_match = re.search(r"(?m)^\[model\]\s*$", text)
    if not model_match:
        raise ValueError("No [model] section found in config.toml")

    insert_at = model_match.end()
    return text[:insert_at] + f"\n{key} = {value}" + text[insert_at:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config.toml")
    args = parser.parse_args()

    config_path = Path(args.config)

    with config_path.open("rb") as fh:
        cfg = tomllib.load(fh)

    kinase_tsv = Path(cfg["paths"]["kinase_tsv"])
    tf_net = Path(cfg["paths"]["tf_net"])

    kinase_df = pd.read_csv(kinase_tsv, sep="\t")
    tf_df = pd.read_csv(tf_net)

    site_proteins = {
        extract_site_protein(site)
        for site in kinase_df["Site"].astype(str)
        if site.strip()
    }

    kinases = {
        kinase.strip()
        for kinase in kinase_df["Kinase"].astype(str)
        if kinase.strip()
    }

    tf_targets = {
        target.strip()
        for target in tf_df["Target"].astype(str)
        if target.strip()
    }

    # Conservative automatic rule:
    # receptor = site protein that is also encoded as kinase and appears in TF target layer
    receptors = sorted(site_proteins & kinases & tf_targets)

    # Kinase layer stimulus candidates:
    # all kinases in the kinase-site prior for this subnetwork
    receptor_kinases = sorted(kinases)

    text = config_path.read_text()

    text = replace_or_insert_model_key(
        text,
        "receptors",
        toml_list(receptors),
    )
    text = replace_or_insert_model_key(
        text,
        "receptor_kinases",
        toml_list(receptor_kinases),
    )

    config_path.write_text(text)

    print(f"Updated {config_path}")
    print(f"receptors        = {receptors}")
    print(f"receptor_kinases = {receptor_kinases}")


if __name__ == "__main__":
    main()