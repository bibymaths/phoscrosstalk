#!/usr/bin/env python3
"""
Build multiplex kinase networks and a unified kinase graph
from KEA/KEA3 kinase network ZIP (TSV files).

- Per-layer graphs: layer_graphs[layer] = nx.Graph
- Unified graph: U (nx.Graph) with edge attributes:
    - layers: set of layer names that support this edge
    - weights: dict[layer] -> float
    - weight_sum, weight_mean, support
"""
import json
import pickle
import zipfile
from pathlib import Path
from typing import Dict

import pandas as pd
import networkx as nx


def load_kinase_network_tsvs(zip_path: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Load all kk_*_net.tsv files from the KEA kinase network zip.

    Returns
    -------
    layer_dfs : dict
        {layer_name: DataFrame(source, target, weight)}
        where layer_name is filename without 'kk_' and '.tsv'
        e.g. 'kk_degene_net.tsv' -> 'degene_net'
    """
    zip_path = Path(zip_path)
    layer_dfs: Dict[str, pd.DataFrame] = {}

    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".tsv"):
                continue

            with zf.open(name) as f:
                df = pd.read_csv(f, sep="\t")

            # basic sanity checks
            required = {"source", "target"}
            if not required.issubset(df.columns):
                raise ValueError(
                    f"{name} missing required columns {required}. "
                    f"Found: {df.columns.tolist()}"
                )

            if "weight" not in df.columns:
                df["weight"] = 1.0

            layer_name = (
                name.replace(".tsv", "")
                    .replace("kk_", "")   # kk_degene_net.tsv -> degene_net
                    .strip()
            )
            layer_dfs[layer_name] = df

    return layer_dfs


def build_layer_graphs(
    layer_dfs: Dict[str, pd.DataFrame],
    directed: bool = False
) -> Dict[str, nx.Graph]:
    """
    Build one NetworkX graph per layer.

    - If multiple edges (u, v) appear in a layer, weights are summed.
    """
    graphs: Dict[str, nx.Graph] = {}
    G_cls = nx.DiGraph if directed else nx.Graph

    for layer, df in layer_dfs.items():
        G = G_cls()

        for _, row in df.iterrows():
            s = row["source"]
            t = row["target"]
            w = float(row.get("weight", 1.0))

            if G.has_edge(s, t):
                # accumulate weights within layer
                G[s][t]["weight"] += w
            else:
                G.add_edge(s, t, weight=w, layer=layer)

        graphs[layer] = G

    return graphs


def build_unified_graph(layer_graphs: Dict[str, nx.Graph]) -> nx.Graph:
    """
    Build a unified multiplex graph U from per-layer graphs.

    For each (u, v) pair across all layers, we store:
        - layers: set of layer names
        - weights: dict[layer] -> weight
        - weight_sum: sum(weights.values())
        - weight_mean: mean(weights.values())
        - support: number of layers
    """
    U = nx.Graph()

    for layer, G in layer_graphs.items():
        for u, v, data in G.edges(data=True):
            w = float(data.get("weight", 1.0))

            if not U.has_node(u):
                U.add_node(u)
            if not U.has_node(v):
                U.add_node(v)

            if U.has_edge(u, v):
                e = U[u][v]
                e["layers"].add(layer)
                e["weights"][layer] = w
            else:
                U.add_edge(
                    u,
                    v,
                    layers={layer},
                    weights={layer: w},
                )

    # derive aggregate attributes
    for u, v, data in U.edges(data=True):
        ws = list(data["weights"].values())
        weight_sum = float(sum(ws))
        weight_mean = float(weight_sum / len(ws))

        data["weight_sum"] = weight_sum
        data["weight_mean"] = weight_mean
        data["support"] = len(ws)

    return U


def summarize_layer_graphs(layer_graphs: Dict[str, nx.Graph]) -> None:
    """
    Print a simple summary (nodes/edges) for each layer.
    """
    print("Per-layer kinase networks:")
    for layer, G in sorted(layer_graphs.items()):
        print(f"  {layer:20s}: {G.number_of_nodes():4d} nodes, "
              f"{G.number_of_edges():7d} edges")
    print()


def summarize_unified_graph(U: nx.Graph) -> None:
    """
    Print high-level summary for unified graph.
    """
    n_nodes = U.number_of_nodes()
    n_edges = U.number_of_edges()
    print(f"Unified graph U: {n_nodes} nodes, {n_edges} edges")

    # quick idea of density and support
    supports = [d["support"] for _, _, d in U.edges(data=True)]
    avg_support = sum(supports) / len(supports)
    print(f"Average support (layers per edge): {avg_support:.2f}")


def main(zip_path: str | Path = "../data/kea2/kinase_networks.zip") -> None:
    layer_dfs = load_kinase_network_tsvs(zip_path)
    layer_graphs = build_layer_graphs(layer_dfs, directed=False)
    U = build_unified_graph(layer_graphs)

    summarize_layer_graphs(layer_graphs)
    summarize_unified_graph(U)

    # 1) Save unified graph in a NetworkX-native format (keeps sets/dicts)
    out_gpickle = Path("unified_kinase_graph.gpickle")
    print(f"Saving unified graph (native, with sets/dicts) to {out_gpickle} ...")
    with open(out_gpickle, "wb") as f:
        pickle.dump(U, f)

    # 2) Build a GraphML-friendly copy (no sets/dicts)
    U_gml = nx.Graph()
    U_gml.add_nodes_from(U.nodes())

    for u, v, data in U.edges(data=True):
        d = dict(data)
        # convert layers (set) -> comma-separated string
        if "layers" in d and isinstance(d["layers"], (set, list, tuple)):
            d["layers"] = ",".join(sorted(d["layers"]))
        # convert weights (dict) -> JSON string
        if "weights" in d and isinstance(d["weights"], dict):
            d["weights"] = json.dumps(d["weights"])
        U_gml.add_edge(u, v, **d)

    out_unified = Path("unified_kinase_graph.graphml")
    print(f"Saving unified graph (GraphML-safe) to {out_unified} ...")
    nx.write_graphml(U_gml, out_unified)

    # 3) Optionally: save each layer as separate GraphML (these are already simple)
    out_dir = Path("layer_graphs")
    out_dir.mkdir(exist_ok=True)
    for layer, G in layer_graphs.items():
        out_file = out_dir / f"{layer}.graphml"
        nx.write_graphml(G, out_file)
        # if you also want native versions:
        # nx.write_gpickle(G, out_dir / f"{layer}.gpickle")


if __name__ == "__main__":
    main()
