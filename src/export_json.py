import json, os
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(__file__))
from graph_builder import build_graph, NODES, EDGES

def export_all():
    G = build_graph()

    # Graph structure
    nodes = [{"id": n["id"], "name": n["name"],
               "interest": n["interest"], "age": n["age"],
               "year": n["year"]} for n in NODES]

    edges = [{"source": u, "target": v,
               "weight": d["weight"],
               "same_interest": d["same_interest"]}
             for u, v, d in G.edges(data=True)]

    # Matrices from saved CSVs
    def load_csv(name):
        path = f"outputs/{name}.csv"
        return np.loadtxt(path, delimiter=",").tolist()

    payload = {
        "nodes":     nodes,
        "edges":     edges,
        "adjacency": load_csv("adjacency"),
        "degree":    load_csv("degree"),
        "incidence": load_csv("incidence"),
        "laplacian": load_csv("laplacian"),
    }

    # Merge embeddings if available
    emb_path = "outputs/embeddings.json"
    if os.path.exists(emb_path):
        with open(emb_path) as f:
            payload["gnn"] = json.load(f)

    out = "outputs/graph_data.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Full frontend payload saved → {out}")

if __name__ == "__main__":
    export_all()