import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Node definitions ──────────────────────────────────────────────────────────
NODES = [
    {"id": 0, "name": "Alice", "age": 20, "interest": "ML",     "year": 2},
    {"id": 1, "name": "Bob",   "age": 21, "interest": "WebDev", "year": 3},
    {"id": 2, "name": "Carol", "age": 20, "interest": "DS",     "year": 2},
    {"id": 3, "name": "Dave",  "age": 22, "interest": "Cyber",  "year": 4},
    {"id": 4, "name": "Eve",   "age": 21, "interest": "ML",     "year": 3},
    {"id": 5, "name": "Frank", "age": 20, "interest": "WebDev", "year": 2},
    {"id": 6, "name": "Grace", "age": 22, "interest": "DS",     "year": 4},
    {"id": 7, "name": "Heidi", "age": 21, "interest": "Cyber",  "year": 3},
]

# ── Edge definitions ──────────────────────────────────────────────────────────
EDGES = [
    (0, 1, {"weight": 0.9, "same_interest": 0}),
    (0, 2, {"weight": 0.8, "same_interest": 0}),
    (1, 3, {"weight": 0.7, "same_interest": 0}),
    (2, 4, {"weight": 0.9, "same_interest": 1}),
    (3, 5, {"weight": 0.6, "same_interest": 1}),
    (4, 6, {"weight": 0.8, "same_interest": 1}),
    (5, 7, {"weight": 0.7, "same_interest": 1}),
    (6, 7, {"weight": 0.9, "same_interest": 1}),
    (1, 4, {"weight": 0.5, "same_interest": 1}),
    (2, 5, {"weight": 0.6, "same_interest": 0}),
]

INTEREST_COLORS = {
    "ML":     "#7F77DD",
    "WebDev": "#1D9E75",
    "DS":     "#EF9F27",
    "Cyber":  "#D85A30",
}


def build_graph():
    G = nx.Graph()
    for n in NODES:
        G.add_node(n["id"], name=n["name"], age=n["age"],
                   interest=n["interest"], year=n["year"])
    for u, v, attr in EDGES:
        G.add_edge(u, v, **attr)
    return G


def visualize_graph(G, save_path="outputs/graph.png"):
    os.makedirs("outputs", exist_ok=True)

    pos = nx.spring_layout(G, seed=42)
    labels    = {n["id"]: f"{n['id']}\n{n['name']}" for n in NODES}
    colors    = [INTEREST_COLORS[G.nodes[n]["interest"]] for n in G.nodes]
    weights   = [G[u][v]["weight"] * 4 for u, v in G.edges()]

    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=900, alpha=0.92)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, edge_color="#534AB7")

    edge_labels = {(u, v): f"w={d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    legend = [mpatches.Patch(color=c, label=i) for i, c in INTEREST_COLORS.items()]
    plt.legend(handles=legend, loc="upper left", fontsize=9)
    plt.title("Social Network Graph — College Friend Group", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Graph saved → {save_path}")


if __name__ == "__main__":
    G = build_graph()
    print(f"Nodes : {G.number_of_nodes()}")
    print(f"Edges : {G.number_of_edges()}")
    print(f"Degree sequence : {sorted(dict(G.degree()).items())}")
    visualize_graph(G)