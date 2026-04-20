import numpy as np
import networkx as nx
import os


def compute_adjacency(G):
    """A[i][j] = edge weight if edge exists, else 0."""
    n = G.number_of_nodes()
    A = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        A[u][v] = d["weight"]
        A[v][u] = d["weight"]      # undirected — symmetric
    return A


def compute_degree(G):
    """D = diagonal matrix of weighted node degrees."""
    n = G.number_of_nodes()
    D = np.zeros((n, n))
    for node in G.nodes():
        D[node][node] = sum(G[node][nb]["weight"] for nb in G.neighbors(node))
    return D


def compute_incidence(G):
    """B[i][e] = 1 if node i is incident to edge e, else 0."""
    nodes  = sorted(G.nodes())
    edges  = list(G.edges())
    n, m   = len(nodes), len(edges)
    B      = np.zeros((n, m))
    for e_idx, (u, v) in enumerate(edges):
        B[u][e_idx] = 1
        B[v][e_idx] = 1
    return B


def compute_laplacian(D, A):
    """L = D - A   (graph Laplacian)."""
    return D - A


def print_matrix(name, M, node_names=None):
    labels = node_names or [str(i) for i in range(M.shape[0])]
    col_w  = 8
    header = f"{'':>{col_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print('─'*60)
    print(header)
    for i, row in enumerate(M):
        row_str = f"{labels[i]:>{col_w}}" + "".join(f"{v:>{col_w}.2f}" for v in row)
        print(row_str)


def save_matrices(A, D, B, L, folder="outputs"):
    os.makedirs(folder, exist_ok=True)
    np.save(f"{folder}/adjacency.npy",  A)
    np.save(f"{folder}/degree.npy",     D)
    np.save(f"{folder}/incidence.npy",  B)
    np.save(f"{folder}/laplacian.npy",  L)
    # also save as CSV for the frontend
    np.savetxt(f"{folder}/adjacency.csv",  A, delimiter=",", fmt="%.2f")
    np.savetxt(f"{folder}/degree.csv",     D, delimiter=",", fmt="%.2f")
    np.savetxt(f"{folder}/incidence.csv",  B, delimiter=",", fmt="%.2f")
    np.savetxt(f"{folder}/laplacian.csv",  L, delimiter=",", fmt="%.2f")
    print(f"All matrices saved to /{folder}")


if __name__ == "__main__":
    from graph_builder import build_graph, NODES

    G     = build_graph()
    names = [n["name"] for n in NODES]

    A = compute_adjacency(G)
    D = compute_degree(G)
    B = compute_incidence(G)
    L = compute_laplacian(D, A)

    print_matrix("Adjacency Matrix  (A)", A, names)
    print_matrix("Degree Matrix     (D)", D, names)
    print_matrix("Incidence Matrix  (B) — rows=nodes, cols=edges", B)
    print_matrix("Laplacian Matrix  (L = D - A)", L, names)

    # Sanity checks
    print("\n── Sanity Checks ──────────────────────────────────────")
    print(f"A is symmetric          : {np.allclose(A, A.T)}")
    print(f"L is symmetric          : {np.allclose(L, L.T)}")
    eigenvalues = np.linalg.eigvalsh(L)
    print(f"L smallest eigenvalue   : {eigenvalues[0]:.6f}  (should be ≈ 0)")
    print(f"L all eigenvalues >= 0  : {np.all(eigenvalues >= -1e-10)}")

    save_matrices(A, D, B, L)