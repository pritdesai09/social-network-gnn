import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from graph_builder import build_graph, NODES

# ── 1. Build node feature matrix X  (shape: [8, 6]) ──────────────────────────
# Features per node: [age_norm, year_norm, is_ML, is_WebDev, is_DS, is_Cyber]

INTEREST_MAP = {"ML": 0, "WebDev": 1, "DS": 2, "Cyber": 3}

def build_node_features():
    rows = []
    for n in NODES:
        age_norm  = (n["age"]  - 20) / 2.0     # normalise 20-22 → 0-1
        year_norm = (n["year"] -  2) / 2.0     # normalise 2-4  → 0-1
        one_hot   = [0.0, 0.0, 0.0, 0.0]
        one_hot[INTEREST_MAP[n["interest"]]] = 1.0
        rows.append([age_norm, year_norm] + one_hot)
    return torch.tensor(rows, dtype=torch.float)


# ── 2. Build edge_index + edge_attr tensors ───────────────────────────────────

def build_edge_tensors(G):
    src, dst, weights, same_int = [], [], [], []
    for u, v, d in G.edges(data=True):
        # add both directions (undirected)
        for a, b in [(u, v), (v, u)]:
            src.append(a);  dst.append(b)
            weights.append(d["weight"])
            same_int.append(float(d["same_interest"]))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(
        list(zip(weights, same_int)), dtype=torch.float
    )
    return edge_index, edge_attr


# ── 3. PyG Data object ────────────────────────────────────────────────────────

def build_pyg_data():
    G            = build_graph()
    x            = build_node_features()
    edge_index, edge_attr = build_edge_tensors(G)
    data         = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_names = [n["name"] for n in NODES]
    return data


# ── 4. GCN model (2 layers) ───────────────────────────────────────────────────

class SocialGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels,     hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Layer 1: aggregate 1-hop neighbours → ReLU
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        # Layer 2: aggregate 2-hop neighbours → final embedding
        h = self.conv2(h, edge_index)
        return h


# ── 5. Train (unsupervised: reconstruct adjacency) ────────────────────────────

def train(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    losses = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)          # [8, out_dim] embeddings
        # Reconstruction loss: dot-product of connected node embeddings
        src = data.edge_index[0]
        dst = data.edge_index[1]
        pos_score = (z[src] * z[dst]).sum(dim=-1)   # should be high for edges
        loss = -pos_score.mean()                     # maximise similarity
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:>3} | Loss: {loss.item():.4f}")
    return losses


# ── 6. Trace computational graph for Node 0 (Alice) ──────────────────────────

def trace_computational_graph(data, target=0):
    G_nx = build_graph()
    names = data.node_names

    hop1 = list(G_nx.neighbors(target))
    hop2 = set()
    for nb in hop1:
        for nb2 in G_nx.neighbors(nb):
            if nb2 != target and nb2 not in hop1:
                hop2.add(nb2)

    print(f"\n{'─'*55}")
    print(f"  Computational graph for Node {target} ({names[target]})")
    print(f"{'─'*55}")
    print(f"  Target node  : {target} ({names[target]})")
    print(f"  1-hop nbrs   : {[(n, names[n]) for n in hop1]}")
    print(f"  2-hop nbrs   : {[(n, names[n]) for n in sorted(hop2)]}")
    print(f"\n  Layer 1 aggregates messages from: {[names[n] for n in hop1]}")
    print(f"  Layer 2 aggregates messages from: {[names[n] for n in sorted(hop2)]}")
    return hop1, sorted(hop2)


# ── 7. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    data  = build_pyg_data()

    print("── PyG Data Object ─────────────────────────────────────")
    print(f"  Node feature matrix x : {data.x.shape}")
    print(f"  Edge index            : {data.edge_index.shape}")
    print(f"  Edge attributes       : {data.edge_attr.shape}")
    print(f"\n  Node features (first 3 rows):")
    for i in range(3):
        print(f"    {data.node_names[i]:>6}: {data.x[i].tolist()}")

    # Trace computational graph BEFORE training
    hop1, hop2 = trace_computational_graph(data, target=0)

    # Build and train model
    model = SocialGCN(
        in_channels=6,       # 6 node features
        hidden_channels=16,
        out_channels=8       # 8-dim embedding
    )
    print(f"\n── Model Architecture ──────────────────────────────────")
    print(model)

    print(f"\n── Training (200 epochs) ───────────────────────────────")
    losses = train(model, data, epochs=200)

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)

    print(f"\n── Node Embeddings  (shape: {embeddings.shape}) ────────")
    for i, name in enumerate(data.node_names):
        vec = [f"{v:.3f}" for v in embeddings[i].tolist()]
        print(f"  {i} {name:>6}: [{', '.join(vec)}]")

    # Target node embedding
    alice_emb = embeddings[0]
    print(f"\n── Alice (Node 0) Embedding ────────────────────────────")
    print(f"  Vector : {alice_emb.tolist()}")
    print(f"  L2 norm: {torch.norm(alice_emb).item():.4f}")

    # Cosine similarity → friend recommendations
    print(f"\n── Friend Recommendations for Alice ────────────────────")
    sims = F.cosine_similarity(alice_emb.unsqueeze(0), embeddings)
    ranked = sorted(
        [(i, sims[i].item(), data.node_names[i])
         for i in range(len(data.node_names)) if i != 0],
        key=lambda x: -x[1]
    )
    for rank, (idx, score, name) in enumerate(ranked, 1):
        already = "✓ friends" if idx in hop1 else "  suggest"
        print(f"  {rank}. {name:>6} (node {idx})  sim={score:.4f}  {already}")

    # Save embeddings for frontend
    os.makedirs("outputs", exist_ok=True)
    emb_dict = {
        "embeddings": embeddings.tolist(),
        "node_names": data.node_names,
        "losses":     losses,
        "hop1":       hop1,
        "hop2":       hop2,
        "similarities": [{"node": i, "name": data.node_names[i],
                           "score": sims[i].item()} for i in range(8)]
    }
    import json
    with open("outputs/embeddings.json", "w") as f:
        json.dump(emb_dict, f, indent=2)
    print(f"\n  Embeddings saved → outputs/embeddings.json")