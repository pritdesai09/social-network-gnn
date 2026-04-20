# Problem Design — Social Network Graph

## Problem Statement
Model a college friend group as a graph and recommend new friendships
using node embeddings from a Graph Convolutional Network (GCN).

## Graph Definition
- Type: Undirected, weighted graph
- Nodes: Users (students)
- Edges: Existing friendships between users

## Nodes (8 users)
| Node ID | Name    | Age | Interest         | Year |
|---------|---------|-----|------------------|------|
| 0       | Alice   | 20  | Machine Learning | 2   |
| 1       | Bob     | 21  | Web Dev          | 3   |
| 2       | Carol   | 20  | Data Science     | 2   |
| 3       | Dave    | 22  | Cybersecurity    | 4   |
| 4       | Eve     | 21  | Machine Learning | 3   |
| 5       | Frank   | 20  | Web Dev          | 2   |
| 6       | Grace   | 22  | Data Science     | 4   |
| 7       | Heidi   | 21  | Cybersecurity    | 3   |

## Node Features (per node, shape: [8 x 3])
| Feature    | Type        | Encoding                          |
|------------|-------------|-----------------------------------|
| age        | numerical   | raw integer (20–22)               |
| interest   | categorical | one-hot (ML, WebDev, DS, Cyber)   |
| year       | numerical   | raw integer (2–4)                 |

## Edges (friendships)
| Edge  | From | To  | Weight |
|-------|------|-----|--------|
| e0    | 0    | 1   | 0.9    |
| e1    | 0    | 2   | 0.8    |
| e2    | 1    | 3   | 0.7    |
| e3    | 2    | 4   | 0.9    |
| e4    | 3    | 5   | 0.6    |
| e5    | 4    | 6   | 0.8    |
| e6    | 5    | 7   | 0.7    |
| e7    | 6    | 7   | 0.9    |
| e8    | 1    | 4   | 0.5    |
| e9    | 2    | 5   | 0.6    |

## Edge Features (per edge, shape: [10 x 2])
| Feature          | Type      | Description                        |
|------------------|-----------|------------------------------------|
| weight           | numerical | strength of friendship (0.0–1.0)  |
| same_interest    | binary    | 1 if both nodes share an interest |

## Target Node for Embedding (Phase 4)
Node 0 (Alice) — trace her 2-hop computational graph

## Task
Node-level: learn embeddings via 2-layer GCN,
use cosine similarity on embeddings to rank friend recommendations.