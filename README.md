# Social Network Graph Analysis

A full-stack graph neural network project that models a college
friend group as a graph and recommends new friendships using GCN embeddings.

## Live Demo
https://pritdesai09.github.io/social-network-gnn/

## Features
- Interactive D3.js graph visualization with drag, zoom, hover tooltips
- Adjacency, Degree, Incidence and Laplacian matrices with heatmap coloring
- 8-dimensional GCN node embeddings visualized per node
- Cosine-similarity friend recommendations for any selected node

## Tech Stack
| Layer    | Tools                              |
|----------|------------------------------------|
| Graph    | NetworkX, NumPy                    |
| GNN      | PyTorch, PyTorch Geometric         |
| Frontend | HTML, CSS, JavaScript, D3.js       |
| DevOps   | Git, GitHub, GitHub Pages          |

## Setup

```bash
# Clone
git clone https://github.com/pritdesai09/social-network-gnn.git
cd social-network-gnn

# Python environment
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run pipeline
python src/graph_builder.py
python src/matrix_utils.py
python src/gnn_model.py
python src/export_json.py

# Serve frontend
python -m http.server 8080
# Open http://localhost:8080/frontend/index.html
```

## Project Structure

social-network-gnn/
├── src/                  # Python scripts
├── frontend/             # HTML, CSS, JS + graph_data.json
├── outputs/              # Matrices (.npy, .csv), embeddings
├── docs/                 # Problem design document
└── requirements.txt

## Phases
1. Project setup & Git
2. Python environment & data modelling
3. Graph construction & matrix computation
4. GCN model & node embeddings
5. Dynamic frontend
6. GitHub Pages deployment