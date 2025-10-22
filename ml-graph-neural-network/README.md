# ML Graph Neural Network

GCN (Graph Convolutional Network) and GAT (Graph Attention Network) implemented from scratch for node classification.

## Features
- Graph Convolutional Layer with normalized adjacency
- Multi-head Graph Attention Layer
- Symmetric adjacency normalization (D^-0.5 A D^-0.5)
- Semi-supervised node classification

## Usage
```python
from src.gnn import GCN, normalize_adj, train_gcn
adj_norm = normalize_adj(adj_matrix)
model = GCN(in_features=1433, hidden_dim=64, n_classes=7)
train_gcn(model, features, adj_norm, labels, train_mask, epochs=200)
```
