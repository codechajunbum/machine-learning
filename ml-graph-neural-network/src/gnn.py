import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = x @ self.weight
        out = adj @ support
        if self.bias is not None:
            out += self.bias
        return out


class GCN(nn.Module):
    def __init__(self, in_features, hidden_dim, n_classes, dropout=0.5):
        super().__init__()
        self.gc1 = GraphConvLayer(in_features, hidden_dim)
        self.gc2 = GraphConvLayer(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.gc2(x, adj)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = out_features // n_heads
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.FloatTensor(2 * self.d_k, 1))
        nn.init.xavier_uniform_(self.a)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        Wh = self.W(x)
        B, N, D = Wh.shape if Wh.dim() == 3 else (1, *Wh.shape)
        Wh = Wh.view(-1, self.n_heads, self.d_k)
        N = Wh.shape[0]
        Wh_i = Wh.unsqueeze(1).expand(N, N, self.n_heads, self.d_k)
        Wh_j = Wh.unsqueeze(0).expand(N, N, self.n_heads, self.d_k)
        e = self.leaky(torch.cat([Wh_i, Wh_j], dim=-1) @ self.a).squeeze(-1).mean(-1)
        zero_vec = -9e15 * torch.ones_like(e)
        attn = torch.where(adj > 0, e, zero_vec)
        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)
        return (attn.unsqueeze(-1) * Wh.unsqueeze(0)).sum(1).view(N, -1)


def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.diag(np.power(adj.sum(1), -0.5))
    return torch.FloatTensor(d @ adj @ d)


def train_gcn(model, features, adj, labels, train_mask, epochs=200, lr=1e-2, weight_decay=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(features, adj)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            acc = (out[train_mask].argmax(1) == labels[train_mask]).float().mean()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} Acc: {acc:.4f}")
