# ML Transformer Classification

Transformer encoder for text classification, built from scratch with Multi-Head Attention, LayerNorm, and GELU activation.

## Features
- Multi-Head Self-Attention with masking
- Positional encoding via learnable embeddings
- Stacked Transformer encoder blocks
- OneCycleLR scheduler with AdamW optimizer

## Usage
```python
from src.transformer import TransformerClassifier, train_classifier
model = TransformerClassifier(vocab_size=30000, d_model=256, n_heads=8, n_layers=4, n_classes=3)
train_classifier(model, train_loader, val_loader, epochs=10)
```
