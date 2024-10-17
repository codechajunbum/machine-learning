# ML Autoencoder

Denoising Autoencoder and Convolutional Autoencoder for unsupervised representation learning.

## Features
- Denoising Autoencoder with configurable hidden layers and Gaussian noise
- Convolutional Autoencoder for image reconstruction
- BatchNorm + ReLU encoder, Sigmoid decoder
- Latent space encoding for downstream tasks

## Usage
```python
from src.autoencoder import DenoisingAutoencoder, train_autoencoder
model = DenoisingAutoencoder(input_dim=784, hidden_dims=[256, 128, 64])
train_autoencoder(model, X_train, epochs=50)
latent = model.encode(torch.FloatTensor(X_test))
```
