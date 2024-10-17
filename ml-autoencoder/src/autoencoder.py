import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], noise_factor=0.3):
        super().__init__()
        self.noise_factor = noise_factor
        dims = [input_dim] + hidden_dims

        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.BatchNorm1d(dims[i+1])]
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(len(dims) - 1, 0, -1):
            if i == 1:
                decoder_layers += [nn.Linear(dims[i], dims[i-1]), nn.Sigmoid()]
            else:
                decoder_layers += [nn.Linear(dims[i], dims[i-1]), nn.ReLU(), nn.BatchNorm1d(dims[i-1])]
        self.decoder = nn.Sequential(*decoder_layers)

    def add_noise(self, x):
        return x + self.noise_factor * torch.randn_like(x)

    def forward(self, x, training=False):
        if training:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x
        encoded = self.encoder(x_noisy)
        return self.decoder(encoded)

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_enc = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        enc = self.encoder(x).view(x.size(0), -1)
        z = self.fc_enc(enc)
        dec_input = self.fc_dec(z).view(-1, 128, 4, 4)
        return self.decoder(dec_input)


def train_autoencoder(model, X, epochs=50, lr=1e-3, batch_size=64, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = torch.FloatTensor(X).to(device)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (batch,) in loader:
            recon = model(batch, training=True) if hasattr(model, 'noise_factor') else model(batch)
            loss = F.mse_loss(recon, batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.6f}")
    return model
