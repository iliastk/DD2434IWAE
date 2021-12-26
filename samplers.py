
import torch
from torch import nn


class GaussianSampler(nn.Module):
    def __init__(self, X_dim, H_dim, Z_dim):
        super(GaussianSampler, self).__init__()
        self.mu_z, self.std_z = None, None
        layers = []
        for layer_dim in H_dim:
            layers.append(nn.Linear(X_dim, layer_dim))
            layers.append(nn.Tanh())
            X_dim = layer_dim
        self.base_encoder = nn.Sequential(*layers)

        self.encoder_mean = nn.Sequential(
            self.base_encoder, nn.Linear(H_dim[-1], Z_dim))
        self.encoder_logvar = nn.Sequential(
            self.base_encoder, nn.Linear(H_dim[-1], Z_dim))

    def forward(self, X):
        mean = self.encoder_mean(X)
        logvar = self.encoder_logvar(X)
        std = torch.exp(logvar / 2)

        z = mean + std * torch.randn_like(std)

        self.std = std
        self.mean = mean

        return z


class BernoulliSampler(nn.Module):
    def __init__(self, X_dim, H_dim, Z_dim):
        super(BernoulliSampler, self).__init__()
        self.mu = None
        layers = []
        for layer_dim in list(reversed(H_dim)):
            layers.append(nn.Linear(Z_dim, layer_dim))
            layers.append(nn.Tanh())
            Z_dim = layer_dim
        self.base_encoder = nn.Sequential(*layers)

        self.encoder_mean = nn.Sequential(
            self.base_encoder, nn.Linear(H_dim[0], X_dim), nn.Sigmoid())

    def forward(self, X):
        self.mean = self.encoder_mean(X)
        return self.mu
