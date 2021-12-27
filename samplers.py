
import torch
from torch import nn


class GaussianSampler(nn.Module):
    def __init__(self, X_dim, H_dim, Z_dim):
        super(GaussianSampler, self).__init__()
        self.mean, self.std = None, None
        layers = []
        for layer_dim in H_dim:
            layers.append(nn.Linear(X_dim, layer_dim))
            layers.append(nn.Tanh())
            X_dim = layer_dim
        self.base_net = nn.Sequential(*layers)

        self.mean_net = nn.Sequential(
            self.base_net, nn.Linear(H_dim[-1], Z_dim))
        self.logvar_net = nn.Sequential(
            self.base_net, nn.Linear(H_dim[-1], Z_dim))


    def forward(self, X):
        mean = self.mean_net(X)
        logvar = self.logvar_net(X)
        std = torch.exp(logvar / 2)

        z = mean + std * torch.randn_like(std)

        self.std = std
        self.mean = mean

        return z


class BernoulliSampler(nn.Module):
    def __init__(self, X_dim, H_dim, Z_dim):
        super(BernoulliSampler, self).__init__()
        self.mean = None
        layers = []
        for layer_dim in list(reversed(H_dim)):
            layers.append(nn.Linear(Z_dim, layer_dim))
            layers.append(nn.Tanh())
            Z_dim = layer_dim
        self.base_net = nn.Sequential(*layers)

        self.mean_net = nn.Sequential(
            self.base_net, nn.Linear(H_dim[0], X_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        self.mean = self.sigmoid(self.mean_net(X))
        return self.mean
