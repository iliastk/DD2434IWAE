import torch.distributions as td
import torch
from torch import nn
import numpy as np
from datasets import BinarizedMNIST

twoPI = torch.tensor(2*np.pi)


class VAE(nn.Module):
    def __init__(self, X_dim, H_dim, Z_dim):
        super(VAE, self).__init__()
        self.X_dim, self.Z_dim = X_dim, Z_dim
        self.Tanh, self.Sigmoid = nn.Tanh(), nn.Sigmoid()
        # encoder
        # deterministic layer
        self.fc1 = nn.Linear(X_dim, H_dim)
        self.fc2 = nn.Linear(H_dim, H_dim)

        # stochastic layer
        self.fc3 = nn.Linear(H_dim, Z_dim)

        # decoder
        # deterministic layer
        self.fc4 = nn.Linear(Z_dim, H_dim)
        self.fc5 = nn.Linear(H_dim, H_dim)

        # stochastic layer
        self.fc6 = nn.Linear(H_dim, X_dim)

    def encode(self, X):
        hidden = self.Tanh(self.fc1(X))
        hidden = self.Tanh(self.fc2(hidden))

        mu_z = self.fc3(hidden)
        std_z = self.fc3(hidden)
        std_z = torch.exp(std_z / 2)

        return mu_z, std_z

    def decode(self, Z):
        hidden = self.Tanh(self.fc4(Z))
        hidden = self.Tanh(self.fc5(hidden))

        mu_x = self.Sigmoid(self.fc6(hidden))

        return mu_x

    def reparametrize(self, mu, std):
        # unit_gaussian = td.Normal(0, 1).rsample(sample_shape=mu.shape)
        unit_gaussian = torch.randn_like(std)
        z = mu + std * unit_gaussian
        return z

    def forward(self, X):
        mu_z, std_z = self.encode(X)
        z = self.reparametrize(mu_z, std_z)
        mu_x = self.decode(z)

        return mu_z, std_z, z, mu_x

    def loss(self, output, target):
        mu_z, std_z, z, mu_x = output
        X = target

        # TODO: Unsqueeze X or sum only in Xdimnesion?
        # posterior: q(z|x, phi) => log(N(mu_z, std_z))
        logQ_ZgivenX = torch.sum(-torch.log(std_z) -
                                 0.5*torch.log(twoPI) -
                                 0.5*torch.pow((z - mu_z)/std_z, 2),
                                 dim=-1)  # sum over last dimension, i.e, content (mu or std) of each batch
        # likelihood: p(x|z, theta) => log(Bernoulli(mu_x))
        logP_XgivenZ = torch.sum(
            X*torch.log(mu_x) + (1-X)*torch.log(mu_x), dim=-1)
        # prior: p(z|theta) => log(N(0,1))
        logP_Z = torch.sum(-0.5*torch.log(twoPI) - 0.5*torch.pow(z, 2), dim=-1)

        # TODO: this is the ELBO, right?
        log_w = logP_Z + logP_XgivenZ - logQ_ZgivenX

        # exp-normalize trick to avoid numerical overflow TODO: is this equiv to softmax?
        max_w = torch.max(log_w, dim=-1)[0]
        w = torch.exp(log_w - max_w)

        normalized_w = w / torch.sum(w, dim=-1)
        loss = torch.sum(normalized_w * log_w)
        loss = -torch.mean(loss)

        # TODO: How to compute log-likelihood p(x) to compare NLL
        # sum over num_samples
        log_px = max_w + torch.log(torch.sum(w, dim=-1))
        log_px = torch.mean(log_px)  # mean over batches
        return loss, log_px
