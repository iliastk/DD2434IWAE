import torch.distributions as td
import torch
from torch import nn
import numpy as np
from datasets import BinarizedMNIST

twoPI = torch.tensor(2*np.pi)


class VAE(nn.Module):
    def __init__(self, X_dim, H_dim, Z_dim, num_samples):
        super(VAE, self).__init__()
        self.num_samples = num_samples
        self.mu_z, self.std_z, self.mu_x = None, None, None
        # encoder network for computing mean and std of a Gaussian proposal q(h|x)
        self.encoder_base = nn.Sequential(
            nn.Linear(X_dim, H_dim),nn.Tanh(),
            nn.Linear(H_dim, H_dim),nn.Tanh())
        self.encoder_q_mean = nn.Sequential(
            self.encoder_base,nn.Linear(H_dim, Z_dim))
        self.encoder_q_logvar = nn.Sequential(
            self.encoder_base,nn.Linear(H_dim, Z_dim))

        # decoder network for computing mean of a Bernoulli likelihood p(x|h)
        self.decoder_p_mean = nn.Sequential(
            nn.Linear(Z_dim, H_dim),nn.Tanh(),
            nn.Linear(H_dim, H_dim),nn.Tanh(),
            nn.Linear(H_dim, X_dim),nn.Sigmoid())

    def encode(self, X):
        mu_z = self.encoder_q_mean(X)
        logvar_z = self.encoder_q_logvar(X)
        std_z = torch.exp(logvar_z / 2)

        z = mu_z + std_z * torch.randn_like(std_z)

        self.std_z = std_z
        self.mu_z = mu_z

        return z
    
    def decode(self, Z):
        self.mu_x = self.decoder_p_mean(Z)
        return self.mu_x

    def forward(self, X):

        X = torch.repeat_interleave(X.unsqueeze(1), self.num_samples, dim=1)

        Z = self.encode(X) 
        mu_x = self.decode(Z)

        loss, log_px = self.loss(Z, X)

        return (Z, self.mu_z, self.std_z, self.mu_x), loss, log_px 

    def loss(self, Z, X):
        mu_z, std_z, mu_x = self.mu_z, self.std_z, self.mu_x

        # likelihood: p(x|z, theta) => log(Bernoulli(mu_x))
        logP_XgivenZ = torch.sum(X * torch.log(mu_x) + (1-X) * torch.log(1 - mu_x), dim=-1)

        # prior: p(z|theta) => log(N(0,1))
        logP_Z = torch.sum(-0.5*torch.log(twoPI) - torch.pow(0.5*Z, 2), dim=-1)
        # logP_Z = torch.sum(-0.5*torch.log(twoPI) - 0.5*torch.pow(z, 2), dim=-1) #TODO: which one is the correct formula?

        # posterior: q(z|x, phi) => log(N(mu_z, std_z))
        logQ_ZgivenX = torch.sum(-torch.log(std_z) -
                                 0.5*torch.log(twoPI) -
                                 0.5*torch.pow((Z - mu_z)/std_z, 2),
                                 dim=-1)  # sum over last dimension, i.e, content (mu or std) of each batch

        # TODO: this is the ELBO, right?
        # computing log w function: log(w) = log(p(x,z)) - log(p(z|x))
        log_w = logP_XgivenZ + logP_Z - logQ_ZgivenX

        # normalized weights through Exp-Normalization trick
        max_w = torch.max(log_w, dim=-1)[0].unsqueeze(1)
        w = torch.exp(log_w - max_w)
        # unsqueeze for broadcast
        normalized_w = w / torch.sum(w, dim=-1).unsqueeze(1)
        # loss signal
        loss = torch.sum(normalized_w * log_w, dim=-1)  # sum over num_samples
        loss = -torch.mean(loss)  # mean over batchs

        # computing log likelihood through Log-Sum-Exp trick
        # TODO: How to compute log-likelihood p(x) to compare NLL
        log_px = max_w + torch.log((1/self.num_samples) * torch.sum(w, dim=-1))
        log_px = torch.mean(log_px)  # mean over batches

        return loss, log_px


def iwae_loss():
   # normalized weights through Exp-Normalization trick
    max_w = torch.max(log_w, dim=-1)[0].unsqueeze(1)
    w = torch.exp(log_w - max_w)
    # unsqueeze for broadcast
    normalized_w = w / torch.sum(w, dim=-1).unsqueeze(1)
    # loss signal
    loss = torch.sum(normalized_w * log_w, dim=-1)  # sum over num_samples
    loss = -torch.mean(loss)  # mean over batchs

    # computing log likelihood through Log-Sum-Exp trick
    # sum over num_samples
    log_px = max_w + torch.log((1/num_samples) * torch.sum(w, dim=-1))
    log_px = torch.mean(log_px)  # mean over batches

    return mu_x, log_px, loss
