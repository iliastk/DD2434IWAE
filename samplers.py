
import torch
from torch import nn

class Sampler(nn.Module):
    def __init__(self, X_dim, hidden_dims, Z_dim, sampler_kind, is_encoder=True):
        super(Sampler, self).__init__()
        self.mu_z, self.std_z = None, None
        self.sampler_kind = sampler_kind
        layer_sizes = [X_dim] + hidden_dims + [Z_dim]
        if not is_encoder:
            layer_sizes.reverse()

        self.input_dim  = X_dim if is_encoder else Z_dim
        output_dim = Z_dim if is_encoder else X_dim

        layers = [nn.Identity()]
        in_out_pairs = list(zip(layer_sizes, layer_sizes[1:]))

        next_dim = self.input_dim
        for prev_dim, next_dim in in_out_pairs[:-1]:
            layers.append(nn.Linear(prev_dim, next_dim))
            layers.append(nn.Tanh())
        
        self.base_net = nn.Sequential(*layers)

        if sampler_kind == 'Gaussian':
            self.mean_net = nn.Sequential(
                self.base_net, nn.Linear(next_dim, output_dim))
            # Why not define exp(encode_logvar) at once?
            self.logvar_net = nn.Sequential(
                self.base_net, nn.Linear(next_dim, output_dim))
        elif sampler_kind == 'Bernoulli':
            self.mean_net = nn.Sequential(
                self.base_net, nn.Linear(next_dim, output_dim), nn.Sigmoid())
        else: assert False

        # What does this do?
        #self.apply(self.init)

        # if not is_encoder:
        #     self.encoder_logvar[-1].bias = torch.tensor(10)
            

        # Bias for the output layer? Why?
        # What is it that we can pass in to this? How does it work?
        # ANS: we pass in the average of all training samples. Not sure why,
        # shouldn't the model be able to learn the bias param by itself?
        # if bias is not None:
        #     self.encoder_mean[-1].bias = torch.nn.Parameter(torch.Tensor(bias))

    def forward(self, X):
        mean = self.mean_net(X)
        if self.sampler_kind == 'Gaussian':
            logvar = self.logvar_net(X)
            std = torch.exp(logvar / 2)
            self.std = std # torch.max(std, torch.tensor(0.1))
            z = mean + std * torch.randn_like(std)
            self.mean = mean
            return z
        else:
            self.mean = self.mean_net(X)
            return self.mean


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
