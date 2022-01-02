import torch
from torch import nn


class Sampler(nn.Module):
    def __init__(self, layer_sizes, sampler_kind):
        super(Sampler, self).__init__()
        self.sampler_kind = sampler_kind
        self.params = None

        self.input_dim = layer_sizes[0]
        self.output_dim = layer_sizes[-1]

        layers = [nn.Identity()]
        in_out_pairs = list(zip(layer_sizes, layer_sizes[1:]))

        for prev_dim, next_dim in in_out_pairs[:-1]:
            layers.append(nn.Linear(prev_dim, next_dim))
            layers.append(nn.Tanh())

        self.base_net = nn.Sequential(*layers)

        if sampler_kind == 'Gaussian':
            self.mean_net = nn.Sequential(
                self.base_net, nn.Linear(next_dim, self.output_dim))
            self.logvar_net = nn.Sequential(
                self.base_net, nn.Linear(next_dim, self.output_dim))
        elif sampler_kind == 'Bernoulli':
            self.mean_net = nn.Sequential(
                self.base_net, nn.Linear(next_dim, self.output_dim), nn.Sigmoid())
        else:
            assert False

        self.set_gpu_use()

    def log_prob(self, V):
        if self.sampler_kind == 'Gaussian':
            pdf = torch.distributions.Normal(self.mean, self.std)
        if self.sampler_kind == 'Bernoulli':
            pdf = torch.distributions.Bernoulli(self.mean)

        return pdf.log_prob(V)

    def forward(self, X):
        if self.sampler_kind == 'Gaussian':
            self.mean = self.mean_net(X)
            logvar = self.logvar_net(X)
            # logvar = log(sigma**2) = 2*log(sigma)
            # std = exp(logvar / 2) = exp(2*log(sigma)/2) = exp(log(sigma)) = sigma
            self.std = torch.exp(logvar / 2) 
            self.Z = self.mean + self.std * torch.randn_like(self.std)
            self.params = (self.Z, self.mean, self.std)
        else:
            self.mean = self.mean_net(X)
            self.params = (self.mean)
        return self.params

    def set_gpu_use(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
