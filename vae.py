import torch.distributions as td
import torch
from torch import nn
import numpy as np
from samplers import GaussianSampler, BernoulliSampler, Sampler

class Encoder(nn.Module):
    def __init__(self, X_dim, H_dim, Z_dim, type='Gaussian'):
        super(Encoder, self).__init__()
        self.input_dim = X_dim
        self.output_dim = Z_dim

        self.layers = []
        for units_prev, hidden_units, units_next in zip([X_dim], H_dim, Z_dim):
            self.layers.append(Sampler([units_prev]+hidden_units+[units_next], sampler_kind=type))
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        Z = X
        return Z


class Decoder(nn.Module):
    def __init__(self, X_dim, H_dim, Z_dim, type='Bernoulli'):
        super(Decoder, self).__init__()     
        self.input_dim = Z_dim
        self.output_dim = X_dim

        self.layers = []
        for units_prev, hidden_units, units_next in zip(Z_dim[:-1], H_dim[:-1], [X_dim][:-1]):
            self.layers.append(
                Sampler([units_prev]+hidden_units+[units_next], sampler_kind='Gaussian'))
        self.layers.append(
            Sampler([units_prev]+hidden_units+[units_next], sampler_kind=type))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, Z):
        for layer in self.layers:
            Z = layer(Z)
        X = Z
        return X

class VAE(nn.Module):
    def __init__(self, X_dim, H_dim, Z_dim, num_samples, encoder='Gaussian', decoder='Bernoulli', bias=None):
        super(VAE, self).__init__()
        self.num_samples = num_samples
        # encoder network - q(z|x)
        self.encoder_layers = []
        for units_prev, hidden_units, units_next in zip([X_dim], H_dim, Z_dim):
            self.encoder_layers.append(Sampler([units_prev]+hidden_units+[units_next],
                    sampler_kind=encoder))
            
        self.encoder_layers = nn.Sequential(*self.encoder_layers)
        # decoder network - p(x|h)
        self.decoder_layers = []
        for units_prev, hidden_units, units_next in zip(Z_dim, H_dim, [X_dim]):
            self.decoder_layers.append(Sampler([units_prev]+hidden_units+[units_next],
                    sampler_kind=decoder))
        self.decoder_layers = nn.Sequential(*self.decoder_layers)

        # TODO: Why I get better results if I dont use the authors initialization?
        self.apply(self.init)
        self.set_bias(bias)
        self.set_gpu_use()
        # prior - p(z)
        # self.prior = torch.distributions.Normal(torch.zeros(Z_dim).to(self.device), torch.ones(Z_dim).to(self.device))
        self.prior = torch.distributions.Normal(0, 1)

    def encode(self, X):
        for encoder in self.encoder_layers:
            X = encoder(X)
        Z = X
        return Z

    def decode(self, Z):
        for decoder in self.decoder_layers:
            Z = decoder(Z)
        X = Z
        return X

    def forward(self, X):

        X = torch.repeat_interleave(X.unsqueeze(
            1), self.num_samples, dim=1).to(self.device)

        Z = self.encode(X)
        self.decode(Z)

        return Z
        # return {"Z": Z, 
        #         "encoder": {
        #             "mean": self.encoder_layers[0].mean, 
        #             "std": self.encoder_layers[0].std
        #         }, 
        #         "decoder": { 
        #             "mean":self.decoder_layers[0].mean
        #         }}

    def init(self, module):
        ''' All models were initialized with the heuristic of Glorot & Bengio (2010). '''
        if type(module) == nn.Linear:
            torch.nn.init.xavier_uniform_(
                module.weight, gain=nn.init.calculate_gain("tanh")
            )
            module.bias.data.fill_(0.01)

    def set_bias(self, bias):
        if bias is not None:
            for decoder in self.decoder_layers:
                decoder.mean_net[-1].bias = torch.nn.Parameter(torch.Tensor(bias))

    def set_gpu_use(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device [{self.device}].')
        self.to(self.device)


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
