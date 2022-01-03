import torch.distributions as td
import torch
from torch import nn
from sampler import Sampler

class Encoder(nn.Module):
    def __init__(self, X_dim, H_dim, Z_dim, type='Gaussian'):
        super(Encoder, self).__init__()
        self.input_dim = X_dim
        self.output_dim = Z_dim[-1]
        self.params = []

        self.layers = []
        for units_prev, hidden_units, units_next in zip([X_dim]+Z_dim, H_dim, Z_dim):
            self.layers.append(Sampler([units_prev]+hidden_units+[units_next], sampler_kind=type))
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, input):
        self.params = []
        for layer in self.layers:
            params = layer(input)
            self.params.append(params)
            input = params[0]
        return self.params

class Decoder(nn.Module):
    def __init__(self, X_dim, H_dim, Z_dim, type='Bernoulli'):
        super(Decoder, self).__init__()     
        self.input_dim = Z_dim[-1]
        self.output_dim = X_dim
        self.params = []

        self.layers = []
        for units_prev, hidden_units, units_next in zip(Z_dim[::-1], H_dim[::-1][:-1], Z_dim[:-1]):
            self.layers.append(
                Sampler([units_prev]+hidden_units+[units_next], sampler_kind='Gaussian'))
        self.layers.append(
            Sampler([Z_dim[0]]+H_dim[0]+[X_dim], sampler_kind=type))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, Z):
        self.params = []
        for z, layer in zip(reversed(Z), self.layers):
            params = layer(z)
            self.params.append(params)
        return self.params

class VAE(nn.Module):
    def __init__(self, X_dim, H_dim, Z_dim, num_samples, encoder='Gaussian', decoder='Bernoulli', bias=None):
        super(VAE, self).__init__()
        self.num_samples = num_samples
        # prior - p(z)
        self.prior = torch.distributions.Normal(0, 1)
        # encoder network - q(z|x)
        self.encoder = Encoder(X_dim, H_dim, Z_dim, type=encoder)
        # decoder network - p(x|h)
        self.decoder = Decoder(X_dim, H_dim, Z_dim, type=decoder)

        self.apply(self.init)
        self.set_bias(bias)
        self.set_gpu_use()

    def encode(self, X):
        return self.encoder(X)

    def decode(self, Z):
        return self.decoder(Z)

    def forward(self, X, num_samples=None):
        if num_samples is None:
            num_samples = self.num_samples
        X = torch.repeat_interleave(X.unsqueeze(
            1), num_samples, dim=1).to(self.device)

        q_params = self.encode(X)
        Z = [p[0] for p in q_params] # list of [(Z, mean, std) x num_stochastic_layers]
        p_params = self.decode(Z)

        return q_params, p_params

    def init(self, module):
        ''' All models were initialized with the heuristic of Glorot & Bengio (2010). '''
        if type(module) == nn.Linear:
            # Burda does
            # A = 1 * sqrt(6/(n_out + n_in)), weights ~ unif(-A, A)
            # bias = 0

            # We did A = 5/3 * sqrt(6/(n_out + n_in))
            # because nn.init.calculate_gain("tanh") = 5/3
            gain = 1.0 # nn.init.calculate_gain("tanh")
            torch.nn.init.xavier_uniform_(
                module.weight, gain=gain
            )
            module.bias.data.fill_(0.001)

    def set_bias(self, bias):
        if bias is not None:
            self.decoder.layers[-1].mean_net[-1].bias = torch.nn.Parameter(torch.Tensor(bias))

    def set_gpu_use(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device [{self.device}].')
        self.to(self.device)

