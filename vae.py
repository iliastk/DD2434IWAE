import torch.distributions as td
import torch
from torch import nn
import numpy as np
from samplers import GaussianSampler, BernoulliSampler

twoPI = torch.tensor(2*np.pi)


class VAE(nn.Module):
    def __init__(self, X_dim, H_dim, Z_dim, num_samples, encoder='Gaussian', decoder='Bernoulli', bias=None, loss_threshold=0.01):
        super(VAE, self).__init__()
        self.num_samples = num_samples
        self.best_test_loss = np.inf
        self.loss_threshold = loss_threshold
        # encoder network - q(z|x)
        if encoder == 'Gaussian':
            self.encoder = GaussianSampler(X_dim, H_dim["encoder"], Z_dim)
        if encoder == 'Bernoulli':
            self.encoder = BernoulliSampler(X_dim, H_dim["encoder"], Z_dim)
        # decoder network - p(x|h)
        if decoder == 'Gaussian':  # for continous value data
            self.decoder = GaussianSampler(X_dim, H_dim["decoder"], Z_dim)
        if decoder == 'Bernoulli':  # for binary value data
            self.decoder = BernoulliSampler(X_dim, H_dim["decoder"], Z_dim)

        # TODO: Why I get better results if I dont use the authors initialization?
        self.apply(self.init)
        self.set_bias(bias)
        self.set_gpu_use()

    def encode(self, X):
        return self.encoder(X)

    def decode(self, Z):
        return self.decoder(Z)

    def forward(self, X):

        X = torch.repeat_interleave(X.unsqueeze(
            1), self.num_samples, dim=1).to(self.device)

        Z = self.encode(X)
        self.decode(Z)

        return {"Z": Z, 
                "encoder": {
                    "mean": self.encoder.mean, 
                    "std": self.encoder.std
                }, 
                "decoder": { 
                    "mean":self.decoder.mean
                }}

    def init(self, module):
        ''' All models were initialized with the heuristic of Glorot & Bengio (2010). '''
        if type(module) == nn.Linear:
            torch.nn.init.xavier_uniform_(
                module.weight, gain=nn.init.calculate_gain("tanh")
            )
            module.bias.data.fill_(0.01)

    def set_bias(self, bias):
        if bias is not None:
            self.decoder.mean_net[-1].bias = torch.nn.Parameter(
                torch.Tensor(bias))

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
