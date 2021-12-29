from torch import nn
import numpy as np
import torch

twoPI = torch.tensor(2*np.pi)


class VAELoss(nn.Module):
    def __init__(self, num_samples):
        super(VAELoss, self).__init__()
        self.num_samples = num_samples
        self.set_gpu_use()

    def forward(self, outputs, target):
        elbo = self.elbo(outputs, target)
        loss = torch.sum(elbo, dim=-1)
        loss = torch.mean(loss)

        NLL = self.NLL(elbo)
        return -loss, NLL

    def elbo(self, output, target):  # TODO: is this log_elbo?
        X = torch.repeat_interleave(target.unsqueeze(
            1), self.num_samples, dim=1).to(self.device)
        Z = output['Z']
        mu_z, std_z, mu_x = output['encoder']['mean'], output['encoder']['std'], output['decoder']['mean']

        # likelihood: p(x|z, theta) => log(Bernoulli(mu_x))
        logP_XgivenZ = torch.sum(
            X * torch.log(mu_x) + (1-X) * torch.log(1 - mu_x), dim=-1)

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
        elbo = logP_XgivenZ + logP_Z - logQ_ZgivenX

        return elbo

    def NLL(self, elbo):
        # normalized weights through Exp-Normalization trick
        max_elbo = torch.max(elbo, dim=-1)[0].unsqueeze(1)
        elbo = torch.exp(elbo - max_elbo)

        # Computes Negative Log Likelihood (p(x)) through Log-Sum-Exp trick
        # TODO: How to compute log-likelihood p(x) to compare NLL
        NLL = max_elbo + \
            torch.log((1/self.num_samples) * torch.sum(elbo, dim=-1))
        NLL = torch.mean(NLL)  # mean over batches

        return NLL

    def set_gpu_use(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


class EarlyStopping:
    """Early stops the training if test NLL doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=True, threshold=0, best_model_dir=None, trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time test NLL improved.
                            Default: 7
            verbose (bool): If True, prints a message for each test NLL improvement.
                            Default: False
            threshold (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            name (str): name for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_NLL = None
        self.early_stop = False
        self.min_NLL = np.Inf
        self.threshold = threshold
        self.best_model_dir = best_model_dir
        self.trace_func = trace_func

    def __call__(self, NLL, loss, epoch, model):

        if self.best_NLL is None:
            self.best_NLL = NLL
            self.save_checkpoint(NLL, model, loss, epoch)
        elif np.abs(NLL) - np.abs(self.best_NLL) > self.threshold:
            self.best_score = NLL
            self.save_checkpoint(NLL, model, loss, epoch)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(
                f"\t\t == EarlyStopping counter: [{self.counter}/{self.patience}] =="
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, NLL, model, loss, epoch):
        """Saves model when test NLL decrease."""
        if self.verbose:
            self.trace_func(
                f"\t\t >>> Test NLL increased ({self.min_NLL:.3f} --> {NLL:.3f}).  Saving model ... <<<"
            )
            # Save
        best_model_filename = f'{self.best_model_dir}/Epoch:{epoch}-Loss:{loss:.2f}-LogPx:{NLL:.2f}.pt'
        torch.save(model.state_dict(), best_model_filename)
        self.NLL_min = NLL
