from torch import nn
import numpy as np
import torch


class VAELoss(nn.Module):
    def __init__(self, num_samples):
        super(VAELoss, self).__init__()
        self.num_samples = num_samples
        self.set_gpu_use()

    def forward(self, outputs, target, model):
        n_elbo = self.elbo(outputs, target, model)
        elbo = torch.mean(n_elbo, dim=-1) # avg over num_samples elbos, not weighted sum

        loss = torch.mean(elbo, dim=0) # avg over batches

        NLL = self.NLL(n_elbo)
        return -loss, -NLL

    def elbo(self, output, target, model):
        X = torch.repeat_interleave(target.unsqueeze(
            1), self.num_samples, dim=1).to(self.device)
        q_params, p_params = output
        Z = [p[0] for p in q_params] # list of [(Z, mean, std) x num_stochastic_layers]
        inner_Z = Z[-1] # deepest Z in network

        # elbo = log(p(x)) + log(p(x|z)) - log(q(z|x)) = prior + likelihood - posterior
        elbo = torch.sum(model.prior.log_prob(inner_Z), dim=-1)
        for q, p, in_p, in_q in zip(model.encoder.layers, reversed(model.decoder.layers), [X]+Z, Z):
            elbo += torch.sum(p.log_prob(in_p), dim=-1) - torch.sum(q.log_prob(in_q), dim=-1)

        return elbo

    def NLL(self, elbo):
        # normalized weights through Exp-Normalization trick
        max_elbo = torch.max(elbo, dim=-1)[0].unsqueeze(1)
        elbo = torch.exp(elbo - max_elbo)

        # Computes Negative Log Likelihood (p(x)) through Log-Sum-Exp trick
        NLL = max_elbo + \
            torch.log((1/self.num_samples) * torch.sum(elbo, dim=-1))
        NLL = torch.mean(NLL)  # mean over batches

        return NLL

    def set_gpu_use(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


class IWAELoss(VAELoss):
    def __init__(self, num_samples):
        super(IWAELoss, self).__init__(num_samples)

    def forward(self, outputs, target, model):
       n_log_w = self.elbo(outputs, target, model)
       # weighted sum:
       max_w = torch.max(n_log_w, dim=-1)[0].unsqueeze(1)
       w = torch.exp(n_log_w - max_w)
       with torch.no_grad():
           normalized_w = w / torch.sum(w, dim=-1).unsqueeze(1)
       elbo = torch.sum(normalized_w * n_log_w, dim=-1)  # sum over num_samples 

       loss = torch.mean(elbo, dim=0)  # avg over batches

       NLL = self.NLL(n_log_w)
       return -loss, -NLL



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
        self.threshold = threshold
        self.best_model_dir = best_model_dir
        self.trace_func = trace_func

    def __call__(self, NLL, loss, epoch, model):

        if self.best_NLL is None:
            self.best_NLL = NLL
            self.save_checkpoint(NLL, model, loss, epoch)
        elif np.abs(NLL) < np.abs(self.best_NLL) - self.threshold:
            self.save_checkpoint(NLL, model, loss, epoch)
            self.counter = 0
        elif epoch % 50 == 0:
            self.save_checkpoint(NLL, model, loss, epoch)
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
               f"\t\t >>> epoch is {epoch}. Saving model ... <<< "
            )
            # Save
        best_model_filename = f'{self.best_model_dir}/Epoch--{epoch}-Loss--{loss:.2f}-NLL_k--{NLL:.2f}.pt'
        torch.save(model.state_dict(), best_model_filename)
        self.best_NLL = min(NLL, self.best_NLL)
