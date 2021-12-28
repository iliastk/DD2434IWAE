from torch import nn
import numpy as np
import torch


class VAELoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(VAELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, outputs, labels):
        v_ccc, a_ccc, d_ccc = (
            self.ccc(outputs[:, 0], labels[:, 0]),
            self.ccc(outputs[:, 1], labels[:, 1]),
            self.ccc(outputs[:, 2], labels[:, 2]),
        )
        v_loss, a_loss, d_loss = 1.0 - v_ccc, 1.0 - a_ccc, 1.0 - d_ccc

        ccc = v_ccc + a_ccc + d_ccc
        loss = self.alpha * v_loss + self.beta * a_loss + self.gamma * d_loss
        # loss = (v_loss + a_loss + d_loss) / 3

        return loss, ccc, v_ccc, a_ccc, d_ccc

    def ccc(self, outputs, labels):
        labels_mean = torch.mean(labels)
        outputs_mean = torch.mean(outputs)
        covariance = (labels - labels_mean) * (outputs - outputs_mean)

        label_var = torch.mean(torch.square(labels - labels_mean))
        outputs_var = torch.mean(torch.square(outputs - outputs_mean))

        ccc = (2.0 * covariance) / (
            label_var + outputs_var + torch.square(labels_mean - outputs_mean)
        )
        return torch.mean(ccc)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, name="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            name (str): name for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mean_ccc_min = np.Inf
        self.delta = delta
        self.name = name
        self.trace_func = trace_func

    def __call__(self, mean_ccc, model):

        score = mean_ccc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(mean_ccc, model)
        elif np.abs(score) - np.abs(self.best_score) > self.delta:
            self.best_score = score
            self.save_checkpoint(mean_ccc, model)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, mean_ccc, model):
        """Saves model when validation loss decrease."""
        import copy

        if self.verbose:
            self.trace_func(
                f"Validation CCC increased ({self.mean_ccc_min:.6f} --> {mean_ccc:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.name)
        self.mean_ccc_min = mean_ccc
