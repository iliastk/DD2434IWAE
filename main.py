from datasets import BinarizedMNIST
from vae import VAE

import torch
import random
import numpy as np

from experiments.base import experiment as base_experiment
from experiments.base2stochastic import experiment as base_2_stochastic_experiment
from experiments.linear import experiment as linear_experiment

from experiment import launch_experiment


def main():
    # TODO: Fancy Logs to TensorBoard (nicetohave)
    # TODO: Gradient clipping
    # TODO: Detect hidden units
    # TODO: Bernoulli Binarization?
    # TODO: Compute NLL/-log(p(x)) from elbo
    # TODO: Create IWAE 
    # TODO: Why not define exp(encode_logvar) at once?

    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    launch_experiment(base_2_stochastic_experiment)
    # launch_experiment(base_experiment)
    # launch_experiment(linear_experiment)


if __name__ == "__main__":
    main()
