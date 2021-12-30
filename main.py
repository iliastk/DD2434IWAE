from datasets import BinarizedMNIST
from vae import VAE

import torch
import random
import numpy as np

from experiments.base import experiment as base_experiment
from experiments.linear import experiment as linear_experiment

from experiment import launch_experiment


def main():
    # TODO: Fancy Logs to TensorBoard (nicetohave)
    # TODO: Instead of hardcoded pdf use samplers.pdf
    # TODO: Gradient clipping
    # TODO: Detect hidden units
    # TODO: Bernoulli Binarization?
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    launch_experiment(base_experiment)
    # launch_experiment(linear_experiment)


if __name__ == "__main__":
    main()
