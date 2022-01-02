from datasets import BinarizedMNIST
from vae import VAE

import torch
import random
import numpy as np

from experiments.vae_k_1_layers_1 import experiment as vae_k_1_layers_1
from experiments.vae_k_50_layers_1 import experiment as vae_k_50_layers_1

from experiments.iwae_k_1_layers_1 import experiment as iwae_k_1_layers_1
from experiments.iwae_k_50_layers_1 import experiment as iwae_k_50_layers_1

from experiments.base2stochastic import experiment as base_2_stochastic_experiment

from experiments.linear import experiment as linear_experiment

from experiment import launch_experiment


def main():
    # TODO: Fancy Logs to TensorBoard (nicetohave)
    # TODO: Gradient clipping
    # TODO: get_units_variances() utils.py:134 @Venia
    # TODO: chop_units() utils.py:139 @Venia
    # TODO: Compute NLL/-log(p(x)) from elbo using L_{5000} when we test
    # TODO: Bernoulli Binarization?
    # TODO: /255.0 Binarization?
    # TODO: Why not define exp(encode_logvar) at once?
    # TODO: Create IWAE
 

    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    launch_experiment(iwae_k_50_layers_1)
    # launch_experiment(vae_k_50_layers_1)

    # launch_experiment(vae_k_1_layers_1)
    # launch_experiment(iwae_k_1_layers_1)

    # launch_experiment(base_2_stochastic_experiment)
    # launch_experiment(linear_experiment)


if __name__ == "__main__":
    main()
