from datasets import BinarizedMNIST
from vae import VAE

import torch
import random
import numpy as np


from experiments.vae_k_1_layers_2 import experiment as vae_k_1_layers_2

from experiments.vae_k_1_layers_1 import experiment as vae_k_1_layers_1
from experiments.vae_k_50_layers_1 import experiment as vae_k_50_layers_1

from experiments.iwae_k_1_layers_1 import experiment as iwae_k_1_layers_1
from experiments.iwae_k_50_layers_1 import experiment as iwae_k_50_layers_1

from experiments.linear import experiment as linear_experiment

from experiment import launch_experiment


def main():
    ''' TODO:
            - Do same experiments as authors:
                1. Demsity Estimation
                    1.1 MNIST: 
                        1.1.1 VAE: 
                            stochastic_layers = 1 & k = [1, 5, 50]
                            stochastic_layers = 2 & k = [1, 5, 50]
                        1.1.2 IWAE:
                            stochastic_layers = 1 & k = [1, 5, 50]
                            stochastic_layers = 2 & k = [1, 5, 50]
                    1.2 OMNIGLOT:
                        1.2.1 VAE: 
                            stochastic_layers = 1 & k = [1, 5, 50]
                            stochastic_layers = 2 & k = [1, 5, 50]
                        1.2.2 IWAE:
                            stochastic_layers = 1 & k = [1, 5, 50]
                            stochastic_layers = 2 & k = [1, 5, 50]
                2. Latent space representation:
                    2.1 Best VAE -> keep training it as IWAE(k=50)
                    2.2 Best IWAE(k=50) -> keep training it as VAE
            
            - Run our own experiments:
                1. Venia's simple experiment: proves IWAEs superiority over VAEs
                2. FashionMNIST

            @Venia:
            - get_units_variances() utils.py:134
            - chop_units() utils.py:139
            - Compute NLL/-log(p(x)) from elbo using L_{5000} when we test 
              (maybe forget about this because it entails refactoring a lot our code)
            - Bernoulli Binarization?
            - /255.0 Binarization?
    '''

    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    launch_experiment(vae_k_1_layers_2)

    # launch_experiment(iwae_k_50_layers_1)
    # launch_experiment(vae_k_50_layers_1)

    # launch_experiment(iwae_k_1_layers_1)
    # launch_experiment(vae_k_1_layers_1)

    # launch_experiment(base_2_stochastic_experiment)
    # launch_experiment(linear_experiment)


if __name__ == "__main__":
    main()
