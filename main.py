from datasets import BinarizedMNIST
from vae import VAE

import torch
import random
import numpy as np

from experiments.base import experiment as base_experiment
from experiment import launch_experiment

def main():
   # TODO: Log to TensorBoard
   torch.manual_seed(123)
   random.seed(123)
   np.random.seed(123)

   launch_experiment(base_experiment)

if __name__ == "__main__":
    main()