import sys
import subprocess
from datasets import BinarizedMNIST
import datasets
from vae import VAE

import torch
import random
import numpy as np

from experiments.vae_k_1_layers_2 import experiment as vae_k_1_layers_2
from experiments.vae_k_5_layers_2 import experiment as vae_k_5_layers_2
from experiments.vae_k_50_layers_2 import experiment as vae_k_50_layers_2

from experiments.vae_k_1_layers_1 import experiment as vae_k_1_layers_1
from experiments.vae_k_1_layers_1_many_workers import experiment as vae_k_1_layers_1_many_workers
from experiments.vae_k_5_layers_1 import experiment as vae_k_5_layers_1
from experiments.vae_k_50_layers_1 import experiment as vae_k_50_layers_1

from experiments.iwae_k_5_layers_2 import experiment as iwae_k_5_layers_2
from experiments.iwae_k_50_layers_2 import experiment as iwae_k_50_layers_2

from experiments.iwae_k_5_layers_1 import experiment as iwae_k_5_layers_1
from experiments.iwae_k_50_layers_1 import experiment as iwae_k_50_layers_1
from experiments.vae_k_5_layers_1_other_seed import experiment as vae_k_5_layers_1_other_seed

from experiments.vae_k_1_layers_1_retrained_as_iwae_k_50_layers_1 import experiment as vae_k_1_layers_1_retrained_as_iwae_k_50_layers_1
from experiments.iwae_k_50_layers_1_retrained_as_vae_k_1 import experiment as iwae_k_50_layers_1_retrained_as_vae_k_1

import experiment

def main():
    print("Usage: python runner.py <experiment-name> [checkpoint]")
    
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    exper_name = sys.argv[1]
    checkpoint = sys.argv[2]
    if checkpoint != 'NONE':
        EPOCH = int(checkpoint[checkpoint.find("Epoch--"):].split("-")[2])
    else:
        EPOCH=0
        checkpoint = None
    EPOCH=0
    print(EPOCH)
    print(exper_name)
    sys.stdout.flush()
    
    exper = eval(exper_name)
    print(exper)
    experiment.launch_experiment(exper, checkpoint_location=checkpoint, epoch=EPOCH)

if __name__ == "__main__":
    main()
