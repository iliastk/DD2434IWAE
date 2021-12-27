from datasets import BinarizedMNIST
from vae import VAE

from datetime import datetime
from pathlib import Path

import torchvision
from torchvision import transforms

import torch
import random
import numpy as np

from evaluation import train_epoch, test_epoch, log_results 

def main():
    # TODO: Proper weight init
    # TODO: Proper bias init
    # TODO: Log to TensorBoard
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    batch_size = 20
    data = {
        "train": BinarizedMNIST(train=True, root_path="./data/"),
        "val": None,
        "test": BinarizedMNIST(train=False, root_path="./data/")
    }
    data_loader = {
        "train": torch.utils.data.DataLoader(dataset=data["train"], batch_size=batch_size, shuffle=True, num_workers=8),
        "val": None,
        "test": torch.utils.data.DataLoader(dataset=data["test"], batch_size=batch_size, shuffle=True, num_workers=8)
    }

    X_dim = 784  # 28x28
    Z_dim = 50
    H_dim = {"encoder": [200, 200], "decoder": [200, 200]}
    num_samples = 1
    model_bias = data["train"].get_bias()
    model = VAE(X_dim, H_dim, Z_dim, num_samples,
                encoder='Gaussian', decoder='Bernoulli', bias=model_bias)
    print(model)

    lr = 0.001  # TODO: Make lr scheduable as in Burda et al.
    beta_1, beta_2, epsilon = 0.9, 0.999, 1e-4
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(beta_1, beta_2), eps=epsilon)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    milestones = np.cumsum([3 ** i for i in range(8)])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=10 ** (-1 / 7), verbose=True
    )

    experiment_name = 'base'
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y-%H:%M:%S")
    best_model_dir = f'results/{experiment_name}/{timestamp}'
    Path(best_model_dir).mkdir(parents=True, exist_ok=True)


    num_epochs = 3280  # TODO: Set epochs like Burda et al.
    for epoch in range(num_epochs):
        train_results = train_epoch(optimizer, scheduler, batch_size, data_loader["train"], model, X_dim)
        test_results = test_epoch(data_loader["test"], batch_size, model, X_dim)
        log_results(best_model_dir, test_results, train_results, epoch, num_epochs, model)


    # TODO: Check if loss is nan
    # TODO: Check if improvement in test loss is too small

main()
