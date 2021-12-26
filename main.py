from datasets import BinarizedMNIST
from vae2 import VAE as VAE_2
from vae import VAE as VAE
from iwae import IWAE

import torchvision
from torchvision import transforms

import torch
import random
import numpy as np


def main():
    # TODO: Proper weight init
    # TODO: Proper bias init
    # TODO: Fix Seed to 123 (Burda et al.)
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    batch_size = 20  # TODO: correct?
    data = {
        "train": BinarizedMNIST(train=True, root_path="./data/"),
        "val": None,
        "test": BinarizedMNIST(train=False, root_path="./data/")
    }
    data_loader = {
        "train": torch.utils.data.DataLoader(dataset=data["train"], batch_size=batch_size, shuffle=True),
        "val": None,
        "test": torch.utils.data.DataLoader(dataset=data["test"], batch_size=batch_size, shuffle=True)
    }

    X_dim = 784  # 28x28
    Z_dim = 50
    H_dim = 200
    num_samples = 1
    model = VAE_2(X_dim, H_dim, Z_dim, num_samples)
#     model = IWAE(X_dim, Z_dim)

    lr = 0.001  # TODO: Make lr scheduable as in Burda et al.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    beta_1, beta_2, epsilon = 0.9, 0.999, 1e-4
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=lr, betas=(beta_1, beta_2), eps=epsilon)

    num_epochs = 10  # TODO: Set epochs like Burda et al.
    for epoch in range(num_epochs):
        for batch_idx, (X, _) in enumerate(data_loader["train"]):
            optimizer.zero_grad()
            X = X.view(batch_size, X_dim)
            outputs, loss, log_px = model(X)
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}],  loss: {:.3f}'.format(epoch +
                                                    1, num_epochs, loss.item()))
        print('Epoch [{}/{}],  negative log-likelihood: {:.3f}'.format(epoch +
                                                                       1, num_epochs, - log_px.item()))


main()
