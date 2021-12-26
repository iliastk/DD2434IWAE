from datasets import BinarizedMNIST
from vae import VAE

import torchvision
from torchvision import transforms

import torch
import random
import numpy as np


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
        "train": torch.utils.data.DataLoader(dataset=data["train"], batch_size=batch_size, shuffle=False),
        "val": None,
        "test": torch.utils.data.DataLoader(dataset=data["test"], batch_size=batch_size, shuffle=False)
    }

    X_dim = 784  # 28x28
    Z_dim = 50
    H_dim = {"encoder": [200, 200], "decoder": [200, 200]}
    num_samples = 1
    model = VAE(X_dim, H_dim, Z_dim, num_samples,
                encoder='Gaussian', decoder='Bernoulli', bias=data["train"].get_train_bias())
#     model = IWAE(X_dim, Z_dim)

    lr = 0.001  # TODO: Make lr scheduable as in Burda et al.
    beta_1, beta_2, epsilon = 0.9, 0.999, 1e-4
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(beta_1, beta_2), eps=epsilon)

    milestones = np.cumsum([3 ** i for i in range(8)])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=10 ** (-1 / 7), verbose=True
    )

    num_epochs = 3280  # TODO: Set epochs like Burda et al.
    for epoch in range(num_epochs):
        for batch_idx, (X, _) in enumerate(data_loader["train"]):
            optimizer.zero_grad()
            X = X.view(batch_size, X_dim)
            outputs, loss, log_px = model(X)
            loss.backward()
            optimizer.step()

        scheduler.step()
        print('Epoch [{}/{}],  loss: {:.3f}'.format(epoch +
                                                    1, num_epochs, loss.item()))
        print('Epoch [{}/{}],  negative log-likelihood: {:.3f}'.format(epoch +
                                                                       1, num_epochs, - log_px.item()))

    # Dirty Testing
    log_px_test = []
    with torch.no_grad():
        for batch_idx, (X, _) in enumerate(data_loader["test"]):
            X = X.view(batch_size, X_dim)
            outputs, log_px, loss = model(X)
            log_px_test.append(-log_px.item())
        print('Negative log-likelihood on test set: {:.3f}'.format(
            torch.mean(torch.tensor(log_px_test))))

    with open('results.txt', 'w') as f:
        f.write('Negative log-likelihood on test set: {:.3f}'.format(
            torch.mean(torch.tensor(log_px_test))))


main()
