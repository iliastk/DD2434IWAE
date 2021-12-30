import copy
from vae import VAE

import torch.utils.data as tud # for TensorDataset, DataLoader


import torchvision
from torchvision import transforms

import torch
import random
import numpy as np

# σ*Z+μ
σ, μ = 2, 1
BATCH_SIZE = 20
orig_model = None

def gen_data(n_samples) -> tud.DataLoader:
    true_z = np.random.normal(size=n_samples)
    ϵ1 = np.random.normal(size=n_samples)
    ϵ2 = np.random.normal(size=n_samples)
    x = σ*true_z + μ
    #x = np.array([σ*true_z + μ+ϵ1, σ*(-true_z) + μ + ϵ2])
    
    tensor_x = torch.Tensor(list(x.T)) # transform to torch tensor
    dataset = tud.TensorDataset(tensor_x) # create your datset
    return tud.DataLoader(dataset=dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False) # create your dataloader

def main():
    global orig_model
    # TODO: Proper weight init
    # TODO: Proper bias init
    # TODO: Log to TensorBoard
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    data_loaders = {
        "train"   : gen_data(1000),
        "test"    : gen_data(1000),
        "val"     : None
    }

    X_dim = 1
    Z_dim = 1
    # Training a linear thing, so we don't even need hidden layers.
    # Not sure if 'sampler.py' supporths that, though.
    hidden_layer_dims = []
    num_samples = 1
    model = VAE(
        X_dim, hidden_layer_dims, Z_dim,
        num_samples,
        encoder='Gaussian',
        decoder='Gaussian',
        bias=μ
    )
    print(model)
    orig_model = copy.deepcopy(model)
    orig_model.load_state_dict(copy.deepcopy(model.state_dict()))
    
    lr = 0.001
    beta_1, beta_2, epsilon = 0.9, 0.999, 1e-4
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(beta_1, beta_2), eps=epsilon)

    milestones = np.cumsum([3 ** i for i in range(8)])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=10 ** (-1 / 7), verbose=True
    )

    num_epochs = 2 # 3280  # TODO: Set epochs like Burda et al.
    for epoch in range(num_epochs):
        for batch_idx, (X,) in enumerate(data_loaders["train"]):
            optimizer.zero_grad()
            X = X.view(BATCH_SIZE, X_dim)
            
            outputs, loss, log_px = model(X)
            #print(f"outputs: {outputs}, loss: {loss}, log_px: {log_px}")
            loss.backward()
            optimizer.step()
            break
        break

        scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}],  loss: {loss.item():.3f}')
        print('Epoch [{}/{}],  negative log-likelihood: {:.3f}'.format(
            epoch + 1, num_epochs, - log_px.item()))

    # PARAMS:
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data)
    print("#######")
    for name, param in orig_model.named_parameters():
        if param.requires_grad:
            print (name, param.data)


    globals()["model"] = model


if __name__ == '__main__':
    main()
