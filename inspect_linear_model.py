import numpy as np
from utils import gen_fake_data
import matplotlib.pyplot as plt
import os
from utils import setup_model

import torch

from experiments.two_clusters import experiment 


def sample_x(model, n_samples):
    res = []
    zs = np.random.normal(size=n_samples)
    for z in zs:
        z = torch.tensor([float(z)])
        x = model.decode(z).item()
        res.append(x)
    return res        

def main():
    model, _ = setup_model(experiment["model"], model_bias=[1])

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
      'results/two_clusters/30-12-2021-12:54:30/Epoch:0-Loss:1.53-LogPx:1.53.pt')

    # path="/home/alex/DD2434IWAE/results/linear2dim/30-12-2021-11:16:03/Epoch:9-Loss:3.93-LogPx:3.93.pt"
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)

    for name, param in model.named_parameters():
            if param.requires_grad:
                print (name, param.data)


    data, _ = gen_fake_data(experiment['data'])
    plt.figure(figsize=(9, 4))
    plt.title("True distr vs VAE distr[]")

    plt.subplot(141)
    plt.hist(data, density=True, bins=30)
    plt.xlabel("x sampled from true distribution")

    x_means = []
    x_stds = []
    zs = np.arange(-3, 3, 0.1)
    for z in zs:
        model.decode(torch.tensor([float(z)]))
        x_means.append(model.decoder.mean.item())
        x_stds.append(model.decoder.std.item())
    

    plt.subplot(142)
    plt.plot(zs, x_means)
    plt.ylabel("μ_x")
    plt.xlabel("z value")

    plt.subplot(143)
    plt.plot(zs, x_stds)
    plt.ylabel("σ_x")
    plt.xlabel("z value")
    

    plt.subplot(144)
    xs = sample_x(model, 10000)
    plt.hist(xs, bins=30, density=True)
    plt.xlabel("x sampled from model")
    plt.show()
    
    return model
    

if __name__ == "__main__":
    model = main()

"""

Result is:

decoder.mean_net.1.weight tensor([[-2.0052],
        [ 1.9879]])
decoder.mean_net.1.bias tensor([0.9941])
decoder.logvar_net.1.weight tensor([[-0.0041],
        [-0.0028]])
decoder.logvar_net.1.bias tensor([-0.0121,  0.0239])

X|Z should be (2Z+1+noise, -2Z+1+noise)

means are correct
logvar weight is correct

exp(logvar.bias) = (0.9879729106308383, 1.0241878939801135)

"""
