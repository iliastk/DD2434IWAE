import utils
import evaluation
import math
from scipy.stats import norm as sci_norm
import numpy as np
from utils import gen_fake_data
import matplotlib.pyplot as plt
import os
from utils import setup_model

import torch

from experiments.two_close_clusters_iwae import experiment

# entropy
def expected_log_likelihood(xs, x_density):
    dx = xs[1] - xs[0]
    # Integral ∫ p(x) log (p(x)) dx
    return sum(np.log(x_density) * x_density * dx)

def integrate_normal_density(mus, sigmas, weights, xs):
    """Suppose X has support over 'xs', and the density of X at x

    is weights[x].

    Let μ(x), σ(x) be defined though `mus`, `sigmas`. Define

    Z|X=x ~ N(μ(x), σ(x)^2). This function computes the
    density of Z.
    """

    res = np.zeros(len(xs))
    for i, _ in enumerate(xs):
        μ, σ = mus[i], sigmas[i]
        res += sci_norm.pdf(xs, loc=μ, scale=σ) * weights[i]
    # Normalize:
    dx = xs[1] - xs[0] # assume equal spacing
    # sum(res)*dx should be 1
    res /= (sum(res)*dx)
    return res    

def sample_x(model, n_samples):
    res = []
    with torch.no_grad():
        for _ in range(n_samples):
            z1 = np.random.normal(size=1)
            z1 = [float(x) for x in z1]
            z1 = torch.tensor([z1])
            x, _, _ = model.decoder.layers[0](z1)
            # For 2 stoch layers if output is gaussian:
            #x, _, _ = model.decoder.layers[1](z2)
            #print(params)
            # x, _, _ = params
            res.append(x.numpy()[0])
    return np.concatenate(res)

def main():
    print(experiment)
    model, _ = setup_model(experiment["model"], model_bias=[1])

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'results/two_close_clusters/07-01-2022-00--39--44/Epoch:0-Loss:1.53-NLL_k:1.48.pt')
    # 'results/two_close_clusters/31-12-2021-11:30:14/Epoch:52-Loss:1.53-LogPx:1.53.pt')
     #'results/two_close_clusters_iwae/04-01-2022-09:30:58/Epoch:5-Loss:0.33-NLL_k:0.33.pt')

    state_dict = torch.load(path)
    model.load_state_dict(state_dict)

    for name, param in model.named_parameters():
            if param.requires_grad:
                print (name, param.data)


    data, _ = gen_fake_data(experiment['data'])
    dl_data, _, _ = utils.setup_data(experiment['data'])
    model_log_l = evaluation.measure_estimated_log_likelihood(dl_data['test'], model,
                                                              num_points=5000)
    print(f"L_{5000} = {model_log_l}")
    
    plt.figure(figsize=(9, 4))
    plt.title("True distr vs VAE distr")

    xs = np.arange(-3, 3, 0.1)
    # 2 close clusters:
    print(len(xs))
    true_distr = (0.5*sci_norm.pdf(xs, loc=-1, scale=0.5) +
                  0.5*sci_norm.pdf(xs, loc=1, scale=0.5))
    print(f"True log-likelihood: {expected_log_likelihood(xs, true_distr)}")
    print(len(true_distr))

    plt.subplot(231)
    plt.hist(data, density=True, bins=40)
    plt.plot(xs, true_distr, '-', color='r')
    plt.xlabel("x sampled from true distribution")

    x_means = []
    x_stds = []
    zs = np.arange(-3, 3, 0.1)
    
    z_means = []
    z_stds = []
    for z in zs:
        #print(model.decoder.params)
        #print(z)
        res = model.decode(torch.tensor([[float(z)]]))
        
        #print("I think this is params:", res)
        zz, mean, std = res[0]
        #print(model.decoder.params)
        x_means.append(mean.item())
        x_stds.append(std.item())
    print("got X|Z")
    for x in xs:
        res = model.encode(torch.tensor([[float(x)]]))
        zz, mean, std = res[0]
        z_means.append(mean.item())
        z_stds.append(std.item())
    print("got Z1|X")
    plt.subplot(233)
    plt.plot(zs, x_means)
    plt.ylabel("μ_x")
    plt.xlabel("z value")

    plt.subplot(234)
    plt.plot(zs, x_stds)
    plt.ylabel("σ_x")
    plt.xlabel("z value")

    plt.subplot(235)
    plt.plot(xs, z_means)
    plt.ylabel("μ_z")
    plt.xlabel("x value")

    plt.subplot(236)
    plt.plot(xs, z_stds)
    plt.ylabel("σ_z")
    plt.xlabel("x value")
    

    plt.subplot(232)
    xs = sample_x(model, 4000)
    #print(xs)
    plt.hist(xs, bins=40, density=True)
    true_model_distr = integrate_normal_density(
        x_means, x_stds, sci_norm.pdf(zs), zs
     )
    plt.plot(zs, true_model_distr, '-', color='r')
    plt.show()
    
    return model
    

if __name__ == "__main__":
    model = main()
