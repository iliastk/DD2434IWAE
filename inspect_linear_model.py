from utils import setup_model

import torch

from experiments.linear import experiment as linear_experiment



def main():
    model, _ = setup_model(linear_experiment["model"], 1)
    path="/home/alex/DD2434IWAE/results/linear2dim/30-12-2021-11:16:03/Epoch:9-Loss:3.93-LogPx:3.93.pt"
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)

    for name, param in model.named_parameters():
            if param.requires_grad:
                print (name, param.data)
    

if __name__ == "__main__":
    main()

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
