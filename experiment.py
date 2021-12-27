from datetime import datetime
from pathlib import Path
import numpy as np

experiment = {
    'name': 'base',
    'seed': 123,
    'data': {
        'name': 'BinarizedMNISt',
        'batch_size': 20,
        'path': './data/',
        'num_workers': 8,
    },
    'model': {
        'type': 'VAE',
        'X_dim': 784,   # input dim
        'Z_dim': 50,    # latent dim
        'encoder': {
            'H_dim': [200, 200],    # deterministic layers
            'type': 'Gaussian'
        },
        'decoder': {
            'H_dim': [200, 200],    # deterministic layers
            'type': 'Bernoulli'
        },
        'num_samples': 1,
    },
    'training': {
        'scheduler': {
            'base_lr': 0.001,
            'gamma': 10 ** (-1/7),
            'milestones': np.cumsum([3 ** i for i in range(8)])
        },
        'optimizer': {
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-4
        },
        'total_epochs': 3280
    }
}


def run_experiment(experiment):
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y-%H:%M:%S")

    results_dir = f'results/{experiment["name"]}/{timestamp}'
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    experiment['results_dir'] = results_dir

    num_epochs = 3280  # TODO: Set epochs like Burda et al.
    for epoch in range(num_epochs):
        train_results = train_epoch(
            optimizer, scheduler, batch_size, data_loader["train"], model, X_dim)
        test_results = test_epoch(
            data_loader["test"], batch_size, model, X_dim)
        log_results(best_model_dir, test_results,
                    train_results, epoch, num_epochs, model)
