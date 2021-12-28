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
        'H_dim': {  # deterministic layer dim
                'encoder': [200, 200],
                'decoder': [200, 200]
        },
        'encoder_type': 'Gaussian',
        'decoder_type': 'Bernoulli',
        'num_samples': 1,
        'loss_th': 0.01,
    },
    'training': {
        'scheduler': {
            'gamma': 10 ** (-1/7),
            'milestones': np.cumsum([3 ** i for i in range(8)])
        },
        'optimizer': {
            'lr': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-4
        },
        'total_epochs': 3280
    }
}