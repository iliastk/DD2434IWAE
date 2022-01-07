import numpy as np

experiment = {
    'name': 'simple_2_layers',
    'seed': 123,
    'data': {
        'name': 'simple_2_layers',
        'batch_size': 20,
        'path': './data/',
        'num_workers': 1,
        'n_samples': 20000
    },
    'model': {
        'type': 'VAE',
        'X_dim': 2,   # input dim
        'Z_dim': [2, 2],    # latent dim
        'H_dim': [[5, 5], [5, 5]],  # deterministic layer dim
        'encoder_type': 'Gaussian',
        'decoder_type': 'Gaussian',
        'num_samples': 1,
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
        'early_stopping': {
            'patience': 7,
            'threshold': 0.01
        },
        'total_epochs': 3280
    }
}
