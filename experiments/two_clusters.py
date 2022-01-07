import numpy as np

experiment = {
    'name': 'two_clusters', # the one that's supposedly bad for VAE
    'seed': 123,
    'data': {
        'name': 'two_clusters',
        'batch_size': 20,
        'path': './data/',
        'num_workers': 1,
        'n_samples': 20
    },
    'model': {
        'type': 'VAE',
        'X_dim': 1,   # input dim
        'Z_dim': [1],    # latent dim
        # SHOULD ADD hidden dim!!!
        'H_dim': [[1]],  # deterministic layer dim (can be empty)
        'encoder_type': 'Gaussian',
        'decoder_type': 'Gaussian',
        'num_samples': 1,
    },
    'training': {
        'scheduler': {
            'gamma': 10 ** (-1/7),
            'milestones': np.cumsum([2 ** i for i in range(8)])
        },
        'optimizer': {
            'lr': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-4
        },
        'early_stopping':{
            'patience': 200,
            'threshold': 0.01
        },
        'total_epochs': 1001
    }
}
