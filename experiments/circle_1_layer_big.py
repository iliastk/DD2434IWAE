import numpy as np

experiment = {
    'name': 'circle2',
    'seed': 123,
    'data': {
        'name': 'circle',
        'batch_size': 20,
        'path': './data/',
        'num_workers': 1,
        'n_samples': 50000
    },
    'model': {
        'type': 'IWAE',
        'X_dim': 2,   # input dim
        'Z_dim': [5],    # latent dim
        'H_dim': [[20, 20, 5]],  # deterministic layer dim
        'encoder_type': 'Gaussian',
        'decoder_type': 'Gaussian',
        'num_samples': 200,
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
