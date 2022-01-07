from typing import List
from datasets import BinarizedMNIST
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from vae import VAE
from loss import VAELoss, IWAELoss, EarlyStopping


def setup_model(params, model_bias):
    X_dim = params['X_dim']
    Z_dim = params['Z_dim']
    H_dim = params['H_dim']
    num_samples = params['num_samples']

    model = VAE(X_dim, H_dim, Z_dim, num_samples,
                encoder=params['encoder_type'], decoder=params['decoder_type'],
                bias=model_bias)

    if params['type'] == 'VAE':
        criterion = VAELoss(num_samples)
    elif params['type'] == 'IWAE':
        criterion = IWAELoss(num_samples)

    print(f'Creating a {params["type"]} model ...')

    return model, criterion


def gen_fake_data(params: dict) -> np.ndarray:
    n_samples = params['n_samples']
    name = params['name']
    if name == 'linear2dim':
        σ, μ = 2, 1.
    
        true_z = np.random.normal(size=n_samples)
        ϵ1 = np.random.normal(size=n_samples)
        ϵ2 = np.random.normal(size=n_samples)
        x = np.array([σ*true_z + μ+ϵ1, σ*(-true_z) + μ + ϵ2])
        # x is dims x samples
        x = x.T  # samples x dims
    elif name == 'two_clusters':
        true_z = np.random.normal(size=n_samples)
        means_x = (true_z > 0) * 2.0 + (true_z <= 0) * (-2.0)
        ϵ = np.random.normal(size=n_samples, scale=0.5)
        x = means_x + ϵ
        μ = 0.0
    elif name == 'two_close_clusters':
        true_z = np.random.normal(size=n_samples)
        means_x = (true_z > 0) * 1.0 + (true_z <= 0) * (-1.0)
        ϵ = np.random.normal(size=n_samples, scale=0.5)
        x = means_x + ϵ
        μ = 0.0
    elif name == 'simple_2_layers':
        means = [[-2, -2], [-2, 2], [2, -2], [2, 2]]
        res = []
        for mean in means:
            ϵ = np.random.normal(size=(n_samples//4, 2), scale=0.7)
            res.append(ϵ + mean)
        x = np.concatenate(res)
        μ = [0.0, 0.0]

    elif name == 'circle':
        thetas = np.random.uniform(0, 2*np.pi, n_samples)
        x, y = 3*np.cos(thetas), 3*np.sin(thetas)
        ϵ = np.random.normal(size=(n_samples, 2), scale=0.4)
        x = np.stack([x, y]).T + ϵ
        μ = [0.0, 0.0]
        
    return x, [μ]


def setup_data(params):
    if params['name'] not in ['linear2dim',
                              'two_clusters',
                              'two_close_clusters',
                              'simple_2_layers',
                              'circle'
                              ]:
        data = {
            'train': BinarizedMNIST(train=True, root_path=params['path']),
            'val': None,
            'test': BinarizedMNIST(train=False, root_path=params['path'])
        }

        data_loader = {
            'train': torch.utils.data.DataLoader(
                dataset=data['train'], batch_size=params['batch_size'],
                shuffle=True, num_workers=params['num_workers']),
            'val': None,
            'test': torch.utils.data.DataLoader(
                dataset=data['test'], batch_size=params['batch_size'],
                shuffle=True, num_workers=params['num_workers'])
        }
        bias = data['train'].get_bias()
        batch_size = params['batch_size']
    else:
        tud = torch.utils.data
        train_x, bias = gen_fake_data(params)
        test_x, bias = gen_fake_data(params)
        train_tensor_x = torch.Tensor(train_x) # transform to torch tensor
        test_tensor_x = torch.Tensor(test_x) # transform to torch tensor
        n_samples = params["n_samples"]
        tensor_dummy_y = torch.Tensor(np.zeros(n_samples))
        train_dataset = tud.TensorDataset(train_tensor_x, tensor_dummy_y)
        test_dataset = tud.TensorDataset(test_tensor_x, tensor_dummy_y)
        data_loader = {'train': tud.DataLoader(dataset=train_dataset,
                                           batch_size=params['batch_size'],
                                           shuffle=False),
                   'test': tud.DataLoader(dataset=test_dataset,
                                          batch_size=params['batch_size'],
                                          shuffle=False),
                   'val': None
                            }
    return data_loader, params['batch_size'], bias


def create_results_dir(name):
    now = datetime.now()
    # Something in Win doesn't like ':' in folder names.
    timestamp = now.strftime("%d-%m-%Y-%H--%M--%S")

    results_dir = f'results/{name}/{timestamp}'
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    return results_dir


def setup_optimizer(params, model_parameters):
    optimizer = torch.optim.Adam(
        model_parameters, lr=params['lr'], betas=(
            params['beta1'], params['beta2']), eps=params['epsilon']
    )
    return optimizer


def setup_scheduler(params, optimizer, start_epoch):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=params['milestones'], gamma=params['gamma'], verbose=True
    )
    for _ in range(start_epoch):
        scheduler.step()
    return scheduler


def setup_early_stopping(params, results_dir):
    early_stopping = EarlyStopping(
        patience=params['patience'], threshold=params['threshold'], best_model_dir=results_dir)
    return early_stopping


def log_results(early_stopping, test_results, train_results, curr_epoch, num_epochs, model, writer, epoch, active_units):

    # Log Train Results
    train_loss, train_NLL = train_results['loss'], train_results['NLL']
    out_result = f'Epoch[{curr_epoch+1}/{num_epochs}],  Train [loss: {train_loss.item():.3f},  NLL_k: {train_NLL.item():.3f}]'

    # Log Test Results
    test_loss, test_NLL = test_results['loss'], test_results['NLL_k']
    out_result = out_result + \
        f'\t == \t Test [loss: {test_loss.item():.3f}, NLL_k:{test_NLL.item():.3f}, active units: {active_units}]'

    print(out_result)    

    # Log to tensorboard
    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/NLL_k', train_NLL, epoch)
    writer.add_scalar('test/loss', test_loss, epoch)
    writer.add_scalar('test/NLL_k', test_NLL, epoch)
    writer.add_scalar('test/actives_l0', active_units[0], epoch)
    if len(active_units) > 1:
        writer.add_scalar('test/actives_l1', active_units[1], epoch)

    early_stopping(test_NLL, test_loss, epoch, model)


def get_units_variances(model, data_loaders: dict, num_points: int = 1000):
    # X = minibatch from 0 to 500 of training data
    # means = list of mean of each layer of encoder/q
    # variances = variances of means[1], means[2], ...

    def gen_zs(model, data_loaders, num_points):
       """
       Generates Z|X_i expectations for `num_points` values X_i.
       Only tested for 1-dim Z (yet)!
       """
       data_loader = data_loaders["test"]
       batch_size = data_loader.batch_size
       num_processed = 0
       def group_zs(ts):
         # ts should have shape (batch_size, 1, z_dim)
         a, b, c = ts.size()
         assert a == batch_size and b == 1, (a, b, c)
         return ts.squeeze() # reshapes to (batch_size, z_dim)
     
       with torch.no_grad():
         means = None
         for batch_idx, (X, _) in enumerate(data_loader):
             X = X.view(batch_size, model.encoder.input_dim)
             q_params, _ = model(X, num_samples=1)
             # q_params is [(Z, mean, std) x num_stochastic_layers]
             batch_means = [group_zs(mean) for (_, mean, _) in q_params]
             if means is None:
               means = batch_means
             else:
               for i, (prev_layer_means, batch_layer_means) in enumerate(
                                                       zip(means, batch_means)):
                 upd_layer_means = torch.cat([prev_layer_means, batch_layer_means], 
                                             dim=0)          
                 means[i] = upd_layer_means
             num_processed += batch_size
             if num_processed >= num_points:
               break
         return means
     
    def compute_covs(zs):
       """
       zs: List[TensorType] one entry for each stochastic layer.
       Each z in zs should have shape (num_points, z_dim).
       This function computes Cov over num_points for each z_dim
       """
       res = []
       for layer in zs:
         z_means = torch.mean(layer, dim=0)
         z_rest = layer - z_means
         z_sq = z_rest**2
         z_covs = torch.mean(z_sq, dim=0)
         z_covs = z_covs.cpu().numpy()
         if len(z_covs.shape) == 0:
             z_covs = np.reshape(z_covs, 1)
         res.append(z_covs)
       return res

    z_means = gen_zs(model, data_loaders, num_points)
    z_covs = compute_covs(z_means)
    return z_covs
   
def num_active_units(variances: list, threshold=0.01):
    res = []
    for layer_vars in variances:
        above = len([v for v in layer_vars if v > threshold])
        res.append(above)
    return res

def chop_units(model, variances, threshold=0.01):
    # See https://github.com/yburda/iwae/blob/master/iwae.py line 288
    # Usage in https://github.com/yburda/iwae/blob/master/experiments.py line 76
    # Author's measure_marginal_log_likelihood() is our test_epoch() pretty much
    pass

def sample_X(model, exper: dict, amount=1):
    """Only works for MNIST (assumes 28x28 and Bernoulli layer).
    See sample_x in inspect_linear_model for Gaussian.
    """
    with torch.no_grad():
        zs = sample_Z(model, exper)

        res = []
        for _ in range(amount):
            extra_noise = 0.3*sample_Z(model, exper)
            ps = model.decode(zs + extra_noise)
            assert len(ps) == 1, "Only Bernoulli supported"
            (ps,) = ps
            ps = ps.view(28, 28)
            res.append(ps.cpu().numpy())
        return res if len(res) > 1 else res[0]

def interpolate_X(model, exper: dict, amount=10):
    """Only works for MNIST (assumes 28x28 and Bernoulli layer).
    See sample_x in inspect_linear_model for Gaussian.
    """
    with torch.no_grad():
        zs1 = sample_Z(model, exper)
        zs2 = sample_Z(model, exper)

        res = []
        for i in range(amount):
            zs = i/amount*zs1 + (amount-i)/amount*zs2
            ps = model.decode(zs)
            assert len(ps) == 1, "Only Bernoulli supported"
            (ps,) = ps
            ps = ps.view(28, 28)
            res.append(ps.cpu().numpy())
        return res if len(res) > 1 else res[0]

def sample_Z(model, exper: dict):
    with torch.no_grad():
        z_dims = exper['model']['Z_dim']
        assert len(z_dims) == 1 # no support for 2 stoch layers yet
        (z_dim,) = z_dims
        from typing import Any
        zs : Any = [float(x) for x in np.random.normal(size=z_dim)]
        zs = torch.tensor([zs])
        return zs


def plot_images(images: List[List[np.ndarray]]):
    import matplotlib.pyplot as plt

    N, M = len(images), len(images[0])

    fig, axs = plt.subplots(N, M)

    for i, xs in enumerate(images):
        assert len(xs) == M
        for j, X in enumerate(xs):
           X = (1-X) * 255
           X = np.array(X, dtype='uint8')
           axs[i,j].imshow(X, cmap='gray')
           axs[i,j].axis('off')
           axs[i,j].grid(True)

    fig.tight_layout()
    return fig, axs


