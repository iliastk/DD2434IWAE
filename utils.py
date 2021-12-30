from typing import Tuple
from datasets import BinarizedMNIST
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from vae import VAE
from loss import VAELoss, EarlyStopping


def setup_model(params, model_bias):
    X_dim = params['X_dim']
    Z_dim = params['Z_dim']
    H_dim = params['H_dim']
    num_samples = params['num_samples']
    if params['type'] == 'VAE':
        model = VAE(X_dim, H_dim, Z_dim, num_samples,
                    encoder=params['encoder_type'], decoder=params['decoder_type'],
                    bias=model_bias)
    print(model)

    criterion = VAELoss(num_samples, bernoulli_decoder =
                        (params['decoder_type']=='Bernoulli'))

    return model, criterion


def gen_fake_data(params: dict) -> np.array: #Tuple[dict, float, float]:
    n_samples = params['n_samples']
    name = params['name']
    # Maybe code a formula into the params somehow. Or generate a static dataset
    # in './data'?
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
    return x, [μ]
    
    

def setup_data(params):
    if params['name'] not in ['linear2dim', 'two_clusters']:
        data = {
            'train': BinarizedMNIST(train=True, root_path=params['path']),
            'val': None,
            'test': BinarizedMNIST(train=False, root_path=params['path'])
        }
    
        data_loader = {
            'train': torch.utils.data.DataLoader(
                dataset=data['train'], batch_size=params['batch_size'],
                shuffle=True, num_workers=8),
            'val': None,
            'test': torch.utils.data.DataLoader(
                dataset=data['test'], batch_size=params['batch_size'],
                shuffle=True, num_workers=8)
        }
        bias = data['train'].get_bias()
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
    timestamp = now.strftime("%d-%m-%Y-%H:%M:%S")

    results_dir = f'results/{name}/{timestamp}'
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    return results_dir


def setup_optimizer(params, model_parameters):
    optimizer = torch.optim.Adam(
        model_parameters, lr=params['lr'], betas=(
            params['beta1'], params['beta2']), eps=params['epsilon']
    )
    return optimizer


def setup_scheduler(params, optimizer):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=params['milestones'], gamma=params['gamma'], verbose=True
    )
    return scheduler


def setup_early_stopping(params, results_dir):
    early_stopping = EarlyStopping(
        patience=params['patience'], threshold=params['threshold'], best_model_dir=results_dir)
    return early_stopping


def log_results(early_stopping, test_results, train_results, curr_epoch, num_epochs, model, writer, epoch):

    # Log Train Results
    train_loss, train_NLL = train_results['loss'], train_results['NLL']
    out_result = f'Epoch[{curr_epoch+1}/{num_epochs}],  Train [loss: {train_loss.item():.3f},  NLL: {train_NLL.item():.3f}]'

    # Log Test Results
    test_loss, test_NLL = test_results['loss'], test_results['NLL']
    out_result = out_result + \
        f'\t == \t Test [loss: {test_loss.item():.3f}, NLL:{test_NLL.item():.3f}]'

    print(out_result)

    # Log to tensorboard
    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/NLL', train_NLL, epoch)
    writer.add_scalar('test/loss', test_loss, epoch)
    writer.add_scalar('test/NLL', test_NLL, epoch)

    early_stopping(test_NLL, test_loss, epoch, model)
