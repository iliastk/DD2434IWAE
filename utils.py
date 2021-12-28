
from datasets import BinarizedMNIST
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from vae import VAE


def setup_model(params, model_bias):
    X_dim = params['X_dim']
    Z_dim = params['Z_dim']
    H_dim = params['H_dim']
    num_samples = params['num_samples']
    if params['type'] == 'VAE':
        model = VAE(X_dim, H_dim, Z_dim, num_samples,
                    encoder=params['encoder_type'], decoder=params['decoder_type'],
                    bias=model_bias, loss_threshold=params['loss_th'])
    print(model)
    return model


def setup_data(params):
    data = {
        "train": BinarizedMNIST(train=True, root_path=params['path']),
        "val": None,
        "test": BinarizedMNIST(train=False, root_path=params['path'])
    }
    data_loader = {
        "train": torch.utils.data.DataLoader(
            dataset=data["train"], batch_size=params['batch_size'], shuffle=True, num_workers=8),
        "val": None,
        "test": torch.utils.data.DataLoader(
            dataset=data["test"], batch_size=params['batch_size'], shuffle=True, num_workers=8)
    }
    bias = data["train"].get_bias()
    return data_loader, params['batch_size'], bias


def create_results_dir(name):
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y-%H:%M:%S")

    results_dir = f'results/name/{timestamp}'
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    return results_dir


def get_optimizer(params, model_parameters):
    optimizer = torch.optim.Adam(
        model_parameters, lr=params['lr'], betas=(
            params['beta1'], params['beta2']), eps=params['epsilon']
    )
    return optimizer


def get_scheduler(params, optimizer):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=params['milestones'], gamma=params['gamma'], verbose=True
    )
    return scheduler


def log_results(best_model_dir, test_results, train_results, curr_epoch, num_epochs, model, writer, epoch):

    # Log Train Results
    train_loss, train_log_px = train_results['loss'], train_results['log_px']
    out_result = f'Epoch[{curr_epoch+1}/{num_epochs}],  Train [loss: {train_loss.item():.3f},  NLL: {train_log_px.item():.3f}]'

    # Log Test Results
    test_loss, test_log_px = test_results['loss'], test_results['log_px']
    out_result = out_result + \
        f'\t == \t Test [loss: {test_loss.item():.3f}, NLL:{test_log_px.item():.3f}]'

    print(out_result)

    # Save
    best_model_filename = f'{best_model_dir}/Epoch:{epoch}-Loss:{test_loss:.2f}-LogPx:{test_log_px:.2f}.pt'
    torch.save(model.state_dict(), best_model_filename)

    # Log to tensorboard
    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/NLL', train_log_px, epoch)
    writer.add_scalar('test/loss', test_loss, epoch)
    writer.add_scalar('test/NLL', test_log_px, epoch)


