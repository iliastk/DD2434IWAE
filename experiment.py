import os
from evaluation import train_epoch, test_epoch, measure_estimated_log_likelihood
from utils import *
from tensorboardX import SummaryWriter

def load(model, path):
    kwargs = {}
    if torch.cuda.device_count() == 0:
        kwargs['map_location'] = torch.device('cpu')
    state_dict = torch.load(path, **kwargs)
    model.load_state_dict(state_dict)
    print(f"Loaded model from path {path}")

def launch_experiment(experiment, checkpoint_location = None, epoch=0):

    results_dir = create_results_dir(experiment["name"])
    writer = SummaryWriter(results_dir)

    data_loader, batch_size, model_bias = setup_data(experiment["data"])
    model, criterion = setup_model(experiment["model"], model_bias)
    if checkpoint_location is not None:
      load(model, checkpoint_location)


    run_train_test(experiment["training"], batch_size,
                   data_loader, criterion, model, results_dir, writer, start_epoch=epoch)
    

def run_train_test(params, batch_size, data_loader, criterion, model, results_dir, writer, start_epoch):
    optimizer = setup_optimizer(params["optimizer"], model.parameters())
    scheduler = setup_scheduler(params["scheduler"], optimizer, start_epoch)
    early_stopping = setup_early_stopping(params['early_stopping'], results_dir)

    num_epochs = params['total_epochs']
    for epoch in range(start_epoch, num_epochs):
        train_results = train_epoch(
            optimizer, scheduler, criterion, batch_size, data_loader["train"], model)
        test_results  = test_epoch(
            data_loader["test"], criterion, batch_size, model)
        z_variances = get_units_variances(model, data_loader)
        active_units = num_active_units(z_variances)
        log_results(early_stopping, test_results, train_results, epoch, num_epochs, model, writer, epoch, active_units)

        if early_stopping.early_stop:
            print("\t\t == Early stopped == ")
            break

