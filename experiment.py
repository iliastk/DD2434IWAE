from evaluation import train_epoch, test_epoch
from utils import *
from tensorboardX import SummaryWriter


def launch_experiment(experiment):

    results_dir = create_results_dir(experiment["name"])
    writer = SummaryWriter(results_dir)

    data_loader, batch_size, model_bias = setup_data(experiment["data"])
    model = setup_model(experiment["model"], model_bias)

    run_train_test(experiment["training"], batch_size,
                   data_loader, model, results_dir, writer)
    

def run_train_test(params, batch_size, data_loader, model, results_dir, writer):
    optimizer = get_optimizer(params["optimizer"], model.parameters())
    scheduler = get_scheduler(params["scheduler"], optimizer)
    num_epochs = params['total_epochs']
    input_dim = model.encoder.base_net[0].in_features
    for epoch in range(num_epochs):
        train_results = train_epoch(
            optimizer, scheduler, batch_size, data_loader["train"], model, input_dim)
        test_results, stop_training = test_epoch(
            data_loader["test"], batch_size, model, input_dim)
        log_results(results_dir, test_results,
                    train_results, epoch, num_epochs, model, writer, epoch)
        # Stop training when loss vanishes
        if (test_results['loss'] or train_results['loss']) is np.nan:
            print(
                f'\t\t == Stopping at epoch [{epoch}/{num_epochs}] because loss vanished ==')
            break
        if stop_training:
            print(
                f'\t\t == Stopping at epoch [{epoch}/{num_epochs}] because training converged ==')
            break

