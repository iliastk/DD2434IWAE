from evaluation import train_epoch, test_epoch, measure_estimated_log_likelihood
from utils import *
from tensorboardX import SummaryWriter

def launch_experiment(experiment):

    results_dir = create_results_dir(experiment["name"])
    writer = SummaryWriter(results_dir)

    data_loader, batch_size, model_bias = setup_data(experiment["data"])
    model, criterion = setup_model(experiment["model"], model_bias)


    run_train_test(experiment["training"], batch_size,
                   data_loader, criterion, model, results_dir, writer)
    

def run_train_test(params, batch_size, data_loader, criterion, model, results_dir, writer):
    optimizer = setup_optimizer(params["optimizer"], model.parameters())
    scheduler = setup_scheduler(params["scheduler"], optimizer)
    early_stopping = setup_early_stopping(params['early_stopping'], results_dir)

    num_epochs = params['total_epochs']
    for epoch in range(num_epochs):
        train_results = train_epoch(
            optimizer, scheduler, criterion, batch_size, data_loader["train"], model)
        test_results  = test_epoch(
            data_loader["test"], criterion, batch_size, model)
        # test_results["NLL_5000"] = measure_estimated_log_likelihood(data_loader["test"], batch_size, model, num_samples=5000)
        log_results(early_stopping, test_results, train_results, epoch, num_epochs, model, writer, epoch)

        if early_stopping.early_stop:
            print("\t\t == Early stopped == ")
            break

