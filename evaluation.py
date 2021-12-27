import torch


def test_epoch(data_loader, batch_size, model, X_dim):
    epoch_log_px = []
    epoch_loss = []
    with torch.no_grad():
        for batch_idx, (X, _) in enumerate(data_loader):
            X = X.view(batch_size, X_dim)
            outputs, batch_loss, batch_log_px = model(X)

            epoch_log_px.append(batch_log_px.item())
            epoch_loss.append(batch_loss.item())
        epoch_log_px = torch.mean(torch.tensor(epoch_log_px))
        epoch_loss = torch.mean(torch.tensor(epoch_loss))

        return {'loss': epoch_loss, 'log_px': epoch_log_px}


def train_epoch(optimizer, scheduler, batch_size, data_loader, model, X_dim):
    epoch_loss = []
    epoch_log_px = []
    for batch_idx, (X, _) in enumerate(data_loader):
        optimizer.zero_grad()
        X = X.view(batch_size, X_dim)
        outputs, batch_loss, batch_log_px = model(X)
        batch_loss.backward()
        optimizer.step()

        epoch_loss.append(batch_loss)
        epoch_log_px.append(batch_log_px)
    epoch_loss = torch.mean(torch.tensor(epoch_loss))
    epoch_log_px = torch.mean(torch.tensor(epoch_log_px))
    scheduler.step()

    return {'loss': epoch_loss, 'log_px': epoch_log_px}


def log_results(best_model_dir, test_results, train_results, curr_epoch, num_epochs, model):

    # Log Train Results
    loss, log_px = train_results['loss'], train_results['log_px']
    out_result = f'Epoch[{curr_epoch+1}/{num_epochs}],  Train [loss: {loss.item():.3f},  NLL: {log_px.item():.3f}]'

    # Log Test Results
    loss, log_px = test_results['loss'], test_results['log_px']
    out_result = out_result + \
        f'\t == \t Test [loss: {loss.item():.3f}, NLL:{log_px.item():.3f}]'
    best_model_filename = f'{best_model_dir}/Loss:{loss:.3f}-LogPx:{log_px:.3f}-Epoch:{curr_epoch}.pt'

    # TODO: Log to tensorboard

    print(out_result)
    torch.save(model.state_dict(), best_model_filename)
