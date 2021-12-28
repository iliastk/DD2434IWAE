import torch


def test_epoch(data_loader, batch_size, model, X_dim):
    epoch_log_px = []
    epoch_loss = []
    stop_training = False
    with torch.no_grad():
        for batch_idx, (X, _) in enumerate(data_loader):
            X = X.view(batch_size, X_dim)
            outputs, batch_loss, batch_log_px = model(X)

            epoch_log_px.append(batch_log_px.item())
            epoch_loss.append(batch_loss.item())
        epoch_log_px = torch.mean(torch.tensor(epoch_log_px))
        epoch_loss = torch.mean(torch.tensor(epoch_loss))

        # Stop training when test_losss does not improve significantly
        if epoch_loss < model.best_test_loss + model.loss_threshold:
            model.best_test_loss = epoch_loss

        else:
            stop_training = True

        return {'loss': epoch_loss, 'log_px': epoch_log_px}, stop_training


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


