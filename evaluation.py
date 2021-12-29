import torch


def test_epoch(epoch, data_loader, criterion, batch_size, model, X_dim, early_stopping):
    epoch_NLL = []
    epoch_loss = []
    with torch.no_grad():
        for batch_idx, (X, _) in enumerate(data_loader):
            X = X.view(batch_size, X_dim)
            output = model(X)
            batch_loss, batch_NLL = criterion(output, X)

            epoch_NLL.append(batch_NLL.item())
            epoch_loss.append(batch_loss.item())
        epoch_NLL = torch.mean(torch.tensor(epoch_NLL))
        epoch_loss = torch.mean(torch.tensor(epoch_loss))

        return {'loss': epoch_loss, 'NLL': epoch_NLL}


def train_epoch(optimizer, scheduler, criterion, batch_size, data_loader, model, X_dim):
    epoch_loss = []
    epoch_NLL = []
    for batch_idx, (X, _) in enumerate(data_loader):
        optimizer.zero_grad()
        X = X.view(batch_size, X_dim)
        output = model(X)
        batch_loss, batch_NLL = criterion(output, X)
        batch_loss.backward()
        optimizer.step()

        epoch_loss.append(batch_loss)
        epoch_NLL.append(batch_NLL)
    epoch_loss = torch.mean(torch.tensor(epoch_loss))
    epoch_NLL = torch.mean(torch.tensor(epoch_NLL))
    scheduler.step()

    return {'loss': epoch_loss, 'NLL': epoch_NLL}


