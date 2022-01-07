import torch
from loss import IWAELoss


def measure_estimated_log_likelihood(data_loader, model,
                                     num_samples=5000,
                                     num_points=500): # Hope they're shuffled!
    NLL = []
    criterion = IWAELoss(num_samples)
    batch_size = data_loader.batch_size
    with torch.no_grad():
        for batch_idx, (X, _) in enumerate(data_loader):
            X = X.view(batch_size, model.encoder.input_dim)
            output = model(X, num_samples)
            batch_loss, batch_NLL = criterion(output, X, model)

            NLL.append(batch_NLL.item())
            num_points -= batch_size
            if num_points <= 0: break
                

        NLL = torch.mean(torch.tensor(NLL))

        return NLL

def test_epoch(data_loader, criterion, batch_size, model):
    epoch_NLL = []
    epoch_loss = []
    with torch.no_grad():
        for batch_idx, (X, _) in enumerate(data_loader):
            if X.size(0) != batch_size: continue
            output = model(X)
            batch_loss, batch_NLL = criterion(output, X, model)

            epoch_NLL.append(batch_NLL.item())
            epoch_loss.append(batch_loss.item())
        epoch_NLL = torch.mean(torch.tensor(epoch_NLL))
        epoch_loss = torch.mean(torch.tensor(epoch_loss))

        return {'loss': epoch_loss, "NLL_k":
                measure_estimated_log_likelihood(data_loader,model,
                                                 num_samples=5000,
                                                 )}


def train_epoch(optimizer, scheduler, criterion, batch_size, data_loader, model):
    epoch_loss = []
    epoch_NLL = []
    for batch_idx, (X, _) in enumerate(data_loader):
        optimizer.zero_grad()
        if X.size(0) != batch_size: continue
        output = model(X)
        batch_loss, batch_NLL = criterion(output, X, model)
        batch_loss.backward()
        optimizer.step()

        epoch_loss.append(batch_loss)
        epoch_NLL.append(batch_NLL)
    epoch_loss = torch.mean(torch.tensor(epoch_loss))
    epoch_NLL = torch.mean(torch.tensor(epoch_NLL))
    scheduler.step()

    return {'loss': epoch_loss, 'NLL': epoch_NLL}


