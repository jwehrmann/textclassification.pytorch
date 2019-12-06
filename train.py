import torch
from torch import nn
from torch import optim
import torchvision
from matplotlib import pyplot as plt
import numpy as np


def train_epoch(
        model, 
        device, 
        train_loader, 
        optimizer, 
        criterion, 
        epoch, 
        log_interval
    ):
    losses = []
    model.train()    
    for batch_idx, (data, target, lengths) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, lengths)
        loss = criterion(output, target)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {:3d} [{:6d}/{:6d} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))            
    return losses

@torch.no_grad()
def test(
        model, 
        device,         
        test_loader
    ):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss().to(device)    
    for data, target, lengths in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data, lengths)
        test_loss += criterion(output, target).item() # sum up batch loss
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    return accuracy


def train(
        model,
        train_loader,
        test_loader,
        device,
        optimizer,
        nb_epochs=3,
        log_interval=100,
        lr_scheduler=None,
    ):    
    criterion = nn.CrossEntropyLoss().to(device)
    history = {'train_loss': [], 'val_acc': []}
    for epoch in range(1, nb_epochs + 1):
        print('\n* * * Training * * *')
        train_loss = train_epoch(
            model=model, 
            device=device, 
            train_loader=train_loader, 
            optimizer=optimizer, 
            criterion=criterion, 
            epoch=epoch, 
            log_interval=log_interval
        )
        if lr_scheduler:
            lr_scheduler.step()
        print('\n* * * Evaluating * * *')
        acc = test(model, device, test_loader)                
        history['val_acc'].append(acc)
        history['train_loss'].extend(train_loss)

    return history


def check_input(model, device):
    dummy_data = torch.ones(5, 30).long().to(device)
    lens = [28]*5
    dummy_pred = model(dummy_data, lens)
    assert dummy_pred.shape == (5, 4), '\nOutput expected: (batch_size, 4) \nOutput found   : {}'.format(dummy_pred.shape)
    print('Passed')
    return dummy_pred
