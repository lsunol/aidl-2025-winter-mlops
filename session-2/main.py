import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model
from sklearn.model_selection import train_test_split
from typing import Tuple

import time


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device: ", device)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

def train_single_epoch(
        train_loader: torch.utils.data.DataLoader,
        network: torch.nn.Module,
        optimizer: torch.optim,
        criterion: torch.nn.functional,
        epoch: int,
        log_interval: int,
        ) -> Tuple[float, float]:
    
    network.train()

    train_loss = []
    acc = 0.
    avg_weight = 0.1

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Compute metrics
        acc += accuracy(outputs=output, labels=target)
        train_loss.append(loss.item())
 
        if batch_idx % log_interval == 0 or batch_idx >= len(train_loader):
            print('Train Epoch: {} [{:5d}/{} ({:3.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
    avg_acc = 100. * acc / len(train_loader.dataset)

    return np.mean(train_loss), avg_acc


def eval_single_epoch(
        test_loader: torch.utils.data.DataLoader, 
        network: torch.nn.Module,
        criterion: torch.nn.functional
        ) -> Tuple[float, float]:

    network.eval()

    test_loss = 0
    acc = 0
    for data, target in test_loader:

        data, target = data.to(device), target.to(device)

        output = network(data)

        # Apply the loss criterion and accumulate the loss
        test_loss += criterion(output, target).item()

        # compute number of correct predictions in the batch
        acc += accuracy(outputs=output, labels=target)

    # test_loss /= len(test_loader.dataset)
    avg_loss = test_loss / len(test_loader)

    # Average accuracy across all correct predictions batches now
    avg_acc = 100. * acc / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, acc, len(test_loader.dataset), avg_acc,
        ))
    return avg_loss, avg_acc

def train_model(
        config,
        plot: bool=True) -> MyModel:
    
    dataset = MyDataset('./mnist-chinese/data/data', './mnist-chinese/chinese_mnist.csv', transform)

    # Dividir el dataset en train (80%) y temp (20%)
    train_set, temp_set = train_test_split(dataset, test_size=0.2)

    # Dividir el conjunto temp en eval (10%) y test (10%)
    val_set, test_set = train_test_split(temp_set, test_size=0.5)

    print("Tamaño del conjunto de entrenamiento:", len(train_set))
    print("Tamaño del conjunto de evaluación:", len(val_set))
    print("Tamaño del conjunto de prueba:", len(test_set))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    eval_loader = torch.utils.data.DataLoader(val_set, batch_size=config['val_batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['val_batch_size'], shuffle=False)
    
    my_model = MyModel().to(device)

    num_epochs = config['num_epochs']
    optimizer = torch.optim.Adam(my_model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    tr_losses = []
    tr_accs = []
    te_losses = []
    te_accs = []

    for epoch in range(num_epochs):
        
        tr_loss, tr_acc = train_single_epoch(train_loader, my_model, optimizer, criterion, epoch, config['log_interval'])
        te_loss, te_acc = eval_single_epoch(eval_loader, my_model, criterion)

        tr_losses.append(tr_loss)
        tr_accs.append(tr_acc)
        te_losses.append(te_loss)
        te_accs.append(te_acc)
        
    if plot:
        plt.figure(figsize=(10, 8))
        plt.subplot(2,1,1)
        plt.xlabel('Epoch')
        plt.ylabel('NLLLoss')
        plt.plot(tr_losses, label='train')
        plt.plot(te_losses, label='eval')
        plt.legend()
        plt.subplot(2,1,2)
        plt.xlabel('Epoch')
        plt.ylabel('Eval Accuracy [%]')
        plt.plot(tr_accs, label='train')
        plt.plot(te_accs, label='eval')
        plt.legend()

        # Show graphics till closed manually
        plt.show()
    return my_model


if __name__ == "__main__":

    start_time = time.time()

    config = {
        "num_epochs": 15,
        "learning_rate": 0.001,
        "batch_size": 64,
        "val_batch_size": 64,
        "log_interval": 20,
    }
    train_model(config, plot=True)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tiempo de ejecución: {execution_time:.2f} segundos")
