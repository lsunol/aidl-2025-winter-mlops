import torch
from torch.utils.data import DataLoader

from model import MyModel
from utils import binary_accuracy, binary_accuracy_with_logits
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(model, train_loader, optimizer):
    model.train()
    accs, losses = [], []
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        # You will need to do y = y.unsqueeze(1).float() to add an output dimension to the labels and cast to the correct type
        y = y.unsqueeze(1).float()

        # Set network gradients to 0.
        optimizer.zero_grad()

        # Forward batch of images through the network
        output = model(x)
        # Compute loss
        loss = F.binary_cross_entropy_with_logits(output, y)

        # Compute backpropagation
        loss.backward()

        # Update parameters of the network
        optimizer.step()

        # Calculate Binary Accuracy
        acc = binary_accuracy_with_logits(output, y)

        losses.append(loss.item())
        accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def eval_single_epoch(model, val_loader):
    accs, losses = [], []
    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            y = y.unsqueeze(1).float()

            # Forward batch of images through the network
            output = model(x)
            
            # Compute loss
            loss = F.binary_cross_entropy_with_logits(output, y)

            # Calculate Binary Accuracy
            acc = binary_accuracy_with_logits(output, y)

            losses.append(loss.item())
            accs.append(acc.item())

    return np.mean(losses), np.mean(accs)


def train_model(config):

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    train_dataset = ImageFolder(root = './session-3/cars-and-flowers-data/dataset/cars_vs_flowers/training_set', transform = data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataset = ImageFolder(root = './session-3/cars-and-flowers-data/dataset/cars_vs_flowers/test_set', transform = data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    my_model = MyModel().to(device)

    optimizer = optim.Adam(my_model.parameters(), config["lr"])
    for epoch in range(config["epochs"]):
        loss, acc = train_single_epoch(my_model, train_loader, optimizer)
        print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
        loss, acc = eval_single_epoch(my_model, test_loader)
        print(f"Eval Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
    
    return my_model


if __name__ == "__main__":

    config = {
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 5,
    }
    my_model = train_model(config)

    
