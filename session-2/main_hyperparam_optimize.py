import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ray
from ray import train, tune
# from ray.tune import trial_dirname_creator

from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model
from sklearn.model_selection import train_test_split
from typing import Tuple
import time
from functools import partial


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

def short_dirname_creator(trial):
    return f"trial_{trial.trial_id}"


def train_single_epoch(
        train_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim,
        criterion: torch.nn.functional,
        device: torch.device
        ):
    
    model.train()

    running_loss = 0.0
    epoch_steps = 0

    print("[TRAIN] Loading data and target to device:", device)

    for i, data in enumerate(train_loader, 0):
    # for data, target in train_loader:

        print("Training!")
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_steps += 1
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(
                "[%d, %5d] loss: %.3f"
                % (epoch + 1, i + 1, running_loss / epoch_steps)
            )
            running_loss = 0.0


def eval_single_epoch(
        val_loader: torch.utils.data.DataLoader, 
        model: torch.nn.Module,
        criterion: torch.nn.functional,
        device: torch.device
        ):

    model.eval()

    print("[EVAL] Loading data and target to device:", device)

    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0

    # for data, target in val_loader:
    for i, data in enumerate(val_loader, 0):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        output = model(data)
        loss = criterion(output, labels)

        val_loss += loss.cpu().numpy()
        val_steps += 1

    # checkpoint_data = {
    #     "epoch": epoch,
    #     "net_state_dict": net.state_dict(),
    #     "optimizer_state_dict": optimizer.state_dict(),
    # }

    # with tempfile.TemporaryDirectory() as checkpoint_dir:
    #     data_path = Path(checkpoint_dir) / "data.pkl"
    #     with open(data_path, "wb") as fp:
    #         pickle.dump(checkpoint_data, fp)

    #     checkpoint = Checkpoint.from_directory(checkpoint_dir)
    #     train.report(
    #         {"loss": val_loss / val_steps, "accuracy": correct / total},
    #         checkpoint=checkpoint,
    #     )

        train.report(
            {"loss": val_loss / val_steps, "accuracy": correct / total}
        )


def train_model(config):

    print("ENTERING TRAIN MODEL")
    print("Creating train and val loaders...")
    train_loader = torch.utils.data.DataLoader(ray.get(train_dataset_ref), batch_size=config['batch_size'], shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(ray.get(val_dataset_ref), batch_size=config['batch_size'], shuffle=False, num_workers=8)

    print("Creating model...")
    my_model = MyModel(hidden_size=config["hidden_size"], dropout=config["dropout"])

    print("Assigning device...")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        print("Cuda devices:", torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            my_model = nn.DataParallel(my_model)
    my_model.to(device)
    print("Device:", device)

    print("Creating optimizer & criterion...")
    num_epochs = config['num_epochs']
    optimizer = torch.optim.Adam(my_model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    print("Starting epochs...")
    for epoch in range(num_epochs):

        print("Training epoch", epoch)
        train_single_epoch(train_loader, my_model, optimizer, criterion, device)
        print("Evaluating epoch", epoch)
        eval_single_epoch(val_loader, my_model, criterion, device)

    train.report({"loss"})

    return my_model


def test_model(config, model, test_dataset):

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    criterion = nn.CrossEntropyLoss()

    print("TESTING MODEL")
    eval_single_epoch(test_loader, model, criterion)



if __name__ == "__main__":

    start_time = time.time()

    ray.init(configure_logging=False)

    dataset = MyDataset('./mnist-chinese/data/data', './mnist-chinese/chinese_mnist.csv', transform)

    # Dividir el dataset en train (80%) y temp (20%)
    train_dataset, temp_set = train_test_split(dataset, test_size=0.2)
    train_dataset_ref = ray.put(train_dataset)

    # Dividir el conjunto temp en eval (10%) y test (10%)
    val_dataset, test_dataset = train_test_split(temp_set, test_size=0.5)
    val_dataset_ref = ray.put(val_dataset)
    test_dataset_ref = ray.put(test_dataset)

    print("Tamaño del conjunto de entrenamiento:", len(train_dataset))
    print("Tamaño del conjunto de evaluación:", len(val_dataset))
    print("Tamaño del conjunto de prueba:", len(test_dataset))

    print("Manual train start...")
    result = train_model(config={
        "learning_rate": 0.001,
        "num_epochs": 1,
        "batch_size": 32,
        "hidden_size": 64,
        "dropout": 0.2
    })
    print("Manual train ended!")

    analysis = tune.run(
        partial(train_model),
        trial_dirname_creator=short_dirname_creator,
        local_dir="C:/temp/ray_results",
        metric="val_loss",
        mode="min",
        num_samples=5,
        # resources_per_trial={"cpu": 8, "gpu": 1},
        config={
            "learning_rate": tune.choice([1e-4, 1e-3, 1e-2, 1e-1]),
            "num_epochs": tune.choice([5, 10, 15]),
            "batch_size": tune.grid_search([16, 64, 128, 256]),
            # "num_of_kernels_in_conv_layers": tune.grid_search()
            # (["relu", "tanh"]),
            "hidden_size": tune.randint(64, 256),
            "dropout": tune.uniform(0.1, 0.5)
        })

    print("Best hyperparameters found were: ", analysis.best_config)
    best_model = train_model(analysis.best_config)
    print(test_model(analysis.best_config, best_model, test_dataset))

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tiempo de ejecución: {execution_time:.2f} segundos")