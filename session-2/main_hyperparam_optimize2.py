import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

from functools import partial
import tempfile
import time
from pathlib import Path

import ray
from ray import tune
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
# from ray.tune import trial_dirname_creator

from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model

from sklearn.model_selection import train_test_split
import ray.cloudpickle as pickle

def train_model(config):

    my_model = MyModel(hidden_size=config['hidden_size'], dropout=config['dropout'])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            my_model = nn.DataParallel(my_model)
    my_model.to(device)

    optimizer = optim.Adam(my_model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            my_model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    trainset, valset, testset = load_data()

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = my_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 20 == 19:  # print every 20 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = my_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        
        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": my_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"loss": val_loss / val_steps, "accuracy": correct / total},
                checkpoint=checkpoint,
            )
    print("Finished Training")





def test_model(config):
    my_model = MyModel(hidden_size=config['hidden_size'], dropout=config['dropout'])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            my_model = nn.DataParallel(my_model)
    my_model.to(device)

    optimizer = optim.Adam(my_model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0

    trainset, valset, testset = load_data()

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = my_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 20 == 19:  # print every 20 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = my_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        
    print("Finished Training")

    # Test loss
    test_loss = 0.0
    test_steps = 0
    total = 0
    correct = 0
    for i, data in enumerate(testloader, 0):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = my_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            test_loss += loss.cpu().numpy()
            test_steps += 1

    # print statistics
    print("test loss: %.3f" % (test_loss))


def load_data():

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    dataset = MyDataset('C:/Users/lluis/GoogleDrive/PG in AI/project/aidl-2025-winter-mlops/session-2/mnist-chinese/data/data', 
                        'C:/Users/lluis/GoogleDrive/PG in AI/project/aidl-2025-winter-mlops/session-2/mnist-chinese/chinese_mnist.csv', 
                        transform)
    
    train_set, temp_set = train_test_split(dataset, test_size=0.2)
    val_set, test_set = train_test_split(temp_set, test_size=0.5)

    return train_set, val_set, test_set

def short_dirname_creator(trial):
    return f"trial_{trial.trial_id}"

if __name__ == "__main__":

    start_time = time.time()

    ray.init(configure_logging=False)

    num_samples=5, 
    max_num_epochs=10, 
    gpus_per_trial=1

    # config={
    #     "hidden_size": 64,
    #     "dropout": 0.5,
    #     "learning_rate": 0.001,
    #     "batch_size": 64
    # }

    # train_model(config)

    config={
        "hidden_size": tune.choice([64, 128, 256]),
        "dropout": tune.uniform(0.2, 0.5),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128])
    }

    # analysis = tune.run(
    #     partial(train_model),
    #     resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
    #     config=config,
    #     num_samples=num_samples,
    #     checkpoint_at_end=False)


    tuner = tune.Tuner(
        trainable=train_model,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=5, 
            trial_dirname_creator=short_dirname_creator,

            )
    )
    results = tuner.fit()

    analysis = tune.run(
        train_model,
        metric="loss",
        mode="min",
        num_samples=5,
        trial_dirname_creator=short_dirname_creator,
        resources_per_trial={"cpu": 6, "gpu": gpus_per_trial},
        config=config)

    print("Best hyperparameters found were: ", analysis.best_config)
    print(test_model(analysis.best_config))

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tiempo de ejecución: {execution_time:.2f} segundos")