import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ray import train
from torch.utils.data import DataLoader
import numpy as np
from ray import tune
import ray

from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

def train_single_epoch(train_loader, model, optimizer, criterion, log_interval):
    model.train()
    train_loss = []
    acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        acc += accuracy(target, output)
        train_loss.append(loss.item())

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_acc = 100. * acc / len(train_loader.dataset)
    return np.mean(train_loss), avg_acc

def eval_single_epoch(test_loader, model, criterion):
    model.eval()
    test_loss = []
    acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss.append(criterion(output, target).item())
            acc += accuracy(target, output)

        test_acc = 100. * acc / len(test_loader.dataset)
        test_loss = np.mean(test_loss)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {acc}/{len(test_loader.dataset)} ({test_acc:.0f}%)\n')

    return test_loss, test_acc

def train_model(config):
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=int(config['batch_size']),
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=int(config['test_batch_size']),
        shuffle=False,
        drop_last=True,
    )

    my_model = MyModel().to(device)
    optimizer = torch.optim.RMSprop(my_model.parameters(), lr=config["learning_rate"])
    criterion = nn.NLLLoss(reduction='mean')

    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(checkpoint_dir, "checkpoint"))
            my_model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_single_epoch(train_loader, my_model, optimizer, criterion, config["log_interval"])
        val_loss, val_acc = eval_single_epoch(val_loader, my_model, criterion)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((my_model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=val_loss, accuracy=val_acc)
    return my_model

def test_model(model, test_dataset):
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=25,  # Can be adjusted
        shuffle=False,
        drop_last=True,
    )

    criterion = nn.NLLLoss(reduction='mean')
    test_loss, test_accuracy = eval_single_epoch(test_loader, model, criterion)
    print(f'Test Accuracy: {test_accuracy}%')
    return test_loss, test_accuracy

if __name__ == "__main__":
    dataset_path = "C:/Users/user/Desktop/Màster IA/Lab/aidl-2024-winter-mlops/session-2/data_chineseMNIST"
    labels_path = "C:/Users/user/Desktop/Màster IA/Lab/aidl-2024-winter-mlops/session-2/chinese_mnist.csv"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Adjust these values as per your dataset
    ])

    my_dataset = MyDataset(dataset_path, labels_path, transform)
    train_size = int(0.7 * len(my_dataset))
    test_size = int(0.2 * len(my_dataset))
    val_size = len(my_dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(my_dataset, [train_size, test_size, val_size])

    ray.init(configure_logging=False)
    analysis = tune.run(
        train_model,
        resources_per_trial={"cpu": 2, "gpu": 1},
        config={
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([16, 32, 64]),
            "test_batch_size": 64,
            "epochs": 10,
            "log_interval": 10
        },
        num_samples=10
    )

    best_model = train_model(analysis.best_config)
    test_loss, test_accuracy = test_model(best_model, test_dataset)
