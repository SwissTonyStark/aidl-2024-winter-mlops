import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model

# Setting the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lists to store loss and accuracy values
train_losses = []
test_losses = []
train_accs = []
test_accs = []
valid_losses = []
valid_accs = []


# Function to train the model for one epoch
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
            print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    avg_acc = 100. * acc / len(train_loader.dataset)
    return np.mean(train_loss), avg_acc


# Function to evaluate the model
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
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, acc, len(test_loader.dataset), test_acc,
        ))

    return test_loss, test_acc


# Main function to train the model
def train_model(config):
    dataset_path = "C:/Users/user/Desktop/Màster IA/Lab/aidl-2024-winter-mlops/session-2/data_chineseMNIST"
    labels_path = "C:/Users/user/Desktop/Màster IA/Lab/aidl-2024-winter-mlops/session-2/chinese_mnist.csv"

    # Transformations including normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Adjust these values based on your dataset
    ])

    # Creating dataset and splitting into train, test, and validation sets
    my_dataset = MyDataset(dataset_path, labels_path, transform)
    train_size = int(0.8 * len(my_dataset))
    test_size = int(0.1 * len(my_dataset))
    valid_size = len(my_dataset) - train_size - test_size
    train_ds, test_ds, valid_ds = random_split(my_dataset, [train_size, test_size, valid_size])

    # Creating DataLoaders for each dataset
    train_loader = DataLoader(dataset=train_ds, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=config['test_batch_size'], shuffle=False, drop_last=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=config['test_batch_size'], shuffle=False, drop_last=True)

    # Model, optimizer, and loss function initialization
    my_model = MyModel().to(device)
    optimizer = torch.optim.RMSprop(my_model.parameters(), lr=config["learning_rate"])
    criterion = nn.NLLLoss()

    best_valid_accuracy = 0.0
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        train_loss, train_acc = train_single_epoch(train_loader, my_model, optimizer, criterion, config["log_interval"])
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        test_loss, test_accuracy = eval_single_epoch(test_loader, my_model, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_accuracy)

        # Evaluating the model on the validation set
        valid_loss, valid_accuracy = eval_single_epoch(valid_loader, my_model, criterion)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_accuracy)

        # Saving the best model based on validation accuracy
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            save_model(my_model, 'best_model.pth')

    # Plotting the training and validation loss and accuracy
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.xlabel('Epoch')
    plt.ylabel('NLLLoss')
    plt.plot(train_losses, label='train')
    plt.plot(test_losses, label='test')
    plt.plot(valid_losses, label='valid')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.plot(train_accs, label='train')
    plt.plot(test_accs, label='test')
    plt.plot(valid_accs, label='valid')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    config = {
        "batch_size": 25,
        "epochs": 10,
        "test_batch_size": 25,
        "learning_rate": 1e-3,
        "log_interval": 100,
    }
    train_model(config)
