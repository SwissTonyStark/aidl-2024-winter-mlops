import torch
from torch.utils.data import DataLoader
from model import MyModel
from utils import binary_accuracy_with_logits  # Usando la función correcta para la precisión
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

def train_single_epoch(model, train_loader, optimizer):
    model.train()
    accs, losses = [], []
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = F.binary_cross_entropy_with_logits(outputs, y.unsqueeze(1).float())
        acc = binary_accuracy_with_logits(y, outputs)  # Usando la función correcta aquí
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accs.append(acc)
    return np.mean(losses), np.mean(accs)

def eval_single_epoch(model, val_loader):
    accs, losses = [], []
    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = F.binary_cross_entropy_with_logits(outputs, y.unsqueeze(1).float())
            acc = binary_accuracy_with_logits(y, outputs)

            losses.append(loss.item())
            accs.append(acc)
    return np.mean(losses), np.mean(accs)

def train_model(config):
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = ImageFolder(
        "C:/Users/user/Desktop/MàsterIA/Lab/pycharm-aidl/aidl-2024-winter-mlops/session-3/cars_vs_flowers/training_set",
        transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    test_dataset = ImageFolder(
        "C:/Users/user/Desktop/MàsterIA/Lab/pycharm-aidl/aidl-2024-winter-mlops/session-3/cars_vs_flowers/test_set",
        transform=data_transforms)
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
        "lr": 3e-4,  # Tasa de aprendizaje ajustada
        "batch_size": 32,  # Mantener o ajustar según sea necesario
        "epochs": 15,  # Aumentar el número de épocas
    }
    my_model = train_model(config)
