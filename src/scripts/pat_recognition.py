"""
Pat recognition training script 
"""

import os
import sys

import mlflow
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from scipy.io import loadmat

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(SRC_DIR)
from src.data.dataset import PapilaOFI, PapilaTorch, PapilaTorchBothEyes
from src.models.cnn import ResNet18BothEyes, ResNet50BothEyes


dataset = loadmat("./data/mat_db/PAPILAv10.2.1.mat")

X = dataset["X"]
Y = dataset["Y"][0]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2021, stratify=Y)


train_transform = transforms.Compose(transforms=[transforms.ToPILImage(),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandomVerticalFlip(p=0.5),
                                                transforms.RandomRotation(degrees=(0, 180)),
                                                transforms.RandomAffine(degrees=(0, 180), translate=(0.1, 0.1), scale=(0.9, 1.1)),
                                                transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                                                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                                                transforms.ColorJitter(brightness=0.1),
                                                transforms.ToTensor(),
                                                 ])
val_transform = transforms.Compose(transforms=[transforms.ToPILImage(), transforms.ToTensor()])

train_dataset = PapilaTorchBothEyes(x_train, y_train, transform=train_transform)
val_dataset = PapilaTorchBothEyes(x_test, y_test, transform=val_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=True)


os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9007"
os.environ['AWS_ACCESS_KEY_ID'] = 'masoud'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'Strong#Pass#2022'

mlflow.set_tracking_uri('http://localhost:5003')
mlflow.set_experiment("PapilaBothEyes")

config = {"lr": 0.0001,
          "epochs": 200,
          "batch_size": 2,
          "num_classes": 2,
          "hidden_units": 1024,
          "hidden_layers": 2,
          "dropout": 0.5,
          "freeze_layers": 0,
          "pretrained": True,
          "weight_decay": 0.0001}

def train(model, data_loader, optimizer, criterion, device):
    
    model.train()
    print('Training...')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in enumerate(data_loader, 0):
        counter += 1
        image, labels = data
        image, labels = image.to(device), labels.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(image)
        # Calculate loss
        loss = criterion(outputs, labels.long())
        train_running_loss += loss.item()
        # Calculate accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backward pass
        loss.backward()
        optimizer.step()
    
    # Loss and accuracy for the epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = (train_running_correct / len(data_loader.dataset))
    return epoch_loss, epoch_acc

def validation(model, data_loader, criterion, device):
    
    model.eval()
    print('Validating...')
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            counter += 1
            image, labels = data
            image, labels = image.to(device), labels.to(device)
            # Forward pass
            outputs = model(image)
            # Calculate loss
            loss = criterion(outputs, labels.long())
            val_running_loss += loss.item()
            # Calculate accuracy
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == labels).sum().item()
            
    # Loss and accuracy for the epoch
    epoch_loss = val_running_loss / counter
    epoch_acc = (val_running_correct / len(data_loader.dataset))
    return epoch_loss, epoch_acc

model = ResNet50BothEyes(num_classes=config["num_classes"],
                         pretrained=config["pretrained"],
                         hidden_units=config["hidden_units"],
                         hidden_layers=config["hidden_layers"],
                         dropout=config["dropout"],
                         freeze_layers=config["freeze_layers"])

device = "cuda:0"
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])


with mlflow.start_run() as run:
    
    for epoch in range(config["epochs"]):
        print(f"[INFO]: Epoch {epoch + 1} of {config['epochs']}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, device)
        val_epoch_loss, val_epoch_acc = validation(model, val_loader, criterion, device)
        
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {val_epoch_loss:.3f}, validation acc: {val_epoch_acc:.3f}")
        print('-'*50)

        mlflow.log_metric("train_loss", train_epoch_loss, step=epoch)
        mlflow.log_metric("train_acc", train_epoch_acc, step=epoch)
        mlflow.log_metric("val_loss", val_epoch_loss, step=epoch)
        mlflow.log_metric("val_acc", val_epoch_acc, step=epoch)
        mlflow.log_params(config)
    mlflow.pytorch.log_model(model, "models")
        
        