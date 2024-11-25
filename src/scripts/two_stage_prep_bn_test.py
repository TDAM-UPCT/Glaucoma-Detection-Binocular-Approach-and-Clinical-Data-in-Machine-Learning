import os
import sys
from typing import Dict

import numpy as np
import torch
from torchinfo import summary
from torchvision import transforms
from torchvision.models import (
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
)
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(SRC_DIR, "data")
sys.path.append(SRC_DIR)

from src.data.dataset import PapilaOFIBE, PapilaOFISE

SAMPLE_NUM = 0

base_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
# print(summary(base_model, input_size=(1, 3, 224, 224)))
# base_model = efficientnet_b0(pretrained=True)
# print(summary(base_model, input_size=(1, 3, 224, 224)))

# Initialize the Weight Transforms
weights = EfficientNet_V2_S_Weights.DEFAULT
preprocess = weights.transforms()
print(preprocess)
# Apply it to the input image
# img_transformed = preprocess(img)

train_transform = transforms.Compose(
    transforms=[
        transforms.ToPILImage(),
        # transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomAffine(
            degrees=(0, 180), translate=(0.1, 0.1), scale=(0.9, 1.1)
        ),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
    ]
)

# TODO: With InceptionV3, use 299x299 images,
val_transform = transforms.Compose(
    transforms=[
        transforms.ToPILImage(),
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
print(train_transform)
print(val_transform)
ofi_se = PapilaOFISE(train_transform=train_transform, test_transform=val_transform)
ofi_be = PapilaOFIBE(train_transform=train_transform, test_transform=val_transform)

ofi_se_dataset = ofi_se.get_kfold(sample_num=SAMPLE_NUM)
ofi_be_dataset = ofi_be.get_kfold(sample_num=SAMPLE_NUM)


class ModelOPS:
    def __init__(self, model: torch.nn.Module, config: Dict):
        self.model = model
        self.config = config

    def fit(self, dataset):
        train_loader = torch.utils.data.DataLoader(
            dataset, self.config["batch_size"], shuffle=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        self.model.to(device)
        class_weights = torch.tensor(
            self.config["class_weights"], dtype=torch.float32
        ).to(device)
        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=self.config["label_smoothing"]
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )

        # Epoch Loop
        for epoch in range(self.config["epochs"]):
            print(f"Epoch: {epoch}/{self.config['epochs']}")
            print("-" * 10)
            self.model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs = preprocess(inputs)
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Loss: {epoch_loss:.4f}")

    def predict(self, dataset):

        y_pred = []
        y_true = []
        data_loader = torch.utils.data.DataLoader(
            dataset, 1, shuffle=False
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = preprocess(inputs)

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model(inputs)
                outputs = F.softmax(outputs, dim=1).cpu().numpy()
                y_pred.extend(outputs[:, 1])
                y_true.extend(labels.cpu().numpy())

        return np.array(y_true), np.array(y_pred)


config = {
    "batch_size": 32,
    "epochs": 30,
    "lr": 0.002,
    "weight_decay": 0.0001,
    "class_weights": [0.48702595, 1.89147287],
    "label_smoothing": 0.1,
}

# Base model config
# 1. Freeze the base model
# 2. Add a new classifier
base_model.requires_grad_(False)
base_model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(1280, 2))
trainable = any(param.requires_grad for param in base_model.classifier.parameters())
print(f"The model is {'trainable' if trainable else 'not trainable'}.")
# base_model.classifier.requires_grad_(True)
# print(summary(base_model, input_size=(1, 3, 384, 384)))


model_ops = ModelOPS(base_model, config)
freeze_layers = 0.3

for train_dataset, val_dataset in ofi_se_dataset:
    print("Stage 1")
    model_ops.fit(train_dataset)
    y_true, y_pred = model_ops.predict(val_dataset)
    y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
    auc = roc_auc_score(y_true, y_pred)
    print(f"AUC: {auc}")
    print("Stage 2")
    print(f"Model parameters: {len(list(model_ops.model.parameters()))}")
    model_ops.model.requires_grad_(True)
    aux_freeze_layers = int(len(list(model_ops.model.parameters())) * freeze_layers)
    for i, param in enumerate(model_ops.model.parameters()):
        if i < aux_freeze_layers:
            param.requires_grad = False

    model_ops.fit(train_dataset)
    y_true, y_pred = model_ops.predict(val_dataset)
    y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
    auc = roc_auc_score(y_true, y_pred)
    print(f"AUC: {auc}")

    break
