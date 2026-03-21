import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.datasets.medmnist_datasets import get_chestmnist
from src.models.cnn_model import CNN

from sklearn.metrics import classification_report
import numpy as np

print(">>> SCRIPT STARTED")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = get_chestmnist("train")
val_dataset = get_chestmnist("val")
test_dataset = get_chestmnist("test")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

model = CNN(in_channels=1, num_classes=14).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=3,
    factor=0.5
)

epochs = 20
best_acc = 0


def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            preds = (probs > 0.5).int()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    acc = (all_preds == all_labels).mean() * 100
    return acc


for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    val_acc = evaluate(model, val_loader)

    scheduler.step(val_acc)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_chestmnist_model.pth")


print("\nTraining finished.")
print("Best validation accuracy:", best_acc)


model.load_state_dict(torch.load("best_chestmnist_model.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).float()

        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, zero_division=0))