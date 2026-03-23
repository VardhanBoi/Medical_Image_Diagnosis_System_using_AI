import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.datasets.medmnist_datasets import get_chestmnist
from src.models.cnn_model import CNN

from sklearn.metrics import classification_report, f1_score
import numpy as np

print(">>> ChestMNIST SCRIPT STARTED")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = get_chestmnist("train", transform=train_transform)
val_dataset = get_chestmnist("val", transform=test_transform)
test_dataset = get_chestmnist("test", transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

model = CNN(in_channels=1, num_classes=14).to(device)

for param in model.model.parameters():
    param.requires_grad = False

for param in model.model.fc.parameters():
    param.requires_grad = True

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0003,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=3,
    factor=0.5
)

epochs = 40
best_f1 = 0
patience = 7
no_improve = 0

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

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return f1

for epoch in range(epochs):

    model.train()
    total_loss = 0

    # Unfreeze after few epochs
    if epoch == 5:
        print("Unfreezing backbone...")
        for param in model.model.parameters():
            param.requires_grad = True

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

    val_f1 = evaluate(model, val_loader)
    scheduler.step(val_f1)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        no_improve = 0
        torch.save(model.state_dict(), "models/chestmnist_resnet.pth")
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break


print("\nTraining finished.")
print("Best validation F1:", best_f1)

model.load_state_dict(torch.load("models/chestmnist_resnet.pth"))
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

f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
print("Macro F1:", f1)