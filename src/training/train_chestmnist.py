"""
ChestMNIST — multi-label classification (14 disease classes + no-finding).
ResNet-34 backbone, BCEWithLogitsLoss with per-class positive weights.

Key fixes vs original:
  - Image size 224×224 (was 64×64 — ResNet pretrained resolution)
  - Consistent threshold (0.5) for both val and test
  - Per-class positive weights to handle severe label imbalance
  - Gradient clipping
  - Phased unfreezing with warmup LR
  - num_workers=0, pin_memory=False for CPU/Windows compatibility
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import classification_report, f1_score

from src.datasets.medmnist_datasets import get_chestmnist
from src.models.cnn_model import CNN

print(">>> ChestMNIST training started")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ─────────────────────────────────────────────────────────────
# TRANSFORMS — 224×224 matches ResNet-34 pretrained resolution
# ─────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.Normalize((0.5,), (0.5,)),
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5,), (0.5,)),
])

train_dataset = get_chestmnist("train", transform=train_transform)
val_dataset   = get_chestmnist("val",   transform=test_transform)
test_dataset  = get_chestmnist("test",  transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False,
                          num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False,
                          num_workers=0, pin_memory=False)

# ─────────────────────────────────────────────────────────────
# PER-CLASS POSITIVE WEIGHTS
# ChestMNIST is severely imbalanced — some diseases appear in
# <1% of images. pos_weight = neg_count / pos_count tells BCE
# to penalise false negatives more heavily for rare diseases.
# ─────────────────────────────────────────────────────────────
all_labels = np.array([label for _, label in train_dataset], dtype=np.float32)
pos_counts = all_labels.sum(axis=0).clip(min=1)
neg_counts = len(all_labels) - pos_counts
pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32).to(device)
print(f"pos_weight range: {pos_weight.min():.1f} – {pos_weight.max():.1f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
model = CNN(in_channels=1, num_classes=14).to(device)

for p in model.model.parameters():
    p.requires_grad = False
for p in model.model.fc.parameters():
    p.requires_grad = True


def resnet_params(layer_name):
    return [p for n, p in model.model.named_parameters() if layer_name in n]


optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7
)

THRESHOLD = 0.5   # single consistent threshold for val and test

def evaluate(loader):
    model.eval()
    all_preds, all_labels_list = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs  = torch.sigmoid(model(images))
            preds  = (probs > THRESHOLD).int()
            all_preds.append(preds.cpu().numpy())
            all_labels_list.append(labels.numpy())
    preds_arr  = np.vstack(all_preds)
    labels_arr = np.vstack(all_labels_list)
    return f1_score(labels_arr, preds_arr, average="macro", zero_division=0)


EPOCHS    = 40
best_f1   = 0.0
patience  = 8
no_improve = 0

for epoch in range(EPOCHS):

    # Phased unfreezing — head needs time to stabilise first
    if epoch == 8:
        print("Unfreezing layer4...")
        for p in model.model.layer4.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam([
            {"params": model.model.fc.parameters(),     "lr": 1e-4},
            {"params": resnet_params("layer4"),          "lr": 1e-5},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7
        )

    if epoch == 14:
        print("Unfreezing layer3...")
        for p in model.model.layer3.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam([
            {"params": model.model.fc.parameters(),     "lr": 5e-5},
            {"params": resnet_params("layer4"),          "lr": 1e-5},
            {"params": resnet_params("layer3"),          "lr": 5e-6},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7
        )

    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_f1   = evaluate(val_loader)
    scheduler.step(val_f1)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1    = val_f1
        no_improve = 0
        torch.save(model.state_dict(), "models/chestmnist_resnet.pth")
        print(f"  ✓ Saved (F1={best_f1:.4f})")
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# ─────────────────────────────────────────────────────────────
# FINAL TEST
# ─────────────────────────────────────────────────────────────
print(f"\nTraining finished. Best val F1: {best_f1:.4f}")
model.load_state_dict(torch.load("models/chestmnist_resnet.pth"))
model.eval()

all_preds, all_labels_list = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        probs  = torch.sigmoid(model(images))
        preds  = (probs > THRESHOLD).int()
        all_preds.append(preds.cpu().numpy())
        all_labels_list.append(labels.numpy())

preds_arr  = np.vstack(all_preds)
labels_arr = np.vstack(all_labels_list)

print("\nClassification Report:")
print(classification_report(labels_arr, preds_arr, zero_division=0))
print("Macro F1:", f1_score(labels_arr, preds_arr, average="macro", zero_division=0))