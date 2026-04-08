"""
DermaMNIST — 7-class skin lesion classification.
ResNet-34 backbone, weighted CrossEntropyLoss.

Key fixes vs original:
  - Critical bug fixed: scheduler.step(f1_score) passed the function
    object — now correctly passes the computed val_f1 float value
  - Critical bug fixed: print(f"... {f1_score:.4f}") crashed —
    f1_score is a function, val_f1 is the computed value
  - Image size 224×224 (was 64×64)
  - Phased unfreezing — full unfreeze at epoch 5 was too early,
    head hadn't learned yet causing feature corruption
  - Gradient clipping added
  - num_workers=0, pin_memory=False for CPU/Windows
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, f1_score

from src.datasets.medmnist_datasets import get_dermamnist
from src.models.cnn_model import CNN

print(">>> DermaMNIST training started")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ─────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_dataset = get_dermamnist("train", transform=train_transform)
val_dataset   = get_dermamnist("val",   transform=test_transform)
test_dataset  = get_dermamnist("test",  transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False,
                          num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False,
                          num_workers=0, pin_memory=False)

# ─────────────────────────────────────────────────────────────
# CLASS WEIGHTS — sqrt-capped inverse frequency
# ─────────────────────────────────────────────────────────────
labels_list  = [int(label) for _, label in train_dataset]
class_counts = Counter(labels_list)
num_classes  = 7
total        = len(labels_list)
print(f"Class counts: {dict(sorted(class_counts.items()))}")

weights = torch.tensor(
    [np.sqrt(total / class_counts[i]) for i in range(num_classes)],
    dtype=torch.float32
).to(device)
print(f"Class weights: {np.round(weights.cpu().numpy(), 2)}")

criterion = nn.CrossEntropyLoss(weight=weights)

# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
model = CNN(in_channels=3, num_classes=num_classes).to(device)

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


def evaluate(loader):
    model.eval()
    all_preds, all_labels_list = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds  = torch.argmax(model(images), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels_list.extend(labels.numpy())
    # f1_score is now the local variable name, not the function
    return f1_score(all_labels_list, all_preds, average="macro", zero_division=0)


EPOCHS     = 50
best_f1    = 0.0
patience   = 8
no_improve = 0

for epoch in range(EPOCHS):

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
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_f1   = evaluate(val_loader)          # computed float, not the function
    scheduler.step(val_f1)                   # fixed: was scheduler.step(f1_score)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1    = val_f1
        no_improve = 0
        torch.save(model.state_dict(), "models/dermamnist_resnet.pth")
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
model.load_state_dict(torch.load("models/dermamnist_resnet.pth"))
model.eval()

all_preds, all_labels_list = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds  = torch.argmax(model(images), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels_list.extend(labels.numpy())

print(f"\nTest Accuracy: {np.mean(np.array(all_preds) == np.array(all_labels_list))*100:.2f}%")
print("\nClassification Report:")
print(classification_report(all_labels_list, all_preds, zero_division=0))
print("Macro F1:", f1_score(all_labels_list, all_preds, average="macro", zero_division=0))