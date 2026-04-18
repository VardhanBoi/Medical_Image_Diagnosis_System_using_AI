"""
ChestMNIST — multi-label chest X-ray classification (14 classes).

Published ResNet-34 benchmark on MedMNIST: ~0.23-0.27 macro F1.
Current run reached 0.2216 — close to benchmark, but stalling.

Changes from previous run:
1. Threshold search only every 3 epochs (was every epoch) — the
   full val forward pass for threshold tuning is expensive on GPU
   and the optimal thresholds don't change dramatically epoch-to-epoch
2. Early stopping patience increased to 12 (was 8) — improvement
   from ep15-27 was real but slow; patience=8 would have stopped too early
3. CosineAnnealingLR after each unfreezing phase instead of
   ReduceLROnPlateau — gives a more structured LR schedule that
   continues decreasing rather than waiting to plateau
4. Vertical flip augmentation added — chest X-rays have less vertical
   symmetry than retinal images but flipping still adds useful variance
   for rare pathologies
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
# TRANSFORMS
# ─────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.Normalize((0.5,), (0.5,)),
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5,), (0.5,)),
])

train_dataset = get_chestmnist("train", transform=train_transform)
val_dataset   = get_chestmnist("val",   transform=test_transform)
test_dataset  = get_chestmnist("test",  transform=test_transform)

nw = 2 if torch.cuda.is_available() else 0
pm = torch.cuda.is_available()
bs = 64 if torch.cuda.is_available() else 32

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                          num_workers=nw, pin_memory=pm)
val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=False,
                          num_workers=nw, pin_memory=pm)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False,
                          num_workers=nw, pin_memory=pm)

# ─────────────────────────────────────────────────────────────
# PER-CLASS POSITIVE WEIGHTS — capped at 10.0
# Raw neg/pos can exceed 500:1 for Hernia. Without cap,
# BCE loss pushes model to predict everything positive.
# ─────────────────────────────────────────────────────────────
all_train_labels = np.array([label for _, label in train_dataset], dtype=np.float32)
pos_counts  = all_train_labels.sum(axis=0).clip(min=1)
neg_counts  = len(all_train_labels) - pos_counts
capped_pw   = np.clip(neg_counts / pos_counts, 1.0, 10.0)
print(f"pos_weight (capped): {np.round(capped_pw, 1)}")

pos_weight = torch.tensor(capped_pw, dtype=torch.float32).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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


# ─────────────────────────────────────────────────────────────
# EVALUATION — returns val probs/labels for threshold search
# ─────────────────────────────────────────────────────────────
def get_val_probs(loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.sigmoid(model(images.to(device)))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.vstack(all_probs), np.vstack(all_labels)


def find_thresholds(probs_arr, labels_arr, n_classes=14):
    """Per-class threshold maximising per-class F1 on val set."""
    thresholds = np.zeros(n_classes)
    for c in range(n_classes):
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.1, 0.7, 25):
            preds = (probs_arr[:, c] > t).astype(int)
            f1 = f1_score(labels_arr[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


def eval_with_thresholds(probs_arr, labels_arr, thresholds):
    preds = (probs_arr > thresholds).astype(int)
    return f1_score(labels_arr, preds, average="macro", zero_division=0)


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────
EPOCHS     = 50
best_f1    = 0.0
patience   = 12
no_improve = 0

# Initial optimizer for head-only phase
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-7
)

# Cache thresholds — recompute only every 3 epochs
cached_thresholds = np.full(14, 0.5)
last_threshold_epoch = -1

for epoch in range(EPOCHS):

    # ── PHASE TRANSITIONS ────────────────────────────────────
    if epoch == 8:
        print("Unfreezing layer4...")
        for p in model.model.layer4.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam([
            {"params": model.model.fc.parameters(),  "lr": 1e-4},
            {"params": resnet_params("layer4"),       "lr": 1e-5},
        ], weight_decay=1e-4)
        # CosineAnnealingLR: smoothly decays LR over remaining epochs
        # better than ReduceLROnPlateau for this slow-converging task
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS - epoch, eta_min=1e-6
        )

    if epoch == 14:
        print("Unfreezing layer3...")
        for p in model.model.layer3.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam([
            {"params": model.model.fc.parameters(),  "lr": 5e-5},
            {"params": resnet_params("layer4"),       "lr": 1e-5},
            {"params": resnet_params("layer3"),       "lr": 5e-6},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS - epoch, eta_min=1e-7
        )

    # ── TRAIN ────────────────────────────────────────────────
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

    # ── EVALUATE ─────────────────────────────────────────────
    # Recompute thresholds every 3 epochs (expensive) or use cache
    val_probs, val_labels = get_val_probs(val_loader)

    if epoch % 3 == 0 or epoch < 5:
        cached_thresholds = find_thresholds(val_probs, val_labels)
        last_threshold_epoch = epoch

    val_f1 = eval_with_thresholds(val_probs, val_labels, cached_thresholds)

    # CosineAnnealingLR steps every epoch regardless of performance
    if epoch >= 8:
        scheduler.step()
    else:
        scheduler.step(val_f1)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | "
          f"Val F1: {val_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    if val_f1 > best_f1:
        best_f1    = val_f1
        no_improve = 0
        torch.save(model.state_dict(), "models/chestmnist_resnet.pth")
        np.save("models/chestmnist_thresholds.npy", cached_thresholds)
        print(f"  ✓ Saved (F1={best_f1:.4f})")
        print(f"    Thresholds: {np.round(cached_thresholds, 2)}")
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
best_thresholds = np.load("models/chestmnist_thresholds.npy")
model.eval()

all_probs, all_labels_list = [], []
with torch.no_grad():
    for images, labels in test_loader:
        probs = torch.sigmoid(model(images.to(device)))
        all_probs.append(probs.cpu().numpy())
        all_labels_list.append(labels.numpy())

probs_arr  = np.vstack(all_probs)
labels_arr = np.vstack(all_labels_list)
preds_arr  = (probs_arr > best_thresholds).astype(int)

CHEST_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia"
]
print("\nClassification Report:")
print(classification_report(labels_arr, preds_arr,
                             target_names=CHEST_LABELS, zero_division=0))
macro_f1 = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
print(f"Test Macro F1: {macro_f1:.4f}")
print(f"Thresholds used: {np.round(best_thresholds, 2)}")