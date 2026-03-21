import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.datasets.medmnist_datasets import get_dermamnist
from src.models.cnn_model import CNN

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# TRANSFORMS (ResNet style)
# -------------------------

train_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])

test_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])

# -------------------------
# DATA
# -------------------------

train_dataset = get_dermamnist("train", transform=train_transform)
val_dataset = get_dermamnist("val", transform=test_transform)
test_dataset = get_dermamnist("test", transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# -------------------------
# CLASS WEIGHTS (smoothed)
# -------------------------

all_labels = []

for _, labels in train_loader:
    all_labels.extend(labels.numpy())

all_labels = np.array(all_labels)

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=all_labels
)

weights = torch.tensor(weights, dtype=torch.float32)
weights = weights ** 0.5   # smooth weights
weights = weights.to(device)

# -------------------------
# MODEL (ResNet18)
# -------------------------

model = CNN(in_channels=3, num_classes=7).to(device)

# Freeze backbone initially
for param in model.model.parameters():
    param.requires_grad = False

for param in model.model.fc.parameters():
    param.requires_grad = True

# -------------------------
# TRAINING SETUP
# -------------------------

criterion = nn.CrossEntropyLoss(weight=weights)

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

epochs = 60
best_acc = 0
patience = 7
no_improve = 0


# -------------------------
# EVALUATION
# -------------------------

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# -------------------------
# TRAINING LOOP
# -------------------------

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
        labels = labels.to(device)

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
        no_improve = 0
        torch.save(model.state_dict(), "best_dermamnist_model.pth")
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break


print("\nTraining finished.")
print("Best validation accuracy:", best_acc)


# -------------------------
# TESTING
# -------------------------

model.load_state_dict(torch.load("best_dermamnist_model.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


test_acc = np.mean(np.array(all_preds) == np.array(all_labels)) * 100

print(f"\nTest Accuracy: {test_acc:.2f}%")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, zero_division=0))


# -------------------------
# METRICS
# -------------------------

f1 = f1_score(all_labels, all_preds, average='macro')
print("Macro F1:", f1)

print("Predicted classes:", np.unique(all_preds))


# -------------------------
# CONFUSION MATRIX
# -------------------------

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(7)
plt.xticks(tick_marks)
plt.yticks(tick_marks)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plt.show()