import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.datasets.medmnist_datasets import get_retinamnist
from src.models.cnn_model import CNN

from sklearn.metrics import classification_report, f1_score,recall_score
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none', label_smoothing=0.05)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)

        if torch.rand(1).item() < 0.01:
            print("pt mean:", pt.mean().item())

        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()/2

print(">>> RetinaMNIST training started")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import json

with open("data/retinamnist_stats.json", "r") as f:
    stats = json.load(f)

mean = tuple(stats["train"]["mean"])
std = tuple(stats["train"]["std"])

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1
    ),
    transforms.Normalize(mean,std)
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Normalize(mean,std)
])

train_dataset = get_retinamnist("train", transform=train_transform)
val_dataset = get_retinamnist("val", transform=test_transform)
test_dataset = get_retinamnist("test", transform=test_transform)

from collections import Counter
import numpy as np

from collections import Counter

labels = []
for _, label in train_dataset:
    labels.append(int(label))

class_counts = Counter(labels)
total = sum(class_counts.values())

print("Class counts:", class_counts)

num_classes = 5

from torch.utils.data import WeightedRandomSampler

# Create sample weights
sample_weights = [1.0 / class_counts[label] for label in labels]
sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler
)

val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

model = CNN(in_channels=3, num_classes=5).to(device)

for param in model.model.parameters():
    param.requires_grad = False

for param in model.model.fc.parameters():
    param.requires_grad = True

criterion = FocalLoss(gamma=2, weight=None)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4,
    weight_decay=1e-4
)

epochs = 50
best_f1 = 0
patience = 15
no_improve = 0

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs1 = model(images)
            outputs2 = model(torch.flip(images, dims=[3]))
            outputs = (outputs1 + outputs2) / 2
            probs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probs, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average=None)
    return f1, weighted_f1, recall

for epoch in range(epochs):

    model.train()
    total_loss = 0

    if epoch == 8:
        print("Unfreezing backbone...")
        for name, param in model.model.named_parameters():
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True

        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4


    if epoch == 15:
        print("Unfreezing entire backbone...")
        for param in model.model.parameters():
            param.requires_grad = True

        for param_group in optimizer.param_groups:
            param_group['lr'] = 1.5e-5

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs1 = model(images)
        outputs2 = model(torch.flip(images, dims=[3]))
        outputs = (outputs1 + outputs2) / 2
        focal_loss = criterion(outputs, labels)

        loss = focal_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    val_f1, val_weighted_f1, recall = evaluate(model, val_loader)
    print("Per-class recall:", recall)
    scheduler.step()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Macro F1: {val_f1:.4f} | Weighted F1: {val_weighted_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        no_improve = 0
        torch.save(model.state_dict(), "models/retinamnist_resnet.pth")
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break    

print("\nTraining finished.")
print("Best validation F1:", best_f1)

model.load_state_dict(torch.load("models/retinamnist_resnet.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probs, dim=1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, zero_division=0))

f1 = f1_score(all_labels, all_preds, average='macro')
print("Macro F1:", f1)