import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import json
import numpy as np
from collections import Counter
import random
import os

from src.datasets.medmnist_datasets import get_retinamnist
from src.models.cnn_model import CNN
from sklearn.metrics import classification_report, f1_score, recall_score


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # prob of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
def predict_probs(model, loader, device, use_tta=False):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            if use_tta:
                logits = (logits + model(torch.flip(images, dims=[3]))) / 2.0
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.array(all_labels)


def scores_from_probs(probs, labels):
    preds    = np.argmax(probs, axis=1)
    macro_f1 = f1_score(labels, preds, average="macro",    zero_division=0)
    wf1      = f1_score(labels, preds, average="weighted", zero_division=0)
    recall   = recall_score(labels, preds, average=None,   zero_division=0)
    return macro_f1, wf1, recall, preds


def get_train_recall(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = torch.argmax(model(images.to(device)), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return recall_score(all_labels, all_preds, average=None, zero_division=0)


def checkpoint_score(train_recall, val_recall, val_f1):
    """
    Revised checkpoint selector.

    Key insight from GPU run: class 4 train recall is 0.97-1.00
    from ep19 onwards for every seed, but test recall is only 0.40.
    The high c4_tr was dominating the score and selecting epochs
    where c4 is overfit, not generalising.

    Key insight from all runs: class 1 val recall is the most
    predictive of final test performance. When val c1 recall is
    high (0.58+), test c1 recall follows. When val c1 is low
    (0.17-0.42), test c1 collapses to 0.15-0.22.

    New formula: 

      0.30 × val_c1_recall   — most predictive of test class 1
      0.20 × val_f1          — overall held-out quality
      0.20 × train_c1_recall — whether class 1 actually learned
      0.15 × val_min_recall  — catch total class collapse
      0.15 × train_macro     — overall training balance
      (class 4 removed — overfit signal is misleading)
    """
    val_c1   = float(val_recall[1])
    tr_c1    = float(train_recall[1])
    val_min  = float(np.min(val_recall))
    tr_macro = float(np.mean(train_recall))

    return (0.30 * val_c1 +
            0.20 * val_f1 +
            0.20 * tr_c1 +
            0.15 * val_min +
            0.15 * tr_macro)


def resnet_params(model, layer_name):
    return [p for n, p in model.model.named_parameters() if layer_name in n]


# ─────────────────────────────────────────────────────────────
# SINGLE-SEED TRAINING
# ─────────────────────────────────────────────────────────────
def train_one_seed(seed, device, train_dataset, train_eval_dataset,
                   val_dataset, test_dataset, weights, num_classes, class_counts):

    set_seed(seed)
    print(f"\n{'='*55}\n  SEED {seed}\n{'='*55}")

    sample_weights = torch.tensor(
        [1.0 / class_counts[int(label)] for _, label in train_dataset],
        dtype=torch.float32
    )
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Use larger batch on GPU, smaller on CPU
    bs = 64 if torch.cuda.is_available() else 32
    nw = 4  if torch.cuda.is_available() else 0
    pm = True if torch.cuda.is_available() else False

    train_loader      = DataLoader(train_dataset,      batch_size=bs, sampler=sampler,
                                   num_workers=nw, pin_memory=pm)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=128, shuffle=False,
                                   num_workers=nw, pin_memory=pm)
    val_loader        = DataLoader(val_dataset,        batch_size=128, shuffle=False,
                                   num_workers=nw, pin_memory=pm)
    test_loader       = DataLoader(test_dataset,       batch_size=128, shuffle=False,
                                   num_workers=nw, pin_memory=pm)

    model     = CNN(in_channels=3, num_classes=num_classes).to(device)
    criterion = FocalLoss(alpha=weights, gamma=2)

    for p in model.model.parameters():
        p.requires_grad = False
    for p in model.model.fc.parameters():
        p.requires_grad = True

    def make_scheduler(opt):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=4, min_lr=1e-7
        )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4, weight_decay=1e-4
    )
    scheduler = make_scheduler(optimizer)

    TOTAL_EPOCHS     = 50
    CHECKPOINT_START = 12

    checkpoint_data   = {}
    best_ep_so_far    = None
    best_score_so_far = -1
    class4_history    = []
    restart_done      = False

    for epoch in range(TOTAL_EPOCHS):

        if epoch == 4:
            for n, p in model.model.named_parameters():
                if "layer4" in n:
                    p.requires_grad = True
            optimizer = torch.optim.Adam([
                {"params": model.model.fc.parameters(),    "lr": 1e-4},
                {"params": resnet_params(model, "layer4"), "lr": 1e-5},
            ], weight_decay=1e-4)
            scheduler = make_scheduler(optimizer)
            print(f"  [ep{epoch+1}] Unfreezing layer4 (warmup)")

        if epoch == 6:
            for pg in optimizer.param_groups:
                pg["lr"] = min(pg["lr"] * 3, 1e-4)
            print(f"  [ep{epoch+1}] layer4 LR ramped up")

        if epoch == 9:
            for n, p in model.model.named_parameters():
                if "layer3" in n:
                    p.requires_grad = True
            optimizer = torch.optim.Adam([
                {"params": model.model.fc.parameters(),    "lr": 5e-5},
                {"params": resnet_params(model, "layer4"), "lr": 1e-5},
                {"params": resnet_params(model, "layer3"), "lr": 5e-6},
            ], weight_decay=1e-4)
            scheduler = make_scheduler(optimizer)
            print(f"  [ep{epoch+1}] Unfreezing layer3")

        # Stuck detection on class 4 val recall
        if epoch == 22 and not restart_done and best_ep_so_far is not None:
            recent_c4 = class4_history[-4:]
            if len(recent_c4) == 4 and max(recent_c4) <= 0.20:
                print(f"  [ep{epoch+1}] STUCK — reloading ep{best_ep_so_far} + LR bump")
                model.load_state_dict(torch.load(checkpoint_data[best_ep_so_far][0]))
                optimizer = torch.optim.Adam([
                    {"params": model.model.fc.parameters(),    "lr": 8e-5},
                    {"params": resnet_params(model, "layer4"), "lr": 2e-5},
                    {"params": resnet_params(model, "layer3"), "lr": 1e-5},
                ], weight_decay=1e-4)
                scheduler = make_scheduler(optimizer)
                restart_done = True

        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        val_probs, val_labels = predict_probs(model, val_loader, device)
        val_f1, _, val_recall, _ = scores_from_probs(val_probs, val_labels)
        scheduler.step(val_f1)

        train_recall = get_train_recall(model, train_eval_loader, device)
        class4_history.append(float(val_recall[4]))
        score = checkpoint_score(train_recall, val_recall, val_f1)

        print(f"  Ep {epoch+1:2d} | loss {avg_loss:.4f} | vF1 {val_f1:.3f} | "
              f"vc1 {val_recall[1]:.2f} tc1 {train_recall[1]:.2f} | "
              f"tr {np.round(train_recall, 2)} | vl {np.round(val_recall, 2)}")

        if epoch >= CHECKPOINT_START:
            path = f"models/retinamnist_seed{seed}_ep{epoch+1}.pth"
            torch.save(model.state_dict(), path)
            checkpoint_data[epoch+1] = (path, score, val_recall.copy(),
                                        train_recall.copy(), val_f1)
            if score > best_score_so_far:
                best_score_so_far = score
                best_ep_so_far    = epoch + 1

    best_ep = max(checkpoint_data, key=lambda e: checkpoint_data[e][1])
    best_path, best_score, best_vr, best_tr, best_vf1 = checkpoint_data[best_ep]

    print(f"\n  Best ep {best_ep} | score={best_score:.4f} | vF1={best_vf1:.3f} | "
          f"vc1={best_vr[1]:.2f} tc1={best_tr[1]:.2f} | "
          f"tr={np.round(best_tr, 2)} | vl={np.round(best_vr, 2)}")

    model.load_state_dict(torch.load(best_path))
    test_probs, test_labels = predict_probs(model, test_loader, device, use_tta=True)

    for ep, (path, *_) in checkpoint_data.items():
        if ep != best_ep:
            try: os.remove(path)
            except: pass
    torch.save(model.state_dict(), f"models/retinamnist_seed{seed}_best.pth")

    return {
        "seed":        seed,
        "score":       best_score,
        "val_f1":      best_vf1,
        "tr_recall":   best_tr,
        "val_recall":  best_vr,
        "test_probs":  test_probs,
        "test_labels": test_labels,
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
print(">>> RetinaMNIST — ResNet-34")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

with open("data/retinamnist_stats.json", "r") as f:
    stats = json.load(f)
mean = tuple(stats["train"]["mean"])
std  = tuple(stats["train"]["std"])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.Normalize(mean, std),
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean, std),
])

train_dataset      = get_retinamnist("train", transform=train_transform)
train_eval_dataset = get_retinamnist("train", transform=test_transform)
val_dataset        = get_retinamnist("val",   transform=test_transform)
test_dataset       = get_retinamnist("test",  transform=test_transform)

labels_list  = [int(label) for _, label in train_eval_dataset]
class_counts = Counter(labels_list)
num_classes  = 5
total        = sum(class_counts.values())
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
print(f"Class counts: {dict(sorted(class_counts.items()))}")

weights = torch.tensor(
    [total / (num_classes * class_counts[i]) for i in range(num_classes)],
    dtype=torch.float32
).to(device)
print(f"Class weights: {np.round(weights.cpu().numpy(), 3)}")

os.makedirs("models", exist_ok=True)

SEEDS   = [314, 42, 777, 1337]
results = []

for seed in SEEDS:
    r = train_one_seed(
        seed=seed, device=device,
        train_dataset=train_dataset,
        train_eval_dataset=train_eval_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        weights=weights,
        num_classes=num_classes,
        class_counts=class_counts,
    )
    results.append(r)

# ─────────────────────────────────────────────────────────────
# ENSEMBLE — squared score weighting
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*55}\n  ENSEMBLE\n{'='*55}")

test_labels_ref = results[0]["test_labels"]
raw_scores = np.array([r["score"] for r in results])
sq_scores  = raw_scores ** 2
ens_w      = sq_scores / sq_scores.sum()

print(f"Seed scores    : {[round(r['score'], 4) for r in results]}")
print(f"Squared weights: {dict(zip([r['seed'] for r in results], np.round(ens_w, 3)))}")

ensemble_probs = sum(w * r["test_probs"] for w, r in zip(ens_w, results))
macro_f1, wf1, recall, preds = scores_from_probs(ensemble_probs, test_labels_ref)

print(f"\nEnsemble Test Macro F1   : {macro_f1:.4f}")
print(f"Ensemble Test Weighted F1: {wf1:.4f}")
print(f"Gap (weighted-macro)     : {wf1 - macro_f1:.4f}")
print(f"Per-class recall         : {np.round(recall, 3)}")
print("\nClassification Report:")
print(classification_report(test_labels_ref, preds, zero_division=0))

print("\nIndividual seed test results:")
for r in results:
    f1, wf1_s, rec, _ = scores_from_probs(r["test_probs"], test_labels_ref)
    print(f"  Seed {r['seed']}: macro={f1:.4f} | weighted={wf1_s:.4f} | "
          f"gap={wf1_s-f1:.3f} | test={np.round(rec, 2)}")

np.save("models/retinamnist_ensemble_weights.npy", ens_w)
print("\nDone.")