# Mini Project: Rice Image Classification — CNN vs Vision Transformer

**Course:** AI and Advanced Machine Learning  
**Goal:** Compare 3 architectures (Exercise CNN, ResNet50, ViT-B/16) with and without pretraining on a rice image dataset.

---

## Team Split

The work is divided into two independent parts that merge at the evaluation step.

| Person | Branch | Responsible for |
|--------|--------|-----------------|
| Person 1 | `cnn` | Conditions E (Exercise CNN) and A+B (ResNet50 scratch + fine-tuned) |
| Person 2 | `vit` | Conditions C+D (ViT-B/16 scratch + fine-tuned) |

Both people use the **same** `data_prep.py`, `train.py`, and `evaluate.py` — only `models.py` differs per branch.  
At the end, both merge into `main` and run `plot_results.py` together to compare all 5 conditions.

### Git workflow

```bash
# Person 1
git checkout -b cnn

# Person 2
git checkout -b vit

# When done — both push their branch, then merge into main together
git push origin cnn   # or vit
```

---

## What We Are Doing

We train 5 models to classify rice grain images into 5 types:
- Arborio, Basmati, Ipsala, Jasmine, Karacadag

| Condition | Model | Training | Covers |
|-----------|-------|----------|--------|
| E | Exercise CNN | from scratch | Module 2 (what we built in class) |
| A | ResNet50 | from scratch | Deep CNN |
| B | ResNet50 | fine-tuned (pretrained) | Transfer learning |
| C | ViT-B/16 | from scratch | Transformer / Attention |
| D | ViT-B/16 | fine-tuned (pretrained) | Transfer learning + Attention |

**The hypothesis:** ViT needs pretraining much more than CNN because it has no spatial inductive bias.

---

## Step 0: What You Need

### On HPOC cluster (or any machine with NVIDIA GPU):

```bash
pip install torch torchvision matplotlib scikit-learn seaborn
```

### Download the dataset

1. Go to https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset
2. Click **Download** and save `archive.zip`
3. Upload to HPOC and run:

```bash
unzip archive.zip -d data/rice
```

You should now have:
```
data/rice/Rice_Image_Dataset/
    Arborio/      (~15,000 images)
    Basmati/      (~15,000 images)
    Ipsala/       (~15,000 images)
    Jasmine/      (~15,000 images)
    Karacadag/    (~15,000 images)
```

Check it worked:
```bash
find data/rice -name "*.jpg" | wc -l   # should print ~75000
```

---

## Step 1: Prepare the Data (`data_prep.py`)

This file handles loading images, resizing them to 224×224, and splitting into train/val/test.

```python
# data_prep.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

DATA_DIR = "data/rice/Rice_Image_Dataset"

# Image transforms — resize + normalize to ImageNet mean/std
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),       # random crop for augmentation
    transforms.RandomHorizontalFlip(),        # flip randomly
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def get_loaders(batch_size=64, seed=42):
    full_dataset = datasets.ImageFolder(DATA_DIR)
    n = len(full_dataset)
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    # Apply transforms
    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform   = val_tf
    test_ds.dataset.transform  = val_tf

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

# Class names: ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
CLASS_NAMES = datasets.ImageFolder(DATA_DIR).classes
```

---

## Step 2: Define the Models (`models.py`)

Three architectures — same interface so training code works for all of them.

```python
# models.py
import torch.nn as nn
import torchvision.models as tv_models

NUM_CLASSES = 5

# ── Condition E: Exercise CNN (from Module 2, adapted for 224×224 color images) ──
class ExerciseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 channels (RGB)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # 224 → 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # 112 → 56
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Conditions A & B: ResNet50 ──
def get_resnet50(pretrained: bool):
    weights = "IMAGENET1K_V2" if pretrained else None
    model = tv_models.resnet50(weights=weights)
    model.fc = nn.Linear(2048, NUM_CLASSES)   # replace final layer
    return model


# ── Conditions C & D: ViT-B/16 ──
def get_vit(pretrained: bool):
    weights = "IMAGENET1K_V1" if pretrained else None
    model = tv_models.vit_b_16(weights=weights)
    model.heads.head = nn.Linear(768, NUM_CLASSES)   # replace final layer
    return model


# Factory function — pick model by condition letter
def get_model(condition: str):
    return {
        "E": ExerciseCNN(),
        "A": get_resnet50(pretrained=False),
        "B": get_resnet50(pretrained=True),
        "C": get_vit(pretrained=False),
        "D": get_vit(pretrained=True),
    }[condition]
```

---

## Step 3: Training Loop (`train.py`)

One script that trains any condition. Call it with `--condition A`, `--condition D`, etc.

```python
# train.py
import argparse, os, json
import torch
import torch.nn as nn
from data_prep import get_loaders
from models import get_model

def train(condition, epochs=20, lr=1e-4, batch_size=64, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training condition {condition} on {device}")

    train_loader, val_loader, _ = get_loaders(batch_size=batch_size, seed=seed)
    model = get_model(condition).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # ── Training ──
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # ── Validation ──
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    # Save model and history
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), f"results/model_{condition}_seed{seed}.pt")
    with open(f"results/history_{condition}_seed{seed}.json", "w") as f:
        json.dump(history, f)

    print(f"Saved to results/model_{condition}_seed{seed}.pt")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["A","B","C","D","E"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.condition, args.epochs, args.lr, seed=args.seed)
```

---

## Step 4: Evaluation (`evaluate.py`)

Loads a saved model and prints test accuracy + confusion matrix.

```python
# evaluate.py
import argparse, json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from data_prep import get_loaders, CLASS_NAMES
from models import get_model

def evaluate(condition, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_loaders(seed=seed)

    model = get_model(condition).to(device)
    model.load_state_dict(torch.load(f"results/model_{condition}_seed{seed}.pt"))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = model(imgs.to(device)).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Condition {condition} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"results/confusion_{condition}.png")
    print(f"Saved confusion matrix to results/confusion_{condition}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["A","B","C","D","E"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate(args.condition, args.seed)
```

---

## Step 5: Plot Learning Curves (`plot_results.py`)

```python
# plot_results.py
import json, glob
import matplotlib.pyplot as plt

CONDITIONS = ["E", "A", "B", "C", "D"]
LABELS = {
    "E": "Exercise CNN (scratch)",
    "A": "ResNet50 (scratch)",
    "B": "ResNet50 (fine-tuned)",
    "C": "ViT-B/16 (scratch)",
    "D": "ViT-B/16 (fine-tuned)",
}

plt.figure(figsize=(10, 5))
for cond in CONDITIONS:
    files = glob.glob(f"results/history_{cond}_seed*.json")
    if not files:
        continue
    # Average over seeds
    all_accs = []
    for f in files:
        with open(f) as fp:
            all_accs.append(json.load(fp)["val_acc"])
    avg = [sum(x)/len(x) for x in zip(*all_accs)]
    plt.plot(avg, label=LABELS[cond])

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("All Conditions — Learning Curves")
plt.legend()
plt.tight_layout()
plt.savefig("results/learning_curves.png")
print("Saved results/learning_curves.png")
```

---

## Step 6: Run Everything on HPOC

Use 3 seeds per condition for statistical confidence. Each person runs their own conditions independently.

### Person 1 — CNN branch

```bash
git checkout cnn

# Condition E — Exercise CNN (~5 min on L40)
python train.py --condition E --seed 42
python train.py --condition E --seed 0
python train.py --condition E --seed 1

# Condition A — ResNet50 from scratch (~20 min)
python train.py --condition A --seed 42
python train.py --condition A --seed 0
python train.py --condition A --seed 1

# Condition B — ResNet50 fine-tuned (~15 min)
python train.py --condition B --seed 42
python train.py --condition B --seed 0
python train.py --condition B --seed 1

# Evaluate CNN conditions
python evaluate.py --condition E
python evaluate.py --condition A
python evaluate.py --condition B

git add results/
git commit -m "CNN results: conditions E, A, B"
git push origin cnn
```

### Person 2 — ViT branch

```bash
git checkout vit

# Condition C — ViT from scratch (~30 min)
python train.py --condition C --seed 42
python train.py --condition C --seed 0
python train.py --condition C --seed 1

# Condition D — ViT fine-tuned (~20 min)
python train.py --condition D --seed 42
python train.py --condition D --seed 0
python train.py --condition D --seed 1

# Evaluate ViT conditions
python evaluate.py --condition C
python evaluate.py --condition D

git add results/
git commit -m "ViT results: conditions C, D"
git push origin vit
```

### Merge and plot together (both people)

```bash
git checkout main
git merge cnn
git merge vit

# Now results/ has all 5 conditions — plot everything
python plot_results.py
```

---

## Step 7: What to Put in the Report

### Expected results table (fill in your numbers)

| Condition | Model | Test Accuracy |
|-----------|-------|--------------|
| E | Exercise CNN (scratch) | ~70–80% |
| A | ResNet50 (scratch) | ~85–90% |
| B | ResNet50 (fine-tuned) | ~97–99% |
| C | ViT-B/16 (scratch) | ~60–75% |
| D | ViT-B/16 (fine-tuned) | ~97–99% |

### Key things to discuss

1. **Why does ViT (scratch) perform worst?**  
   ViT has no spatial inductive bias — it doesn't assume nearby pixels are related. CNN assumes locality, which happens to be true for images. ViT needs to learn this from data, requiring much more data or pretraining.

2. **Why does fine-tuning close the gap?**  
   Both B and D start from ImageNet weights — they already know edges, textures, and shapes. Fine-tuning adapts this general knowledge to rice grain specifics in a few epochs.

3. **What does the confusion matrix show?**  
   Some rice types look visually similar (e.g. Jasmine vs Basmati — both long grain). The Exercise CNN likely confuses these most, while fine-tuned models distinguish them better.

---

## File Structure Summary

```
ai-project-cnn-vs-vit/
    README.md               ← this file (main branch)
    data_prep.py            ← shared: dataset loading (both people use this)
    train.py                ← shared: training loop (both people use this)
    evaluate.py             ← shared: test accuracy + confusion matrix
    plot_results.py         ← shared: final comparison plots (run after merge)
    models.py               ← each branch has this; Person 1 defines E/A/B, Person 2 defines C/D
    results/                ← generated automatically; merged from both branches
        model_E_seed42.pt
        model_A_seed42.pt   ← from cnn branch
        model_B_seed42.pt
        model_C_seed42.pt   ← from vit branch
        model_D_seed42.pt
        learning_curves.png ← generated after merge
```

### Branch ownership

| File | Who writes it |
|------|--------------|
| `data_prep.py` | Person 1 (set up once, shared) |
| `train.py` | Person 1 (set up once, shared) |
| `evaluate.py` | Person 1 (set up once, shared) |
| `models.py` (ExerciseCNN, ResNet50) | Person 1 on `cnn` branch |
| `models.py` (ViT-B/16) | Person 2 on `vit` branch |
| `plot_results.py` | Either person, run after merge |
