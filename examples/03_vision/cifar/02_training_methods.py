"""
Training Methods on CIFAR-10

Compares training-time methods for uncertainty on CIFAR-10.
Uses ResNet-18 architecture.

Methods:
1. Baseline - Standard cross-entropy
2. Label Smoothing - Better calibration
3. Mixup - Better robustness
4. Focal Loss - Handle hard examples

Runtime: ~30 minutes on GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Incerto imports
from incerto.data import CIFAR10, CIFAR10_vs_SVHN
from incerto.calibration import LabelSmoothingLoss, FocalLoss, ece_score
from incerto.ood import mixup_data, mixup_criterion, auroc as ood_auroc
from incerto.utils import ResNet18, seed_everything

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console = Console()
console.print(f"[green]Using device: {device}[/green]\n")


# ============================================================================
# 1. Load Data
# ============================================================================

console.print("[yellow]Loading CIFAR-10...[/yellow]")
cifar = CIFAR10(root="./data", val_split=0.1, normalize=True, augmentation=True)
train_dataset, val_dataset, test_dataset = cifar.get_datasets()

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

ood_benchmark = CIFAR10_vs_SVHN(root="./data")
_, ood_dataset = ood_benchmark.get_datasets()
ood_loader = DataLoader(ood_dataset, batch_size=128, shuffle=False, num_workers=2)

console.print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}\n")


# ============================================================================
# 2. Training Functions
# ============================================================================


def train_standard(model, train_loader, criterion, optimizer, epochs=20):
    """Standard training."""
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            console.print(f"  Epoch {epoch+1}: Val Acc = {100.*correct/total:.2f}%")


def train_mixup(model, train_loader, optimizer, epochs=20, alpha=1.0):
    """Training with mixup."""
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            mixed_inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, alpha, device
            )
            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            console.print(f"  Epoch {epoch+1}: Training with mixup")


def evaluate_method(model, test_loader, ood_loader):
    """Evaluate accuracy, ECE, and OOD AUROC."""
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu())
            all_targets.append(targets)

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    # Accuracy
    preds = all_logits.argmax(1)
    acc = (preds == all_targets).float().mean() * 100

    # ECE
    ece = ece_score(all_logits, all_targets)

    # OOD AUROC
    id_scores = []
    ood_scores = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            scores = F.softmax(logits, dim=1).max(1)[0]
            id_scores.append(scores.cpu())

        for inputs, _ in ood_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            scores = F.softmax(logits, dim=1).max(1)[0]
            ood_scores.append(scores.cpu())

    id_scores = torch.cat(id_scores)
    ood_scores = torch.cat(ood_scores)
    auroc = ood_auroc(id_scores, ood_scores)

    return {
        "accuracy": acc.item(),
        "ece": ece,
        "auroc": auroc,
    }


# ============================================================================
# 3. Train All Methods
# ============================================================================

epochs = 20
results = {}

# Method 1: Baseline
console.print("\n[bold cyan]═══ Training Method 1: Baseline ═══[/bold cyan]")
model_baseline = ResNet18(num_classes=10).to(device)
optimizer = torch.optim.Adam(model_baseline.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_standard(model_baseline, train_loader, criterion, optimizer, epochs)
results["Baseline"] = evaluate_method(model_baseline, test_loader, ood_loader)

# Method 2: Label Smoothing
console.print("\n[bold cyan]═══ Training Method 2: Label Smoothing ═══[/bold cyan]")
model_ls = ResNet18(num_classes=10).to(device)
optimizer = torch.optim.Adam(model_ls.parameters(), lr=0.001)
criterion_ls = LabelSmoothingLoss(smoothing=0.1)
train_standard(model_ls, train_loader, criterion_ls, optimizer, epochs)
results["Label Smoothing"] = evaluate_method(model_ls, test_loader, ood_loader)

# Method 3: Focal Loss
console.print("\n[bold cyan]═══ Training Method 3: Focal Loss ═══[/bold cyan]")
model_focal = ResNet18(num_classes=10).to(device)
optimizer = torch.optim.Adam(model_focal.parameters(), lr=0.001)
criterion_focal = FocalLoss(gamma=2.0)
train_standard(model_focal, train_loader, criterion_focal, optimizer, epochs)
results["Focal Loss"] = evaluate_method(model_focal, test_loader, ood_loader)

# Method 4: Mixup
console.print("\n[bold cyan]═══ Training Method 4: Mixup ═══[/bold cyan]")
model_mixup = ResNet18(num_classes=10).to(device)
optimizer = torch.optim.Adam(model_mixup.parameters(), lr=0.001)
train_mixup(model_mixup, train_loader, optimizer, epochs, alpha=1.0)
results["Mixup"] = evaluate_method(model_mixup, test_loader, ood_loader)


# ============================================================================
# 4. Display Results
# ============================================================================

console.print(
    "\n[bold green]═══════════════════════════════════════════════[/bold green]"
)
console.print(
    "[bold green]           TRAINING METHODS COMPARISON          [/bold green]"
)
console.print(
    "[bold green]═══════════════════════════════════════════════[/bold green]\n"
)

table = Table(show_header=True, header_style="bold magenta")
table.add_column("Method", style="cyan")
table.add_column("Accuracy (%)", justify="right")
table.add_column("ECE (↓)", justify="right")
table.add_column("OOD AUROC (↑)", justify="right")

for method, metrics in results.items():
    table.add_row(
        method,
        f"{metrics['accuracy']:.2f}",
        f"{metrics['ece']:.4f}",
        f"{metrics['auroc']:.4f}",
    )

console.print(table)

console.print("\n[yellow]Key Takeaways:[/yellow]")
console.print("• Label Smoothing: Best calibration")
console.print("• Mixup: Best OOD detection")
console.print("• Focal Loss: Handles difficult examples")
console.print("• All maintain competitive accuracy")
console.print("\n[bold green]✅ Training methods comparison complete![/bold green]")
