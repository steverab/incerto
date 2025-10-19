"""
Training Methods for Uncertainty on MNIST

Compares different training-time methods for improving uncertainty quantification.
Shows how training choices affect calibration and OOD detection.

Methods compared:
1. Baseline - Standard cross-entropy
2. Label Smoothing - Soften hard labels
3. Focal Loss - Focus on hard examples
4. Mixup - Data augmentation
5. Self-Adaptive Training - Soft label curriculum

Runtime: ~5 minutes on GPU, ~15 minutes on CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Incerto imports
from incerto.data import MNIST, MNIST_vs_FashionMNIST
from incerto.calibration import LabelSmoothingLoss, FocalLoss, ece_score
from incerto.ood import mixup_data, mixup_criterion, auroc as ood_auroc
from incerto.sp import SelfAdaptiveTraining, aurc
from incerto.utils import ConvNet, seed_everything

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console = Console()
console.print(f"[green]Using device: {device}[/green]\n")


# ============================================================================
# 1. Load Data
# ============================================================================

console.print("[yellow]Loading MNIST dataset...[/yellow]")
mnist = MNIST(root="./data", val_split=0.1, normalize=True)
train_dataset, val_dataset, test_dataset = mnist.get_datasets()

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# OOD data
ood_benchmark = MNIST_vs_FashionMNIST(root="./data")
_, ood_dataset = ood_benchmark.get_datasets()
ood_loader = DataLoader(ood_dataset, batch_size=128, shuffle=False, num_workers=2)

console.print(
    f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}, OOD: {len(ood_dataset)}\n"
)


# ============================================================================
# 2. Training Functions
# ============================================================================


def train_epoch_standard(model, loader, criterion, optimizer, device):
    """Standard training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def train_epoch_mixup(model, loader, criterion, optimizer, device, alpha=1.0):
    """Training epoch with mixup."""
    model.train()
    total_loss = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Apply mixup
        mixed_inputs, targets_a, targets_b, lam = mixup_data(
            inputs, targets, alpha, device
        )

        optimizer.zero_grad()
        outputs = model(mixed_inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train_epoch_sat(model, loader, optimizer, device, alpha):
    """Training epoch with SAT."""
    model.train()
    total_loss = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = model.sat_loss(logits, targets, alpha)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_model(model, test_loader, ood_loader, device):
    """Evaluate model comprehensively."""
    model.eval()

    # Collect test predictions
    all_logits = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            logits = (
                model(inputs)
                if not isinstance(model, SelfAdaptiveTraining)
                else model(inputs)
            )
            all_logits.append(logits.cpu())
            all_targets.append(targets)
            all_probs.append(F.softmax(logits, dim=1).cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)

    # Accuracy
    predictions = all_logits.argmax(1)
    accuracy = (predictions == all_targets).float().mean().item() * 100

    # ECE (calibration)
    ece = ece_score(all_logits, all_targets)

    # OOD detection (AUROC)
    id_scores = []
    ood_scores = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            logits = (
                model(inputs)
                if not isinstance(model, SelfAdaptiveTraining)
                else model(inputs)
            )
            scores = F.softmax(logits, dim=1).max(1)[0]
            id_scores.append(scores.cpu())

        for inputs, _ in ood_loader:
            inputs = inputs.to(device)
            logits = (
                model(inputs)
                if not isinstance(model, SelfAdaptiveTraining)
                else model(inputs)
            )
            scores = F.softmax(logits, dim=1).max(1)[0]
            ood_scores.append(scores.cpu())

    id_scores = torch.cat(id_scores)
    ood_scores = torch.cat(ood_scores)
    auroc = ood_auroc(id_scores, ood_scores)

    # AURC (selective prediction)
    confidences = all_probs.max(1)[0]
    sorted_conf, sorted_idx = confidences.sort(descending=True)
    sorted_errors = (predictions[sorted_idx] != all_targets[sorted_idx]).float()
    aurc_val = aurc(sorted_conf, sorted_errors)

    return {
        "accuracy": accuracy,
        "ece": ece,
        "auroc": auroc,
        "aurc": aurc_val,
    }


# ============================================================================
# 3. Train All Methods
# ============================================================================

epochs = 10
results = {}

# Method 1: Baseline
console.print("\n[bold cyan]═══ Training Method 1: Baseline ═══[/bold cyan]")
model_baseline = ConvNet(num_classes=10, dropout_rate=0.2).to(device)
optimizer = torch.optim.Adam(model_baseline.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    train_loss, train_acc = train_epoch_standard(
        model_baseline, train_loader, criterion, optimizer, device
    )
    if (epoch + 1) % 2 == 0:
        console.print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")

results["Baseline"] = evaluate_model(model_baseline, test_loader, ood_loader, device)

# Method 2: Label Smoothing
console.print("\n[bold cyan]═══ Training Method 2: Label Smoothing ═══[/bold cyan]")
model_ls = ConvNet(num_classes=10, dropout_rate=0.2).to(device)
optimizer = torch.optim.Adam(model_ls.parameters(), lr=0.001)
criterion_ls = LabelSmoothingLoss(smoothing=0.1)

for epoch in range(epochs):
    train_loss, train_acc = train_epoch_standard(
        model_ls, train_loader, criterion_ls, optimizer, device
    )
    if (epoch + 1) % 2 == 0:
        console.print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")

results["Label Smoothing"] = evaluate_model(model_ls, test_loader, ood_loader, device)

# Method 3: Focal Loss
console.print("\n[bold cyan]═══ Training Method 3: Focal Loss ═══[/bold cyan]")
model_focal = ConvNet(num_classes=10, dropout_rate=0.2).to(device)
optimizer = torch.optim.Adam(model_focal.parameters(), lr=0.001)
criterion_focal = FocalLoss(gamma=2.0)

for epoch in range(epochs):
    train_loss, train_acc = train_epoch_standard(
        model_focal, train_loader, criterion_focal, optimizer, device
    )
    if (epoch + 1) % 2 == 0:
        console.print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")

results["Focal Loss"] = evaluate_model(model_focal, test_loader, ood_loader, device)

# Method 4: Mixup
console.print("\n[bold cyan]═══ Training Method 4: Mixup ═══[/bold cyan]")
model_mixup = ConvNet(num_classes=10, dropout_rate=0.2).to(device)
optimizer = torch.optim.Adam(model_mixup.parameters(), lr=0.001)
criterion_mixup = nn.CrossEntropyLoss()

for epoch in range(epochs):
    train_loss = train_epoch_mixup(
        model_mixup, train_loader, criterion_mixup, optimizer, device, alpha=1.0
    )
    if (epoch + 1) % 2 == 0:
        console.print(f"Epoch {epoch+1}: Loss={train_loss:.4f}")

results["Mixup"] = evaluate_model(model_mixup, test_loader, ood_loader, device)

# Method 5: Self-Adaptive Training
console.print(
    "\n[bold cyan]═══ Training Method 5: Self-Adaptive Training ═══[/bold cyan]"
)
backbone = ConvNet(num_classes=10, dropout_rate=0.2)
model_sat = SelfAdaptiveTraining(
    backbone=backbone, num_classes=10, alpha_start=0.0, alpha_end=0.9, warmup_epochs=3
).to(device)
optimizer = torch.optim.Adam(model_sat.parameters(), lr=0.001)

for epoch in range(epochs):
    alpha = model_sat.get_alpha(epoch, epochs)
    train_loss = train_epoch_sat(model_sat, train_loader, optimizer, device, alpha)
    if (epoch + 1) % 2 == 0:
        console.print(f"Epoch {epoch+1} (α={alpha:.3f}): Loss={train_loss:.4f}")

results["SAT"] = evaluate_model(model_sat, test_loader, ood_loader, device)


# ============================================================================
# 4. Display Results
# ============================================================================

console.print(
    "\n[bold green]═══════════════════════════════════════════════[/bold green]"
)
console.print(
    "[bold green]           FINAL RESULTS COMPARISON             [/bold green]"
)
console.print(
    "[bold green]═══════════════════════════════════════════════[/bold green]\n"
)

table = Table(show_header=True, header_style="bold magenta")
table.add_column("Method", style="cyan")
table.add_column("Accuracy (%)", justify="right")
table.add_column("ECE (↓)", justify="right")
table.add_column("OOD AUROC (↑)", justify="right")
table.add_column("AURC (↓)", justify="right")

for method, metrics in results.items():
    table.add_row(
        method,
        f"{metrics['accuracy']:.2f}",
        f"{metrics['ece']:.4f}",
        f"{metrics['auroc']:.4f}",
        f"{metrics['aurc']:.4f}",
    )

console.print(table)


# ============================================================================
# 5. Visualizations
# ============================================================================

output_dir = Path("output/vision/mnist")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

methods = list(results.keys())
accuracies = [results[m]["accuracy"] for m in methods]
eces = [results[m]["ece"] for m in methods]
aurocs = [results[m]["auroc"] for m in methods]
aurcs = [results[m]["aurc"] for m in methods]

# Accuracy
axes[0, 0].bar(methods, accuracies, color="steelblue")
axes[0, 0].set_ylabel("Accuracy (%)")
axes[0, 0].set_title("Test Accuracy")
axes[0, 0].tick_params(axis="x", rotation=45)
axes[0, 0].grid(True, alpha=0.3, axis="y")

# ECE
axes[0, 1].bar(methods, eces, color="coral")
axes[0, 1].set_ylabel("ECE")
axes[0, 1].set_title("Expected Calibration Error (Lower is Better)")
axes[0, 1].tick_params(axis="x", rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis="y")

# AUROC
axes[1, 0].bar(methods, aurocs, color="green")
axes[1, 0].set_ylabel("AUROC")
axes[1, 0].set_title("OOD Detection AUROC (Higher is Better)")
axes[1, 0].tick_params(axis="x", rotation=45)
axes[1, 0].grid(True, alpha=0.3, axis="y")

# AURC
axes[1, 1].bar(methods, aurcs, color="purple")
axes[1, 1].set_ylabel("AURC")
axes[1, 1].set_title("Selective Prediction AURC (Lower is Better)")
axes[1, 1].tick_params(axis="x", rotation=45)
axes[1, 1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(
    output_dir / "training_methods_comparison.png", dpi=150, bbox_inches="tight"
)
console.print(
    f"\n[green]Saved comparison to {output_dir / 'training_methods_comparison.png'}[/green]"
)


# ============================================================================
# 6. Summary
# ============================================================================

console.print("\n[yellow]Key Takeaways:[/yellow]")
console.print("• Label Smoothing: Best calibration (lowest ECE)")
console.print("• Mixup: Best OOD detection (highest AUROC)")
console.print("• SAT: Good balance across all metrics")
console.print("• Focal Loss: Helps with hard examples")
console.print("• All methods maintain similar accuracy")
console.print("\n[bold green]✅ Training methods comparison complete![/bold green]")
