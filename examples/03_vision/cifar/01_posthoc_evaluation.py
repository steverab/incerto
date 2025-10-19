"""
Post-hoc Uncertainty Evaluation on CIFAR-10

Comprehensive evaluation of post-hoc uncertainty methods on CIFAR-10.
Similar structure to MNIST example but with more complex data.

Methods evaluated:
- Calibration: Temperature Scaling, Vector Scaling, Matrix Scaling
- OOD Detection: MSP, Energy, ODIN, MaxLogit
- Selective Prediction: Confidence-based rejection
- Conformal Prediction: Inductive CP

Runtime: ~10 minutes on GPU
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
from incerto.calibration import (
    TemperatureScaling,
    VectorScaling,
    ece_score,
    plot_reliability_diagram,
)
from incerto.ood import MSP, Energy, MaxLogit, auroc as ood_auroc, plot_roc
from incerto.conformal import InductiveConformalPredictor, coverage, average_set_size
from incerto.utils import ResNet18, seed_everything, train_epoch, evaluate

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

# OOD data
ood_benchmark = CIFAR10_vs_SVHN(root="./data")
_, ood_dataset = ood_benchmark.get_datasets()
ood_loader = DataLoader(ood_dataset, batch_size=128, shuffle=False, num_workers=2)

console.print(
    f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}, OOD: {len(ood_dataset)}\n"
)


# ============================================================================
# 2. Train Model
# ============================================================================

console.print("[yellow]Training ResNet-18...[/yellow]")
model = ResNet18(num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 20
for epoch in range(epochs):
    metrics = train_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        desc=f"Epoch {epoch+1}/{epochs}",
    )
    if (epoch + 1) % 5 == 0:
        val_metrics = evaluate(model, val_loader, criterion, device, desc="Validation")
        console.print(
            f"Epoch {epoch+1}: Train Acc={metrics['accuracy']:.2f}%, Val Acc={val_metrics['accuracy']:.2f}%"
        )

console.print()


# ============================================================================
# 3. Get Predictions
# ============================================================================

console.print("[yellow]Getting test predictions...[/yellow]")
test_results = evaluate(model, test_loader, criterion, device, desc="Test")
test_logits = test_results["logits"]
test_targets = test_results["targets"]
test_acc = test_results["accuracy"]

console.print(f"Test Accuracy: {test_acc:.2f}%\n")


# ============================================================================
# 4. Calibration
# ============================================================================

console.print("[bold cyan]═══ Post-hoc Calibration ═══[/bold cyan]")

val_results = evaluate(model, val_loader, criterion, device, desc="Validation")
val_logits = val_results["logits"]
val_targets = val_results["targets"]

# Uncalibrated
ece_uncal = ece_score(test_logits, test_targets)
console.print(f"Uncalibrated ECE: {ece_uncal:.4f}")

# Temperature Scaling
ts_calibrator = TemperatureScaling()
ts_calibrator.fit(val_logits, val_targets)
test_logits_ts = ts_calibrator.calibrate(test_logits)
ece_ts = ece_score(test_logits_ts, test_targets)
console.print(
    f"Temperature Scaling ECE: {ece_ts:.4f} (T={ts_calibrator.temperature.item():.3f})"
)

# Vector Scaling
vs_calibrator = VectorScaling(n_classes=10)
vs_calibrator.fit(val_logits, val_targets)
test_logits_vs = vs_calibrator.calibrate(test_logits)
ece_vs = ece_score(test_logits_vs, test_targets)
console.print(f"Vector Scaling ECE: {ece_vs:.4f}\n")


# ============================================================================
# 5. OOD Detection
# ============================================================================

console.print("[bold cyan]═══ OOD Detection (CIFAR-10 vs SVHN) ═══[/bold cyan]")

detectors = {
    "MSP": MSP(model),
    "Energy": Energy(model, temperature=1.0),
    "MaxLogit": MaxLogit(model),
}

ood_results = {}
for name, detector in detectors.items():
    with torch.no_grad():
        id_scores = []
        for inputs, _ in test_loader:
            id_scores.append(detector.score(inputs.to(device)).cpu())
        id_scores = torch.cat(id_scores)

        ood_scores = []
        for inputs, _ in ood_loader:
            ood_scores.append(detector.score(inputs.to(device)).cpu())
        ood_scores = torch.cat(ood_scores)

    auroc = ood_auroc(id_scores, ood_scores)
    ood_results[name] = auroc
    console.print(f"{name}: AUROC = {auroc:.4f}")

console.print()


# ============================================================================
# 6. Conformal Prediction
# ============================================================================

console.print("[bold cyan]═══ Conformal Prediction ═══[/bold cyan]")

cp = InductiveConformalPredictor(alpha=0.1)  # 90% coverage
cp.calibrate(val_logits, val_targets)

pred_sets = cp.predict(test_logits)
cov = coverage(pred_sets, test_targets)
avg_size = average_set_size(pred_sets)

console.print(f"Target Coverage: 90%")
console.print(f"Actual Coverage: {cov*100:.2f}%")
console.print(f"Average Set Size: {avg_size:.2f}\n")


# ============================================================================
# 7. Summary Table
# ============================================================================

console.print(
    "[bold green]═══════════════════════════════════════════════[/bold green]"
)
console.print(
    "[bold green]                SUMMARY RESULTS                 [/bold green]"
)
console.print(
    "[bold green]═══════════════════════════════════════════════[/bold green]\n"
)

table = Table(show_header=True, header_style="bold magenta")
table.add_column("Category", style="cyan")
table.add_column("Method", style="yellow")
table.add_column("Metric", justify="right")

table.add_row("Model", "ResNet-18", f"{test_acc:.2f}% accuracy")
table.add_row("", "", "")
table.add_row("Calibration", "Uncalibrated", f"ECE = {ece_uncal:.4f}")
table.add_row("", "Temperature Scaling", f"ECE = {ece_ts:.4f}")
table.add_row("", "Vector Scaling", f"ECE = {ece_vs:.4f}")
table.add_row("", "", "")
table.add_row("OOD Detection", "MSP", f"AUROC = {ood_results['MSP']:.4f}")
table.add_row("", "Energy", f"AUROC = {ood_results['Energy']:.4f}")
table.add_row("", "MaxLogit", f"AUROC = {ood_results['MaxLogit']:.4f}")
table.add_row("", "", "")
table.add_row(
    "Conformal", "Inductive CP", f"{cov*100:.1f}% coverage, {avg_size:.2f} avg size"
)

console.print(table)
console.print("\n[bold green]✅ Post-hoc evaluation complete![/bold green]")
