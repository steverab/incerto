"""
Ensemble Methods for Uncertainty on MNIST

Demonstrates ensemble-based uncertainty quantification methods.
Compares single model vs. deep ensembles vs. MC Dropout.

Methods:
1. Single Model - Baseline
2. Deep Ensemble (5 models) - Multiple independent models
3. MC Dropout - Dropout at test time

Runtime: ~10 minutes on GPU, ~30 minutes on CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from rich.console import Console

# Incerto imports
from incerto.data import MNIST
from incerto.calibration import ece_score
from incerto.utils import seed_everything

console = Console()
seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console.print(f"[green]Using device: {device}[/green]\n")


# ============================================================================
# 1. Model Definitions
# ============================================================================


class StandardCNN(nn.Module):
    """Standard CNN for MNIST."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MCDropoutCNN(nn.Module):
    """CNN with MC Dropout."""

    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, n_samples=1):
        """Forward with MC sampling."""
        if n_samples == 1 or self.training:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            return x
        else:
            # Multiple forward passes with dropout active
            self.dropout1.train()
            self.dropout2.train()
            outputs = []
            for _ in range(n_samples):
                x_temp = x
                x_temp = F.relu(self.conv1(x_temp))
                x_temp = F.relu(self.conv2(x_temp))
                x_temp = F.max_pool2d(x_temp, 2)
                x_temp = self.dropout1(x_temp)
                x_temp = torch.flatten(x_temp, 1)
                x_temp = F.relu(self.fc1(x_temp))
                x_temp = self.dropout2(x_temp)
                x_temp = self.fc2(x_temp)
                outputs.append(x_temp)
            # Average predictions
            return torch.stack(outputs).mean(0)


# ============================================================================
# 2. Load Data
# ============================================================================

console.print("[yellow]Loading MNIST dataset...[/yellow]")
mnist = MNIST(root="./data", val_split=0.1, normalize=True)
train_dataset, val_dataset, test_dataset = mnist.get_datasets()

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

console.print(
    f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}\n"
)


# ============================================================================
# 3. Training Function
# ============================================================================


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """Train a single model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100.0 * val_correct / val_total
        if (epoch + 1) % 2 == 0:
            console.print(
                f"  Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.2f}%"
            )

    return model


# ============================================================================
# 4. Train Methods
# ============================================================================

# Method 1: Single Model
console.print("\n[bold cyan]═══ Method 1: Single Model ═══[/bold cyan]")
single_model = StandardCNN().to(device)
single_model = train_model(single_model, train_loader, val_loader, epochs=10)

# Method 2: Deep Ensemble (5 models)
console.print("\n[bold cyan]═══ Method 2: Deep Ensemble (5 models) ═══[/bold cyan]")
n_ensemble = 5
ensemble_models = []
for i in range(n_ensemble):
    console.print(f"\n[yellow]Training model {i+1}/{n_ensemble}...[/yellow]")
    seed_everything(42 + i)  # Different seed for diversity
    model = StandardCNN().to(device)
    model = train_model(model, train_loader, val_loader, epochs=10)
    ensemble_models.append(model)

# Method 3: MC Dropout
console.print("\n[bold cyan]═══ Method 3: MC Dropout ═══[/bold cyan]")
mc_model = MCDropoutCNN(dropout_rate=0.3).to(device)
mc_model = train_model(mc_model, train_loader, val_loader, epochs=10)


# ============================================================================
# 5. Evaluation
# ============================================================================

console.print("\n[yellow]Evaluating all methods...[/yellow]")


def evaluate_single(model, test_loader):
    """Evaluate single model."""
    model.eval()
    all_logits = []
    all_targets = []
    all_entropies = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)

            # Compute entropy (uncertainty)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)

            all_logits.append(logits.cpu())
            all_targets.append(targets)
            all_entropies.append(entropy.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    all_entropies = torch.cat(all_entropies)

    preds = all_logits.argmax(1)
    acc = (preds == all_targets).float().mean() * 100
    ece = ece_score(all_logits, all_targets)

    return {
        "accuracy": acc.item(),
        "ece": ece,
        "avg_entropy": all_entropies.mean().item(),
        "entropies": all_entropies,
        "correct": (preds == all_targets),
    }


def evaluate_ensemble(models, test_loader):
    """Evaluate ensemble."""
    for model in models:
        model.eval()

    all_logits = []
    all_targets = []
    all_entropies = []
    all_mutual_info = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)

            # Get predictions from all models
            ensemble_logits = []
            for model in models:
                logits = model(inputs)
                ensemble_logits.append(logits)

            ensemble_logits = torch.stack(ensemble_logits)

            # Average predictions (predictive distribution)
            avg_logits = ensemble_logits.mean(0)
            avg_probs = F.softmax(avg_logits, dim=1)

            # Total uncertainty (entropy of average)
            total_entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum(dim=1)

            # Aleatoric uncertainty (average entropy)
            individual_probs = F.softmax(ensemble_logits, dim=2)
            individual_entropies = -(
                individual_probs * torch.log(individual_probs + 1e-10)
            ).sum(dim=2)
            aleatoric = individual_entropies.mean(0)

            # Epistemic uncertainty (mutual information)
            epistemic = total_entropy - aleatoric

            all_logits.append(avg_logits.cpu())
            all_targets.append(targets)
            all_entropies.append(total_entropy.cpu())
            all_mutual_info.append(epistemic.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    all_entropies = torch.cat(all_entropies)
    all_mutual_info = torch.cat(all_mutual_info)

    preds = all_logits.argmax(1)
    acc = (preds == all_targets).float().mean() * 100
    ece = ece_score(all_logits, all_targets)

    return {
        "accuracy": acc.item(),
        "ece": ece,
        "avg_entropy": all_entropies.mean().item(),
        "avg_mutual_info": all_mutual_info.mean().item(),
        "entropies": all_entropies,
        "mutual_info": all_mutual_info,
        "correct": (preds == all_targets),
    }


def evaluate_mc_dropout(model, test_loader, n_samples=10):
    """Evaluate MC Dropout."""
    model.eval()
    all_logits = []
    all_targets = []
    all_entropies = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs, n_samples=n_samples)
            probs = F.softmax(logits, dim=1)

            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)

            all_logits.append(logits.cpu())
            all_targets.append(targets)
            all_entropies.append(entropy.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    all_entropies = torch.cat(all_entropies)

    preds = all_logits.argmax(1)
    acc = (preds == all_targets).float().mean() * 100
    ece = ece_score(all_logits, all_targets)

    return {
        "accuracy": acc.item(),
        "ece": ece,
        "avg_entropy": all_entropies.mean().item(),
        "entropies": all_entropies,
        "correct": (preds == all_targets),
    }


# Evaluate all methods
results = {}
results["Single Model"] = evaluate_single(single_model, test_loader)
results["Deep Ensemble"] = evaluate_ensemble(ensemble_models, test_loader)
results["MC Dropout"] = evaluate_mc_dropout(mc_model, test_loader, n_samples=10)

# Display results
console.print("\n[bold green]Results:[/bold green]")
for method, res in results.items():
    console.print(f"\n[cyan]{method}:[/cyan]")
    console.print(f"  Accuracy: {res['accuracy']:.2f}%")
    console.print(f"  ECE: {res['ece']:.4f}")
    console.print(f"  Avg Entropy: {res['avg_entropy']:.4f}")
    if "avg_mutual_info" in res:
        console.print(f"  Avg Mutual Info: {res['avg_mutual_info']:.4f}")


# ============================================================================
# 6. Visualizations
# ============================================================================

output_dir = Path("output/vision/mnist")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Uncertainty distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (method, res) in enumerate(results.items()):
    ax = axes[idx]
    entropies = res["entropies"].numpy()
    correct = res["correct"].numpy()

    ax.hist(entropies[correct], bins=30, alpha=0.6, label="Correct", color="green")
    ax.hist(entropies[~correct], bins=30, alpha=0.6, label="Incorrect", color="red")
    ax.set_xlabel("Entropy (Uncertainty)")
    ax.set_ylabel("Count")
    ax.set_title(f"{method}")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    output_dir / "ensemble_uncertainty_distributions.png", dpi=150, bbox_inches="tight"
)
console.print(
    f"\n[green]Saved uncertainty distributions to {output_dir / 'ensemble_uncertainty_distributions.png'}[/green]"
)

# Plot 2: Comparison metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

methods = list(results.keys())
accuracies = [results[m]["accuracy"] for m in methods]
eces = [results[m]["ece"] for m in methods]
entropies = [results[m]["avg_entropy"] for m in methods]

axes[0].bar(methods, accuracies, color="steelblue")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Test Accuracy")
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].bar(methods, eces, color="coral")
axes[1].set_ylabel("ECE")
axes[1].set_title("Calibration Error")
axes[1].grid(True, alpha=0.3, axis="y")

axes[2].bar(methods, entropies, color="green")
axes[2].set_ylabel("Average Entropy")
axes[2].set_title("Uncertainty")
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(output_dir / "ensemble_comparison.png", dpi=150, bbox_inches="tight")
console.print(
    f"[green]Saved comparison to {output_dir / 'ensemble_comparison.png'}[/green]"
)


# ============================================================================
# 7. Summary
# ============================================================================

console.print("\n" + "=" * 60)
console.print("[bold]SUMMARY[/bold]")
console.print("=" * 60)
console.print("\n[yellow]Key Takeaways:[/yellow]")
console.print("• Deep Ensemble: Best overall uncertainty quantification")
console.print("• MC Dropout: Efficient alternative (1 model, multiple passes)")
console.print("• Single Model: Least reliable uncertainty estimates")
console.print("• Ensemble provides epistemic + aleatoric uncertainty")
console.print("• Trade-off: Ensemble (5x cost) vs MC Dropout (10x inference)")
console.print("\n[bold green]✅ Ensemble methods comparison complete![/bold green]")
