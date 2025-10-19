"""
Training Methods for Uncertainty on Tabular Data

Demonstrates training-time methods for improving uncertainty on tabular classification.
Compares standard training vs uncertainty-aware training.

Concepts covered:
- Label smoothing for calibration
- Focal loss for hard examples
- Mixup for robustness
- Training vs post-hoc approaches

Dataset: Breast Cancer (binary classification, 30 features)
Runtime: ~30 seconds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

# Incerto imports
from incerto.calibration import LabelSmoothingLoss, FocalLoss, ece_score
from incerto.ood import mixup_data, mixup_criterion
from incerto.utils import MLP, seed_everything

# Set random seed
seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ============================================================================
# 1. Load and Prepare Data
# ============================================================================

print("Loading Breast Cancer dataset...")
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Dataset: Breast Cancer Wisconsin")
print(f"Features: {X.shape[1]}, Classes: 2 (Benign/Malignant)")
print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")


# ============================================================================
# 2. Training Function
# ============================================================================


def train_model(model, X_train, y_train, criterion, method_name, use_mixup=False):
    """Train model with given criterion."""
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Training with {method_name}...")
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        if use_mixup:
            # Apply mixup
            mixed_x, y_a, y_b, lam = mixup_data(
                X_train_t, y_train_t, alpha=0.2, device=device
            )
            outputs = model(mixed_x)
            loss = mixup_criterion(nn.CrossEntropyLoss(), outputs, y_a, y_b, lam)
        else:
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train_t)
                train_acc = (train_outputs.argmax(1) == y_train_t).float().mean() * 100
            print(
                f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Train Acc={train_acc:.2f}%"
            )

    return model


# ============================================================================
# 3. Train Models with Different Methods
# ============================================================================

models = {}
results = {}

# Convert test data
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.LongTensor(y_test).to(device)

# Method 1: Baseline (Cross-Entropy)
print("-" * 60)
model_baseline = MLP(input_dim=X.shape[1], hidden_dims=[64, 32], num_classes=2).to(
    device
)
criterion_baseline = nn.CrossEntropyLoss()
models["Baseline"] = train_model(
    model_baseline, X_train, y_train, criterion_baseline, "Baseline (CE)"
)

# Method 2: Label Smoothing
print("\n" + "-" * 60)
model_ls = MLP(input_dim=X.shape[1], hidden_dims=[64, 32], num_classes=2).to(device)
criterion_ls = LabelSmoothingLoss(smoothing=0.1)
models["Label Smoothing"] = train_model(
    model_ls, X_train, y_train, criterion_ls, "Label Smoothing"
)

# Method 3: Focal Loss
print("\n" + "-" * 60)
model_focal = MLP(input_dim=X.shape[1], hidden_dims=[64, 32], num_classes=2).to(device)
criterion_focal = FocalLoss(gamma=2.0)
models["Focal Loss"] = train_model(
    model_focal, X_train, y_train, criterion_focal, "Focal Loss"
)

# Method 4: Mixup
print("\n" + "-" * 60)
model_mixup = MLP(input_dim=X.shape[1], hidden_dims=[64, 32], num_classes=2).to(device)
models["Mixup"] = train_model(
    model_mixup, X_train, y_train, None, "Mixup", use_mixup=True
)

print("\n" + "=" * 60)


# ============================================================================
# 4. Evaluate All Models
# ============================================================================

print("\nEvaluating models...")
print("-" * 60)

for name, model in models.items():
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_preds = test_logits.argmax(1)
        test_probs = F.softmax(test_logits, dim=1)

        # Metrics
        acc = (test_preds == y_test_t).float().mean() * 100
        ece = ece_score(test_logits, y_test_t)

        # Confidence statistics
        confidences = test_probs.max(1)[0]
        avg_conf = confidences.mean() * 100
        correct_mask = test_preds == y_test_t
        avg_conf_correct = (
            confidences[correct_mask].mean() * 100 if correct_mask.sum() > 0 else 0
        )
        avg_conf_incorrect = (
            confidences[~correct_mask].mean() * 100 if (~correct_mask).sum() > 0 else 0
        )

        results[name] = {
            "accuracy": acc.item(),
            "ece": ece,
            "avg_confidence": avg_conf.item(),
            "conf_correct": avg_conf_correct.item(),
            "conf_incorrect": avg_conf_incorrect.item(),
        }

        print(f"{name}:")
        print(f"  Accuracy: {acc:.2f}%")
        print(f"  ECE: {ece:.4f}")
        print(f"  Avg Confidence: {avg_conf:.2f}%")
        print(f"  Conf (Correct): {avg_conf_correct:.2f}%")
        print(f"  Conf (Incorrect): {avg_conf_incorrect:.2f}%")
        print()


# ============================================================================
# 5. Visualizations
# ============================================================================

output_dir = Path("output/tabular")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

methods = list(results.keys())
accuracies = [results[m]["accuracy"] for m in methods]
eces = [results[m]["ece"] for m in methods]

# Accuracy
axes[0].bar(methods, accuracies, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Test Accuracy (Higher is Better)")
axes[0].set_ylim([min(accuracies) - 2, max(accuracies) + 2])
axes[0].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.3, f"{v:.2f}%", ha="center", va="bottom")

# ECE
axes[1].bar(methods, eces, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
axes[1].set_ylabel("ECE")
axes[1].set_title("Expected Calibration Error (Lower is Better)")
axes[1].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(eces):
    axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig(
    output_dir / "training_methods_comparison.png", dpi=150, bbox_inches="tight"
)
print(f"Saved comparison to {output_dir / 'training_methods_comparison.png'}")

# Plot 2: Confidence analysis
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(methods))
width = 0.25

conf_correct = [results[m]["conf_correct"] for m in methods]
conf_incorrect = [results[m]["conf_incorrect"] for m in methods]
avg_conf = [results[m]["avg_confidence"] for m in methods]

bars1 = ax.bar(
    x - width,
    conf_correct,
    width,
    label="Correct Predictions",
    color="green",
    alpha=0.7,
)
bars2 = ax.bar(
    x, conf_incorrect, width, label="Incorrect Predictions", color="red", alpha=0.7
)
bars3 = ax.bar(x + width, avg_conf, width, label="Average", color="blue", alpha=0.7)

ax.set_ylabel("Confidence (%)")
ax.set_title("Confidence Analysis by Training Method")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(output_dir / "confidence_analysis.png", dpi=150, bbox_inches="tight")
print(f"Saved confidence analysis to {output_dir / 'confidence_analysis.png'}")


# ============================================================================
# 6. Summary
# ============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("Training Method Comparison:")
print()

# Find best method for each metric
best_acc = max(methods, key=lambda m: results[m]["accuracy"])
best_ece = min(methods, key=lambda m: results[m]["ece"])

for method in methods:
    acc = results[method]["accuracy"]
    ece = results[method]["ece"]
    marker_acc = " ← Best Accuracy" if method == best_acc else ""
    marker_ece = " ← Best ECE" if method == best_ece else ""
    print(f"{method}:")
    print(f"  Accuracy: {acc:.2f}%{marker_acc}")
    print(f"  ECE: {ece:.4f}{marker_ece}")

print()
print("Key Takeaways:")
print("• Label Smoothing improves calibration with minimal accuracy cost")
print("• Focal Loss helps when class imbalance exists")
print("• Mixup improves robustness and generalization")
print("• Training-time methods > Post-hoc when you can retrain")
print("• Lower confidence on incorrect predictions = better uncertainty")
print("=" * 60)
