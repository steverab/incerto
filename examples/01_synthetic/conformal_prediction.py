"""
Conformal Prediction Basics on Synthetic 2D Data

Introduction to conformal prediction using simple 2D synthetic data.
Shows how to create prediction sets with guaranteed coverage.

Concepts covered:
- What is conformal prediction?
- Inductive conformal prediction
- Coverage guarantees
- Prediction set sizes
- Efficiency vs coverage trade-off

Runtime: ~15 seconds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Incerto imports
from incerto.conformal import InductiveConformalPredictor
from incerto.conformal.metrics import coverage, average_set_size
from incerto.utils import seed_everything

# Set random seed
seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ============================================================================
# 1. Generate Synthetic Data
# ============================================================================


def generate_3class_data(n_samples=200):
    """Generate 3-class 2D data."""
    n_per_class = n_samples // 3

    # Class 0: Left cluster
    X0 = np.random.randn(n_per_class, 2) * 0.4 + np.array([-2, 0])
    y0 = np.zeros(n_per_class)

    # Class 1: Right cluster
    X1 = np.random.randn(n_per_class, 2) * 0.4 + np.array([2, 0])
    y1 = np.ones(n_per_class)

    # Class 2: Top cluster
    X2 = np.random.randn(n_per_class, 2) * 0.4 + np.array([0, 2.5])
    y2 = np.ones(n_per_class) * 2

    X = np.vstack([X0, X1, X2]).astype(np.float32)
    y = np.concatenate([y0, y1, y2]).astype(np.int64)

    return X, y


print("Generating synthetic 3-class data...")
X_train, y_train = generate_3class_data(n_samples=300)
X_calib, y_calib = generate_3class_data(n_samples=150)
X_test, y_test = generate_3class_data(n_samples=150)

print(f"Train: {len(X_train)}, Calibration: {len(X_calib)}, Test: {len(X_test)}\n")


# ============================================================================
# 2. Train Classifier
# ============================================================================


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


print("Training classifier...")
model = SimpleMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 25 == 0:
        acc = (outputs.argmax(1) == y_train_t).float().mean() * 100
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.2f}%")

print()


# ============================================================================
# 3. Conformal Prediction
# ============================================================================

print("Setting up conformal predictor...")

# Get calibration logits
model.eval()
X_calib_t = torch.FloatTensor(X_calib).to(device)
y_calib_t = torch.LongTensor(y_calib).to(device)

with torch.no_grad():
    calib_logits = model(X_calib_t)

# Create conformal predictor
cp = InductiveConformalPredictor(alpha=0.1)  # 90% coverage guarantee
cp.calibrate(calib_logits, y_calib_t)

print(
    f"Conformal predictor calibrated with α={cp.alpha} (target coverage: {(1-cp.alpha)*100}%)"
)
print(f"Quantile threshold: {cp.q_hat:.4f}\n")


# ============================================================================
# 4. Evaluate on Test Set
# ============================================================================

print("Evaluating on test set...")

X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.LongTensor(y_test).to(device)

with torch.no_grad():
    test_logits = model(X_test_t)
    test_probs = F.softmax(test_logits, dim=1)

# Get prediction sets
pred_sets = cp.predict(test_logits)

# Compute metrics
cov = coverage(pred_sets, y_test_t)
avg_size = average_set_size(pred_sets)

print(f"Coverage: {cov:.4f} (target: {(1-cp.alpha):.2f})")
print(f"Average set size: {avg_size:.2f}")
print()

# Analyze set sizes
set_sizes = torch.tensor([len(s) for s in pred_sets])
print("Prediction set size distribution:")
for size in range(1, 4):
    count = (set_sizes == size).sum().item()
    pct = count / len(set_sizes) * 100
    print(f"  Size {size}: {count} ({pct:.1f}%)")


# ============================================================================
# 5. Visualizations
# ============================================================================

output_dir = Path("output/synthetic")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Prediction sets on 2D space
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Standard predictions
ax = axes[0]
with torch.no_grad():
    preds = test_logits.argmax(1).cpu().numpy()

scatter = ax.scatter(
    X_test[:, 0], X_test[:, 1], c=preds, cmap="RdYlBu", edgecolors="k", s=100, alpha=0.7
)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("Standard Predictions (Point Estimates)")
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax)

# Conformal prediction sets
ax = axes[1]
# Color by set size
colors = ["green", "orange", "red"]  # singleton, doublet, triplet
color_map = [colors[len(s) - 1] for s in pred_sets]

scatter = ax.scatter(
    X_test[:, 0], X_test[:, 1], c=color_map, edgecolors="k", s=100, alpha=0.7
)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("Conformal Prediction Sets (by size)")
ax.grid(True, alpha=0.3)

# Custom legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="green", label="Singleton (1 class)"),
    Patch(facecolor="orange", label="Doublet (2 classes)"),
    Patch(facecolor="red", label="Triplet (3 classes)"),
]
ax.legend(handles=legend_elements, loc="best")

plt.tight_layout()
plt.savefig(output_dir / "conformal_predictions.png", dpi=150, bbox_inches="tight")
print(f"\nSaved predictions to {output_dir / 'conformal_predictions.png'}")

# Plot 2: Coverage vs set size
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Try different alpha values
alphas = np.linspace(0.01, 0.3, 20)
coverages = []
avg_sizes = []

for alpha in alphas:
    cp_temp = InductiveConformalPredictor(alpha=alpha)
    cp_temp.calibrate(calib_logits, y_calib_t)
    sets = cp_temp.predict(test_logits)
    coverages.append(coverage(sets, y_test_t))
    avg_sizes.append(average_set_size(sets))

# Coverage vs alpha
axes[0].plot(alphas, coverages, "b-", linewidth=2, label="Actual coverage")
axes[0].plot(alphas, 1 - alphas, "r--", linewidth=2, label="Target coverage")
axes[0].set_xlabel("α (miscoverage rate)")
axes[0].set_ylabel("Coverage")
axes[0].set_title("Coverage Guarantee")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Set size vs alpha
axes[1].plot(alphas, avg_sizes, "g-", linewidth=2)
axes[1].set_xlabel("α (miscoverage rate)")
axes[1].set_ylabel("Average Set Size")
axes[1].set_title("Efficiency (smaller is better)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "coverage_vs_efficiency.png", dpi=150, bbox_inches="tight")
print(f"Saved coverage analysis to {output_dir / 'coverage_vs_efficiency.png'}")


# ============================================================================
# 6. Summary
# ============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Target Coverage:  {(1-cp.alpha)*100:.0f}%")
print(f"Actual Coverage:  {cov*100:.2f}%")
print(f"Average Set Size: {avg_size:.2f}")
print()
print("Key Takeaways:")
print("• Conformal prediction provides coverage guarantees")
print("• No assumptions about data distribution needed")
print("• Trade-off: higher coverage → larger prediction sets")
print("• Singleton sets = high confidence predictions")
print("• Larger sets = model uncertainty about true class")
print("=" * 60)
