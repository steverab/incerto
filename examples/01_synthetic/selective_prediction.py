"""
Selective Prediction Basics on Synthetic 2D Data

Introduction to selective prediction (prediction with rejection) using 2D synthetic data.
Shows the trade-off between coverage and risk.

Concepts covered:
- What is selective prediction?
- Confidence-based rejection
- Risk-coverage curves
- AURC (Area Under Risk-Coverage)
- Selective accuracy

Runtime: ~15 seconds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Incerto imports
from incerto.sp import SoftmaxThreshold
from incerto.sp.metrics import aurc, risk, coverage as sp_coverage
from incerto.sp.visual import plot_risk_coverage
from incerto.utils import seed_everything

# Set random seed
seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ============================================================================
# 1. Generate Synthetic Data with Ambiguous Region
# ============================================================================


def generate_data_with_boundary(n_samples=300):
    """Generate data with clear and ambiguous regions."""
    n_per_class = n_samples // 2

    # Class 0: Left side (easy)
    X0_easy = np.random.randn(n_per_class // 2, 2) * 0.3 + np.array([-2, 0])
    # Class 0: Near boundary (hard)
    X0_hard = np.random.randn(n_per_class // 2, 2) * 0.3 + np.array([-0.3, 0])

    # Class 1: Right side (easy)
    X1_easy = np.random.randn(n_per_class // 2, 2) * 0.3 + np.array([2, 0])
    # Class 1: Near boundary (hard)
    X1_hard = np.random.randn(n_per_class // 2, 2) * 0.3 + np.array([0.3, 0])

    X0 = np.vstack([X0_easy, X0_hard])
    X1 = np.vstack([X1_easy, X1_hard])

    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int64)

    return X, y


print("Generating synthetic data with ambiguous boundary region...")
X_train, y_train = generate_data_with_boundary(n_samples=400)
X_test, y_test = generate_data_with_boundary(n_samples=200)

print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")


# ============================================================================
# 2. Train Classifier
# ============================================================================


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

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
# 3. Selective Prediction
# ============================================================================

print("Evaluating selective prediction...")

model.eval()
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.LongTensor(y_test).to(device)

with torch.no_grad():
    test_logits = model(X_test_t)
    test_probs = F.softmax(test_logits, dim=1)

# Standard accuracy (no rejection)
preds = test_logits.argmax(1)
base_acc = (preds == y_test_t).float().mean() * 100
print(f"Standard accuracy (100% coverage): {base_acc:.2f}%")

# Selective prediction with different thresholds
thresholds = [0.6, 0.7, 0.8, 0.9, 0.95]

print("\nSelective Prediction Results:")
print("Threshold | Coverage | Accuracy | Risk")
print("-" * 45)

results = []
for thresh in thresholds:
    predictor = SoftmaxThreshold(threshold=thresh)
    predictions, selected_mask = predictor.predict(test_logits)

    # Compute metrics
    cov = sp_coverage(selected_mask)
    if cov > 0:
        selected_preds = predictions[selected_mask]
        selected_targets = y_test_t[selected_mask]
        acc = (selected_preds == selected_targets).float().mean() * 100
        err = 100 - acc
    else:
        acc = 0
        err = 0

    print(f"  {thresh:.2f}    |  {cov*100:5.1f}%  |  {acc:5.2f}%  | {err:5.2f}%")
    results.append({"threshold": thresh, "coverage": cov, "accuracy": acc, "risk": err})

# Compute AURC
confidences = test_probs.max(1)[0]
sorted_conf, sorted_idx = confidences.sort(descending=True)
sorted_errors = (preds[sorted_idx] != y_test_t[sorted_idx]).float()
aurc_val = aurc(sorted_conf, sorted_errors)

print(f"\nAURC (Area Under Risk-Coverage): {aurc_val:.4f}")
print("(Lower is better)\n")


# ============================================================================
# 4. Visualizations
# ============================================================================

output_dir = Path("output/synthetic")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Confidence map with rejected regions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Create mesh
h = 0.05
x_min, x_max = -3.5, 3.5
y_min, y_max = -2, 2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)

with torch.no_grad():
    Z = model(grid)
    Z_probs = F.softmax(Z, dim=1).cpu().numpy()
    Z_conf = Z_probs.max(axis=1)
    Z_pred = Z.argmax(1).cpu().numpy()

Z_conf = Z_conf.reshape(xx.shape)
Z_pred = Z_pred.reshape(xx.shape)

# Left: Confidence map
ax = axes[0]
contour = ax.contourf(xx, yy, Z_conf, levels=20, cmap="RdYlGn", alpha=0.6)
plt.colorbar(contour, ax=ax, label="Confidence")
ax.scatter(
    X_test[:, 0], X_test[:, 1], c=y_test, cmap="RdYlBu", edgecolors="k", s=50, alpha=0.7
)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("Model Confidence Map")
ax.grid(True, alpha=0.3)

# Right: Rejection regions
ax = axes[1]
# Show which points would be rejected at threshold 0.8
predictor = SoftmaxThreshold(threshold=0.8)
_, selected_mask = predictor.predict(test_logits)
selected_mask_np = selected_mask.cpu().numpy()

ax.contourf(
    xx, yy, Z_conf, levels=[0, 0.8, 1.0], colors=["red", "lightgreen"], alpha=0.3
)
ax.scatter(
    X_test[selected_mask_np, 0],
    X_test[selected_mask_np, 1],
    c="green",
    edgecolors="k",
    s=100,
    alpha=0.7,
    label="Predicted",
    marker="o",
)
ax.scatter(
    X_test[~selected_mask_np, 0],
    X_test[~selected_mask_np, 1],
    c="red",
    edgecolors="k",
    s=100,
    alpha=0.7,
    label="Rejected",
    marker="X",
)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("Selective Prediction (threshold=0.8)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "selective_prediction_map.png", dpi=150, bbox_inches="tight")
print(f"Saved prediction map to {output_dir / 'selective_prediction_map.png'}")

# Plot 2: Risk-coverage curve
fig, ax = plt.subplots(figsize=(8, 6))
plot_risk_coverage(test_logits, y_test_t, confidences, ax=ax, show_aurc=True)
plt.savefig(output_dir / "risk_coverage_curve.png", dpi=150, bbox_inches="tight")
print(f"Saved risk-coverage curve to {output_dir / 'risk_coverage_curve.png'}")

# Plot 3: Coverage vs Accuracy trade-off
fig, ax = plt.subplots(figsize=(8, 6))
coverages_plot = [r["coverage"] * 100 for r in results]
accuracies_plot = [r["accuracy"] for r in results]

ax.plot(coverages_plot, accuracies_plot, "bo-", linewidth=2, markersize=8)
ax.axhline(
    y=base_acc, color="r", linestyle="--", label=f"Base accuracy ({base_acc:.1f}%)"
)
ax.set_xlabel("Coverage (%)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Coverage vs Accuracy Trade-off")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 105])
ax.set_ylim([base_acc - 5, 100])

plt.tight_layout()
plt.savefig(output_dir / "coverage_accuracy_tradeoff.png", dpi=150, bbox_inches="tight")
print(f"Saved trade-off curve to {output_dir / 'coverage_accuracy_tradeoff.png'}")


# ============================================================================
# 5. Summary
# ============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Base Accuracy (100% coverage): {base_acc:.2f}%")
print(f"Selective Accuracy (80% coverage): {results[2]['accuracy']:.2f}%")
print(f"AURC: {aurc_val:.4f}")
print()
print("Key Takeaways:")
print("• Selective prediction rejects uncertain examples")
print("• Trade-off: reject more → higher accuracy, lower coverage")
print("• Useful when cost of error is high")
print("• AURC measures overall risk-coverage performance")
print("• Low confidence ≈ boundary region → reject!")
print("=" * 60)
