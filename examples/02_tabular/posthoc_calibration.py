"""
Post-hoc Calibration on Tabular Data

Demonstrates calibration methods on the Wine dataset from sklearn.
Shows how to apply post-hoc calibration to tabular classification.

Concepts covered:
- Training on tabular data
- Post-hoc calibration methods
- Calibration metrics (ECE, MCE, Brier)
- Reliability diagrams for tabular data

Dataset: Wine Recognition (3 classes, 13 features)
Runtime: ~10 seconds
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

# Incerto imports
from incerto.calibration import (
    TemperatureScaling,
    VectorScaling,
    MatrixScaling,
    ece_score,
    mce_score,
    brier_score,
    plot_reliability_diagram,
)
from incerto.utils import MLP, seed_everything, train_epoch, evaluate

# Set random seed
seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ============================================================================
# 1. Load and Prepare Data
# ============================================================================

print("Loading Wine dataset...")
data = load_wine()
X, y = data.data, data.target

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val).to(device)
y_val_t = torch.LongTensor(y_val).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.LongTensor(y_test).to(device)

print(f"Dataset: {data.DESCR.split(':', 1)[0]}")
print(f"Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\n")


# ============================================================================
# 2. Train MLP Classifier
# ============================================================================

print("Training MLP classifier...")
model = MLP(
    input_dim=X.shape[1],
    hidden_dims=[64, 32],
    num_classes=3,
    dropout_rate=0.3,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
train_losses = []
val_accs = []

for epoch in range(50):
    # Train
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    # Validate
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_preds = val_outputs.argmax(1)
        val_acc = (val_preds == y_val_t).float().mean() * 100

    train_losses.append(loss.item())
    val_accs.append(val_acc.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Val Acc={val_acc:.2f}%")

print()


# ============================================================================
# 3. Evaluate Uncalibrated Model
# ============================================================================

print("Evaluating uncalibrated model...")
model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_preds = test_logits.argmax(1)
    test_acc = (test_preds == y_test_t).float().mean() * 100

# Compute uncalibrated metrics
ece_uncal = ece_score(test_logits, y_test_t)
mce_uncal = mce_score(test_logits, y_test_t)
brier_uncal = brier_score(test_logits, y_test_t)

print(f"Test Accuracy: {test_acc:.2f}%")
print(f"ECE: {ece_uncal:.4f}")
print(f"MCE: {mce_uncal:.4f}")
print(f"Brier Score: {brier_uncal:.4f}\n")


# ============================================================================
# 4. Apply Post-hoc Calibration
# ============================================================================

print("Applying post-hoc calibration methods...")

# Get validation logits for calibration
with torch.no_grad():
    val_logits = model(X_val_t)

# Calibration methods
calibrators = {
    "Temperature Scaling": TemperatureScaling(),
    "Vector Scaling": VectorScaling(n_classes=3),
    "Matrix Scaling": MatrixScaling(n_classes=3),
}

results = {
    "Uncalibrated": {
        "ECE": ece_uncal,
        "MCE": mce_uncal,
        "Brier": brier_uncal,
    }
}

# Fit and evaluate each calibrator
for name, calibrator in calibrators.items():
    # Fit on validation set
    calibrator.fit(val_logits, y_val_t)

    # Calibrate test logits
    test_logits_cal = calibrator.calibrate(test_logits)

    # Compute calibrated metrics
    ece_cal = ece_score(test_logits_cal, y_test_t)
    mce_cal = mce_score(test_logits_cal, y_test_t)
    brier_cal = brier_score(test_logits_cal, y_test_t)

    results[name] = {
        "ECE": ece_cal,
        "MCE": mce_cal,
        "Brier": brier_cal,
    }

    print(f"{name}:")
    print(f"  ECE: {ece_cal:.4f} ({(ece_uncal - ece_cal)/ece_uncal*100:+.1f}%)")
    print(f"  MCE: {mce_cal:.4f} ({(mce_uncal - mce_cal)/mce_uncal*100:+.1f}%)")
    print(
        f"  Brier: {brier_cal:.4f} ({(brier_uncal - brier_cal)/brier_uncal*100:+.1f}%)"
    )
    print()


# ============================================================================
# 5. Visualizations
# ============================================================================

output_dir = Path("output/tabular")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Comparison table
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("tight")
ax.axis("off")

# Prepare table data
methods = list(results.keys())
table_data = []
for method in methods:
    row = [
        method,
        f"{results[method]['ECE']:.4f}",
        f"{results[method]['MCE']:.4f}",
        f"{results[method]['Brier']:.4f}",
    ]
    table_data.append(row)

table = ax.table(
    cellText=table_data,
    colLabels=["Method", "ECE ↓", "MCE ↓", "Brier ↓"],
    cellLoc="center",
    loc="center",
    colWidths=[0.4, 0.2, 0.2, 0.2],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header
for i in range(4):
    table[(0, i)].set_facecolor("#4CAF50")
    table[(0, i)].set_text_props(weight="bold", color="white")

# Color best results
for col_idx, metric in enumerate(["ECE", "MCE", "Brier"], start=1):
    values = [results[m][metric] for m in methods]
    best_idx = np.argmin(values)
    table[(best_idx + 1, col_idx)].set_facecolor("#E8F5E9")

plt.title(
    "Calibration Methods Comparison on Wine Dataset", fontsize=14, weight="bold", pad=20
)
plt.savefig(output_dir / "comparison_table.png", dpi=150, bbox_inches="tight")
print(f"Saved comparison table to {output_dir / 'comparison_table.png'}")

# Plot 2: Reliability diagrams
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Uncalibrated
plot_reliability_diagram(test_logits, y_test_t, ax=axes[0], n_bins=10)
axes[0].set_title(f"Uncalibrated (ECE={ece_uncal:.4f})")

# Calibrated versions
for idx, (name, calibrator) in enumerate(calibrators.items(), start=1):
    test_logits_cal = calibrator.calibrate(test_logits)
    plot_reliability_diagram(test_logits_cal, y_test_t, ax=axes[idx], n_bins=10)
    ece_cal = results[name]["ECE"]
    axes[idx].set_title(f"{name} (ECE={ece_cal:.4f})")

plt.tight_layout()
plt.savefig(output_dir / "reliability_diagrams.png", dpi=150, bbox_inches="tight")
print(f"Saved reliability diagrams to {output_dir / 'reliability_diagrams.png'}")


# ============================================================================
# 6. Summary
# ============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Dataset: Wine Recognition ({len(X)} samples, {X.shape[1]} features)")
print(f"Test Accuracy: {test_acc:.2f}%")
print()
print("Calibration Results:")
for method in results:
    ece = results[method]["ECE"]
    improvement = (
        ((ece_uncal - ece) / ece_uncal * 100) if method != "Uncalibrated" else 0
    )
    marker = "✓ Best" if ece == min(r["ECE"] for r in results.values()) else ""
    print(f"  {method:20s}: ECE={ece:.4f} ({improvement:+.1f}%) {marker}")
print()
print("Key Takeaways:")
print("• Tabular data benefits from calibration like images")
print("• Vector/Matrix Scaling can outperform Temperature Scaling")
print("• Post-hoc methods preserve accuracy while improving calibration")
print("• Use validation set for fitting calibrators")
print("=" * 60)
