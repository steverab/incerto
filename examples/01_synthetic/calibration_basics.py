"""
Calibration Basics on Synthetic 2D Data

This example demonstrates calibration concepts using simple 2D synthetic data.
Perfect for understanding calibration visually before applying to real datasets.

Concepts covered:
- What is calibration?
- Temperature scaling
- Reliability diagrams
- ECE (Expected Calibration Error)

Runtime: ~30 seconds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Incerto imports
from incerto.calibration import (
    TemperatureScaling,
    ece_score,
    plot_reliability_diagram,
    plot_confidence_histogram,
)
from incerto.utils import seed_everything, plot_2d_classification

# Set random seed
seed_everything(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ============================================================================
# 1. Generate Synthetic 2D Data
# ============================================================================


def generate_2d_classification_data(n_samples=1000, noise=0.3):
    """Generate 2D spiral classification data."""
    n_per_class = n_samples // 3

    # Class 0: Circle
    theta = np.random.uniform(0, 2 * np.pi, n_per_class)
    r = np.random.uniform(0.5, 1.0, n_per_class)
    X0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    y0 = np.zeros(n_per_class)

    # Class 1: Outer ring
    theta = np.random.uniform(0, 2 * np.pi, n_per_class)
    r = np.random.uniform(2.0, 2.5, n_per_class)
    X1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    y1 = np.ones(n_per_class)

    # Class 2: Top cluster
    X2 = np.random.randn(n_per_class, 2) * 0.3 + np.array([0, 3.5])
    y2 = np.ones(n_per_class) * 2

    # Combine
    X = np.vstack([X0, X1, X2]).astype(np.float32)
    y = np.concatenate([y0, y1, y2]).astype(np.int64)

    # Add noise
    X += np.random.randn(*X.shape).astype(np.float32) * noise

    return X, y


print("Generating synthetic 2D data...")
X_train, y_train = generate_2d_classification_data(n_samples=600, noise=0.2)
X_val, y_val = generate_2d_classification_data(n_samples=200, noise=0.2)
X_test, y_test = generate_2d_classification_data(n_samples=200, noise=0.2)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\n")


# ============================================================================
# 2. Simple MLP Model
# ============================================================================


class SimpleMLP(nn.Module):
    """Simple 2-layer MLP."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================================
# 3. Train Model (Intentionally Overconfident)
# ============================================================================

print("Training model (with high learning rate to induce overconfidence)...")
model = SimpleMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # High LR → overconfident
criterion = nn.CrossEntropyLoss()

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val).to(device)
y_val_t = torch.LongTensor(y_val).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.LongTensor(y_test).to(device)

# Training
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
            val_acc = (val_outputs.argmax(1) == y_val_t).float().mean() * 100
        print(
            f"Epoch {epoch+1}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}, Val Acc={val_acc:.2f}%"
        )

print()


# ============================================================================
# 4. Evaluate Uncalibrated Model
# ============================================================================

print("Evaluating uncalibrated model...")
model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_probs = F.softmax(test_logits, dim=1)
    test_preds = test_logits.argmax(1)
    test_acc = (test_preds == y_test_t).float().mean() * 100

# Compute ECE
ece_uncal = ece_score(test_logits, y_test_t)

print(f"Test Accuracy: {test_acc:.2f}%")
print(f"ECE (Uncalibrated): {ece_uncal:.4f}")
print()


# ============================================================================
# 5. Calibrate with Temperature Scaling
# ============================================================================

print("Calibrating with Temperature Scaling...")
calibrator = TemperatureScaling()

# Fit on validation set
with torch.no_grad():
    val_logits = model(X_val_t)

calibrator.fit(val_logits, y_val_t)
print(f"Learned temperature: {calibrator.temperature.item():.4f}")

# Calibrate test logits
test_logits_cal = calibrator.calibrate(test_logits)

# Compute calibrated ECE
ece_cal = ece_score(test_logits_cal, y_test_t)
print(f"ECE (Calibrated): {ece_cal:.4f}")
print(f"Improvement: {((ece_uncal - ece_cal) / ece_uncal * 100):.1f}% reduction\n")


# ============================================================================
# 6. Visualizations
# ============================================================================

output_dir = Path("output/synthetic")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Decision boundaries
print("Creating visualizations...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Uncalibrated
plt.sca(axes[0])
plot_2d_classification(X_test, y_test, model, device, title="Uncalibrated Model")


# Calibrated (wrap model with calibrator)
class CalibratedModel(nn.Module):
    def __init__(self, model, calibrator):
        super().__init__()
        self.model = model
        self.calibrator = calibrator

    def forward(self, x):
        logits = self.model(x)
        return self.calibrator.calibrate(logits)


cal_model = CalibratedModel(model, calibrator).to(device)
plt.sca(axes[1])
plot_2d_classification(X_test, y_test, cal_model, device, title="Calibrated Model")

plt.tight_layout()
plt.savefig(output_dir / "decision_boundaries.png", dpi=150, bbox_inches="tight")
print(f"Saved decision boundaries to {output_dir / 'decision_boundaries.png'}")

# Plot 2: Reliability diagrams
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0] = plot_reliability_diagram(test_logits, y_test_t, ax=axes[0], n_bins=10)
axes[0].set_title(f"Uncalibrated (ECE={ece_uncal:.4f})")

axes[1] = plot_reliability_diagram(test_logits_cal, y_test_t, ax=axes[1], n_bins=10)
axes[1].set_title(f"Calibrated (ECE={ece_cal:.4f})")

plt.tight_layout()
plt.savefig(output_dir / "reliability_diagrams.png", dpi=150, bbox_inches="tight")
print(f"Saved reliability diagrams to {output_dir / 'reliability_diagrams.png'}")

# Plot 3: Confidence histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_confidence_histogram(test_logits, n_bins=20, ax=axes[0])
axes[0].set_title("Uncalibrated Confidence")

plot_confidence_histogram(test_logits_cal, n_bins=20, ax=axes[1])
axes[1].set_title("Calibrated Confidence")

plt.tight_layout()
plt.savefig(output_dir / "confidence_histograms.png", dpi=150, bbox_inches="tight")
print(f"Saved confidence histograms to {output_dir / 'confidence_histograms.png'}")


# ============================================================================
# 7. Summary
# ============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Test Accuracy:       {test_acc:.2f}%")
print(f"ECE (Uncalibrated):  {ece_uncal:.4f}")
print(f"ECE (Calibrated):    {ece_cal:.4f}")
print(
    f"Improvement:         {((ece_uncal - ece_cal) / ece_uncal * 100):.1f}% reduction"
)
print()
print("Key Takeaways:")
print("• Uncalibrated models can be overconfident (high ECE)")
print("• Temperature scaling improves calibration without changing predictions")
print("• Calibrated models have confidence scores that better match accuracy")
print("• Check reliability diagrams: should be close to diagonal")
print("=" * 60)
