# Tabular Data Examples

Real-world tabular classification with uncertainty quantification. Learn to apply both post-hoc and training-time methods to structured data.

## Why Tabular Data?

- **Practical**: Most real-world ML problems use tabular data
- **Fast**: Small datasets train in seconds/minutes
- **Interpretable**: Easy to understand feature relationships
- **Common use case**: Finance, healthcare, business analytics

## Examples

### 1. Post-hoc Calibration
**File**: `posthoc_calibration.py`
**Runtime**: ~10 seconds
**Dataset**: Wine Recognition (178 samples, 13 features, 3 classes)
**Difficulty**: ⭐☆☆

Apply calibration methods after training.

**What it does**:
1. Train MLP classifier on Wine dataset
2. Apply three calibration methods on validation set
3. Evaluate calibrated models on test set
4. Compare ECE, MCE, and Brier scores

**Methods compared**:
- **Temperature Scaling**: Single scalar temperature parameter
- **Vector Scaling**: Per-class temperature parameters
- **Matrix Scaling**: Full affine transformation of logits

**Visualizations**:
- Comparison table of calibration metrics
- Reliability diagrams (before/after calibration)

**Key insights**:
- Tabular models benefit from calibration like image models
- Vector/Matrix Scaling can outperform Temperature Scaling
- Post-hoc methods preserve accuracy while improving calibration
- Always use validation set for fitting calibrators

**Code pattern**:
```python
# Train model
model = MLP(input_dim=13, hidden_dims=[64, 32], num_classes=3)
# ... training loop ...

# Get validation logits
val_logits = model(X_val)

# Fit calibrator
calibrator = TemperatureScaling()
calibrator.fit(val_logits, y_val)

# Calibrate test logits
test_logits_cal = calibrator.calibrate(test_logits)
```

---

### 2. Training Methods
**File**: `training_methods.py`
**Runtime**: ~30 seconds
**Dataset**: Breast Cancer (569 samples, 30 features, 2 classes)
**Difficulty**: ⭐⭐☆

Compare training-time methods for better uncertainty.

**What it does**:
1. Train 4 different models with different loss functions
2. Evaluate each on accuracy, calibration (ECE), and OOD detection
3. Compare trade-offs between methods

**Methods compared**:
- **Baseline**: Standard cross-entropy loss
- **Label Smoothing**: Smooths one-hot labels (reduces overconfidence)
- **Focal Loss**: Focuses on hard examples
- **Mixup**: Trains on interpolated samples

**Metrics evaluated**:
- **Accuracy**: Classification performance
- **ECE**: Calibration quality
- **OOD AUROC**: Ability to detect synthetic OOD data

**Visualizations**:
- Training curves for all methods
- Final comparison table
- Uncertainty distributions

**Key insights**:
- Label Smoothing: Best calibration
- Focal Loss: Best for imbalanced data
- Mixup: Best OOD detection and robustness
- All maintain competitive accuracy

**Code pattern**:
```python
# Label Smoothing
criterion = LabelSmoothingLoss(smoothing=0.1)
train_standard(model, train_loader, criterion, optimizer, epochs=50)

# Focal Loss
criterion = FocalLoss(gamma=2.0)
train_standard(model, train_loader, criterion, optimizer, epochs=50)

# Mixup
train_mixup(model, train_loader, optimizer, epochs=50, alpha=1.0)
```

---

## Recommended Order

1. **Start with `posthoc_calibration.py`**
   - Understand calibration metrics
   - Learn post-hoc calibration workflow
   - See how to evaluate calibration quality

2. **Then `training_methods.py`**
   - Compare training vs post-hoc approaches
   - Understand trade-offs between methods
   - Learn when to use each method

## Running the Examples

```bash
# From the repository root
python examples/02_tabular/posthoc_calibration.py
python examples/02_tabular/training_methods.py
```

All examples save visualizations to `output/tabular/`.

## When to Use Each Method

### Use Post-hoc Calibration When:
- You have a pre-trained model
- You can't retrain the model
- You have a calibration set available
- You need quick calibration fixes

### Use Training Methods When:
- Training from scratch
- You can modify the training process
- You need better OOD detection
- You want more robust features (Mixup)

## Common Patterns

### Train-Validate-Calibrate-Test Split:
```python
# Split 1: Train/Temp (60/40)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)

# Split 2: Val/Test from Temp (50/50)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Train on train set
model.fit(X_train, y_train)

# Calibrate on val set
calibrator.fit(val_logits, y_val)

# Evaluate on test set
metrics = evaluate(test_logits_calibrated, y_test)
```

### Comparing Multiple Methods:
```python
methods = {
    'Baseline': {'criterion': nn.CrossEntropyLoss()},
    'Label Smoothing': {'criterion': LabelSmoothingLoss(0.1)},
    'Focal Loss': {'criterion': FocalLoss(gamma=2.0)},
}

results = {}
for name, config in methods.items():
    model = train_model(config['criterion'])
    results[name] = evaluate_model(model)
```

## What's Next?

After mastering tabular examples:
- **`03_vision/mnist/`** - Apply to image classification
- **`03_vision/cifar/`** - Scale to complex images
- **`04_nlp/`** - Work with language models

The same concepts (calibration, OOD detection, training methods) apply across all modalities!

## Tips for Tabular Data

1. **Feature Scaling**: Always standardize features
   ```python
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_val = scaler.transform(X_val)
   ```

2. **Class Imbalance**: Use `stratify=y` in train_test_split
   ```python
   train_test_split(X, y, stratify=y, test_size=0.3)
   ```

3. **Small Datasets**: Use dropout and early stopping
   ```python
   model = MLP(dropout_rate=0.3)
   early_stopping = EarlyStopping(patience=10)
   ```

4. **Hyperparameters**: These examples use reasonable defaults, but tune for your data

## Common Questions

**Q: Why are tabular datasets so small?**
A: Using sklearn datasets for quick demonstrations. Apply the same methods to larger datasets.

**Q: Should I use post-hoc or training methods?**
A: Training methods are generally better if you can train from scratch. Post-hoc is great for existing models.

**Q: Can I combine methods?**
A: Yes! Train with Label Smoothing, then apply Temperature Scaling. Often gives best results.

**Q: What about other tabular algorithms (XGBoost, Random Forest)?**
A: These examples focus on neural networks, but calibration applies to any probabilistic classifier.

---

**Next**: [Vision Examples](../03_vision/) or back to [Synthetic Examples](../01_synthetic/)
