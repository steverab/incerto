# Computer Vision Examples

Image classification with uncertainty quantification. From simple MNIST digits to complex CIFAR-10 natural images.

## Directory Structure

```
03_vision/
├── mnist/              # Handwritten digits (28x28 grayscale)
│   ├── 01_posthoc_evaluation.py
│   ├── 02_training_methods.py
│   └── 03_ensemble_methods.py
└── cifar/              # Natural images (32x32 RGB)
    ├── 01_posthoc_evaluation.py
    ├── 02_training_methods.py
    └── 03_advanced_training.py
```

## Learning Progression

### Start with MNIST
MNIST is perfect for learning because:
- **Simple**: Grayscale digits, clear patterns
- **Fast**: Trains in minutes on CPU
- **Well-studied**: Easy to interpret results
- **Low complexity**: Focus on uncertainty, not architecture

### Progress to CIFAR-10
CIFAR-10 is the next step because:
- **Realistic**: Natural images with background clutter
- **Challenging**: Requires better architectures (ResNet)
- **Longer training**: ~30 min, shows importance of methods
- **Better OOD**: CIFAR vs SVHN is classic benchmark

---

## MNIST Examples

**Dataset**: 60k train, 10k test, 28x28 grayscale, 10 classes (digits 0-9)
**Architecture**: Simple ConvNet (2 conv layers + 2 fc layers)
**OOD Dataset**: FashionMNIST (clothing items)

### 01_posthoc_evaluation.py
**Runtime**: ~5 minutes
**Difficulty**: ⭐⭐☆

Comprehensive post-hoc uncertainty evaluation.

**Sections**:
1. **Calibration**: Temperature Scaling, Vector Scaling, reliability diagrams
2. **OOD Detection**: MSP, Energy, ODIN, Mahalanobis scores vs FashionMNIST
3. **Selective Prediction**: Risk-coverage curves, confidence thresholding
4. **Conformal Prediction**: Prediction sets with coverage guarantees

**Key takeaway**: Post-hoc methods are powerful and easy to apply without retraining.

---

### 02_training_methods.py
**Runtime**: ~5 minutes
**Difficulty**: ⭐⭐☆

Compare training-time methods for uncertainty.

**Methods**:
1. **Baseline**: Standard cross-entropy
2. **Label Smoothing**: Reduces overconfidence
3. **Focal Loss**: Handles difficult examples
4. **Mixup**: Data augmentation for robustness
5. **SAT**: Self-Adaptive Training for selective prediction

**Evaluation**:
- Accuracy on MNIST test set
- ECE (calibration)
- AUROC (OOD detection on FashionMNIST)
- AURC (selective prediction)

**Key takeaway**: Training methods can improve multiple aspects of uncertainty simultaneously.

---

### 03_ensemble_methods.py
**Runtime**: ~10 minutes
**Difficulty**: ⭐⭐☆

Ensemble-based uncertainty quantification.

**Methods**:
1. **Single Model**: Baseline
2. **Deep Ensemble**: Train 5 independent models
3. **MC Dropout**: Multiple stochastic forward passes

**Uncertainty metrics**:
- Predictive entropy (disagreement between models/samples)
- Mutual information (epistemic uncertainty)
- Expected probability (aleatoric uncertainty)

**Key takeaway**: Ensembles capture epistemic uncertainty (model uncertainty) better than single models. Deep ensembles generally outperform MC Dropout.

---

## CIFAR-10 Examples

**Dataset**: 50k train, 10k test, 32x32 RGB, 10 classes (animals/vehicles)
**Architecture**: ResNet-18
**OOD Dataset**: SVHN (street view house numbers)

### 01_posthoc_evaluation.py
**Runtime**: ~10 minutes
**Difficulty**: ⭐⭐⭐

Post-hoc evaluation on complex images.

**Methods**:
- Calibration: Temperature, Vector, Matrix Scaling
- OOD: MSP, Energy, MaxLogit
- Conformal Prediction with different alpha values

**Key takeaway**: More complex data requires stronger architectures, but same uncertainty methods apply.

---

### 02_training_methods.py
**Runtime**: ~30 minutes
**Difficulty**: ⭐⭐⭐

Training methods comparison on CIFAR-10.

**Methods**:
1. Baseline
2. Label Smoothing
3. Focal Loss
4. Mixup

**Key takeaway**: Training methods show bigger improvements on complex data. Mixup particularly helps with robustness.

---

### 03_advanced_training.py
**Runtime**: ~30 minutes
**Difficulty**: ⭐⭐⭐

Advanced training techniques.

**Methods**:
1. **Evidential Deep Learning**: Learn higher-order uncertainty
2. **Focal Loss**: Focus on hard examples
3. **Confidence Penalty**: Explicitly regularize confidence

**Key takeaway**: Advanced methods can model different types of uncertainty (epistemic vs aleatoric).

---

## Recommended Learning Path

### For Beginners:
1. `mnist/01_posthoc_evaluation.py` - Learn all post-hoc methods
2. `mnist/02_training_methods.py` - Compare training approaches
3. `mnist/03_ensemble_methods.py` - Understand ensembles

### For Practitioners:
1. `mnist/02_training_methods.py` - Quick training methods intro
2. `cifar/01_posthoc_evaluation.py` - Post-hoc on realistic data
3. `cifar/02_training_methods.py` - Training on realistic data

### For Researchers:
1. `cifar/02_training_methods.py` - Baseline comparison
2. `cifar/03_advanced_training.py` - Advanced techniques
3. `mnist/03_ensemble_methods.py` - Ensemble methods deep dive

## Running the Examples

```bash
# MNIST examples
python examples/03_vision/mnist/01_posthoc_evaluation.py
python examples/03_vision/mnist/02_training_methods.py
python examples/03_vision/mnist/03_ensemble_methods.py

# CIFAR-10 examples
python examples/03_vision/cifar/01_posthoc_evaluation.py
python examples/03_vision/cifar/02_training_methods.py
python examples/03_vision/cifar/03_advanced_training.py
```

Visualizations saved to `output/vision/mnist/` and `output/vision/cifar/`.

## Common Patterns

### Post-hoc Evaluation Workflow:
```python
# 1. Train model
model = ConvNet(num_classes=10).to(device)
train_model(model, train_loader, epochs=10)

# 2. Get logits
val_logits = get_logits(model, val_loader)
test_logits = get_logits(model, test_loader)

# 3. Calibrate
calibrator = TemperatureScaling()
calibrator.fit(val_logits, val_targets)
test_logits_cal = calibrator.calibrate(test_logits)

# 4. Evaluate
ece = ece_score(test_logits_cal, test_targets)
```

### Training Methods Workflow:
```python
# Train multiple models
results = {}
for name, criterion in methods.items():
    model = ConvNet(num_classes=10).to(device)
    train(model, train_loader, criterion, epochs=10)
    results[name] = evaluate(model, test_loader, ood_loader)

# Compare
compare_methods(results)
```

### Ensemble Workflow:
```python
# Train ensemble
models = [ConvNet(num_classes=10).to(device) for _ in range(5)]
for model in models:
    train(model, train_loader, epochs=10)

# Get predictions
predictions = [model(x) for model in models]
mean_pred = torch.stack(predictions).mean(0)
entropy = predictive_entropy(predictions)
```

## Tips for Vision Examples

### 1. Data Augmentation
MNIST: Minimal augmentation (or none)
```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

CIFAR: Strong augmentation
```python
transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
```

### 2. Architecture Choice
- MNIST: Simple ConvNet is sufficient
- CIFAR: Use ResNet-18 or better

### 3. Training Time
- Reduce epochs for faster testing (2-3 epochs)
- Use GPU if available
- MNIST: Fast even on CPU
- CIFAR: GPU strongly recommended

### 4. OOD Detection
Classic pairings:
- MNIST vs FashionMNIST (same distribution family)
- CIFAR-10 vs SVHN (different domains)
- CIFAR-10 vs CIFAR-100 (semantic shift)

### 5. Hyperparameters
Default values in examples are reasonable but not optimal. For best results:
- Tune learning rate
- Adjust batch size based on GPU memory
- Increase epochs for final models

## Performance Expectations

### MNIST
- **Accuracy**: >98% (should be easy)
- **ECE**: 0.01-0.05 after calibration
- **OOD AUROC**: 0.85-0.95 (FashionMNIST)

### CIFAR-10
- **Accuracy**: 85-92% (ResNet-18)
- **ECE**: 0.02-0.08 after calibration
- **OOD AUROC**: 0.75-0.90 (SVHN)

## Common Questions

**Q: Why is MNIST accuracy lower than expected?**
A: Check data normalization and architecture. Should easily hit >98%.

**Q: CIFAR training is too slow?**
A: Reduce epochs to 5-10 for testing. Use GPU. Reduce batch size if OOM.

**Q: Which OOD dataset should I use?**
A: MNIST→FashionMNIST and CIFAR-10→SVHN are standard. Try CIFAR-10→CIFAR-100 for semantic shift.

**Q: Should I use post-hoc or training methods?**
A: Training methods give better results but require retraining. Post-hoc is great for existing models.

**Q: Deep Ensemble vs MC Dropout?**
A: Deep Ensemble is generally better but more expensive (5x training time). MC Dropout is a good lightweight alternative.

**Q: How many ensemble members?**
A: 5-10 is typical. Diminishing returns after 10. Even 3 helps significantly.

---

**Next**: [NLP Examples](../04_nlp/) or back to [Tabular Examples](../02_tabular/)
