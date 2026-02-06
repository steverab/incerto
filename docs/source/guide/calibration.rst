Calibration Guide
=================

Model calibration ensures that predicted confidence scores accurately reflect empirical accuracy. A well-calibrated model's confidence estimates are trustworthy: if it predicts 80% confidence, it should be correct 80% of the time.

Why Calibration Matters
-----------------------

Modern neural networks are often **overconfident** - they produce high confidence scores even for incorrect predictions. This is especially problematic in:

- **Safety-critical applications**: Medical diagnosis, autonomous vehicles
- **Decision making**: Where confidence guides resource allocation
- **Human-AI collaboration**: Users need trustworthy uncertainty estimates

Example of miscalibration:
   A classifier predicts 95% confidence but is only correct 70% of the time.

Calibration Metrics
-------------------

Expected Calibration Error (ECE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most common calibration metric. Divides predictions into bins by confidence and measures the gap between confidence and accuracy:

.. code-block:: python

   from incerto.calibration import ece_score

   # logits: model outputs (N, num_classes)
   # labels: ground truth (N,)
   ece = ece_score(logits, labels, n_bins=15)
   print(f"ECE: {ece:.4f}")  # Lower is better, 0 = perfect calibration

**Interpretation**:
   - ECE < 0.05: Well calibrated
   - ECE 0.05-0.15: Moderate miscalibration
   - ECE > 0.15: Severe miscalibration

Maximum Calibration Error (MCE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The worst-case calibration error across all bins:

.. code-block:: python

   from incerto.calibration import mce_score

   mce = mce_score(logits, labels, n_bins=15)
   # MCE >= ECE always

Adaptive ECE
^^^^^^^^^^^^

Uses adaptive binning (equal mass per bin) instead of uniform bins:

.. code-block:: python

   from incerto.calibration import adaptive_ece_score

   adaptive_ece = adaptive_ece_score(logits, labels, n_bins=15)
   # More stable for imbalanced datasets

Brier Score
^^^^^^^^^^^

Measures both calibration and accuracy:

.. code-block:: python

   from incerto.calibration import brier_score

   bs = brier_score(logits, labels)
   # Lower is better, ranges from 0 to 1

Negative Log-Likelihood (NLL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Standard loss function, sensitive to miscalibration:

.. code-block:: python

   from incerto.calibration import nll

   loss = nll(logits, labels)
   # Lower is better

Post-Hoc Calibration Methods
-----------------------------

Post-hoc methods calibrate a trained model using a held-out calibration set (typically part of the validation data).

Temperature Scaling
^^^^^^^^^^^^^^^^^^^

**Best for**: Most scenarios, especially with good validation data

The simplest and most effective method. Scales logits by a learned temperature T:

.. math::

   p_i^{calibrated} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}

.. code-block:: python

   from incerto.calibration import TemperatureScaling

   # Fit on validation data
   calibrator = TemperatureScaling()
   calibrator.fit(val_logits, val_labels)

   # Apply to test data
   calibrated_dist = calibrator.predict(test_logits)
   calibrated_probs = calibrated_dist.probs

   # Save for later use
   calibrator.save('temperature_calibrator.pt')

   print(f"Learned temperature: {calibrator.temperature.item():.4f}")
   # T > 1: Model is overconfident (common)
   # T < 1: Model is underconfident (rare)

**Advantages**:
   - Single parameter (very stable)
   - Preserves predicted class (argmax unchanged)
   - Fast to optimize

**Disadvantages**:
   - Same scaling for all classes

Vector Scaling
^^^^^^^^^^^^^^

**Best for**: Class imbalance, when different classes have different confidence patterns

Learns a different temperature for each class:

.. code-block:: python

   from incerto.calibration import VectorScaling

   calibrator = VectorScaling(n_classes=10)
   calibrator.fit(val_logits, val_labels, max_iters=50)

   calibrated_dist = calibrator.predict(test_logits)

   print(repr(calibrator))
   # VectorScaling(n_classes=10, temperature_range=[0.8, 2.1])

**Advantages**:
   - More flexible than temperature scaling
   - Can handle class-specific miscalibration

**Disadvantages**:
   - More parameters (more validation data needed)
   - May not preserve argmax predictions

Matrix Scaling
^^^^^^^^^^^^^^

**Best for**: When you have lots of validation data and need maximum flexibility

Most general affine transformation: :math:`z_{calibrated} = W z + b`

.. code-block:: python

   from incerto.calibration import MatrixScaling

   calibrator = MatrixScaling(n_classes=10)
   calibrator.fit(val_logits, val_labels, max_iters=50)

   calibrated_dist = calibrator.predict(test_logits)

**Advantages**:
   - Most flexible parametric method
   - Can fix complex miscalibration patterns

**Disadvantages**:
   - Many parameters (n_classes²)
   - Risk of overfitting
   - Does not preserve argmax

Dirichlet Calibration
^^^^^^^^^^^^^^^^^^^^^

**Best for**: Research settings, when you need distribution-level calibration

Learns a matrix and bias like matrix scaling but with better theoretical properties:

.. code-block:: python

   from incerto.calibration import DirichletCalibrator

   # With L2 regularization
   calibrator = DirichletCalibrator(n_classes=10, mu=0.01)
   calibrator.fit(val_logits, val_labels, max_iters=100)

   calibrated_dist = calibrator.predict(test_logits)

**Reference**: Kull et al., "Beyond temperature scaling" (NeurIPS 2019)

Isotonic Regression
^^^^^^^^^^^^^^^^^^^

**Best for**: Non-parametric, flexible calibration

Fits a monotonic (isotonic) mapping per class:

.. code-block:: python

   from incerto.calibration import IsotonicRegressionCalibrator

   calibrator = IsotonicRegressionCalibrator(out_of_bounds='clip')
   calibrator.fit(val_logits, val_labels)

   calibrated_dist = calibrator.predict(test_logits)

**Advantages**:
   - Very flexible (non-parametric)
   - Can capture complex patterns

**Disadvantages**:
   - Needs more validation data
   - Risk of overfitting
   - Can change predicted class

Histogram Binning
^^^^^^^^^^^^^^^^^

**Best for**: Research baselines

Bins predictions and uses empirical frequencies:

.. code-block:: python

   from incerto.calibration import HistogramBinningCalibrator

   calibrator = HistogramBinningCalibrator(n_bins=10)
   calibrator.fit(val_logits, val_labels)

   calibrated_dist = calibrator.predict(test_logits)

Platt Scaling
^^^^^^^^^^^^^

**Best for**: Binary classification

Logistic regression per class (one-vs-rest for multiclass):

.. code-block:: python

   from incerto.calibration import PlattScalingCalibrator

   calibrator = PlattScalingCalibrator()
   calibrator.fit(val_logits, val_labels)

   calibrated_dist = calibrator.predict(test_logits)

Beta Calibration
^^^^^^^^^^^^^^^^

**Best for**: Binary classification

More flexible than Platt scaling, uses Beta distribution:

.. code-block:: python

   from incerto.calibration import BetaCalibrator

   # For binary classification
   logits_binary = torch.randn(100, 2)
   labels_binary = torch.randint(0, 2, (100,))

   calibrator = BetaCalibrator(method='mle')
   calibrator.fit(logits_binary, labels_binary)

   calibrated_dist = calibrator.predict(test_logits_binary)

**Reference**: Kull et al., "Beta calibration" (AISTATS 2017)

Training-Time Calibration
-------------------------

These methods train models to be calibrated from the start.

Label Smoothing
^^^^^^^^^^^^^^^

Prevents overconfidence by softening one-hot labels:

.. code-block:: python

   from incerto.calibration import LabelSmoothingLoss

   criterion = LabelSmoothingLoss(smoothing=0.1)

   # Training loop
   for epoch in range(epochs):
       for inputs, targets in train_loader:
           outputs = model(inputs)
           loss = criterion(outputs, targets)
           loss.backward()
           optimizer.step()

**Effect**: Targets become (1-ε) for correct class, ε/(C-1) for others

Focal Loss
^^^^^^^^^^

Focuses on hard examples, reduces overconfidence on easy examples:

.. code-block:: python

   from incerto.calibration import FocalLoss

   criterion = FocalLoss(gamma=2.0)  # gamma=0 → standard CE

   # Use in training loop
   loss = criterion(outputs, targets)

**Reference**: Lin et al., "Focal Loss" (ICCV 2017)

Confidence Penalty
^^^^^^^^^^^^^^^^^^

Explicitly penalizes overconfident predictions:

.. code-block:: python

   from incerto.calibration import ConfidencePenalty

   criterion = ConfidencePenalty(beta=0.1)

   loss = criterion(outputs, targets)

Complete Calibration Workflow
------------------------------

Here's a complete example showing the recommended workflow:

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader
   from incerto.calibration import (
       TemperatureScaling,
       ece_score,
       nll,
       brier_score
   )

   # 1. Train your model normally
   model = YourModel()
   # ... training code ...

   # 2. Split validation data: 50% for model selection, 50% for calibration
   val_loader_1, val_loader_2 = split_validation_data()

   # Use val_loader_1 for early stopping, model selection, etc.

   # 3. Get predictions on calibration set (val_loader_2)
   model.eval()
   all_logits, all_labels = [], []
   with torch.no_grad():
       for inputs, labels in val_loader_2:
           logits = model(inputs)
           all_logits.append(logits)
           all_labels.append(labels)

   cal_logits = torch.cat(all_logits)
   cal_labels = torch.cat(all_labels)

   # 4. Measure calibration before
   ece_before = ece_score(cal_logits, cal_labels)
   print(f"ECE before calibration: {ece_before:.4f}")

   # 5. Fit calibrator
   calibrator = TemperatureScaling()
   calibrator.fit(cal_logits, cal_labels)

   # 6. Save calibrator
   calibrator.save('calibrator.pt')

   # 7. Measure calibration after
   calibrated = calibrator.predict(cal_logits)
   ece_after = ece_score(calibrated.logits, cal_labels)
   print(f"ECE after calibration: {ece_after:.4f}")

   # 8. Use on test set
   test_logits = model(test_inputs)
   test_calibrated = calibrator.predict(test_logits)
   test_probs = test_calibrated.probs  # Use these for decisions

Visualization
-------------

Reliability diagrams show calibration visually:

.. code-block:: python

   from incerto.calibration import plot_reliability_diagram
   import matplotlib.pyplot as plt

   # Create reliability diagram
   fig, ax = plt.subplots(1, 2, figsize=(12, 5))

   # Before calibration
   plot_reliability_diagram(logits, labels, n_bins=10, ax=ax[0])
   ax[0].set_title('Before Calibration')

   # After calibration
   calibrated = calibrator.predict(logits)
   plot_reliability_diagram(calibrated.logits, labels, n_bins=10, ax=ax[1])
   ax[1].set_title('After Calibration')

   plt.tight_layout()
   plt.savefig('calibration_comparison.png')

A well-calibrated model shows points near the diagonal.

Best Practices
--------------

1. **Always use a separate calibration set**
      Never calibrate on training data or the same validation data used for early stopping

2. **Start with temperature scaling**
      It's simple, effective, and rarely overfits

3. **Use enough calibration data**
      - Temperature scaling: 500+ samples minimum
      - Vector scaling: 1000+ samples
      - Matrix scaling: 5000+ samples
      - Isotonic regression: 2000+ samples

4. **Evaluate on a held-out test set**
      Calibration can overfit just like accuracy

5. **Monitor multiple metrics**
      Use ECE, MCE, and NLL together for a complete picture

6. **Save calibrators with your models**
      Always deploy calibrated predictions to production

7. **Re-calibrate when data shifts**
      If deployment data differs from training, re-fit calibrator

Common Pitfalls
---------------

❌ **Calibrating on training data**
      Will appear perfect but won't generalize

❌ **Using too complex calibrators with little data**
      Matrix scaling with 500 samples will overfit

❌ **Ignoring class imbalance**
      Use classwise-ECE to detect per-class issues

❌ **Not saving calibrators**
      Model + calibrator should be versioned together

❌ **Expecting perfect calibration**
      ECE < 0.05 is excellent; perfect calibration is impossible

Comparison of Methods
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 20 30

   * - Method
     - Parameters
     - Data Needed
     - Preserves Argmax?
     - Best Use Case
   * - Temperature Scaling
     - 1
     - 500+
     - ✅ Yes
     - Default choice, most scenarios
   * - Vector Scaling
     - C
     - 1000+
     - ❌ No
     - Class imbalance
   * - Matrix Scaling
     - C²
     - 5000+
     - ❌ No
     - Lots of data, complex patterns
   * - Dirichlet
     - C²
     - 5000+
     - ❌ No
     - Research, distribution calibration
   * - Isotonic
     - Non-param
     - 2000+
     - ❌ No
     - Flexible, non-linear patterns
   * - Platt Scaling
     - 2C
     - 1000+
     - ❌ No
     - Binary classification
   * - Beta
     - Non-param
     - 1000+
     - ❌ No
     - Binary classification

References
----------

1. Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
2. Kull et al., "Beyond temperature scaling" (NeurIPS 2019)
3. Kull et al., "Beta calibration" (AISTATS 2017)
4. Müller et al., "When does label smoothing help?" (NeurIPS 2019)
5. Nixon et al., "Measuring Calibration in Deep Learning" (CVPR 2019)

See Also
--------

- :doc:`../api/calibration` - Complete API reference
- :doc:`quickstart` - Quick start guide
- :doc:`conformal` - Prediction sets with coverage guarantees
- :doc:`selective` - Selective prediction with abstention
