Distribution Shift Detection Guide
==================================

Distribution shift detection identifies when deployment data differs from training data. Critical for maintaining model performance in production.

Why Shift Detection Matters
----------------------------

Model performance degrades when data distribution changes:

- **Covariate shift**: P(X) changes, P(Y|X) stays same
   Example: Camera angle changes but objects remain same

- **Label shift**: P(Y) changes, P(X|Y) stays same
   Example: Disease prevalence changes seasonally

- **Concept drift**: P(Y|X) changes
   Example: User preferences change over time

Detecting shifts enables proactive retraining and monitoring.

Two-Sample Tests
----------------

Test if two datasets come from same distribution.

Maximum Mean Discrepancy (MMD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Best for**: Non-parametric, works in any dimension

Kernel-based test measuring distance between distributions:

.. code-block:: python

   from incerto.shift import MMDShiftDetector

   detector = MMDShiftDetector(sigma=1.0)

   # Fit on reference (training) data
   detector.fit(reference_loader)

   # Compute shift score
   shift_score = detector.score(deployment_loader)

   print(f"MMD score: {shift_score:.4f}")
   # Higher score = more shift detected

   # Save for monitoring
   detector.save('mmd_detector.pt')

**Advantages**:
   - Non-parametric
   - Theoretically sound
   - Works well empirically

**Disadvantages**:
   - Requires choosing kernel bandwidth (sigma)
   - O(n²) complexity (can be approximated)

Energy Distance
^^^^^^^^^^^^^^^

**Best for**: Alternative to MMD, similar properties

.. code-block:: python

   from incerto.shift import EnergyShiftDetector

   detector = EnergyShiftDetector()
   detector.fit(reference_loader)

   shift_score = detector.score(deployment_loader)

Kolmogorov-Smirnov Test
^^^^^^^^^^^^^^^^^^^^^^^^

**Best for**: 1D features, interpretable

Per-feature univariate test:

.. code-block:: python

   from incerto.shift import KSShiftDetector

   detector = KSShiftDetector()
   detector.fit(reference_loader)

   # Returns maximum KS statistic across features
   shift_score = detector.score(deployment_loader)

**Advantages**:
   - Simple, interpretable
   - Exact p-values available

**Disadvantages**:
   - Only tests marginal distributions
   - Misses multivariate dependencies

Classifier Two-Sample Test
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Best for**: High dimensions, model-agnostic

Train classifier to distinguish reference from test data:

.. code-block:: python

   from incerto.shift import ClassifierShiftDetector

   detector = ClassifierShiftDetector()
   detector.fit(reference_loader)

   # Fit classifier and compute shift score
   shift_score = detector.score(deployment_loader)

   # Score is how well classifier can distinguish datasets
   # 0.5 = no shift (random guessing)
   # 1.0 = complete shift (perfect separation)

**Intuition**: If distributions differ, classifier can tell them apart

Label Shift Detection
---------------------

Detect and quantify label distribution changes:

.. code-block:: python

   from incerto.shift import LabelShiftDetector

   detector = LabelShiftDetector(num_classes=10)

   # Fit on source data
   detector.fit(model, source_loader, validation_loader)

   # Estimate target label distribution
   target_dist = detector.estimate_target_distribution(
       model,
       target_loader
   )

   print(f"Source dist: {detector.source_label_dist}")
   print(f"Target dist: {target_dist}")

   # Quantify shift magnitude
   shift_mag = detector.compute_shift_magnitude(
       model,
       target_loader,
       metric='tvd'  # Total variation distance
   )

   print(f"Label shift magnitude: {shift_mag:.4f}")

Importance Weighting
^^^^^^^^^^^^^^^^^^^^

Adapt to covariate shift via importance weighting:

.. code-block:: python

   from incerto.shift import ImportanceWeightingShift

   # Estimate importance weights
   iw = ImportanceWeightingShift(method='logistic', alpha=0.01)
   iw.fit(source_features, target_features)

   # Get weights for source samples
   weights = iw.compute_weights(source_features)

   # Use in training
   for x, y in source_loader:
       loss = model(x, y)
       weighted_loss = iw.weighted_loss(loss, weights)
       weighted_loss.backward()

Complete Monitoring Workflow
-----------------------------

.. code-block:: python

   from incerto.shift import MMDShiftDetector
   import time

   # 1. Train model and save reference data
   model = train_model(train_loader)
   detector = MMDShiftDetector(sigma=1.0)
   detector.fit(train_loader)  # Reference distribution

   # Save baseline
   detector.save('shift_detector.pt')

   # 2. Production monitoring loop
   shift_scores = []
   timestamps = []

   while True:
       # Get recent production data
       prod_data = get_recent_production_data()

       # Detect shift
       shift_score = detector.score(prod_data)

       shift_scores.append(shift_score)
       timestamps.append(time.time())

       # Alert if shift detected
       if shift_score > threshold:
           print(f"⚠️  Distribution shift detected!")
           print(f"Shift score: {shift_score:.4f}")
           # Trigger retraining, alert team, etc.

       # Wait before next check
       time.sleep(3600)  # Check hourly

Evaluation Metrics
------------------

**Shift score**:
   Method-specific score quantifying shift magnitude. Higher values indicate
   more distributional difference.

**Interpreting scores**:

.. code-block:: python

   from incerto.shift import MMDShiftDetector

   detector = MMDShiftDetector(sigma=1.0)
   detector.fit(reference_loader)

   # Score production data
   shift_score = detector.score(production_loader)

   # Thresholds depend on your data - calibrate on validation set
   if shift_score > calibrated_threshold:
       print("Significant shift detected!")

**Area Under Shift Curve**:
   For gradual shifts, plot shift score over time

Best Practices
--------------

1. **Establish baseline on clean training data**
      Reference distribution should be well-defined

2. **Monitor continuously**
      Shifts can be gradual or sudden

3. **Use multiple detectors**
      Different methods catch different shift types

4. **Set meaningful thresholds**
      Based on validation data, not test data

5. **Combine with performance monitoring**
      Shift detection + accuracy drop = retrain

6. **Save detector state**
      Allows consistent monitoring over time

7. **Consider feature-level monitoring**
      Track individual feature distributions

Common Shift Scenarios
----------------------

**Natural drift**:
   Gradual changes over time (user preferences, seasonality)

**Sudden shifts**:
   Abrupt changes (new data source, hardware change)

**Adversarial shifts**:
   Intentional distribution changes (attacks)

**Data quality issues**:
   Preprocessing bugs, sensor failures

References
----------

1. Gretton et al., "A Kernel Two-Sample Test" (JMLR 2012)
2. Lipton et al., "Detecting and Correcting for Label Shift" (ICML 2018)
3. Rabanser et al., "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift" (NeurIPS 2019)
4. Quinonero-Candela et al., "Dataset Shift in Machine Learning" (MIT Press 2009)

See Also
--------

- :doc:`../api/shift` - Complete API reference
- :doc:`ood` - Out-of-distribution detection
- :doc:`calibration` - Model calibration under shift
