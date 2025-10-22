Conformal Prediction Guide
==========================

Conformal prediction provides **prediction sets with statistical coverage guarantees**. Instead of a single prediction, you get a set of labels that contains the true label with high probability (e.g., 90%).

Why Conformal Prediction
-------------------------

Standard neural networks give you:
   - Single prediction (may be wrong)
   - Confidence score (often miscalibrated)

Conformal prediction gives you:
   - **Set of plausible labels**
   - **Provable coverage guarantee**: P(y ∈ C(x)) ≥ 1 - α

Key advantage: Coverage guarantee holds **without assumptions** about the data distribution.

Basic Concepts
--------------

**Miscoverage level (α)**:
   Desired error rate (e.g., α = 0.1 for 90% coverage)

**Prediction set C(x)**:
   Set of labels that might be correct

**Coverage guarantee**:
   True label is in the prediction set at least (1 - α) of the time

Example:
   With α = 0.1, prediction set {cat, dog, fox} contains true label ≥90% of the time

Inductive Conformal Prediction
-------------------------------

The most common approach. Split data into:
   - Training set: Train your model
   - Calibration set: Compute conformity scores
   - Test set: Make prediction sets

.. code-block:: python

   from incerto.conformal import inductive_conformal

   # Train model normally
   model = train_model(train_loader)

   # Create conformal predictor
   alpha = 0.1  # 90% coverage
   predictor = inductive_conformal(
       model,
       calibration_loader,
       alpha=alpha
   )

   # Get prediction sets
   for x, y in test_loader:
       pred_sets = predictor(x)

       # pred_sets[i] is a set of plausible labels for x[i]
       for i, pred_set in enumerate(pred_sets):
           print(f"Prediction set: {pred_set}")
           print(f"True label: {y[i].item()}")
           print(f"Covered: {y[i].item() in pred_set}")

Methods
-------

Score Functions
^^^^^^^^^^^^^^^

Different conformity scores lead to different prediction sets:

**Softmax score** (Adaptive Prediction Sets):

.. code-block:: python

   from incerto.conformal import APS

   predictor = APS(model, alpha=0.1)
   predictor.fit(calibration_loader)

   # Smaller sets, adapts to uncertainty
   prediction_sets = predictor.predict(test_data)

**Cumulative score** (RAPS):

.. code-block:: python

   from incerto.conformal import RAPS

   # Regularized Adaptive Prediction Sets
   predictor = RAPS(model, alpha=0.1, k_reg=2, lambda_reg=0.01)
   predictor.fit(calibration_loader)

   prediction_sets = predictor.predict(test_data)

Complete Workflow
-----------------

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader, random_split
   from incerto.conformal import APS

   # 1. Split data: train / calibration / test
   dataset = load_dataset()
   n = len(dataset)
   n_train = int(0.7 * n)
   n_cal = int(0.15 * n)
   n_test = n - n_train - n_cal

   train_data, cal_data, test_data = random_split(
       dataset, [n_train, n_cal, n_test]
   )

   train_loader = DataLoader(train_data, batch_size=32)
   cal_loader = DataLoader(cal_data, batch_size=32)
   test_loader = DataLoader(test_data, batch_size=32)

   # 2. Train model
   model = YourModel()
   train_model(model, train_loader)

   # 3. Create conformal predictor
   alpha = 0.1  # 90% coverage
   predictor = APS(model, alpha=alpha)

   # 4. Calibrate
   predictor.fit(cal_loader)

   # 5. Evaluate coverage
   covered, set_sizes = [], []
   for x, y in test_loader:
       pred_sets = predictor.predict(x)

       for i in range(len(y)):
           pred_set = pred_sets[i]
           covered.append(y[i].item() in pred_set)
           set_sizes.append(len(pred_set))

   coverage = sum(covered) / len(covered)
   avg_size = sum(set_sizes) / len(set_sizes)

   print(f"Empirical coverage: {coverage:.3f}")  # Should be ≥ 0.90
   print(f"Average set size: {avg_size:.2f}")

Regression
----------

For regression, predict **intervals** instead of sets:

.. code-block:: python

   from incerto.conformal import conformalized_quantile_regression

   # Predict quantiles
   model = QuantileRegressionModel()  # Predicts upper/lower quantiles
   model.train(train_loader)

   # Create conformal intervals
   intervals = conformalized_quantile_regression(
       model,
       calibration_loader,
       alpha=0.1
   )

   # Intervals contain true value ≥90% of the time
   for x, y in test_loader:
       lower, upper = intervals(x)
       print(f"Interval: [{lower:.2f}, {upper:.2f}]")
       print(f"True value: {y:.2f}")
       print(f"Covered: {lower <= y <= upper}")

Best Practices
--------------

1. **Use enough calibration data**
      At least 1000 samples for reliable coverage

2. **Don't tune α on test data**
      Choose α based on requirements, not test performance

3. **Monitor set sizes**
      Smaller is better (more informative)

4. **Combine with calibration**
      Well-calibrated models produce smaller sets

5. **Use exchangeability**
      Calibration and test data should be i.i.d.

Evaluation Metrics
------------------

**Coverage**:
   Fraction of test samples where true label is in prediction set

.. code-block:: python

   coverage = sum(y in pred_set for y, pred_set in zip(labels, pred_sets)) / len(labels)
   # Should be ≥ 1 - α

**Average set size**:
   How many labels in each set (smaller is better)

.. code-block:: python

   avg_size = sum(len(s) for s in pred_sets) / len(pred_sets)

References
----------

1. Vovk et al., "Algorithmic Learning in a Random World" (2005)
2. Papadopoulos et al., "Inductive Conformal Prediction" (2002)
3. Romano et al., "Classification with Valid and Adaptive Coverage" (NeurIPS 2020)
4. Angelopoulos & Bates, "Conformal Prediction: A Gentle Introduction" (2021)

See Also
--------

- :doc:`../api/conformal` - Complete API reference
- :doc:`calibration` - Calibration improves set sizes
- :doc:`selective` - Selective prediction
