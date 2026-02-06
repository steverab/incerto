Quick Start
===========

This guide provides quick examples to get you started with incerto's main features.

Calibration
-----------

Calibrate your model's confidence scores to match empirical accuracy:

.. code-block:: python

   import torch
   from incerto.calibration import TemperatureScaling, ece_score

   # Assume you have a trained model and validation data
   model = YourModel()
   val_logits, val_labels = get_validation_data()

   # Post-hoc calibration
   calibrator = TemperatureScaling()
   calibrator.fit(val_logits, val_labels)

   # Get calibrated predictions
   test_logits = model(test_data)
   calibrated = calibrator.predict(test_logits)

   # Evaluate calibration
   ece_before = ece_score(test_logits, test_labels)
   ece_after = ece_score(calibrated.logits, test_labels)
   print(f"ECE before: {ece_before:.4f}, after: {ece_after:.4f}")

Out-of-Distribution Detection
------------------------------

Detect when inputs are outside the training distribution:

.. code-block:: python

   from incerto.ood import Energy
   from incerto.ood import auroc

   # Initialize OOD detector
   detector = Energy(model, temperature=1.0)

   # Get scores (higher = more OOD)
   id_scores = detector.score(in_distribution_data)
   ood_scores = detector.score(out_of_distribution_data)

   # Evaluate
   auc = auroc(id_scores, ood_scores)
   print(f"OOD Detection AUROC: {auc:.4f}")

Conformal Prediction
--------------------

Generate prediction sets with coverage guarantees:

.. code-block:: python

   from incerto.conformal import inductive_conformal

   # Calibrate conformal predictor
   alpha = 0.1  # Desired coverage: 1 - alpha = 90%
   predictor = inductive_conformal(
       model,
       calibration_loader,
       alpha=alpha
   )

   # Get prediction sets
   prediction_sets = predictor(test_data)

   # Each set is guaranteed to contain the true label
   # with probability >= 90%
   for i, pred_set in enumerate(prediction_sets[:5]):
       print(f"Sample {i}: Classes {pred_set}")

Selective Prediction
--------------------

Abstain from uncertain predictions:

.. code-block:: python

   from incerto.sp import SoftmaxThreshold

   # Create selective predictor (wraps your trained model)
   selector = SoftmaxThreshold(model)
   selector.eval()

   # Get logits and confidence scores
   with torch.no_grad():
       logits, confidence = selector(test_data, return_confidence=True)

   predictions = logits.argmax(dim=-1)

   # Reject low-confidence predictions
   threshold = 0.9
   rejected = selector.reject(confidence, threshold)
   coverage = (~rejected).float().mean()
   acc = (predictions[~rejected] == labels[~rejected]).float().mean()
   print(f"Coverage: {coverage:.2%}, Selective accuracy: {acc:.4f}")

LLM Uncertainty
---------------

Quantify uncertainty in language model outputs:

.. code-block:: python

   from incerto.llm import TokenEntropy, SelfConsistency

   # Token-level uncertainty
   logits = llm(input_ids)  # shape: (batch, seq_len, vocab)
   token_entropy = TokenEntropy.compute(logits)

   # Sampling-based uncertainty
   responses = [
       llm.generate(prompt, do_sample=True, temperature=1.0)
       for _ in range(10)
   ]

   sc_result = SelfConsistency.compute(responses)
   print(f"Agreement rate: {sc_result['agreement_rate']:.2f}")
   print(f"Unique answers: {sc_result['num_unique']}")

Bayesian Deep Learning
----------------------

Approximate Bayesian inference for epistemic uncertainty:

.. code-block:: python

   from incerto.bayesian import MCDropout

   # Enable dropout during inference
   mc_dropout = MCDropout(model, num_samples=10)

   # Get uncertainty estimates (returns tuple, not dict)
   mean_pred, variance = mc_dropout.predict(test_data)

   print(f"Mean predictions shape: {mean_pred.shape}")
   print(f"Average predictive variance: {variance.mean():.4f}")

Active Learning
---------------

Select informative samples for labeling:

.. code-block:: python

   from incerto.active import EntropyAcquisition, UncertaintySampling

   # Initialize active learning strategy
   strategy = UncertaintySampling(
       model,
       acquisition_fn=EntropyAcquisition()
   )

   # Query most uncertain samples
   unlabeled_pool = get_unlabeled_data()
   query_indices = strategy.query(
       unlabeled_pool,
       n_samples=100
   )

   # Label and retrain with selected samples
   labeled_samples = unlabeled_pool[query_indices]

Distribution Shift Detection
----------------------------

Detect shifts between training and deployment data:

.. code-block:: python

   from incerto.shift import MMDShiftDetector

   # Create MMD shift detector
   detector = MMDShiftDetector(sigma=1.0)

   # Fit on reference (training) distribution
   detector.fit(train_loader)

   # Score production data (higher = more shift)
   shift_score = detector.score(production_loader)
   baseline = detector.score(train_loader)

   print(f"MMD shift score: {shift_score:.6f}")
   print(f"Baseline score: {baseline:.6f}")

   if shift_score > 2 * baseline:
       print("Significant distribution shift detected!")

Saving and Loading Models
--------------------------

All calibrators, OOD detectors, and shift detectors support serialization:

.. code-block:: python

   from incerto.calibration import TemperatureScaling
   from incerto.ood import Energy
   from incerto.shift import MMDShiftDetector

   # Calibrator example
   calibrator = TemperatureScaling()
   calibrator.fit(val_logits, val_labels)

   # Save to file
   calibrator.save('calibrator.pt')

   # Load from file (need to create instance first)
   new_calibrator = TemperatureScaling()
   new_calibrator.load_state_dict(torch.load('calibrator.pt', weights_only=True))

   # Or use state_dict directly
   state = calibrator.state_dict()
   torch.save(state, 'calibrator_state.pt')

   # OOD detector example (model not saved, only detector state)
   detector = Energy(model, temperature=2.0)
   detector.save('detector_state.pt')

   # When loading, provide the model again
   loaded_detector = Energy.load('detector_state.pt', model)

   # Shift detector example
   shift_detector = MMDShiftDetector(sigma=1.5)
   shift_detector.fit(reference_loader)
   shift_detector.save('shift_detector.pt')

   # Load using class method
   loaded_shift = MMDShiftDetector.load('shift_detector.pt')

Next Steps
----------

- Explore detailed guides for each module
- Check out the :doc:`../examples/calibration` and other examples
- Read the :doc:`../api/calibration` reference
