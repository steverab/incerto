LLM Uncertainty Quantification Guide
====================================

Large Language Models (LLMs) need special uncertainty quantification methods due to their autoregressive nature and scale.

Why LLM Uncertainty Matters
----------------------------

LLMs can be confidently wrong:
   - Hallucinations: Generate plausible but false information
   - Factual errors: Incorrect facts stated with high confidence
   - Out-of-domain: Uncertain when query is outside training

Uncertainty helps:
   - Detect hallucinations
   - Trigger fact-checking
   - Decide when to abstain

Unique Challenges
-----------------

**Autoregressive generation**:
   Uncertainty compounds over sequence

**No explicit probabilities**:
   Many models don't expose logits

**Scale**:
   Billions of parameters, expensive to run multiple times

**Calibration**:
   Model confidence often poor

Methods
-------

Token-Level Entropy
^^^^^^^^^^^^^^^^^^^

**Best for**: Simple baseline, available for any model

Measure entropy at each token:

.. code-block:: python

   from incerto.llm import TokenEntropy

   # Get logits from model
   logits = model(input_ids)  # (batch, seq_len, vocab)

   # Compute token-level entropy
   token_entropy = TokenEntropy.compute(logits)

   # Average entropy per sequence
   avg_entropy = token_entropy.mean(dim=1)

   print(f"Average entropy: {avg_entropy}")
   # Higher entropy = more uncertain

**Interpretation**:
   - Low entropy: Model confident in next token
   - High entropy: Model uncertain

Self-Consistency
^^^^^^^^^^^^^^^^

**Best for**: Question answering, factual queries

Sample multiple responses and measure agreement:

.. code-block:: python

   from incerto.llm import SelfConsistency

   # Generate multiple responses
   responses = [
       model.generate(prompt, do_sample=True, temperature=1.0)
       for _ in range(10)
   ]

   # Compute self-consistency metrics
   result = SelfConsistency.compute(responses)

   print(f"Agreement rate: {result['agreement_rate']:.2f}")
   print(f"Unique answers: {result['num_unique']}")
   print(f"Most common: {result['most_common']}")

   # High agreement = high confidence
   if result['agreement_rate'] > 0.7:
       print("Model is confident")
   else:
       print("Model is uncertain - consider fact-checking")

**Advantages**:
   - Model-agnostic
   - Captures semantic uncertainty
   - Effective for factual queries

**Disadvantages**:
   - Expensive (multiple generations)
   - Requires sampling capability

**Reference**: Wang et al., "Self-Consistency Improves Chain of Thought Reasoning" (ICLR 2023)

Semantic Entropy
^^^^^^^^^^^^^^^^

**Best for**: Measuring meaning-level uncertainty

Group semantically equivalent responses:

.. code-block:: python

   from incerto.llm import SemanticEntropy

   responses = generate_multiple_responses(prompt, n=10)

   # Compute semantic entropy (groups similar answers)
   semantic_ent = SemanticEntropy.compute(
       responses,
       similarity_threshold=0.8
   )

   print(f"Semantic entropy: {semantic_ent:.4f}")
   # Accounts for paraphrasing

Verbalized Uncertainty
^^^^^^^^^^^^^^^^^^^^^^

**Best for**: Modern chat models with instruction following

Ask model to express uncertainty:

.. code-block:: python

   from incerto.llm import VerbalizedUncertainty

   # Prompt model to express uncertainty
   prompt = """
   Answer the question and express your confidence (0-100%).

   Question: {question}
   Answer with format: [Answer] | Confidence: [0-100]
   """

   response = model.generate(prompt.format(question=query))

   # Parse confidence
   uncertainty = VerbalizedUncertainty.parse(response)

   print(f"Model confidence: {uncertainty['confidence']}")
   print(f"Answer: {uncertainty['answer']}")

**Advantages**:
   - Natural for chat models
   - Fast (single forward pass)

**Disadvantages**:
   - Requires instruction-tuned model
   - May be miscalibrated

Sequence-Level Metrics
^^^^^^^^^^^^^^^^^^^^^^

Aggregate token probabilities across sequence:

.. code-block:: python

   from incerto.llm import SequenceLevelUncertainty

   # Generate with log probabilities
   output = model.generate(
       input_ids,
       return_dict_in_generate=True,
       output_scores=True
   )

   # Compute sequence-level metrics
   metrics = SequenceLevelUncertainty.compute(output.scores)

   print(f"Mean log probability: {metrics['mean_logprob']:.4f}")
   print(f"Perplexity: {metrics['perplexity']:.4f}")
   print(f"Entropy: {metrics['entropy']:.4f}")

Complete Example
----------------

Uncertainty-aware generation:

.. code-block:: python

   from incerto.llm import SelfConsistency, TokenEntropy
   import torch

   def generate_with_uncertainty(model, prompt, n_samples=10):
       """Generate answer with uncertainty estimate."""

       # 1. Generate multiple responses
       responses = []
       all_entropies = []

       for _ in range(n_samples):
           output = model.generate(
               prompt,
               do_sample=True,
               temperature=1.0,
               return_dict_in_generate=True,
               output_scores=True
           )

           response = tokenizer.decode(output.sequences[0])
           responses.append(response)

           # Compute token entropy
           logits = torch.stack(output.scores, dim=1)
           entropy = TokenEntropy.compute(logits).mean()
           all_entropies.append(entropy.item())

       # 2. Measure self-consistency
       consistency = SelfConsistency.compute(responses)

       # 3. Aggregate uncertainties
       uncertainty_score = {
           'agreement_rate': consistency['agreement_rate'],
           'num_unique': consistency['num_unique'],
           'avg_entropy': sum(all_entropies) / len(all_entropies),
           'most_common_answer': consistency['most_common']
       }

       return uncertainty_score

   # Use it
   result = generate_with_uncertainty(model, "What is the capital of France?")

   if result['agreement_rate'] > 0.8:
       print(f"High confidence answer: {result['most_common_answer']}")
   else:
       print(f"Low confidence - {result['num_unique']} different answers")
       print("Consider fact-checking or abstaining")

Hallucination Detection
-----------------------

Combine uncertainty signals to detect hallucinations:

.. code-block:: python

   def detect_hallucination(model, prompt, claim):
       """Check if model's claim is likely hallucinated."""

       # 1. Generate multiple times
       responses = [
           model.generate(prompt, do_sample=True)
           for _ in range(10)
       ]

       # 2. Check consistency
       consistency = SelfConsistency.compute(responses)

       # 3. Check if claim appears in responses
       claim_freq = sum(claim in r for r in responses) / len(responses)

       # Decision logic
       if consistency['agreement_rate'] < 0.3:
           return "LIKELY HALLUCINATION - Low consistency"

       if claim_freq < 0.5:
           return "POSSIBLE HALLUCINATION - Claim not robust"

       return "LIKELY CORRECT - High confidence"

   # Example
   prompt = "Who won the Nobel Prize in Physics in 2099?"
   claim = "Alice Johnson"  # Hallucinated future event

   result = detect_hallucination(model, prompt, claim)
   print(result)  # "LIKELY HALLUCINATION"

Best Practices
--------------

1. **Use multiple uncertainty signals**
      Combine token entropy, self-consistency, perplexity

2. **Calibrate on validation set**
      LLM uncertainties often miscalibrated

3. **Consider computational cost**
      Self-consistency requires multiple generations

4. **Domain-specific thresholds**
      Factual queries vs. creative tasks need different thresholds

5. **Validate with human evaluation**
      Check if uncertainty correlates with correctness

6. **Use for selective generation**
      Abstain or request human review when uncertain

Applications
------------

**Question Answering**:
   Detect when model doesn't know answer

**Fact Checking**:
   Flag claims for verification

**Content Moderation**:
   Review uncertain generations manually

**Interactive Systems**:
   Ask clarifying questions when uncertain

**Retrieval-Augmented Generation**:
   Retrieve more context when uncertain

References
----------

1. Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in LLMs" (ICLR 2023)
2. Kadavath et al., "Language Models (Mostly) Know What They Know" (2022)
3. Kuhn et al., "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation" (ICLR 2023)
4. Lin et al., "Teaching Models to Express Their Uncertainty" (2022)

See Also
--------

- :doc:`../api/llm` - Complete API reference
- :doc:`calibration` - Calibration for language models
- :doc:`selective` - Selective generation
