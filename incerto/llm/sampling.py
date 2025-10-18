"""
Sampling-based uncertainty quantification for LLMs.

These methods generate multiple samples (e.g., with temperature sampling)
and measure disagreement/diversity to estimate uncertainty.
"""

from __future__ import annotations
from typing import List, Callable
import torch
import numpy as np
from collections import Counter


class SelfConsistency:
    """
    Self-consistency via majority voting across samples.

    Generate N samples and measure agreement. Higher agreement
    indicates lower uncertainty. Proposed by Wang et al. (2023).

    Reference:
        Wang et al., "Self-Consistency Improves Chain of Thought Reasoning"
        ICLR 2023.
    """

    @staticmethod
    def compute(
        responses: List[str],
        normalize_fn: Callable[[str], str] | None = None,
    ) -> dict:
        """
        Compute self-consistency from multiple responses.

        Args:
            responses: List of generated text responses
            normalize_fn: Optional function to normalize responses
                         (e.g., extract answers, lowercase, strip)

        Returns:
            Dictionary with:
                - agreement_rate: Fraction agreeing with majority
                - entropy: Entropy over response distribution
                - top_response: Most common response
                - num_unique: Number of unique responses
        """
        if normalize_fn is not None:
            responses = [normalize_fn(r) for r in responses]

        # Count occurrences
        counts = Counter(responses)
        total = len(responses)

        # Most common response and its frequency
        top_response, top_count = counts.most_common(1)[0]
        agreement_rate = top_count / total

        # Entropy over response distribution
        probs = np.array([count / total for count in counts.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        return {
            "agreement_rate": agreement_rate,
            "entropy": entropy,
            "top_response": top_response,
            "num_unique": len(counts),
            "confidence": agreement_rate,  # Alias
        }


class LexicalSimilarity:
    """
    Measure lexical similarity across samples.

    Compute exact match rate, token overlap, or edit distance
    to quantify how similar the generations are.
    """

    @staticmethod
    def exact_match_rate(responses: List[str]) -> float:
        """
        Compute fraction of responses that exactly match the most common.

        Args:
            responses: List of generated text responses

        Returns:
            Exact match rate (0-1)
        """
        counts = Counter(responses)
        top_count = counts.most_common(1)[0][1]
        return top_count / len(responses)

    @staticmethod
    def pairwise_token_overlap(responses: List[str]) -> float:
        """
        Average pairwise token overlap (Jaccard similarity).

        Args:
            responses: List of generated text responses

        Returns:
            Average Jaccard similarity across all pairs
        """
        if len(responses) < 2:
            return 1.0

        # Tokenize
        token_sets = [set(r.split()) for r in responses]

        # Compute pairwise Jaccard
        similarities = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                intersection = len(token_sets[i] & token_sets[j])
                union = len(token_sets[i] | token_sets[j])
                if union > 0:
                    similarities.append(intersection / union)

        return np.mean(similarities) if similarities else 0.0


class VarianceRatio:
    """
    Variance ratio for classification/multiple choice.

    Measures disagreement in predictions across samples.
    Defined as: VR = 1 - (most_common_count / total_samples)
    """

    @staticmethod
    def compute(predictions: List[int]) -> float:
        """
        Compute variance ratio.

        Args:
            predictions: List of predicted class indices

        Returns:
            Variance ratio (0-1), higher = more uncertainty
        """
        counts = Counter(predictions)
        top_count = counts.most_common(1)[0][1]
        return 1.0 - (top_count / len(predictions))


class PredictiveEntropy:
    """
    Predictive entropy across multiple sampled sequences.

    Average the probability distributions from multiple samples
    and compute entropy. Higher values indicate disagreement.
    """

    @staticmethod
    def compute(logit_samples: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute predictive entropy from multiple samples.

        Args:
            logit_samples: List of logit tensors, each of shape
                          (seq_len, vocab_size) or (vocab_size,)

        Returns:
            Predictive entropy (scalar or per-position)
        """
        import torch.nn.functional as F

        # Stack samples
        logits_stacked = torch.stack(logit_samples, dim=0)  # (n_samples, ...)

        # Average probabilities across samples
        probs = F.softmax(logits_stacked, dim=-1)
        mean_probs = probs.mean(dim=0)

        # Compute entropy
        log_mean_probs = torch.log(mean_probs + 1e-10)
        entropy = -(mean_probs * log_mean_probs).sum(dim=-1)

        return entropy


class MutualInformation:
    """
    Mutual information between predictions and model (aleatoric vs epistemic).

    MI = E[H(y|x, θ)] - H(E[y|x, θ])
       = Expected entropy - Entropy of expected distribution

    High MI indicates epistemic uncertainty (model uncertainty).
    """

    @staticmethod
    def compute(logit_samples: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute mutual information.

        Args:
            logit_samples: List of logit tensors from different samples

        Returns:
            Mutual information value
        """
        import torch.nn.functional as F

        logits_stacked = torch.stack(logit_samples, dim=0)
        probs = F.softmax(logits_stacked, dim=-1)

        # Expected entropy: E[H(y|x, θ)]
        log_probs = F.log_softmax(logits_stacked, dim=-1)
        entropies = -(probs * log_probs).sum(dim=-1)
        expected_entropy = entropies.mean(dim=0)

        # Entropy of expected: H(E[y|x, θ])
        mean_probs = probs.mean(dim=0)
        log_mean_probs = torch.log(mean_probs + 1e-10)
        entropy_of_expected = -(mean_probs * log_mean_probs).sum(dim=-1)

        # Mutual information
        mi = expected_entropy - entropy_of_expected

        return mi


class SemanticEntropy:
    """
    Semantic entropy - entropy over semantically clustered responses.

    Clusters responses by meaning (not exact text) and computes
    entropy. Requires a semantic similarity model.

    Reference:
        Kuhn et al., "Semantic Uncertainty: Linguistic Invariances for
        Uncertainty Estimation in Natural Language Generation", ICLR 2023.
    """

    @staticmethod
    def compute(
        responses: List[str],
        similarity_threshold: float = 0.85,
        embedding_model=None,
    ) -> dict:
        """
        Compute semantic entropy by clustering similar responses.

        Args:
            responses: List of generated text responses
            similarity_threshold: Threshold for considering responses similar
            embedding_model: Optional sentence embedding model (e.g., SentenceTransformer)
                           If None, falls back to lexical similarity

        Returns:
            Dictionary with:
                - semantic_entropy: Entropy over semantic clusters
                - num_clusters: Number of semantic clusters found
                - clusters: List of cluster assignments
        """
        if len(responses) == 0:
            return {"semantic_entropy": 0.0, "num_clusters": 0, "clusters": []}

        if len(responses) == 1:
            return {"semantic_entropy": 0.0, "num_clusters": 1, "clusters": [0]}

        # Compute pairwise similarities
        if embedding_model is not None:
            # Use semantic embeddings
            embeddings = embedding_model.encode(responses)
            from sklearn.metrics.pairwise import cosine_similarity

            similarities = cosine_similarity(embeddings)
        else:
            # Fallback to lexical similarity
            n = len(responses)
            similarities = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        similarities[i, j] = 1.0
                    else:
                        # Simple word overlap
                        words_i = set(responses[i].split())
                        words_j = set(responses[j].split())
                        if len(words_i | words_j) > 0:
                            similarities[i, j] = len(words_i & words_j) / len(
                                words_i | words_j
                            )

        # Cluster using threshold
        clusters = [-1] * len(responses)
        cluster_id = 0

        for i in range(len(responses)):
            if clusters[i] == -1:
                # Start new cluster
                clusters[i] = cluster_id
                # Add similar responses to same cluster
                for j in range(i + 1, len(responses)):
                    if similarities[i, j] >= similarity_threshold:
                        clusters[j] = cluster_id
                cluster_id += 1

        # Count cluster sizes
        cluster_counts = Counter(clusters)
        cluster_probs = np.array(
            [count / len(responses) for count in cluster_counts.values()]
        )

        # Compute entropy
        semantic_entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))

        return {
            "semantic_entropy": semantic_entropy,
            "num_clusters": len(cluster_counts),
            "clusters": clusters,
        }


class EnsembleDisagreement:
    """
    Disagreement rate across an ensemble of models or sampling strategies.

    Measures how often different samples/models produce different outputs.
    """

    @staticmethod
    def compute(predictions: List[List[int]]) -> float:
        """
        Compute disagreement rate.

        Args:
            predictions: List of prediction lists, each containing
                        predicted classes for different examples

        Returns:
            Disagreement rate (0-1)
        """
        if len(predictions) < 2:
            return 0.0

        n_samples = len(predictions[0])
        disagreements = 0

        for i in range(n_samples):
            # Get all predictions for this sample
            sample_preds = [preds[i] for preds in predictions]
            # Check if all agree
            if len(set(sample_preds)) > 1:
                disagreements += 1

        return disagreements / n_samples
