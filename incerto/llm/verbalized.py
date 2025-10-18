"""
Verbalized uncertainty for LLMs.

These methods use prompting to elicit uncertainty estimates directly
from the language model through natural language.
"""

from __future__ import annotations
from typing import Callable
import re


class VerbalizedConfidence:
    """
    Ask the model to verbalize its confidence.

    Prompt: "How confident are you in this answer? (0-100%)"
    Extract and return the confidence score.
    """

    @staticmethod
    def extract_percentage(response: str) -> float | None:
        """
        Extract a percentage value from text.

        Args:
            response: Text containing a percentage

        Returns:
            Confidence value (0-1) or None if not found
        """
        # Look for patterns like "85%", "85 percent", "0.85"
        patterns = [
            r"(\d+(?:\.\d+)?)\s*%",  # 85%
            r"(\d+(?:\.\d+)?)\s*percent",  # 85 percent
            r"confidence.*?(\d+(?:\.\d+)?)",  # confidence: 85
            r"(\d+(?:\.\d+)?)(?:/100)?",  # 85 or 85/100
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                # Normalize to 0-1
                if value > 1.0:
                    value = value / 100.0
                return max(0.0, min(1.0, value))

        return None

    @staticmethod
    def get_confidence_prompt(question: str, answer: str) -> str:
        """
        Generate a prompt to elicit confidence.

        Args:
            question: The original question
            answer: The model's answer

        Returns:
            Prompt string
        """
        return f"""Question: {question}
Your answer: {answer}

On a scale from 0% to 100%, how confident are you that your answer is correct?
Provide only the percentage value."""


class PTrue:
    """
    P(True) - asking the model the probability its answer is correct.

    Prompt: "What is the probability that your answer is correct?"
    """

    @staticmethod
    def get_ptrue_prompt(question: str, answer: str) -> str:
        """
        Generate P(True) prompt.

        Args:
            question: The original question
            answer: The model's answer

        Returns:
            Prompt string
        """
        return f"""Question: {question}
Your answer: {answer}

What is the probability (between 0.0 and 1.0) that your answer is correct?
Provide only the numerical probability value."""

    @staticmethod
    def extract_probability(response: str) -> float | None:
        """
        Extract probability value from response.

        Args:
            response: Text containing a probability

        Returns:
            Probability value (0-1) or None
        """
        # Look for decimal numbers between 0 and 1
        patterns = [
            r"(\d+\.\d+)",  # 0.85
            r"(\d+)%",  # 85%
            r"(\d+)\s+percent",  # 85 percent
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if value > 1.0:
                    value = value / 100.0
                return max(0.0, min(1.0, value))

        return None


class SelfEvaluation:
    """
    Multi-turn self-critique for uncertainty.

    Ask the model to evaluate its own answer and look for inconsistencies.
    """

    @staticmethod
    def get_critique_prompt(question: str, answer: str) -> str:
        """
        Generate self-critique prompt.

        Args:
            question: The original question
            answer: The model's answer

        Returns:
            Prompt for self-evaluation
        """
        return f"""Question: {question}
Proposed answer: {answer}

Please critically evaluate this answer. Consider:
1. Is it factually accurate?
2. Is it complete?
3. Are there any potential errors or ambiguities?
4. What is the likelihood this answer is correct?

Provide your evaluation and a confidence score (0-100%)."""


class BidirectionalConsistency:
    """
    Check consistency by asking the question in different ways.

    If the model gives different answers to equivalent questions,
    it indicates high uncertainty.
    """

    @staticmethod
    def paraphrase_prompts(question: str) -> list[str]:
        """
        Generate paraphrased versions of a question.

        Args:
            question: Original question

        Returns:
            List of paraphrased questions
        """
        # Simple template-based paraphrasing
        # In practice, you'd use a paraphrasing model
        paraphrases = [
            question,
            f"Can you tell me: {question}",
            f"What would you say about: {question}",
            f"I'd like to know: {question}",
        ]
        return paraphrases

    @staticmethod
    def compute_consistency(answers: list[str]) -> float:
        """
        Compute consistency across answers.

        Args:
            answers: List of answers to paraphrased questions

        Returns:
            Consistency score (0-1), higher = more consistent
        """
        if len(answers) < 2:
            return 1.0

        # Simple exact match
        unique_answers = len(set(answers))
        consistency = 1.0 - ((unique_answers - 1) / (len(answers) - 1))

        return consistency
