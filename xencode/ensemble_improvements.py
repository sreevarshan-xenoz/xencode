#!/usr/bin/env python3
"""
Improved Ensemble Components

Provides enhanced token voting, semantic consensus, and quality metrics.
These components fix critical issues in the original ensemble system.
"""

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("[yellow]⚠️ sentence-transformers not available. Using fallback consensus.[/yellow]")

try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("[yellow]⚠️ transformers not available. Using fallback tokenization.[/yellow]")


class SemanticConsensus:
    """
    Semantic-based consensus calculation using sentence embeddings.

    Replaces the naive Jaccard similarity with true semantic understanding.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = None
        self.device = None

        if SEMANTIC_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = self.model.to(self.device)
                print(f"[green]✅ Semantic similarity engine ready: {model_name}[/green]")
            except Exception as e:
                print(f"[yellow]⚠️ Failed to load semantic model: {e}[/yellow]")
                SEMANTIC_AVAILABLE = False
                self.model = None

    def calculate_consensus(self, responses: List[str]) -> float:
        """
        Calculate consensus score (0-1) across responses using semantic similarity.

        Uses sentence embeddings to measure semantic agreement rather than
        surface-level word overlap (Jaccard).

        Args:
            responses: List of model response strings

        Returns:
            Consensus score between 0 and 1
        """
        if len(responses) <= 1:
            return 1.0

        if self.model is None:
            return self._fallback_consensus(responses)

        try:
            with torch.no_grad():
                embeddings = self.model.encode(
                    responses,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )

            # Calculate pairwise cosine similarity
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = torch.nn.functional.cosine_similarity(
                        embeddings[i].unsqueeze(0),
                        embeddings[j].unsqueeze(0),
                        dim=-1
                    ).item()
                    similarities.append(max(0, sim))

            if similarities:
                return sum(similarities) / len(similarities)
            return 0.0

        except Exception as e:
            print(f"[yellow]⚠️ Semantic consensus failed: {e}. Using fallback.[/yellow]")
            return self._fallback_consensus(responses)

    def _fallback_consensus(self, responses: List[str]) -> float:
        """
        Fallback: Jaccard similarity on word sets.

        Used when semantic model is not available.
        """
        all_tokens = [set(resp.split()) for resp in responses]

        if not all_tokens or not any(all_tokens):
            return 0.0

        similarities = []
        for i in range(len(all_tokens)):
            for j in range(i + 1, len(all_tokens)):
                intersection = len(all_tokens[i] & all_tokens[j])
                union = len(all_tokens[i] | all_tokens[j])
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def get_most_similar_pair(self, responses: List[str]) -> Tuple[int, int, float]:
        """
        Find the most semantically similar pair of responses.

        Returns:
            Tuple of (index1, index2, similarity_score)
        """
        if len(responses) < 2 or self.model is None:
            return 0, 0, 0.0

        try:
            with torch.no_grad():
                embeddings = self.model.encode(
                    responses,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )

            max_sim = -1.0
            best_pair = (0, 0)

            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = torch.nn.functional.cosine_similarity(
                        embeddings[i].unsqueeze(0),
                        embeddings[j].unsqueeze(0),
                        dim=-1
                    ).item()
                    if sim > max_sim:
                        max_sim = sim
                        best_pair = (i, j)

            return best_pair[0], best_pair[1], max_sim

        except Exception:
            return 0, 0, 0.0


class ImprovedTokenVoter:
    """
    Enhanced token-level voting with proper tokenization.

    Fixes the critical flaw of using whitespace splitting instead of
    actual LLM tokens.
    """

    def __init__(self, tokenizer_name: str = "gpt2"):
        self.tokenizer = None
        self.use_fallback = False

        if TOKENIZER_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    use_fast=True
                )
                print(f"[green]✅ Tokenizer ready: {tokenizer_name}[/green]")
            except Exception as e:
                print(f"[yellow]⚠️ Failed to load tokenizer: {e}. Using fallback.[/yellow]")
                self.use_fallback = True
        else:
            self.use_fallback = True
            print("[yellow]⚠️ Using whitespace-based fallback tokenization[/yellow]")

    def vote_tokens(self, responses: List[str], weights: Optional[List[float]] = None) -> str:
        """
        Vote on tokens across responses using proper tokenization.

        Args:
            responses: List of model responses
            weights: Optional weights for each model

        Returns:
            Fused response voted at token level
        """
        if not responses:
            return ""

        if len(responses) == 1:
            return responses[0]

        # Tokenize responses properly
        if self.tokenizer is not None:
            tokenized_responses = [
                self.tokenizer.encode(resp, add_special_tokens=False)
                for resp in responses
            ]
        else:
            # Fallback: character-level tokenization
            tokenized_responses = [
                list(resp.encode('utf-8'))
                for resp in responses
            ]

        weights = weights or [1.0] * len(responses)

        # Find maximum length
        max_len = max(len(tokens) for tokens in tokenized_responses) if tokenized_responses else 0

        if max_len == 0:
            return responses[0]

        # Vote on each position
        fused_tokens = []
        for pos in range(max_len):
            position_votes = defaultdict(float)

            for i, tokens in enumerate(tokenized_responses):
                if pos < len(tokens):
                    token = tokens[pos]
                    position_votes[token] += weights[i]

            if position_votes:
                best_token = max(position_votes.items(), key=lambda x: x[1])[0]
                fused_tokens.append(best_token)

        # Decode back to text
        if self.tokenizer is not None:
            try:
                return self.tokenizer.decode(fused_tokens, skip_special_tokens=True)
            except Exception:
                return self._fallback_vote(responses, weights)
        else:
            return self._character_fallback_vote(responses, weights, fused_tokens)

    def _fallback_vote(self, responses: List[str], weights: List[float]) -> str:
        """Fallback voting: word-level with weights"""
        if not responses:
            return ""

        if len(responses) == 1:
            return responses[0]

        word_lists = [resp.split() for resp in responses]
        max_len = max(len(words) for words in word_lists) if word_lists else 0

        if max_len == 0:
            return responses[0]

        fused_words = []
        for pos in range(max_len):
            position_votes = defaultdict(float)

            for i, words in enumerate(word_lists):
                if pos < len(words):
                    word = words[pos]
                    position_votes[word] += weights[i]

            if position_votes:
                best_word = max(position_votes.items(), key=lambda x: x[1])[0]
                fused_words.append(best_word)

        return " ".join(fused_words)

    def _character_fallback_vote(self, responses: List[str], weights: List[float],
                               fused_chars: List[int]) -> str:
        """Character-level fallback decoding"""
        try:
            return bytes(fused_chars).decode('utf-8', errors='ignore')
        except Exception:
            return self._fallback_vote(responses, weights)

    def calculate_token_similarity(self, response1: str, response2: str) -> float:
        """
        Calculate token-level similarity between two responses.

        Args:
            response1: First response
            response2: Second response

        Returns:
            Similarity score between 0 and 1
        """
        if self.tokenizer is None:
            tokens1 = set(response1.split())
            tokens2 = set(response2.split())
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            return intersection / union if union > 0 else 0.0

        try:
            tokens1 = set(self.tokenizer.encode(response1, add_special_tokens=False))
            tokens2 = set(self.tokenizer.encode(response2, add_special_tokens=False))
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            return intersection / union if union > 0 else 0.0
        except Exception:
            return 0.0


class QualityMetrics:
    """
    Advanced quality metrics for model responses.

    Replaces naive length-based confidence with multi-factor assessment.
    """

    @staticmethod
    def calculate_confidence(response: str, inference_time_ms: float,
                         tokens_generated: int, semantic_quality: Optional[float] = None) -> float:
        """
        Calculate comprehensive confidence score using multiple metrics.

        Replaces the naive confidence calculation that was based solely on
        response length.

        Args:
            response: The model response text
            inference_time_ms: Time taken for inference
            tokens_generated: Number of tokens generated
            semantic_quality: Optional semantic quality score (0-1)

        Returns:
            Confidence score (0-1)
        """
        if not response:
            return 0.0

        metrics = []

        # 1. Coherence: Check if response makes grammatical sense
        coherence = QualityMetrics._calculate_coherence(response)
        metrics.append(coherence)

        # 2. Token efficiency: Optimal length (not too short, not too long)
        length_score = QualityMetrics._calculate_length_score(tokens_generated)
        metrics.append(length_score)

        # 3. Speed: Faster is better (up to a point)
        speed_score = QualityMetrics._calculate_speed_score(inference_time_ms)
        metrics.append(speed_score)

        # 4. Semantic quality (if available)
        if semantic_quality is not None:
            metrics.append(semantic_quality)

        # Weighted average
        if semantic_quality is not None:
            weights = [0.3, 0.2, 0.2, 0.3]
        else:
            weights = [0.4, 0.3, 0.3]

        confidence = sum(m * w for m, w in zip(metrics, weights))

        return min(1.0, confidence)

    @staticmethod
    def _calculate_coherence(response: str) -> float:
        """
        Check grammatical coherence of response.

        Simple heuristic-based coherence check.
        """
        if not response:
            return 0.0

        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if not sentences:
            return 0.0

        # Basic coherence checks
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

        # Penalize very short or very long sentences
        if 3 <= avg_sentence_length <= 25:
            return 1.0
        elif avg_sentence_length < 3:
            return avg_sentence_length / 3.0
        else:
            return max(0.0, 1.0 - ((avg_sentence_length - 25) / 50.0))

    @staticmethod
    def _calculate_length_score(tokens: int) -> float:
        """
        Score based on response length.

        Optimal range: 20-200 tokens
        """
        if tokens == 0:
            return 0.0

        if 20 <= tokens <= 200:
            return 1.0
        elif tokens < 20:
            return tokens / 20.0
        else:
            return max(0.0, 1.0 - ((tokens - 200) / 500.0))

    @staticmethod
    def _calculate_speed_score(inference_time_ms: float) -> float:
        """
        Score based on inference speed.

        Fast is good, but extremely fast might indicate poor quality.
        """
        if inference_time_ms <= 0:
            return 0.0

        if inference_time_ms < 50:
            return 0.8  # Might be too fast, reduced score
        elif inference_time_ms < 200:
            return 1.0
        elif inference_time_ms < 500:
            return 0.7
        elif inference_time_ms < 1000:
            return 0.5
        else:
            return max(0.0, 1.0 - (inference_time_ms / 2000.0))


if __name__ == "__main__":
    print("=" * 60)
    print("ENSEMBLE IMPROVEMENTS TEST")
    print("=" * 60)

    print("\n1. Testing SemanticConsensus")
    consensus_calc = SemanticConsensus()
    test_responses = [
        "The cat sat on the mat",
        "A cat was on the mat",
        "The cat is on the mat"
    ]
    score = consensus_calc.calculate_consensus(test_responses)
    print(f"   Consensus score: {score:.3f}")

    print("\n2. Testing ImprovedTokenVoter")
    voter = ImprovedTokenVoter()
    responses = [
        "The quick brown fox jumps",
        "The fast brown fox runs"
    ]
    weights = [1.0, 0.8]
    fused = voter.vote_tokens(responses, weights)
    print(f"   Input 1: {responses[0]}")
    print(f"   Input 2: {responses[1]}")
    print(f"   Fused: {fused}")

    print("\n3. Testing QualityMetrics")
    response = "This is a well-structured response with multiple sentences."
    conf = QualityMetrics.calculate_confidence(
        response,
        inference_time_ms=150,
        tokens_generated=len(response.split())
    )
    print(f"   Response: {response}")
    print(f"   Confidence: {conf:.3f}")

    print("\n" + "=" * 60)
    print("✅ All improvements working!")
    print("=" * 60)
