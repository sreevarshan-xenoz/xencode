#!/usr/bin/env python3
"""
Optimized Token Voter for Xencode Ensemble System

Optimized version of the token voting algorithm with improved performance
and reduced computational overhead.
"""

import asyncio
import json
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("[yellow]⚠️ transformers not installed. Using fallback tokenization.[/yellow]")


class VotingStrategy(Enum):
    """Different voting strategies for token selection"""
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    FREQUENCY_BASED = "frequency_based"


@dataclass
class VotingResult:
    """Result of the token voting process"""
    token: str
    confidence: float
    votes: int
    total_responses: int


class OptimizedTokenVoter:
    """Optimized token voter with multiple voting strategies and performance improvements"""

    def __init__(self, tokenizer_name: str = "gpt2", strategy: VotingStrategy = VotingStrategy.MAJORITY):
        self.strategy = strategy
        self.tokenizer = None
        self.use_fallback = False

        if TOKENIZER_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    use_fast=True
                )
                print(f"[green]✅ Optimized tokenizer ready: {tokenizer_name}[/green]")
            except Exception as e:
                print(f"[yellow]⚠️ Failed to load tokenizer: {e}. Using fallback.[/yellow]")
                self.use_fallback = True
        else:
            self.use_fallback = True

    def vote_tokens(self, responses: List[str], weights: Optional[List[float]] = None,
                   strategy: Optional[VotingStrategy] = None) -> str:
        """
        Optimized token voting with multiple strategies.

        Args:
            responses: List of model responses
            weights: Optional weights for each model
            strategy: Voting strategy to use

        Returns:
            Fused response based on token voting
        """
        if not responses:
            return ""

        if len(responses) == 1:
            return responses[0]

        # Use provided strategy or default
        current_strategy = strategy or self.strategy
        weights = weights or [1.0] * len(responses)

        # Tokenize responses efficiently
        if self.tokenizer is not None:
            # Use vectorized tokenization for better performance
            tokenized_responses = self.tokenizer(responses, add_special_tokens=False, 
                                               padding=True, return_tensors="np")
            token_ids = tokenized_responses['input_ids']
        else:
            # Fallback: simple character-level tokenization
            tokenized_responses = [self._simple_tokenize(resp) for resp in responses]
            # Pad sequences to same length
            max_len = max(len(tokens) for tokens in tokenized_responses) if tokenized_responses else 0
            tokenized_responses = [tokens + ['<PAD>'] * (max_len - len(tokens)) for tokens in tokenized_responses]
            token_ids = np.array([[ord(c) if isinstance(c, str) and len(c) == 1 else hash(c) % 10000 
                                  for c in tokens] for tokens in tokenized_responses])

        # Perform voting based on strategy
        if current_strategy == VotingStrategy.MAJORITY:
            fused_tokens = self._majority_vote(token_ids, weights)
        elif current_strategy == VotingStrategy.WEIGHTED:
            fused_tokens = self._weighted_vote(token_ids, weights)
        elif current_strategy == VotingStrategy.CONFIDENCE_WEIGHTED:
            fused_tokens = self._confidence_weighted_vote(token_ids, weights)
        elif current_strategy == VotingStrategy.FREQUENCY_BASED:
            fused_tokens = self._frequency_based_vote(token_ids, weights)
        else:
            fused_tokens = self._majority_vote(token_ids, weights)  # Default fallback

        # Decode back to text
        if self.tokenizer is not None:
            try:
                # Filter out padding tokens before decoding
                non_pad_mask = fused_tokens != self.tokenizer.pad_token_id
                clean_tokens = fused_tokens[non_pad_mask]
                return self.tokenizer.decode(clean_tokens, skip_special_tokens=True)
            except Exception:
                # Fallback to simple joining
                return self._fallback_decode(fused_tokens)
        else:
            # For fallback tokenization, join the characters/strings
            return ''.join([chr(t) if 32 <= t <= 126 else '<UNK>' for t in fused_tokens if t != 0])

    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple character-level tokenization for fallback"""
        return list(text)

    def _majority_vote(self, token_ids: np.ndarray, weights: List[float]) -> np.ndarray:
        """Majority voting: select token that appears most often at each position"""
        num_positions = token_ids.shape[1]
        fused_tokens = np.zeros(num_positions, dtype=token_ids.dtype)

        for pos in range(num_positions):
            # Get all tokens at this position
            pos_tokens = token_ids[:, pos]
            
            # Count unique tokens and their frequencies
            unique_tokens, counts = np.unique(pos_tokens, return_counts=True)
            
            # Select the most frequent token
            if len(counts) > 0:
                max_idx = np.argmax(counts)
                fused_tokens[pos] = unique_tokens[max_idx]
            else:
                fused_tokens[pos] = 0  # Default to pad/empty

        return fused_tokens

    def _weighted_vote(self, token_ids: np.ndarray, weights: List[float]) -> np.ndarray:
        """Weighted voting based on model weights"""
        num_positions = token_ids.shape[1]
        num_models = token_ids.shape[0]
        fused_tokens = np.zeros(num_positions, dtype=token_ids.dtype)

        # Convert weights to numpy array for vectorized operations
        weights_array = np.array(weights).reshape(-1, 1)

        for pos in range(num_positions):
            # Get tokens at this position
            pos_tokens = token_ids[:, pos]
            
            # Create a weighted vote dictionary
            vote_weights = defaultdict(float)
            for i, token in enumerate(pos_tokens):
                vote_weights[token] += weights_array[i, 0]
            
            # Select token with highest total weight
            if vote_weights:
                best_token = max(vote_weights.items(), key=lambda x: x[1])[0]
                fused_tokens[pos] = best_token

        return fused_tokens

    def _confidence_weighted_vote(self, token_ids: np.ndarray, weights: List[float]) -> np.ndarray:
        """Vote considering both model weights and token confidence"""
        num_positions = token_ids.shape[1]
        fused_tokens = np.zeros(num_positions, dtype=token_ids.dtype)

        for pos in range(num_positions):
            pos_tokens = token_ids[:, pos]
            
            # Calculate weighted confidence for each unique token
            token_confidences = defaultdict(float)
            for i, token in enumerate(pos_tokens):
                # Combine model weight with positional confidence (simplified)
                confidence = weights[i] * (1.0 / (1.0 + pos * 0.01))  # Position penalty
                token_confidences[token] += confidence
            
            # Select token with highest confidence
            if token_confidences:
                best_token = max(token_confidences.items(), key=lambda x: x[1])[0]
                fused_tokens[pos] = best_token

        return fused_tokens

    def _frequency_based_vote(self, token_ids: np.ndarray, weights: List[float]) -> np.ndarray:
        """Vote based on frequency of token appearance across all positions"""
        # Calculate global token frequencies
        flat_tokens = token_ids.flatten()
        global_freq = Counter(flat_tokens)
        
        num_positions = token_ids.shape[1]
        fused_tokens = np.zeros(num_positions, dtype=token_ids.dtype)

        for pos in range(num_positions):
            pos_tokens = token_ids[:, pos]
            
            # Score tokens based on global frequency and local presence
            token_scores = {}
            for token in set(pos_tokens):
                if token in global_freq:
                    # Combine local presence and global frequency
                    local_present = np.sum(pos_tokens == token)
                    score = local_present * global_freq[token] * weights[np.where(pos_tokens == token)[0][0]]
                    token_scores[token] = score
            
            if token_scores:
                best_token = max(token_scores.items(), key=lambda x: x[1])[0]
                fused_tokens[pos] = best_token

        return fused_tokens

    def _fallback_decode(self, tokens: np.ndarray) -> str:
        """Fallback decoding when tokenizer fails"""
        # Try to convert tokens back to readable text
        result = []
        for token in tokens:
            if 32 <= token <= 126:  # Printable ASCII
                result.append(chr(token))
            elif token == 0:  # Padding
                continue
            else:
                result.append('?')
        return ''.join(result)

    def calculate_token_similarity(self, response1: str, response2: str) -> float:
        """Calculate token-level similarity between two responses using optimized method"""
        if not response1 or not response2:
            return 0.0

        if self.tokenizer is not None:
            try:
                tokens1 = set(self.tokenizer.encode(response1, add_special_tokens=False))
                tokens2 = set(self.tokenizer.encode(response2, add_special_tokens=False))
                intersection = len(tokens1 & tokens2)
                union = len(tokens1 | tokens2)
                return intersection / union if union > 0 else 0.0
            except Exception:
                # Fallback to simple method
                pass

        # Fallback: simple word-level similarity
        words1 = set(response1.split())
        words2 = set(response2.split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def batch_vote_tokens(self, response_batches: List[List[str]], 
                         weights_batch: Optional[List[List[float]]] = None) -> List[str]:
        """
        Batch process multiple sets of responses for efficiency.
        
        Args:
            response_batches: List of response sets to vote on
            weights_batch: Optional list of weight sets corresponding to each response set
            
        Returns:
            List of fused responses
        """
        results = []
        for i, responses in enumerate(response_batches):
            weights = weights_batch[i] if weights_batch and i < len(weights_batch) else None
            result = self.vote_tokens(responses, weights)
            results.append(result)
        return results


# Example usage and performance comparison
if __name__ == "__main__":
    import time
    
    # Sample responses for testing
    sample_responses = [
        "The weather is sunny today and it's perfect for a walk in the park.",
        "Today is a sunny day, ideal for outdoor activities and relaxation.",
        "It's a beautiful sunny day, great for spending time outdoors."
    ]
    
    print("🧪 Testing Optimized Token Voter...")
    
    # Test different strategies
    voter = OptimizedTokenVoter()
    
    strategies = [
        VotingStrategy.MAJORITY,
        VotingStrategy.WEIGHTED,
        VotingStrategy.CONFIDENCE_WEIGHTED,
        VotingStrategy.FREQUENCY_BASED
    ]
    
    for strategy in strategies:
        start_time = time.time()
        result = voter.vote_tokens(sample_responses, strategy=strategy)
        end_time = time.time()
        
        print(f"\n{strategy.value.upper()} Strategy:")
        print(f"Result: {result}")
        print(f"Time: {(end_time - start_time)*1000:.2f}ms")
    
    # Test batch processing
    print(f"\n📦 Testing batch processing...")
    start_time = time.time()
    batch_responses = [sample_responses] * 5  # Process 5 identical batches
    batch_weights = [[1.0, 1.2, 0.8]] * 5  # Different weights for each batch
    batch_results = voter.batch_vote_tokens(batch_responses, batch_weights)
    end_time = time.time()
    
    print(f"Processed {len(batch_results)} batches in {(end_time - start_time)*1000:.2f}ms")
    print(f"Average per batch: {(end_time - start_time)*1000/len(batch_results):.2f}ms")