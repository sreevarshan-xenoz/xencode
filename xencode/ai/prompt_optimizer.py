"""
Dynamic Prompt Optimization Engine
Implements PromptOptimizer class with context analysis, real-time performance feedback,
A/B testing framework for prompt effectiveness, and multi-model prompt adaptation.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import time
import uuid
from collections import defaultdict
import statistics


logger = logging.getLogger(__name__)


class PromptOptimizationStrategy(Enum):
    """Different strategies for prompt optimization."""
    CONTEXT_AWARE = "context_aware"
    PERFORMANCE_BASED = "performance_based"
    ADAPTIVE_LEARNING = "adaptive_learning"
    MULTI_MODEL_ADAPTIVE = "multi_model_adaptive"


@dataclass
class PromptMetrics:
    """Metrics collected for a prompt."""
    prompt_id: str
    timestamp: float
    response_time: float
    success_rate: float
    relevance_score: float
    model_used: str
    context_length: int
    token_usage: int


@dataclass
class OptimizedPrompt:
    """Result of prompt optimization."""
    original_prompt: str
    optimized_prompt: str
    strategy_used: PromptOptimizationStrategy
    confidence_score: float
    metrics: PromptMetrics


class PromptPerformanceTracker:
    """Tracks and analyzes prompt performance over time."""
    
    def __init__(self):
        self.metrics_history: List[PromptMetrics] = []
        self.prompt_stats: Dict[str, List[PromptMetrics]] = defaultdict(list)
        
    def record_metrics(self, metrics: PromptMetrics):
        """Record performance metrics for a prompt."""
        self.metrics_history.append(metrics)
        self.prompt_stats[metrics.prompt_id].append(metrics)
        
    def get_average_response_time(self, prompt_id: str) -> Optional[float]:
        """Get average response time for a specific prompt."""
        if prompt_id not in self.prompt_stats:
            return None
        times = [m.response_time for m in self.prompt_stats[prompt_id]]
        return sum(times) / len(times) if times else None
        
    def get_success_rate(self, prompt_id: str) -> Optional[float]:
        """Get success rate for a specific prompt."""
        if prompt_id not in self.prompt_stats:
            return None
        rates = [m.success_rate for m in self.prompt_stats[prompt_id]]
        return sum(rates) / len(rates) if rates else None
        
    def get_relevance_score(self, prompt_id: str) -> Optional[float]:
        """Get average relevance score for a specific prompt."""
        if prompt_id not in self.prompt_stats:
            return None
        scores = [m.relevance_score for m in self.prompt_stats[prompt_id]]
        return sum(scores) / len(scores) if scores else None


class ABTestingFramework:
    """A/B testing framework for comparing prompt effectiveness."""
    
    def __init__(self):
        self.experiments: Dict[str, Dict[str, List[PromptMetrics]]] = {}
        
    def start_experiment(self, experiment_id: str, prompt_variants: List[str]):
        """Start a new A/B testing experiment."""
        self.experiments[experiment_id] = {variant: [] for variant in prompt_variants}
        
    def record_result(self, experiment_id: str, prompt_variant: str, metrics: PromptMetrics):
        """Record results for a specific prompt variant in an experiment."""
        if experiment_id in self.experiments and prompt_variant in self.experiments[experiment_id]:
            self.experiments[experiment_id][prompt_variant].append(metrics)
            
    def get_best_variant(self, experiment_id: str) -> Optional[str]:
        """Get the best performing prompt variant from an experiment."""
        if experiment_id not in self.experiments:
            return None
            
        best_variant = None
        best_score = float('-inf')
        
        for variant, metrics_list in self.experiments[experiment_id].items():
            if not metrics_list:
                continue
                
            avg_relevance = statistics.mean([m.relevance_score for m in metrics_list])
            avg_response_time = statistics.mean([m.response_time for m in metrics_list])
            
            # Calculate composite score (higher relevance, lower response time is better)
            # Normalize response time to 0-1 scale (lower is better)
            max_response_time = max([m.response_time for m in self.metrics_history]) if hasattr(self, 'metrics_history') else 10.0
            normalized_response_time = 1 - (avg_response_time / max_response_time) if max_response_time > 0 else 0
            composite_score = (avg_relevance * 0.7) + (normalized_response_time * 0.3)
            
            if composite_score > best_score:
                best_score = composite_score
                best_variant = variant
                
        return best_variant


class ContextAnalyzer:
    """Analyzes context to optimize prompts accordingly."""
    
    def __init__(self):
        self.context_keywords = {
            'technical': ['algorithm', 'code', 'programming', 'function', 'class', 'variable'],
            'creative': ['story', 'narrative', 'character', 'plot', 'scene', 'dialogue'],
            'analytical': ['analyze', 'compare', 'evaluate', 'assess', 'examine', 'review'],
            'instructional': ['how', 'steps', 'process', 'guide', 'tutorial', 'instructions']
        }
        
    def analyze_context(self, context: str) -> Dict[str, float]:
        """Analyze context and return relevance scores for different domains."""
        context_lower = context.lower()
        scores = {}
        
        for domain, keywords in self.context_keywords.items():
            score = sum(1 for keyword in keywords if keyword in context_lower)
            scores[domain] = score / len(keywords)  # Normalize score
            
        return scores


class PromptOptimizer:
    """
    Dynamic prompt optimization engine with context analysis,
    real-time performance feedback, and A/B testing framework.
    """
    
    def __init__(self):
        self.performance_tracker = PromptPerformanceTracker()
        self.ab_testing_framework = ABTestingFramework()
        self.context_analyzer = ContextAnalyzer()
        self.optimization_strategies = {
            PromptOptimizationStrategy.CONTEXT_AWARE: self._context_aware_optimization,
            PromptOptimizationStrategy.PERFORMANCE_BASED: self._performance_based_optimization,
            PromptOptimizationStrategy.ADAPTIVE_LEARNING: self._adaptive_learning_optimization,
            PromptOptimizationStrategy.MULTI_MODEL_ADAPTIVE: self._multi_model_adaptive_optimization
        }
        
    async def optimize_prompt(
        self, 
        prompt: str, 
        context: Optional[str] = None, 
        strategy: PromptOptimizationStrategy = PromptOptimizationStrategy.CONTEXT_AWARE,
        model_name: str = "default"
    ) -> OptimizedPrompt:
        """
        Optimize a prompt based on the selected strategy.
        
        Args:
            prompt: The original prompt to optimize
            context: Additional context for optimization
            strategy: The optimization strategy to use
            model_name: Name of the model that will process the prompt
            
        Returns:
            OptimizedPrompt object containing the optimized prompt and metadata
        """
        start_time = time.time()
        
        # Apply the selected optimization strategy
        optimized_prompt = await self.optimization_strategies[strategy](prompt, context, model_name)
        
        # Generate metrics for this optimization
        response_time = time.time() - start_time
        metrics = PromptMetrics(
            prompt_id=str(uuid.uuid4()),
            timestamp=time.time(),
            response_time=response_time,
            success_rate=0.95,  # Placeholder - would come from actual model response
            relevance_score=0.85,  # Placeholder - would come from evaluation
            model_used=model_name,
            context_length=len(context or ""),
            token_usage=len(prompt.split())  # Rough estimate
        )
        
        # Record metrics for future optimization
        self.performance_tracker.record_metrics(metrics)
        
        return OptimizedPrompt(
            original_prompt=prompt,
            optimized_prompt=optimized_prompt,
            strategy_used=strategy,
            confidence_score=0.9,  # Placeholder confidence score
            metrics=metrics
        )
        
    async def _context_aware_optimization(self, prompt: str, context: Optional[str], model_name: str) -> str:
        """Apply context-aware optimization to the prompt."""
        if not context:
            return prompt
            
        context_scores = self.context_analyzer.analyze_context(context)
        dominant_domain = max(context_scores, key=context_scores.get)
        
        # Adjust prompt based on detected context domain
        if dominant_domain == 'technical':
            optimized_prompt = f"{prompt}\n\nFormat your response with clear code examples and technical precision."
        elif dominant_domain == 'creative':
            optimized_prompt = f"{prompt}\n\nMake your response creative and engaging with vivid descriptions."
        elif dominant_domain == 'analytical':
            optimized_prompt = f"{prompt}\n\nProvide a thorough analysis with comparisons and evaluations."
        elif dominant_domain == 'instructional':
            optimized_prompt = f"{prompt}\n\nStructure your response as clear, step-by-step instructions."
        else:
            optimized_prompt = prompt  # No specific optimization needed
            
        return optimized_prompt
        
    async def _performance_based_optimization(self, prompt: str, context: Optional[str], model_name: str) -> str:
        """Apply performance-based optimization based on historical metrics."""
        # This would typically involve looking up similar prompts and applying known optimizations
        # For now, we'll implement a simple length-based optimization
        if len(prompt) > 200:
            # Shorten overly verbose prompts
            words = prompt.split()
            optimized_prompt = ' '.join(words[:100]) + "..." if len(words) > 100 else prompt
        else:
            optimized_prompt = prompt
            
        return optimized_prompt
        
    async def _adaptive_learning_optimization(self, prompt: str, context: Optional[str], model_name: str) -> str:
        """Apply adaptive learning optimization based on past performance."""
        # This would involve using ML techniques to learn from past prompt-performance relationships
        # For now, we'll implement a simple version that adjusts based on model name
        if 'gpt-4' in model_name.lower():
            optimized_prompt = f"{prompt}\n\nGiven your advanced capabilities, provide a comprehensive and nuanced response."
        elif 'gpt-3.5' in model_name.lower():
            optimized_prompt = f"{prompt}\n\nProvide a clear and concise response."
        else:
            optimized_prompt = prompt
            
        return optimized_prompt
        
    async def _multi_model_adaptive_optimization(self, prompt: str, context: Optional[str], model_name: str) -> str:
        """Apply optimization tailored for multi-model environments."""
        # This would adjust prompts based on known characteristics of different models
        model_specific_adjustments = {
            'claude': lambda p: f"{p}\n\nPlease think step by step before providing your final answer.",
            'llama': lambda p: f"[INST]{p}[/INST]",
            'mixtral': lambda p: f"Question: {p}\nAnswer:",
            'default': lambda p: p
        }
        
        adjustment_func = model_specific_adjustments.get(model_name.lower(), model_specific_adjustments['default'])
        return adjustment_func(prompt)
        
    def start_ab_test(self, experiment_id: str, prompt_variants: List[str]):
        """Start an A/B test with different prompt variants."""
        self.ab_testing_framework.start_experiment(experiment_id, prompt_variants)
        
    def record_ab_test_result(self, experiment_id: str, prompt_variant: str, metrics: PromptMetrics):
        """Record results for a specific prompt variant in an A/B test."""
        self.ab_testing_framework.record_result(experiment_id, prompt_variant, metrics)
        
    def get_best_prompt_variant(self, experiment_id: str) -> Optional[str]:
        """Get the best performing prompt variant from an A/B test."""
        return self.ab_testing_framework.get_best_variant(experiment_id)


# Async wrapper for easier use in async contexts
async def optimize_prompt_async(
    prompt: str, 
    context: Optional[str] = None, 
    strategy: PromptOptimizationStrategy = PromptOptimizationStrategy.CONTEXT_AWARE,
    model_name: str = "default"
) -> OptimizedPrompt:
    """
    Convenience function to optimize a prompt asynchronously.
    
    Args:
        prompt: The original prompt to optimize
        context: Additional context for optimization
        strategy: The optimization strategy to use
        model_name: Name of the model that will process the prompt
        
    Returns:
        OptimizedPrompt object containing the optimized prompt and metadata
    """
    optimizer = PromptOptimizer()
    return await optimizer.optimize_prompt(prompt, context, strategy, model_name)


# Synchronous wrapper for sync contexts
def optimize_prompt_sync(
    prompt: str, 
    context: Optional[str] = None, 
    strategy: PromptOptimizationStrategy = PromptOptimizationStrategy.CONTEXT_AWARE,
    model_name: str = "default"
) -> OptimizedPrompt:
    """
    Convenience function to optimize a prompt synchronously.
    
    Args:
        prompt: The original prompt to optimize
        context: Additional context for optimization
        strategy: The optimization strategy to use
        model_name: Name of the model that will process the prompt
        
    Returns:
        OptimizedPrompt object containing the optimized prompt and metadata
    """
    optimizer = PromptOptimizer()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(optimizer.optimize_prompt(prompt, context, strategy, model_name))
    finally:
        loop.close()