#!/usr/bin/env python3
"""
Benchmark Recommendations Engine

Generates model recommendations based on benchmark results.
Provides cost/quality tradeoff analysis and use-case specific recommendations.

Features:
- Generate model recommendations based on benchmarks
- Cost/quality tradeoff analysis
- Use-case specific recommendations
- Historical performance tracking
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .benchmark_store import BenchmarkStore


@dataclass
class ModelScore:
    """Aggregated score for a model"""
    provider: str
    model: str
    task_type: str
    
    # Scores (0-100)
    performance_score: float
    quality_score: float
    cost_score: float
    overall_score: float
    
    # Metrics
    avg_latency_ms: float
    avg_accuracy: float
    avg_cost_per_request: float
    sample_count: int
    
    # Trend
    trend: str = "stable"  # improving, declining, stable
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "task_type": self.task_type,
            "performance_score": self.performance_score,
            "quality_score": self.quality_score,
            "cost_score": self.cost_score,
            "overall_score": self.overall_score,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_accuracy": self.avg_accuracy,
            "avg_cost_per_request": self.avg_cost_per_request,
            "sample_count": self.sample_count,
            "trend": self.trend,
        }


@dataclass
class Recommendation:
    """Model recommendation"""
    task_type: str
    use_case: str
    
    # Primary recommendation
    recommended_provider: str
    recommended_model: str
    confidence: float  # 0-1
    
    # Reasoning
    reasons: List[str]
    
    # Alternatives
    alternatives: List[Dict[str, Any]]
    
    # Tradeoffs
    tradeoffs: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "use_case": self.use_case,
            "recommended_provider": self.recommended_provider,
            "recommended_model": self.recommended_model,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "alternatives": self.alternatives,
            "tradeoffs": self.tradeoffs,
        }


class RecommendationsEngine:
    """
    Generates model recommendations based on benchmark data
    
    Features:
    - Multi-criteria scoring
    - Cost/quality optimization
    - Use-case specific recommendations
    - Historical trend analysis
    """
    
    # Scoring weights
    WEIGHTS = {
        "performance": 0.35,  # Latency, throughput
        "quality": 0.45,      # Accuracy, consistency
        "cost": 0.20,         # Cost efficiency
    }
    
    # Use case profiles
    USE_CASE_PROFILES = {
        "production": {
            "performance": 0.40,
            "quality": 0.50,
            "cost": 0.10,
        },
        "development": {
            "performance": 0.30,
            "quality": 0.30,
            "cost": 0.40,
        },
        "experimental": {
            "performance": 0.20,
            "quality": 0.30,
            "cost": 0.50,
        },
        "realtime": {
            "performance": 0.60,
            "quality": 0.30,
            "cost": 0.10,
        },
        "high_quality": {
            "performance": 0.20,
            "quality": 0.70,
            "cost": 0.10,
        },
    }
    
    def __init__(self, store: Optional[BenchmarkStore] = None):
        """
        Initialize recommendations engine
        
        Args:
            store: Optional benchmark result store
        """
        self.store = store or BenchmarkStore()
        self._results: List[Dict[str, Any]] = []
        self._model_scores: Dict[str, ModelScore] = {}
        self._historical_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """
        Add a benchmark result for analysis
        
        Args:
            result: Benchmark result dictionary
        """
        self._results.append(result)
        
        # Update model scores
        self._update_model_scores()
    
    def add_results(self, results: List[Dict[str, Any]]) -> None:
        """Add multiple results"""
        for result in results:
            self._results.append(result)
        self._update_model_scores()
    
    def load_from_store(self, limit: int = 1000) -> None:
        """Load results from store"""
        # Get recent results
        all_results = []
        
        # Get aggregate stats to understand available data
        stats = self.store.get_aggregate_stats()
        
        # For now, use in-memory results
        # In production, would query store for historical data
    
    def _update_model_scores(self) -> None:
        """Recalculate model scores from results"""
        # Group results by model
        model_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for result in self._results:
            key = f"{result.get('provider', 'unknown')}/{result.get('model', 'unknown')}"
            model_results[key].append(result)
        
        # Calculate scores for each model
        for model_key, results in model_results.items():
            if not results:
                continue
            
            provider, model = model_key.split("/", 1) if "/" in model_key else ("unknown", model_key)
            
            # Get task types
            task_types = set(r.get("task_type", "general") for r in results)
            
            for task_type in task_types:
                task_results = [r for r in results if r.get("task_type") == task_type]
                
                if not task_results:
                    continue
                
                # Calculate averages
                avg_latency = sum(r.get("latency_ms", 0) for r in task_results) / len(task_results)
                avg_accuracy = sum(r.get("accuracy_score", 0) for r in task_results) / len(task_results)
                avg_cost = sum(r.get("cost_per_request", 0) for r in task_results) / len(task_results)
                
                # Calculate scores (0-100)
                performance_score = self._calculate_performance_score(avg_latency, task_results)
                quality_score = self._calculate_quality_score(avg_accuracy, task_results)
                cost_score = self._calculate_cost_score(avg_cost, task_results)
                
                # Overall score
                overall_score = (
                    performance_score * self.WEIGHTS["performance"] +
                    quality_score * self.WEIGHTS["quality"] +
                    cost_score * self.WEIGHTS["cost"]
                )
                
                # Determine trend (simplified)
                trend = self._calculate_trend(task_results)
                
                score = ModelScore(
                    provider=provider,
                    model=model,
                    task_type=task_type,
                    performance_score=performance_score,
                    quality_score=quality_score,
                    cost_score=cost_score,
                    overall_score=overall_score,
                    avg_latency_ms=avg_latency,
                    avg_accuracy=avg_accuracy,
                    avg_cost_per_request=avg_cost,
                    sample_count=len(task_results),
                    trend=trend,
                )
                
                self._model_scores[f"{model_key}/{task_type}"] = score
    
    def _calculate_performance_score(self, avg_latency: float, results: List[Dict]) -> float:
        """Calculate performance score (0-100)"""
        # Lower latency = higher score
        # Assume 100ms is excellent, 2000ms is poor
        if avg_latency <= 0:
            return 0
        
        latency_score = max(0, 100 - (avg_latency / 20))
        
        # Bonus for high throughput
        avg_throughput = sum(r.get("tokens_per_sec", 0) for r in results) / len(results)
        throughput_bonus = min(20, avg_throughput / 5)
        
        return min(100, latency_score + throughput_bonus)
    
    def _calculate_quality_score(self, avg_accuracy: float, results: List[Dict]) -> float:
        """Calculate quality score (0-100)"""
        # Accuracy is already 0-1, scale to 0-100
        base_score = avg_accuracy * 100
        
        # Bonus for consistency
        if len(results) > 1:
            accuracies = [r.get("accuracy_score", 0) for r in results]
            variance = sum((a - avg_accuracy) ** 2 for a in accuracies) / len(accuracies)
            consistency_bonus = max(0, 20 - (variance * 100))
            base_score += consistency_bonus
        
        return min(100, base_score)
    
    def _calculate_cost_score(self, avg_cost: float, results: List[Dict]) -> float:
        """Calculate cost score (0-100)"""
        # Lower cost = higher score
        # Assume $0 is excellent, $0.10 per request is poor
        if avg_cost <= 0:
            return 100
        
        cost_score = max(0, 100 - (avg_cost * 1000))
        return cost_score
    
    def _calculate_trend(self, results: List[Dict]) -> str:
        """Calculate performance trend"""
        if len(results) < 3:
            return "stable"
        
        # Sort by timestamp
        sorted_results = sorted(
            results,
            key=lambda r: r.get("timestamp", ""),
        )
        
        # Compare recent vs older
        recent = sorted_results[-3:]
        older = sorted_results[:3]
        
        recent_accuracy = sum(r.get("accuracy_score", 0) for r in recent) / len(recent)
        older_accuracy = sum(r.get("accuracy_score", 0) for r in older) / len(older)
        
        if recent_accuracy > older_accuracy * 1.1:
            return "improving"
        elif recent_accuracy < older_accuracy * 0.9:
            return "declining"
        
        return "stable"
    
    def get_model_scores(
        self,
        task_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[ModelScore]:
        """
        Get model scores
        
        Args:
            task_type: Filter by task type
            limit: Maximum results
            
        Returns:
            List of ModelScore sorted by overall score
        """
        scores = list(self._model_scores.values())
        
        if task_type:
            scores = [s for s in scores if s.task_type == task_type]
        
        # Sort by overall score
        scores.sort(key=lambda s: s.overall_score, reverse=True)
        
        return scores[:limit]
    
    def get_recommendations(
        self,
        task_type: str,
        use_case: str = "production",
        max_budget: Optional[float] = None,
        max_latency_ms: Optional[float] = None,
        min_accuracy: Optional[float] = None,
    ) -> Recommendation:
        """
        Get model recommendations
        
        Args:
            task_type: Type of task
            use_case: Use case profile (production, development, etc.)
            max_budget: Maximum cost per request
            max_latency_ms: Maximum acceptable latency
            min_accuracy: Minimum accuracy score
            
        Returns:
            Recommendation with alternatives
        """
        # Get use case weights
        profile = self.USE_CASE_PROFILES.get(use_case, self.USE_CASE_PROFILES["production"])
        
        # Get scores for task type
        scores = self.get_model_scores(task_type=task_type)
        
        if not scores:
            # Fallback recommendation
            return Recommendation(
                task_type=task_type,
                use_case=use_case,
                recommended_provider="ollama",
                recommended_model="llama3.2",
                confidence=0.5,
                reasons=["No benchmark data available, using default"],
                alternatives=[],
                tradeoffs={},
            )
        
        # Apply filters
        filtered_scores = []
        for score in scores:
            if max_budget is not None and score.avg_cost_per_request > max_budget:
                continue
            if max_latency_ms is not None and score.avg_latency_ms > max_latency_ms:
                continue
            if min_accuracy is not None and score.avg_accuracy < min_accuracy:
                continue
            filtered_scores.append(score)
        
        if not filtered_scores:
            filtered_scores = scores  # Relax filters
        
        # Calculate weighted scores based on use case
        for score in filtered_scores:
            score.overall_score = (
                score.performance_score * profile["performance"] +
                score.quality_score * profile["quality"] +
                score.cost_score * profile["cost"]
            )
        
        # Sort by adjusted score
        filtered_scores.sort(key=lambda s: s.overall_score, reverse=True)
        
        # Top recommendation
        top = filtered_scores[0]
        
        # Generate reasons
        reasons = self._generate_reasons(top, profile)
        
        # Alternatives
        alternatives = [
            {
                "provider": s.provider,
                "model": s.model,
                "overall_score": s.overall_score,
                "avg_latency_ms": s.avg_latency_ms,
                "avg_accuracy": s.avg_accuracy,
                "avg_cost": s.avg_cost_per_request,
            }
            for s in filtered_scores[1:4]
        ]
        
        # Tradeoffs
        tradeoffs = self._generate_tradeoffs(top, filtered_scores[1] if len(filtered_scores) > 1 else None)
        
        # Confidence based on sample size
        confidence = min(0.95, 0.5 + (top.sample_count * 0.05))
        
        return Recommendation(
            task_type=task_type,
            use_case=use_case,
            recommended_provider=top.provider,
            recommended_model=top.model,
            confidence=confidence,
            reasons=reasons,
            alternatives=alternatives,
            tradeoffs=tradeoffs,
        )
    
    def _generate_reasons(self, score: ModelScore, profile: Dict) -> List[str]:
        """Generate recommendation reasons"""
        reasons = []
        
        if score.performance_score >= 80:
            reasons.append(f"Excellent performance ({score.avg_latency_ms:.0f}ms avg latency)")
        elif score.performance_score >= 60:
            reasons.append(f"Good performance ({score.avg_latency_ms:.0f}ms avg latency)")
        
        if score.quality_score >= 80:
            reasons.append(f"High accuracy ({score.avg_accuracy:.2f} avg)")
        elif score.quality_score >= 60:
            reasons.append(f"Reliable quality ({score.avg_accuracy:.2f} avg)")
        
        if score.cost_score >= 80:
            reasons.append(f"Cost-efficient (${score.avg_cost_per_request:.4f}/request)")
        elif score.avg_cost_per_request <= 0:
            reasons.append("Free to use")
        
        if score.trend == "improving":
            reasons.append("Performance trending upward")
        
        if score.sample_count >= 10:
            reasons.append(f"Based on {score.sample_count} benchmark runs")
        
        return reasons
    
    def _generate_tradeoffs(self, top: ModelScore, second: Optional[ModelScore]) -> Dict[str, str]:
        """Generate tradeoff analysis"""
        tradeoffs = {}
        
        if second:
            if second.avg_latency_ms < top.avg_latency_ms:
                tradeoffs["speed"] = f"{second.provider}/{second.model} is {top.avg_latency_ms - second.avg_latency_ms:.0f}ms faster"
            
            if second.avg_accuracy > top.avg_accuracy:
                tradeoffs["accuracy"] = f"{second.provider}/{second.model} has {second.avg_accuracy - top.avg_accuracy:.2f} higher accuracy"
            
            if second.avg_cost_per_request < top.avg_cost_per_request:
                tradeoffs["cost"] = f"{second.provider}/{second.model} is cheaper per request"
        
        if top.avg_cost_per_request > 0:
            tradeoffs["budget"] = "Consider free alternatives for development"
        
        return tradeoffs
    
    def get_cost_quality_analysis(self, task_type: str) -> Dict[str, Any]:
        """
        Get cost vs quality analysis
        
        Args:
            task_type: Task type
            
        Returns:
            Analysis with Pareto frontier
        """
        scores = self.get_model_scores(task_type=task_type)
        
        if not scores:
            return {"error": "No data available"}
        
        # Find Pareto optimal models (best quality for given cost)
        pareto_frontier = []
        
        for score in scores:
            is_dominated = False
            
            for other in scores:
                if other == score:
                    continue
                
                # Other is better in both dimensions
                if other.quality_score >= score.quality_score and other.cost_score > score.cost_score:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_frontier.append(score)
        
        return {
            "task_type": task_type,
            "total_models": len(scores),
            "pareto_optimal": len(pareto_frontier),
            "frontier": [s.to_dict() for s in pareto_frontier],
            "all_models": [s.to_dict() for s in scores],
        }
    
    def get_historical_performance(
        self,
        provider: str,
        model: str,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Get historical performance trend
        
        Args:
            provider: Provider name
            model: Model name
            days: Number of days
            
        Returns:
            Historical performance data
        """
        # In production, would query store for time-series data
        # For now, return current aggregate
        
        stats = self.store.get_aggregate_stats(provider=provider, model=model)
        
        return {
            "provider": provider,
            "model": model,
            "days": days,
            "current_metrics": stats,
            "trend": "stable",  # Would calculate from time-series
        }# Aliases for backward compatibility with monitoring/__init__.py
BenchmarkRecommendations = RecommendationsEngine
ModelRecommendation = Recommendation


# Global instance
_recommendations_instance: Optional[RecommendationsEngine] = None



def get_recommendations_engine(store: Optional[BenchmarkStore] = None) -> RecommendationsEngine:
    """Get global recommendations engine instance"""
    global _recommendations_instance
    if _recommendations_instance is None:
        _recommendations_instance = RecommendationsEngine(store)
    return _recommendations_instance


def generate_recommendations(
    store: Optional[BenchmarkStore] = None,
    task_type: str = "general",
    use_case: str = "production",
) -> Dict[str, Any]:
    """
    Convenience function to generate model recommendations from benchmark data.

    Args:
        store: Optional BenchmarkStore with benchmark data.
        task_type: Type of task to recommend for.
        use_case: Use case profile (production, development, etc.).

    Returns:
        Recommendation dictionary.
    """
    engine = get_recommendations_engine(store=store)
    rec = engine.get_recommendations(task_type=task_type, use_case=use_case)
    return rec.to_dict()
