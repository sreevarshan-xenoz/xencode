#!/usr/bin/env python3
"""
AI/ML Performance Metrics for Xencode Phase 6

Prometheus metrics integration for monitoring ensemble reasoning,
RLHF tuning progress, and Ollama optimization performance.
"""

import time
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Info
from rich.console import Console

console = Console()

# Ensemble Metrics
ensemble_requests_total = Counter(
    'xencode_ensemble_requests_total',
    'Total ensemble reasoning requests',
    ['method', 'model_count', 'status']
)

ensemble_inference_time = Histogram(
    'xencode_ensemble_inference_seconds',
    'Ensemble inference time in seconds',
    ['method', 'model_count'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

ensemble_consensus_score = Histogram(
    'xencode_ensemble_consensus_score',
    'Ensemble consensus score (0-1)',
    ['method'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

ensemble_cache_hits = Counter(
    'xencode_ensemble_cache_hits_total',
    'Total ensemble cache hits',
    ['cache_type']
)

# RLHF Metrics
rlhf_training_loss = Gauge(
    'xencode_rlhf_training_loss',
    'Current RLHF training loss'
)

rlhf_perplexity = Gauge(
    'xencode_rlhf_perplexity',
    'Current RLHF model perplexity'
)

rlhf_code_quality_score = Histogram(
    'xencode_rlhf_code_quality_score',
    'RLHF code quality improvement score',
    ['task_type'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Ollama Optimizer Metrics
ollama_model_pulls = Counter(
    'xencode_ollama_model_pulls_total',
    'Total Ollama model pulls',
    ['model', 'quantization', 'status']
)

ollama_benchmark_time = Histogram(
    'xencode_ollama_benchmark_seconds',
    'Ollama model benchmark time',
    ['model'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

ollama_memory_usage = Gauge(
    'xencode_ollama_memory_usage_mb',
    'Ollama model memory usage in MB',
    ['model']
)

# System Performance Metrics
system_performance_score = Gauge(
    'xencode_system_performance_score',
    'Overall system performance score (0-100)'
)

sub_50ms_achievement_rate = Gauge(
    'xencode_sub_50ms_achievement_rate',
    'Percentage of requests achieving <50ms target'
)

smape_improvement = Gauge(
    'xencode_smape_improvement_percent',
    'SMAPE improvement percentage over baseline'
)


class AIMetricsCollector:
    """Collects and reports AI/ML performance metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.sub_50ms_requests = 0
        
    def record_ensemble_request(self, method: str, model_count: int, 
                              inference_time_ms: float, consensus_score: float,
                              success: bool, cache_hit: bool = False):
        """Record ensemble reasoning metrics"""
        status = "success" if success else "error"
        
        # Record request
        ensemble_requests_total.labels(
            method=method, 
            model_count=str(model_count), 
            status=status
        ).inc()
        
        if success:
            # Record inference time
            ensemble_inference_time.labels(
                method=method,
                model_count=str(model_count)
            ).observe(inference_time_ms / 1000.0)
            
            # Record consensus score
            ensemble_consensus_score.labels(method=method).observe(consensus_score)
            
            # Track sub-50ms achievement
            self.total_requests += 1
            if inference_time_ms < 50:
                self.sub_50ms_requests += 1
            
            # Update achievement rate
            achievement_rate = (self.sub_50ms_requests / self.total_requests) * 100
            sub_50ms_achievement_rate.set(achievement_rate)
        
        # Record cache hits
        if cache_hit:
            ensemble_cache_hits.labels(cache_type="hit").inc()
        else:
            ensemble_cache_hits.labels(cache_type="miss").inc()
    
    def record_rlhf_training(self, loss: float, perplexity: float):
        """Record RLHF training metrics"""
        rlhf_training_loss.set(loss)
        rlhf_perplexity.set(perplexity)
    
    def record_rlhf_quality(self, task_type: str, quality_score: float):
        """Record RLHF code quality metrics"""
        rlhf_code_quality_score.labels(task_type=task_type).observe(quality_score)
    
    def record_ollama_pull(self, model: str, quantization: str, success: bool):
        """Record Ollama model pull metrics"""
        status = "success" if success else "error"
        ollama_model_pulls.labels(
            model=model,
            quantization=quantization,
            status=status
        ).inc()
    
    def record_ollama_benchmark(self, model: str, benchmark_time_ms: float, 
                               memory_usage_mb: float):
        """Record Ollama benchmark metrics"""
        ollama_benchmark_time.labels(model=model).observe(benchmark_time_ms / 1000.0)
        ollama_memory_usage.labels(model=model).set(memory_usage_mb)
    
    def update_system_performance(self, score: float):
        """Update overall system performance score"""
        system_performance_score.set(score)
    
    def update_smape_improvement(self, improvement_percent: float):
        """Update SMAPE improvement metric"""
        smape_improvement.set(improvement_percent)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        uptime_hours = (time.time() - self.start_time) / 3600
        
        return {
            "uptime_hours": uptime_hours,
            "total_requests": self.total_requests,
            "sub_50ms_requests": self.sub_50ms_requests,
            "sub_50ms_rate": (self.sub_50ms_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            "current_performance_score": system_performance_score._value._value if hasattr(system_performance_score._value, '_value') else 0,
            "smape_improvement": smape_improvement._value._value if hasattr(smape_improvement._value, '_value') else 0
        }
    
    def display_metrics_summary(self):
        """Display metrics summary in console"""
        summary = self.get_performance_summary()
        
        console.print("\n[bold blue]📊 AI/ML Performance Metrics[/bold blue]")
        console.print(f"• Uptime: {summary['uptime_hours']:.1f} hours")
        console.print(f"• Total Requests: {summary['total_requests']}")
        console.print(f"• Sub-50ms Achievement: {summary['sub_50ms_rate']:.1f}%")
        console.print(f"• System Performance: {summary['current_performance_score']:.1f}/100")
        console.print(f"• SMAPE Improvement: {summary['smape_improvement']:.1f}%")
        
        # Status indicators
        if summary['sub_50ms_rate'] >= 90:
            console.print("🎯 [green]LEVIATHAN STATUS: DOMINATING[/green]")
        elif summary['sub_50ms_rate'] >= 70:
            console.print("⚡ [yellow]LEVIATHAN STATUS: CRUSHING[/yellow]")
        else:
            console.print("🔥 [red]LEVIATHAN STATUS: AWAKENING[/red]")


# Global metrics collector instance
metrics_collector = AIMetricsCollector()


def get_metrics_collector() -> AIMetricsCollector:
    """Get the global metrics collector"""
    return metrics_collector


# Convenience functions for easy integration
def record_ensemble_success(method: str, model_count: int, inference_time_ms: float,
                          consensus_score: float, cache_hit: bool = False):
    """Record successful ensemble request"""
    metrics_collector.record_ensemble_request(
        method, model_count, inference_time_ms, consensus_score, True, cache_hit
    )


def record_ensemble_error(method: str, model_count: int):
    """Record failed ensemble request"""
    metrics_collector.record_ensemble_request(method, model_count, 0, 0, False)


def record_sub_50ms_achievement():
    """Quick function to record sub-50ms achievement"""
    metrics_collector.total_requests += 1
    metrics_collector.sub_50ms_requests += 1
    achievement_rate = (metrics_collector.sub_50ms_requests / metrics_collector.total_requests) * 100
    sub_50ms_achievement_rate.set(achievement_rate)


if __name__ == "__main__":
    # Demo metrics collection
    console.print("[bold green]🤖 AI/ML Metrics Demo[/bold green]\n")
    
    collector = get_metrics_collector()
    
    # Simulate some metrics
    collector.record_ensemble_request("vote", 2, 35.5, 0.87, True, False)
    collector.record_ensemble_request("weighted", 3, 42.1, 0.92, True, True)
    collector.record_ensemble_request("consensus", 2, 28.3, 0.95, True, False)
    
    collector.record_rlhf_training(1.25, 3.48)
    collector.record_rlhf_quality("refactor", 0.85)
    
    collector.record_ollama_pull("llama3.1:8b", "q4_0", True)
    collector.record_ollama_benchmark("llama3.1:8b", 38.2, 4096.5)
    
    collector.update_system_performance(94.3)
    collector.update_smape_improvement(10.2)
    
    # Display summary
    collector.display_metrics_summary()
    
    console.print("\n[bold blue]🐉 The leviathan's metrics are being tracked![/bold blue]")