#!/usr/bin/env python3
"""
Benchmark Engine

Core execution engine for running benchmarks against AI providers.
Measures performance, quality, and cost metrics.

Features:
- Automated provider/model benchmarking
- Performance metrics (latency, throughput, tokens/sec)
- Quality metrics (accuracy, consistency scores)
- Cost efficiency calculations
- Comparative analysis across providers
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import aiohttp

from .benchmark_store import BenchmarkStore


class TaskType(Enum):
    """Benchmark task types"""
    CODE_GENERATION = "code_generation"
    CHAT = "chat"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    QUESTION_ANSWERING = "question_answering"
    GENERAL = "general"


@dataclass
class BenchmarkTask:
    """A single benchmark task definition"""
    name: str
    task_type: TaskType
    prompt: str
    expected_output: Optional[str] = None
    expected_pattern: Optional[str] = None  # Regex pattern for validation
    max_tokens: int = 1024
    timeout_seconds: int = 60
    weight: float = 1.0  # Weight for scoring
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result from executing a benchmark task"""
    task_name: str
    task_type: str
    provider: str
    model: str
    run_id: str
    
    # Performance metrics
    latency_ms: float
    tokens_per_sec: float
    throughput_rps: float
    
    # Quality metrics
    accuracy_score: float
    consistency_score: float
    quality_rating: float
    
    # Cost metrics
    cost_per_request: float
    cost_per_1k_tokens: float
    
    # Token counts
    input_tokens: int
    output_tokens: int
    total_tokens: int
    
    # Execution details
    success: bool
    error_message: Optional[str] = None
    raw_response: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "run_id": self.run_id,
            "task_name": self.task_name,
            "task_type": self.task_type,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "tokens_per_sec": self.tokens_per_sec,
            "throughput_rps": self.throughput_rps,
            "accuracy_score": self.accuracy_score,
            "consistency_score": self.consistency_score,
            "quality_rating": self.quality_rating,
            "cost_per_request": self.cost_per_request,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "success": self.success,
            "error_message": self.error_message,
            "raw_response": self.raw_response,
            "timestamp": self.timestamp.isoformat(),
        }


class BenchmarkMetrics:
    """
    Lightweight metrics wrapper for benchmark results.

    Provides aggregated metric access from a BenchmarkResult
    for use by the monitoring package and recommendation engine.
    """

    def __init__(self, result: BenchmarkResult):
        self.latency_ms = result.latency_ms
        self.tokens_per_sec = result.tokens_per_sec
        self.throughput_rps = result.throughput_rps
        self.accuracy_score = result.accuracy_score
        self.consistency_score = result.consistency_score
        self.quality_rating = result.quality_rating
        self.cost_per_request = result.cost_per_request
        self.cost_per_1k_tokens = result.cost_per_1k_tokens
        self.input_tokens = result.input_tokens
        self.output_tokens = result.output_tokens
        self.total_tokens = result.total_tokens
        self.success = result.success

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkMetrics":
        """Create from a result dictionary"""
        return cls(
            BenchmarkResult(
                task_name=data.get("task_name", ""),
                task_type=data.get("task_type", ""),
                provider=data.get("provider", ""),
                model=data.get("model", ""),
                run_id=data.get("run_id", ""),
                latency_ms=data.get("latency_ms", 0),
                tokens_per_sec=data.get("tokens_per_sec", 0),
                throughput_rps=data.get("throughput_rps", 0),
                accuracy_score=data.get("accuracy_score", 0),
                consistency_score=data.get("consistency_score", 0),
                quality_rating=data.get("quality_rating", 0),
                cost_per_request=data.get("cost_per_request", 0),
                cost_per_1k_tokens=data.get("cost_per_1k_tokens", 0),
                input_tokens=data.get("input_tokens", 0),
                output_tokens=data.get("output_tokens", 0),
                total_tokens=data.get("total_tokens", 0),
                success=data.get("success", True),
            )
        )

    @classmethod
    def from_results(cls, results: List[BenchmarkResult]) -> "BenchmarkMetrics":
        """Create aggregate metrics from a list of results"""
        if not results:
            return cls(BenchmarkResult(
                task_name="", task_type="", provider="", model="", run_id="",
                latency_ms=0, tokens_per_sec=0, throughput_rps=0,
                accuracy_score=0, consistency_score=0, quality_rating=0,
                cost_per_request=0, cost_per_1k_tokens=0,
                input_tokens=0, output_tokens=0, total_tokens=0,
                success=False,
            ))
        successful = [r for r in results if r.success]
        count = len(successful) or 1
        return cls(BenchmarkResult(
            task_name="", task_type="", provider="", model="", run_id="",
            latency_ms=sum(r.latency_ms for r in successful) / count,
            tokens_per_sec=sum(r.tokens_per_sec for r in successful) / count,
            throughput_rps=sum(r.throughput_rps for r in successful) / count,
            accuracy_score=sum(r.accuracy_score for r in successful) / count,
            consistency_score=sum(r.consistency_score for r in successful) / count,
            quality_rating=sum(r.quality_rating for r in successful) / count,
            cost_per_request=sum(r.cost_per_request for r in successful) / count,
            cost_per_1k_tokens=sum(r.cost_per_1k_tokens for r in successful) / count,
            input_tokens=sum(r.input_tokens for r in successful),
            output_tokens=sum(r.output_tokens for r in successful),
            total_tokens=sum(r.total_tokens for r in successful),
            success=True,
        ))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "latency_ms": self.latency_ms,
            "tokens_per_sec": self.tokens_per_sec,
            "throughput_rps": self.throughput_rps,
            "accuracy_score": self.accuracy_score,
            "consistency_score": self.consistency_score,
            "quality_rating": self.quality_rating,
            "cost_per_request": self.cost_per_request,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "success": self.success,
        }


class BenchmarkEngine:
    """
    Core benchmark execution engine
    
    Features:
    - Concurrent task execution
    - Multiple provider support
    - Comprehensive metrics collection
    - Quality scoring
    """
    
    # Provider endpoint mappings
    PROVIDER_ENDPOINTS = {
        "ollama": "http://localhost:11434/api/generate",
        "openai": "https://api.openai.com/v1/chat/completions",
        "anthropic": "https://api.anthropic.com/v1/messages",
        "qwen": "https://chat.qwen.ai/api/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    }
    
    # Cost estimates per 1K tokens (USD)
    COST_PER_1K_TOKENS = {
        "ollama": 0.0,  # Local, free
        "openai/gpt-4": 0.03,
        "openai/gpt-3.5-turbo": 0.002,
        "anthropic/claude-3-opus": 0.015,
        "anthropic/claude-3-sonnet": 0.003,
        "qwen": 0.0,  # Free tier
        "openrouter": 0.001,  # Average
    }
    
    def __init__(self, store: Optional[BenchmarkStore] = None):
        """
        Initialize benchmark engine
        
        Args:
            store: Optional benchmark result store
        """
        self.store = store or BenchmarkStore()
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def execute_task(
        self,
        task: BenchmarkTask,
        provider: str,
        model: str,
        run_id: Optional[str] = None,
    ) -> BenchmarkResult:
        """
        Execute a single benchmark task
        
        Args:
            task: Benchmark task to execute
            provider: Provider name
            model: Model name
            run_id: Optional run ID for grouping
            
        Returns:
            BenchmarkResult with metrics
        """
        run_id = run_id or str(uuid.uuid4())
        start_time = time.perf_counter()
        
        try:
            # Execute against provider
            response, output_tokens = await self._call_provider(
                provider=provider,
                model=model,
                prompt=task.prompt,
                max_tokens=task.max_tokens,
                timeout=task.timeout_seconds,
            )
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            # Calculate metrics
            input_tokens = len(task.prompt.split())  # Approximate
            total_tokens = input_tokens + output_tokens
            tokens_per_sec = (total_tokens / latency_ms) * 1000 if latency_ms > 0 else 0
            throughput_rps = 1000 / latency_ms if latency_ms > 0 else 0
            
            # Calculate quality scores
            accuracy_score = self._calculate_accuracy(
                response=response,
                expected=task.expected_output,
                pattern=task.expected_pattern,
            )
            
            # Calculate cost
            cost_key = f"{provider}/{model}".lower()
            cost_per_1k = self.COST_PER_1K_TOKENS.get(cost_key, 0.001)
            cost_per_request = (total_tokens / 1000) * cost_per_1k
            
            # Create result
            result = BenchmarkResult(
                task_name=task.name,
                task_type=task.task_type.value if isinstance(task.task_type, TaskType) else task.task_type,
                provider=provider,
                model=model,
                run_id=run_id,
                latency_ms=latency_ms,
                tokens_per_sec=tokens_per_sec,
                throughput_rps=throughput_rps,
                accuracy_score=accuracy_score,
                consistency_score=accuracy_score,  # Simplified for single run
                quality_rating=accuracy_score,
                cost_per_request=cost_per_request,
                cost_per_1k_tokens=cost_per_1k,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                success=True,
                raw_response=response,
            )
            
            # Store result
            self.store.save_result(result.to_dict())
            
            return result
            
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            result = BenchmarkResult(
                task_name=task.name,
                task_type=task.task_type.value if isinstance(task.task_type, TaskType) else task.task_type,
                provider=provider,
                model=model,
                run_id=run_id,
                latency_ms=latency_ms,
                tokens_per_sec=0,
                throughput_rps=0,
                accuracy_score=0,
                consistency_score=0,
                quality_rating=0,
                cost_per_request=0,
                cost_per_1k_tokens=0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                success=False,
                error_message=str(e),
            )
            
            self.store.save_result(result.to_dict())
            return result
    
    async def _call_provider(
        self,
        provider: str,
        model: str,
        prompt: str,
        max_tokens: int,
        timeout: int,
    ) -> tuple[str, int]:
        """
        Call provider API and get response
        
        Args:
            provider: Provider name
            model: Model name
            prompt: Input prompt
            max_tokens: Maximum output tokens
            timeout: Request timeout
            
        Returns:
            Tuple of (response_text, output_tokens)
        """
        session = await self._get_session()
        
        endpoint = self.PROVIDER_ENDPOINTS.get(provider)
        if not endpoint:
            # Mock response for unknown providers
            await asyncio.sleep(0.1)  # Simulate latency
            return f"Mock response for {provider}/{model}", 10
        
        try:
            # Build request based on provider
            if provider == "ollama":
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                
                async with session.post(endpoint, json=payload, timeout=timeout) as resp:
                    data = await resp.json()
                    response = data.get("response", "")
                    output_tokens = len(response.split())
                    return response, output_tokens
                    
            elif provider in ["openai", "qwen", "openrouter"]:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                }
                
                async with session.post(endpoint, json=payload, timeout=timeout) as resp:
                    data = await resp.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        response = data["choices"][0]["message"]["content"]
                    else:
                        response = str(data)
                    output_tokens = len(response.split())
                    return response, output_tokens
                    
            elif provider == "anthropic":
                payload = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }
                
                async with session.post(endpoint, json=payload, timeout=timeout) as resp:
                    data = await resp.json()
                    response = data.get("content", "")
                    output_tokens = len(response.split())
                    return response, output_tokens
            
            else:
                # Unknown provider
                await asyncio.sleep(0.1)
                return "Mock response", 10
                
        except asyncio.TimeoutError:
            raise TimeoutError(f"Provider {provider} timed out")  from None
        except Exception as e:
            raise RuntimeError(f"Provider {provider} error: {e}")  from e
    
    def _calculate_accuracy(
        self,
        response: str,
        expected: Optional[str],
        pattern: Optional[str],
    ) -> float:
        """
        Calculate accuracy score for response
        
        Args:
            response: Model response
            expected: Expected output
            pattern: Expected regex pattern
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        import re
        
        if not response:
            return 0.0
        
        if expected:
            # Exact match or contains
            if response.strip() == expected.strip():
                return 1.0
            elif expected.strip() in response:
                return 0.8
            else:
                # Fuzzy match based on common tokens
                response_tokens = set(response.lower().split())
                expected_tokens = set(expected.lower().split())
                if expected_tokens:
                    overlap = len(response_tokens & expected_tokens) / len(expected_tokens)
                    return min(1.0, overlap * 1.5)  # Boost for partial matches
        
        if pattern:
            try:
                if re.search(pattern, response):
                    return 1.0
            except re.error:
                pass
        
        # No validation criteria, assume success
        return 0.7
    
    async def run_benchmark_suite(
        self,
        tasks: List[BenchmarkTask],
        providers: List[tuple[str, str]],  # List of (provider, model) tuples
        suite_name: str = "custom_suite",
        concurrent: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a complete benchmark suite
        
        Args:
            tasks: List of benchmark tasks
            providers: List of (provider, model) tuples to benchmark
            suite_name: Name for the benchmark run
            concurrent: Whether to run tasks concurrently
            
        Returns:
            Summary of benchmark results
        """
        run_id = str(uuid.uuid4())
        
        # Save run metadata
        self.store.save_run(
            run_id=run_id,
            suite_name=suite_name,
            config={"tasks": len(tasks), "providers": len(providers)},
            total_tasks=len(tasks) * len(providers),
        )
        
        results = []
        
        if concurrent:
            # Run all tasks concurrently
            coroutines = []
            for task in tasks:
                for provider, model in providers:
                    coroutines.append(
                        self.execute_task(task, provider, model, run_id)
                    )
            
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            
        else:
            # Run sequentially
            for task in tasks:
                for provider, model in providers:
                    result = await self.execute_task(task, provider, model, run_id)
                    results.append(result)
        
        # Update run status
        completed = sum(1 for r in results if isinstance(r, BenchmarkResult) and r.success)
        self.store.update_run_status(run_id, "completed", completed)
        
        # Generate summary
        return self._generate_summary(results, run_id, suite_name)
    
    def _generate_summary(
        self,
        results: List[Any],
        run_id: str,
        suite_name: str,
    ) -> Dict[str, Any]:
        """Generate summary statistics from results"""
        valid_results = [r for r in results if isinstance(r, BenchmarkResult)]
        successful = [r for r in valid_results if r.success]
        
        if not successful:
            return {
                "run_id": run_id,
                "suite_name": suite_name,
                "total_tasks": len(results),
                "successful": 0,
                "failed": len(results),
                "results": [],
            }
        
        # Calculate aggregates
        avg_latency = sum(r.latency_ms for r in successful) / len(successful)
        avg_tokens_per_sec = sum(r.tokens_per_sec for r in successful) / len(successful)
        avg_accuracy = sum(r.accuracy_score for r in successful) / len(successful)
        avg_cost = sum(r.cost_per_request for r in successful) / len(successful)
        
        # Group by provider
        provider_stats = {}
        for result in successful:
            key = f"{result.provider}/{result.model}"
            if key not in provider_stats:
                provider_stats[key] = []
            provider_stats[key].append(result)
        
        # Calculate per-provider averages
        provider_summary = {}
        for key, prov_results in provider_stats.items():
            provider_summary[key] = {
                "count": len(prov_results),
                "avg_latency_ms": sum(r.latency_ms for r in prov_results) / len(prov_results),
                "avg_tokens_per_sec": sum(r.tokens_per_sec for r in prov_results) / len(prov_results),
                "avg_accuracy": sum(r.accuracy_score for r in prov_results) / len(prov_results),
                "avg_cost": sum(r.cost_per_request for r in prov_results) / len(prov_results),
            }
        
        return {
            "run_id": run_id,
            "suite_name": suite_name,
            "total_tasks": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "avg_latency_ms": avg_latency,
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "avg_accuracy_score": avg_accuracy,
            "avg_cost_per_request": avg_cost,
            "provider_summary": provider_summary,
            "results": [r.to_dict() for r in successful],
        }


# Factory & convenience functions for monitoring/__init__.py


def create_benchmark_engine(store: Optional[BenchmarkStore] = None) -> BenchmarkEngine:
    """
    Create a new BenchmarkEngine instance.

    Args:
        store: Optional BenchmarkStore to attach.

    Returns:
        A new BenchmarkEngine instance.
    """
    return BenchmarkEngine(store=store or BenchmarkStore())


async def run_benchmark(
    engine: 'BenchmarkEngine',
    tasks: List['BenchmarkTask'],
    providers: List[tuple[str, str]],
    suite_name: str = "run",
    concurrent: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to run a benchmark suite.

    Args:
        engine: A BenchmarkEngine instance.
        tasks: List of benchmark tasks.
        providers: List of (provider, model) tuples.
        suite_name: Name for this benchmark run.
        concurrent: Whether to run tasks concurrently.

    Returns:
        Summary dictionary of benchmark results.
    """
    return await engine.run_benchmark_suite(tasks, providers, suite_name, concurrent)
