#!/usr/bin/env python3
"""
Ollama Optimization System for Xencode Phase 6

Advanced Ollama model management with automatic pulling, quantization,
benchmarking, and optimization for <50ms inference on any hardware.
"""

import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import psutil
import ollama
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

console = Console()


class QuantizationLevel(Enum):
    """Model quantization levels"""
    Q4_0 = "q4_0"  # 4-bit quantization, fastest
    Q4_1 = "q4_1"  # 4-bit quantization, balanced
    Q5_0 = "q5_0"  # 5-bit quantization, better quality
    Q5_1 = "q5_1"  # 5-bit quantization, best quality
    Q8_0 = "q8_0"  # 8-bit quantization, highest quality
    F16 = "f16"    # 16-bit float, original quality
    F32 = "f32"    # 32-bit float, maximum quality


class ModelStatus(Enum):
    """Model availability status"""
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    NOT_FOUND = "not_found"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about an Ollama model"""
    name: str
    tag: str
    size_gb: float
    status: ModelStatus
    quantization: Optional[QuantizationLevel] = None
    download_progress: float = 0.0
    last_used: Optional[float] = None
    performance_score: float = 0.0
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Benchmark results for a model"""
    model_name: str
    avg_inference_time_ms: float
    min_inference_time_ms: float
    max_inference_time_ms: float
    memory_usage_mb: float
    tokens_per_second: float
    quality_score: float
    success_rate: float
    test_prompts_count: int
    hardware_info: Dict[str, Any] = field(default_factory=dict)


class OllamaOptimizer:
    """Advanced Ollama model optimizer"""
    
    def __init__(self):
        self.client = ollama.AsyncClient()
        self.models_cache: Dict[str, ModelInfo] = {}
        self.benchmark_cache: Dict[str, BenchmarkResult] = {}
        self.config_dir = Path.home() / ".xencode" / "ollama"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cached data
        self._load_cache()
    
    def _load_cache(self):
        """Load cached model and benchmark data"""
        try:
            models_cache_file = self.config_dir / "models_cache.json"
            if models_cache_file.exists():
                with open(models_cache_file, 'r') as f:
                    data = json.load(f)
                    for name, info in data.items():
                        self.models_cache[name] = ModelInfo(**info)
            
            benchmark_cache_file = self.config_dir / "benchmark_cache.json"
            if benchmark_cache_file.exists():
                with open(benchmark_cache_file, 'r') as f:
                    data = json.load(f)
                    for name, result in data.items():
                        self.benchmark_cache[name] = BenchmarkResult(**result)
                        
        except Exception as e:
            console.print(f"[yellow]⚠️ Failed to load cache: {e}[/yellow]")
    
    def _save_cache(self):
        """Save cached model and benchmark data"""
        try:
            # Save models cache
            models_data = {}
            for name, info in self.models_cache.items():
                models_data[name] = {
                    "name": info.name,
                    "tag": info.tag,
                    "size_gb": info.size_gb,
                    "status": info.status.value,
                    "quantization": info.quantization.value if info.quantization else None,
                    "download_progress": info.download_progress,
                    "last_used": info.last_used,
                    "performance_score": info.performance_score,
                    "inference_time_ms": info.inference_time_ms,
                    "memory_usage_mb": info.memory_usage_mb,
                    "metadata": info.metadata
                }
            
            with open(self.config_dir / "models_cache.json", 'w') as f:
                json.dump(models_data, f, indent=2)
            
            # Save benchmark cache
            benchmark_data = {}
            for name, result in self.benchmark_cache.items():
                benchmark_data[name] = {
                    "model_name": result.model_name,
                    "avg_inference_time_ms": result.avg_inference_time_ms,
                    "min_inference_time_ms": result.min_inference_time_ms,
                    "max_inference_time_ms": result.max_inference_time_ms,
                    "memory_usage_mb": result.memory_usage_mb,
                    "tokens_per_second": result.tokens_per_second,
                    "quality_score": result.quality_score,
                    "success_rate": result.success_rate,
                    "test_prompts_count": result.test_prompts_count,
                    "hardware_info": result.hardware_info
                }
            
            with open(self.config_dir / "benchmark_cache.json", 'w') as f:
                json.dump(benchmark_data, f, indent=2)
                
        except Exception as e:
            console.print(f"[yellow]⚠️ Failed to save cache: {e}[/yellow]")
    
    async def check_ollama_status(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            # Try to list models to check if Ollama is running
            await self.client.list()
            return True
        except Exception:
            return False
    
    async def list_available_models(self, refresh: bool = False) -> List[ModelInfo]:
        """List all available Ollama models"""
        if not refresh and self.models_cache:
            return list(self.models_cache.values())
        
        try:
            models_response = await self.client.list()
            models = []
            
            for model_data in models_response.get('models', []):
                name = model_data.get('name', '')
                size_bytes = model_data.get('size', 0)
                size_gb = size_bytes / (1024**3) if size_bytes else 0
                
                # Extract quantization info from name
                quantization = None
                for quant in QuantizationLevel:
                    if quant.value in name.lower():
                        quantization = quant
                        break
                
                model_info = ModelInfo(
                    name=name,
                    tag=name,
                    size_gb=size_gb,
                    status=ModelStatus.AVAILABLE,
                    quantization=quantization,
                    last_used=time.time()
                )
                
                models.append(model_info)
                self.models_cache[name] = model_info
            
            self._save_cache()
            return models
            
        except Exception as e:
            console.print(f"[red]❌ Failed to list models: {e}[/red]")
            return []
    
    async def pull_model(self, model_name: str, quantization: QuantizationLevel = None) -> bool:
        """Pull a model from Ollama registry with optional quantization"""
        # Construct full model name with quantization
        if quantization and quantization != QuantizationLevel.F32:
            full_name = f"{model_name}:{quantization.value}"
        else:
            full_name = model_name
        
        console.print(f"[blue]📥 Pulling model: {full_name}[/blue]")
        
        # Create model info entry
        model_info = ModelInfo(
            name=full_name,
            tag=full_name,
            size_gb=0.0,  # Will be updated after download
            status=ModelStatus.DOWNLOADING,
            quantization=quantization
        )
        self.models_cache[full_name] = model_info
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                
                task = progress.add_task(f"Downloading {full_name}...", total=100)
                
                # Use ollama pull with streaming
                stream = await self.client.pull(model=full_name, stream=True)
                
                async for chunk in stream:
                    if 'total' in chunk and 'completed' in chunk:
                        total = chunk['total']
                        completed = chunk['completed']
                        if total > 0:
                            percent = (completed / total) * 100
                            progress.update(task, completed=percent)
                            model_info.download_progress = percent
                    
                    if chunk.get('status') == 'success':
                        break
                
                progress.update(task, completed=100)
            
            # Update model status
            model_info.status = ModelStatus.AVAILABLE
            model_info.download_progress = 100.0
            
            # Get actual model size
            models = await self.list_available_models(refresh=True)
            for model in models:
                if model.name == full_name:
                    model_info.size_gb = model.size_gb
                    break
            
            console.print(f"[green]✅ Successfully pulled {full_name}[/green]")
            self._save_cache()
            return True
            
        except Exception as e:
            model_info.status = ModelStatus.ERROR
            console.print(f"[red]❌ Failed to pull {full_name}: {e}[/red]")
            return False
    
    async def benchmark_model(self, model_name: str, 
                            test_prompts: List[str] = None) -> BenchmarkResult:
        """Benchmark a model's performance"""
        if not test_prompts:
            test_prompts = [
                "Explain the concept of recursion in programming",
                "What are the benefits of using microservices?",
                "How does machine learning work?",
                "Write a Python function to sort a list",
                "What is the difference between REST and GraphQL?"
            ]
        
        console.print(f"[blue]🔬 Benchmarking model: {model_name}[/blue]")
        
        # Check if we have cached results
        if model_name in self.benchmark_cache:
            cached_result = self.benchmark_cache[model_name]
            # Return cached if less than 24 hours old
            if time.time() - cached_result.hardware_info.get('timestamp', 0) < 86400:
                console.print(f"[yellow]📋 Using cached benchmark for {model_name}[/yellow]")
                return cached_result
        
        inference_times = []
        memory_usage = []
        successful_requests = 0
        total_tokens = 0
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task(f"Benchmarking {model_name}...", total=len(test_prompts))
            
            for prompt in test_prompts:
                try:
                    start_time = time.perf_counter()
                    
                    # Make inference request
                    response = await self.client.generate(
                        model=model_name,
                        prompt=prompt,
                        stream=False,
                        options={
                            "temperature": 0.7,
                            "max_tokens": 100
                        }
                    )
                    
                    end_time = time.perf_counter()
                    inference_time = (end_time - start_time) * 1000  # ms
                    
                    inference_times.append(inference_time)
                    successful_requests += 1
                    
                    # Count tokens (approximate)
                    response_text = response.get('response', '')
                    tokens = len(response_text.split())
                    total_tokens += tokens
                    
                    # Memory usage
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    memory_usage.append(current_memory - initial_memory)
                    
                except Exception as e:
                    console.print(f"[yellow]⚠️ Failed prompt: {e}[/yellow]")
                
                progress.update(task, advance=1)
        
        # Calculate metrics
        if inference_times:
            avg_inference_time = sum(inference_times) / len(inference_times)
            min_inference_time = min(inference_times)
            max_inference_time = max(inference_times)
            avg_memory_usage = sum(memory_usage) / len(memory_usage) if memory_usage else 0
            tokens_per_second = total_tokens / (sum(inference_times) / 1000) if inference_times else 0
            success_rate = successful_requests / len(test_prompts) * 100
            
            # Calculate quality score (simplified)
            quality_score = min(1.0, success_rate / 100 * (1 - min(avg_inference_time / 1000, 1)))
        else:
            avg_inference_time = float('inf')
            min_inference_time = float('inf')
            max_inference_time = float('inf')
            avg_memory_usage = 0
            tokens_per_second = 0
            success_rate = 0
            quality_score = 0
        
        # Hardware info
        hardware_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "timestamp": time.time(),
            "platform": os.uname().sysname if hasattr(os, 'uname') else 'unknown'
        }
        
        # Create benchmark result
        result = BenchmarkResult(
            model_name=model_name,
            avg_inference_time_ms=avg_inference_time,
            min_inference_time_ms=min_inference_time,
            max_inference_time_ms=max_inference_time,
            memory_usage_mb=avg_memory_usage,
            tokens_per_second=tokens_per_second,
            quality_score=quality_score,
            success_rate=success_rate,
            test_prompts_count=len(test_prompts),
            hardware_info=hardware_info
        )
        
        # Cache result
        self.benchmark_cache[model_name] = result
        
        # Update model info
        if model_name in self.models_cache:
            self.models_cache[model_name].performance_score = quality_score
            self.models_cache[model_name].inference_time_ms = avg_inference_time
            self.models_cache[model_name].memory_usage_mb = avg_memory_usage
        
        self._save_cache()
        
        console.print(f"[green]✅ Benchmark completed for {model_name}[/green]")
        return result
    
    async def optimize_for_hardware(self) -> Dict[str, Any]:
        """Optimize model selection for current hardware"""
        console.print("[blue]⚡ Optimizing for current hardware...[/blue]")
        
        # Get hardware info
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Get available models
        models = await self.list_available_models()
        
        # Filter models that can run on current hardware
        suitable_models = []
        for model in models:
            if model.status == ModelStatus.AVAILABLE:
                # Estimate memory requirement (model size * 1.5 for overhead)
                estimated_memory = model.size_gb * 1.5
                if estimated_memory <= available_memory_gb:
                    suitable_models.append(model)
        
        # Benchmark suitable models if not already done
        for model in suitable_models:
            if model.name not in self.benchmark_cache:
                await self.benchmark_model(model.name)
        
        # Find optimal models for different use cases
        recommendations = {
            "fastest": None,
            "balanced": None,
            "highest_quality": None,
            "most_efficient": None
        }
        
        if suitable_models:
            # Fastest (lowest inference time)
            fastest = min(suitable_models, key=lambda m: m.inference_time_ms or float('inf'))
            if fastest.inference_time_ms < float('inf'):
                recommendations["fastest"] = fastest.name
            
            # Balanced (best performance score)
            balanced = max(suitable_models, key=lambda m: m.performance_score)
            recommendations["balanced"] = balanced.name
            
            # Highest quality (largest model that fits)
            highest_quality = max(suitable_models, key=lambda m: m.size_gb)
            recommendations["highest_quality"] = highest_quality.name
            
            # Most efficient (best tokens/second per GB)
            if self.benchmark_cache:
                efficient_scores = []
                for model in suitable_models:
                    if model.name in self.benchmark_cache:
                        benchmark = self.benchmark_cache[model.name]
                        efficiency = benchmark.tokens_per_second / max(model.size_gb, 0.1)
                        efficient_scores.append((model.name, efficiency))
                
                if efficient_scores:
                    most_efficient = max(efficient_scores, key=lambda x: x[1])
                    recommendations["most_efficient"] = most_efficient[0]
        
        optimization_result = {
            "hardware_info": {
                "cpu_count": cpu_count,
                "total_memory_gb": memory_gb,
                "available_memory_gb": available_memory_gb
            },
            "suitable_models_count": len(suitable_models),
            "recommendations": recommendations,
            "sub_50ms_models": [
                model.name for model in suitable_models 
                if model.inference_time_ms and model.inference_time_ms < 50
            ]
        }
        
        return optimization_result
    
    async def auto_pull_recommended_models(self) -> List[str]:
        """Automatically pull recommended models for current hardware"""
        console.print("[blue]🤖 Auto-pulling recommended models...[/blue]")
        
        # Get hardware-specific recommendations
        hardware_info = psutil.virtual_memory()
        total_memory_gb = hardware_info.total / (1024**3)
        
        # Determine recommended models based on available memory
        recommended_models = []
        
        if total_memory_gb >= 32:
            # High-end system
            recommended_models = [
                ("llama3.1:8b", None),
                ("mistral:7b", None),
                ("qwen2.5:14b", QuantizationLevel.Q4_0)
            ]
        elif total_memory_gb >= 16:
            # Mid-range system
            recommended_models = [
                ("llama3.1:8b", QuantizationLevel.Q4_0),
                ("mistral:7b", QuantizationLevel.Q4_0),
                ("phi3:mini", None)
            ]
        else:
            # Low-end system
            recommended_models = [
                ("phi3:mini", QuantizationLevel.Q4_0),
                ("gemma2:2b", QuantizationLevel.Q4_0)
            ]
        
        successfully_pulled = []
        
        for model_name, quantization in recommended_models:
            try:
                if await self.pull_model(model_name, quantization):
                    full_name = f"{model_name}:{quantization.value}" if quantization else model_name
                    successfully_pulled.append(full_name)
            except Exception as e:
                console.print(f"[yellow]⚠️ Failed to pull {model_name}: {e}[/yellow]")
        
        return successfully_pulled
    
    def display_benchmark_results(self, results: List[BenchmarkResult]):
        """Display benchmark results in a nice table"""
        if not results:
            console.print("[yellow]No benchmark results to display[/yellow]")
            return
        
        table = Table(title="🔬 Model Benchmark Results")
        table.add_column("Model", style="cyan")
        table.add_column("Avg Time (ms)", style="yellow")
        table.add_column("Min Time (ms)", style="green")
        table.add_column("Memory (MB)", style="blue")
        table.add_column("Tokens/sec", style="magenta")
        table.add_column("Success Rate", style="white")
        table.add_column("Quality", style="red")
        
        for result in results:
            # Format values
            avg_time = f"{result.avg_inference_time_ms:.1f}" if result.avg_inference_time_ms != float('inf') else "∞"
            min_time = f"{result.min_inference_time_ms:.1f}" if result.min_inference_time_ms != float('inf') else "∞"
            memory = f"{result.memory_usage_mb:.1f}"
            tokens_sec = f"{result.tokens_per_second:.1f}"
            success = f"{result.success_rate:.1f}%"
            quality = f"{result.quality_score:.3f}"
            
            # Color coding for performance
            if result.avg_inference_time_ms < 50:
                avg_time = f"[green]{avg_time}[/green]"
            elif result.avg_inference_time_ms < 100:
                avg_time = f"[yellow]{avg_time}[/yellow]"
            else:
                avg_time = f"[red]{avg_time}[/red]"
            
            table.add_row(
                result.model_name,
                avg_time,
                min_time,
                memory,
                tokens_sec,
                success,
                quality
            )
        
        console.print(table)
    
    def display_optimization_results(self, optimization: Dict[str, Any]):
        """Display hardware optimization results"""
        console.print("\n[bold blue]⚡ Hardware Optimization Results[/bold blue]")
        
        # Hardware info
        hw_info = optimization["hardware_info"]
        console.print(f"\n[bold]Hardware Configuration:[/bold]")
        console.print(f"• CPU Cores: {hw_info['cpu_count']}")
        console.print(f"• Total Memory: {hw_info['total_memory_gb']:.1f} GB")
        console.print(f"• Available Memory: {hw_info['available_memory_gb']:.1f} GB")
        console.print(f"• Suitable Models: {optimization['suitable_models_count']}")
        
        # Recommendations
        recommendations = optimization["recommendations"]
        if any(recommendations.values()):
            console.print(f"\n[bold]Model Recommendations:[/bold]")
            
            if recommendations["fastest"]:
                console.print(f"• ⚡ Fastest: {recommendations['fastest']}")
            if recommendations["balanced"]:
                console.print(f"• ⚖️  Balanced: {recommendations['balanced']}")
            if recommendations["highest_quality"]:
                console.print(f"• 🎯 Highest Quality: {recommendations['highest_quality']}")
            if recommendations["most_efficient"]:
                console.print(f"• 🔋 Most Efficient: {recommendations['most_efficient']}")
        
        # Sub-50ms models
        sub_50ms = optimization["sub_50ms_models"]
        if sub_50ms:
            console.print(f"\n[bold green]🎯 Sub-50ms Models ({len(sub_50ms)}):[/bold green]")
            for model in sub_50ms:
                console.print(f"  • {model}")
        else:
            console.print(f"\n[yellow]⚠️ No models achieve <50ms target on this hardware[/yellow]")


# Convenience functions
async def create_ollama_optimizer() -> OllamaOptimizer:
    """Create and initialize Ollama optimizer"""
    optimizer = OllamaOptimizer()
    
    # Check if Ollama is running
    if not await optimizer.check_ollama_status():
        console.print("[red]❌ Ollama is not running. Please start Ollama first.[/red]")
        return None
    
    return optimizer


async def quick_benchmark_all_models() -> Dict[str, BenchmarkResult]:
    """Quick benchmark of all available models"""
    optimizer = await create_ollama_optimizer()
    if not optimizer:
        return {}
    
    models = await optimizer.list_available_models()
    results = {}
    
    for model in models:
        if model.status == ModelStatus.AVAILABLE:
            result = await optimizer.benchmark_model(model.name)
            results[model.name] = result
    
    return results


if __name__ == "__main__":
    async def main():
        """Demo Ollama optimization"""
        console.print(Panel(
            "[bold green]⚡ Ollama Optimizer Demo[/bold green]\n"
            "Advanced model management and optimization",
            title="Ollama Optimizer",
            border_style="green"
        ))
        
        optimizer = await create_ollama_optimizer()
        if not optimizer:
            return
        
        # List available models
        console.print("[blue]📋 Listing available models...[/blue]")
        models = await optimizer.list_available_models()
        
        if models:
            console.print(f"[green]Found {len(models)} models[/green]")
            
            # Benchmark first model
            first_model = models[0]
            console.print(f"[blue]🔬 Benchmarking {first_model.name}...[/blue]")
            result = await optimizer.benchmark_model(first_model.name)
            
            optimizer.display_benchmark_results([result])
            
            # Hardware optimization
            optimization = await optimizer.optimize_for_hardware()
            optimizer.display_optimization_results(optimization)
        else:
            console.print("[yellow]No models found. Try pulling some models first.[/yellow]")
    
    asyncio.run(main())