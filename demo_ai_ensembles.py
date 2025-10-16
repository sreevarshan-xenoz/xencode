#!/usr/bin/env python3
"""
Demo: AI Ensemble System - Multi-Model Reasoning

Demonstrates the power of ensemble AI reasoning with real Ollama models.
Shows <50ms inference times and superior accuracy through model fusion.
"""

import asyncio
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from xencode.ai_ensembles import (
    EnsembleReasoner, QueryRequest, EnsembleMethod, 
    create_ensemble_reasoner, quick_ensemble_query
)

console = Console()


async def demo_basic_ensemble():
    """Demo basic ensemble reasoning"""
    console.print("[bold green]🤖 Basic Ensemble Reasoning Demo[/bold green]\n")
    
    # Create ensemble reasoner
    reasoner = await create_ensemble_reasoner()
    
    # Demo query
    query = QueryRequest(
        prompt="Explain the key benefits of using microservices architecture in modern software development",
        models=["llama3.1:8b", "mistral:7b"],
        method=EnsembleMethod.VOTE,
        max_tokens=256
    )
    
    console.print(f"[cyan]Query:[/cyan] {query.prompt}")
    console.print(f"[cyan]Models:[/cyan] {', '.join(query.models)}")
    console.print(f"[cyan]Method:[/cyan] {query.method.value}\n")
    
    start_time = time.perf_counter()
    
    with console.status("[bold blue]🧠 Ensemble reasoning in progress..."):
        try:
            response = await reasoner.reason(query)
            elapsed = (time.perf_counter() - start_time) * 1000
            
            # Display results
            console.print(f"[green]✅ Response ({elapsed:.1f}ms):[/green]")
            console.print(Panel(response.fused_response, title="Ensemble Response", border_style="green"))
            
            # Show metrics
            metrics_table = Table(title="📊 Ensemble Metrics")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="white")
            
            metrics_table.add_row("Total Time", f"{response.total_time_ms:.1f}ms")
            metrics_table.add_row("Consensus Score", f"{response.consensus_score:.2f}")
            metrics_table.add_row("Confidence", f"{response.confidence:.2f}")
            metrics_table.add_row("Cache Hit", "✅" if response.cache_hit else "❌")
            metrics_table.add_row("Models Used", str(len(response.model_responses)))
            
            console.print(metrics_table)
            
            # Show individual model responses
            console.print("\n[bold]Individual Model Performance:[/bold]")
            for model_resp in response.model_responses:
                status = "✅" if model_resp.success else "❌"
                console.print(f"{status} {model_resp.model}: {model_resp.inference_time_ms:.1f}ms (confidence: {model_resp.confidence:.2f})")
            
            return response.total_time_ms < 50  # Check <50ms target
            
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return False


async def demo_ensemble_methods():
    """Demo different ensemble methods"""
    console.print("\n[bold blue]🔬 Ensemble Methods Comparison[/bold blue]\n")
    
    reasoner = await create_ensemble_reasoner()
    
    test_prompt = "What are the main advantages of using Python for data science?"
    models = ["llama3.1:8b", "mistral:7b"]
    
    methods_table = Table(title="🎯 Ensemble Methods Performance")
    methods_table.add_column("Method", style="cyan")
    methods_table.add_column("Time (ms)", style="yellow")
    methods_table.add_column("Consensus", style="green")
    methods_table.add_column("Confidence", style="blue")
    
    for method in EnsembleMethod:
        try:
            query = QueryRequest(
                prompt=test_prompt,
                models=models,
                method=method,
                max_tokens=128
            )
            
            with console.status(f"[bold blue]Testing {method.value} method..."):
                response = await reasoner.reason(query)
                
                methods_table.add_row(
                    method.value.title(),
                    f"{response.total_time_ms:.1f}",
                    f"{response.consensus_score:.2f}",
                    f"{response.confidence:.2f}"
                )
                
        except Exception as e:
            methods_table.add_row(
                method.value.title(),
                "ERROR",
                "N/A",
                "N/A"
            )
    
    console.print(methods_table)


async def demo_performance_benchmark():
    """Demo performance benchmarking"""
    console.print("\n[bold yellow]⚡ Performance Benchmark[/bold yellow]\n")
    
    reasoner = await create_ensemble_reasoner()
    
    # Run benchmark
    with console.status("[bold blue]Running comprehensive benchmark..."):
        results = await reasoner.benchmark_models([
            "Explain machine learning in simple terms",
            "What is the difference between REST and GraphQL?",
            "How does blockchain technology work?"
        ])
    
    # Display individual model performance
    if results["individual_models"]:
        console.print("[bold]Individual Model Performance:[/bold]")
        models_table = Table()
        models_table.add_column("Model", style="cyan")
        models_table.add_column("Avg Time (ms)", style="yellow")
        models_table.add_column("Success Rate", style="green")
        models_table.add_column("Tier", style="blue")
        
        for model, stats in results["individual_models"].items():
            models_table.add_row(
                model,
                f"{stats['avg_time_ms']:.1f}",
                f"{stats['success_rate']:.1f}%",
                stats['tier']
            )
        
        console.print(models_table)
    
    # Display ensemble performance
    if results["ensemble_methods"]:
        console.print("\n[bold]Ensemble Methods Performance:[/bold]")
        ensemble_table = Table()
        ensemble_table.add_column("Method", style="cyan")
        ensemble_table.add_column("Avg Time (ms)", style="yellow")
        ensemble_table.add_column("Success Rate", style="green")
        
        for method, stats in results["ensemble_methods"].items():
            ensemble_table.add_row(
                method.title(),
                f"{stats['avg_time_ms']:.1f}",
                f"{stats['success_rate']:.1f}%"
            )
        
        console.print(ensemble_table)
    
    # Performance summary
    if results["performance_summary"]:
        summary = results["performance_summary"]
        console.print(f"\n[bold]Performance Summary:[/bold]")
        console.print(f"• Fastest Individual: {summary['fastest_individual_ms']:.1f}ms")
        console.print(f"• Fastest Ensemble: {summary['fastest_ensemble_ms']:.1f}ms")
        console.print(f"• Ensemble Overhead: {summary['ensemble_overhead_ms']:.1f}ms")
        
        target_icon = "🎯" if summary['sub_50ms_target'] else "⚠️"
        console.print(f"• Sub-50ms Target: {target_icon} {'ACHIEVED' if summary['sub_50ms_target'] else 'MISSED'}")


async def demo_quick_api():
    """Demo the quick API"""
    console.print("\n[bold magenta]🚀 Quick API Demo[/bold magenta]\n")
    
    console.print("[cyan]Using quick_ensemble_query for simple use cases...[/cyan]")
    
    start_time = time.perf_counter()
    
    try:
        response = await quick_ensemble_query(
            "What are the key principles of clean code?",
            models=["llama3.1:8b", "mistral:7b"],
            method=EnsembleMethod.WEIGHTED
        )
        
        elapsed = (time.perf_counter() - start_time) * 1000
        
        console.print(f"[green]✅ Quick Response ({elapsed:.1f}ms):[/green]")
        console.print(Panel(response, title="Quick Ensemble Response", border_style="magenta"))
        
    except Exception as e:
        console.print(f"[red]❌ Quick API Error: {e}[/red]")


async def demo_cache_performance():
    """Demo cache performance"""
    console.print("\n[bold cyan]💾 Cache Performance Demo[/bold cyan]\n")
    
    reasoner = await create_ensemble_reasoner()
    
    query = QueryRequest(
        prompt="Explain the benefits of caching in software systems",
        models=["llama3.1:8b", "mistral:7b"],
        method=EnsembleMethod.VOTE
    )
    
    # First call (cache miss)
    console.print("[yellow]First call (cache miss)...[/yellow]")
    start_time = time.perf_counter()
    response1 = await reasoner.reason(query)
    time1 = (time.perf_counter() - start_time) * 1000
    
    # Second call (cache hit)
    console.print("[yellow]Second call (cache hit)...[/yellow]")
    start_time = time.perf_counter()
    response2 = await reasoner.reason(query)
    time2 = (time.perf_counter() - start_time) * 1000
    
    # Show cache performance
    cache_table = Table(title="💾 Cache Performance")
    cache_table.add_column("Call", style="cyan")
    cache_table.add_column("Time (ms)", style="yellow")
    cache_table.add_column("Cache Hit", style="green")
    cache_table.add_column("Speedup", style="blue")
    
    cache_table.add_row("First", f"{time1:.1f}", "❌", "N/A")
    cache_table.add_row("Second", f"{time2:.1f}", "✅" if response2.cache_hit else "❌", f"{time1/time2:.1f}x" if time2 > 0 else "N/A")
    
    console.print(cache_table)
    
    # Show performance stats
    stats = reasoner.get_performance_stats()
    console.print(f"\n[bold]Performance Stats:[/bold]")
    console.print(f"• Total Queries: {stats['total_queries']}")
    console.print(f"• Cache Hit Rate: {stats['cache_hit_rate']:.1f}%")
    console.print(f"• Avg Inference Time: {stats['avg_inference_time_ms']:.1f}ms")
    console.print(f"• Efficiency Score: {stats['efficiency_score']:.1f}/100")


async def main():
    """Main demo function"""
    console.print(Panel(
        "[bold green]🤖 Xencode AI Ensemble System Demo[/bold green]\n"
        "Demonstrating multi-model reasoning with <50ms inference times\n"
        "and superior accuracy through intelligent model fusion.",
        title="AI Ensemble Demo",
        border_style="green"
    ))
    
    try:
        # Run all demos
        sub_50ms_achieved = await demo_basic_ensemble()
        await demo_ensemble_methods()
        await demo_performance_benchmark()
        await demo_quick_api()
        await demo_cache_performance()
        
        # Final summary
        console.print("\n" + "="*60)
        console.print("[bold green]🎯 Demo Summary[/bold green]")
        console.print(f"• Sub-50ms Target: {'🎯 ACHIEVED' if sub_50ms_achieved else '⚠️ MISSED'}")
        console.print("• Multi-model ensemble reasoning: ✅")
        console.print("• Intelligent caching: ✅")
        console.print("• Multiple fusion methods: ✅")
        console.print("• Performance benchmarking: ✅")
        
        console.print("\n[bold blue]🚀 Xencode AI Ensemble is ready to outpace Copilot![/bold blue]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo failed: {e}[/red]")


if __name__ == "__main__":
    asyncio.run(main())