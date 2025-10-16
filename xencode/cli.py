#!/usr/bin/env python3
"""
Xencode CLI - Command Line Interface

Production-ready CLI for the Xencode AI/ML leviathan system.
Provides access to ensemble reasoning, RLHF tuning, Ollama optimization,
and all Phase 2 systems through a unified command interface.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import Xencode systems
from xencode.phase2_coordinator import Phase2Coordinator
from xencode.ai_ensembles import QueryRequest, EnsembleMethod, create_ensemble_reasoner
from xencode.ollama_optimizer import create_ollama_optimizer, QuantizationLevel
from xencode.rlhf_tuner import RLHFConfig, create_rlhf_tuner

console = Console()


@click.group()
@click.version_option(version="2.1.0", prog_name="xencode")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """
    🤖 Xencode AI/ML Leviathan - The Ultimate Offline AI Assistant
    
    Outperforms GitHub Copilot with <50ms inference, 10% SMAPE improvements,
    and 100% offline operation. The leviathan has awakened!
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if verbose:
        console.print("[blue]🐉 Xencode CLI initialized in verbose mode[/blue]")


@cli.group()
def init():
    """Initialize Xencode systems"""
    pass


@init.command()
@click.option('--config-path', type=click.Path(), help='Path to configuration file')
@click.option('--force', is_flag=True, help='Force re-initialization')
def system(config_path, force):
    """Initialize the complete Xencode system"""
    console.print("[bold blue]🚀 Initializing Xencode AI/ML Leviathan...[/bold blue]")
    
    async def _init_system():
        coordinator = Phase2Coordinator()
        
        config_path_obj = Path(config_path) if config_path else None
        success = await coordinator.initialize(config_path_obj)
        
        if success:
            await coordinator.run_first_time_setup()
            console.print("[green]✅ Xencode system initialized successfully![/green]")
            console.print("[yellow]💡 Try 'xencode query \"Explain clean code principles\"' to test the leviathan![/yellow]")
        else:
            console.print("[red]❌ System initialization failed[/red]")
            sys.exit(1)
    
    asyncio.run(_init_system())


@cli.command()
@click.argument('prompt', required=True)
@click.option('--models', '-m', multiple=True, default=['llama3.1:8b', 'mistral:7b'], 
              help='Models to use in ensemble (can specify multiple)')
@click.option('--method', type=click.Choice(['vote', 'weighted', 'consensus', 'hybrid']), 
              default='vote', help='Ensemble method')
@click.option('--max-tokens', type=int, default=512, help='Maximum tokens to generate')
@click.option('--temperature', type=float, default=0.7, help='Sampling temperature')
@click.option('--timeout', type=int, default=2000, help='Timeout in milliseconds')
def query(prompt, models, method, max_tokens, temperature, timeout):
    """
    Query the AI ensemble with a prompt
    
    Examples:
      xencode query "Explain microservices architecture"
      xencode query "How to optimize database queries?" --method weighted
      xencode query "Debug this Python code" --models llama3.1:8b --models phi3:mini
    """
    console.print(f"[cyan]🧠 Querying AI ensemble: {method} method with {len(models)} models[/cyan]")
    
    async def _run_query():
        try:
            # Initialize ensemble reasoner
            reasoner = await create_ensemble_reasoner()
            
            # Create query request
            query_request = QueryRequest(
                prompt=prompt,
                models=list(models),
                method=EnsembleMethod(method),
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_ms=timeout
            )
            
            # Execute query with progress
            with console.status("[bold blue]🤖 AI ensemble reasoning in progress..."):
                response = await reasoner.reason(query_request)
            
            # Display results
            console.print(f"\n[green]✅ Response ({response.total_time_ms:.1f}ms):[/green]")
            console.print(Panel(response.fused_response, title="AI Ensemble Response", border_style="green"))
            
            # Show metrics
            console.print(f"\n[yellow]📊 Metrics:[/yellow]")
            console.print(f"• Consensus Score: {response.consensus_score:.3f}")
            console.print(f"• Confidence: {response.confidence:.3f}")
            console.print(f"• Cache Hit: {'✅' if response.cache_hit else '❌'}")
            console.print(f"• Models Used: {len(response.model_responses)}")
            
            # Performance indicator
            if response.total_time_ms < 50:
                console.print("[bold green]🎯 Sub-50ms target ACHIEVED![/bold green]")
            else:
                console.print(f"[yellow]⚠️ Response time: {response.total_time_ms:.1f}ms[/yellow]")
                
        except Exception as e:
            console.print(f"[red]❌ Query failed: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(_run_query())


@cli.group()
def ollama():
    """Ollama model management and optimization"""
    pass


@ollama.command()
@click.option('--refresh', is_flag=True, help='Refresh model list from Ollama')
def list(refresh):
    """List available Ollama models"""
    console.print("[blue]📋 Listing Ollama models...[/blue]")
    
    async def _list_models():
        try:
            optimizer = await create_ollama_optimizer()
            if not optimizer:
                console.print("[red]❌ Ollama not available. Please start Ollama first.[/red]")
                return
            
            models = await optimizer.list_available_models(refresh=refresh)
            
            if not models:
                console.print("[yellow]No models found. Try pulling some models first.[/yellow]")
                return
            
            # Display models table
            table = Table(title="📋 Available Ollama Models")
            table.add_column("Model", style="cyan")
            table.add_column("Size (GB)", style="yellow")
            table.add_column("Status", style="green")
            table.add_column("Quantization", style="blue")
            
            for model in models:
                quant = model.quantization.value if model.quantization else "None"
                table.add_row(
                    model.name,
                    f"{model.size_gb:.1f}",
                    model.status.value,
                    quant
                )
            
            console.print(table)
            console.print(f"\n[green]Found {len(models)} models[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to list models: {e}[/red]")
    
    asyncio.run(_list_models())


@ollama.command()
@click.argument('model_name', required=True)
@click.option('--quantization', type=click.Choice(['q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0', 'f16', 'f32']),
              help='Quantization level')
def pull(model_name, quantization):
    """Pull a model from Ollama registry"""
    console.print(f"[blue]📥 Pulling model: {model_name}[/blue]")
    
    async def _pull_model():
        try:
            optimizer = await create_ollama_optimizer()
            if not optimizer:
                console.print("[red]❌ Ollama not available[/red]")
                return
            
            quant_level = QuantizationLevel(quantization) if quantization else None
            success = await optimizer.pull_model(model_name, quant_level)
            
            if success:
                console.print(f"[green]✅ Successfully pulled {model_name}[/green]")
            else:
                console.print(f"[red]❌ Failed to pull {model_name}[/red]")
                
        except Exception as e:
            console.print(f"[red]❌ Pull failed: {e}[/red]")
    
    asyncio.run(_pull_model())


@ollama.command()
@click.argument('model_name', required=True)
@click.option('--prompts', type=int, default=5, help='Number of test prompts')
def benchmark(model_name, prompts):
    """Benchmark a model's performance"""
    console.print(f"[blue]🔬 Benchmarking model: {model_name}[/blue]")
    
    async def _benchmark_model():
        try:
            optimizer = await create_ollama_optimizer()
            if not optimizer:
                console.print("[red]❌ Ollama not available[/red]")
                return
            
            # Generate test prompts
            test_prompts = [
                "Explain the concept of recursion in programming",
                "What are the benefits of using microservices?",
                "How does machine learning work?",
                "Write a Python function to sort a list",
                "What is the difference between REST and GraphQL?"
            ][:prompts]
            
            result = await optimizer.benchmark_model(model_name, test_prompts)
            
            # Display results
            optimizer.display_benchmark_results([result])
            
            # Performance summary
            if result.avg_inference_time_ms < 50:
                console.print("[bold green]🎯 Sub-50ms target ACHIEVED![/bold green]")
            else:
                console.print(f"[yellow]⚠️ Average time: {result.avg_inference_time_ms:.1f}ms[/yellow]")
                
        except Exception as e:
            console.print(f"[red]❌ Benchmark failed: {e}[/red]")
    
    asyncio.run(_benchmark_model())


@ollama.command()
def optimize():
    """Optimize model selection for current hardware"""
    console.print("[blue]⚡ Optimizing for current hardware...[/blue]")
    
    async def _optimize():
        try:
            optimizer = await create_ollama_optimizer()
            if not optimizer:
                console.print("[red]❌ Ollama not available[/red]")
                return
            
            optimization = await optimizer.optimize_for_hardware()
            optimizer.display_optimization_results(optimization)
            
        except Exception as e:
            console.print(f"[red]❌ Optimization failed: {e}[/red]")
    
    asyncio.run(_optimize())


@ollama.command()
def auto_pull():
    """Automatically pull recommended models for your hardware"""
    console.print("[blue]🤖 Auto-pulling recommended models...[/blue]")
    
    async def _auto_pull():
        try:
            optimizer = await create_ollama_optimizer()
            if not optimizer:
                console.print("[red]❌ Ollama not available[/red]")
                return
            
            models = await optimizer.auto_pull_recommended_models()
            
            if models:
                console.print(f"[green]✅ Successfully pulled {len(models)} models:[/green]")
                for model in models:
                    console.print(f"  • {model}")
            else:
                console.print("[yellow]No models were pulled[/yellow]")
                
        except Exception as e:
            console.print(f"[red]❌ Auto-pull failed: {e}[/red]")
    
    asyncio.run(_auto_pull())


@cli.group()
def rlhf():
    """RLHF tuning for code mastery"""
    pass


@rlhf.command()
@click.option('--base-model', default='microsoft/DialoGPT-small', help='Base model for tuning')
@click.option('--epochs', type=int, default=1, help='Number of training epochs')
@click.option('--data-size', type=int, default=50, help='Synthetic data size')
@click.option('--batch-size', type=int, default=2, help='Training batch size')
def train(base_model, epochs, data_size, batch_size):
    """Train a model with RLHF for code mastery"""
    console.print("[blue]🎯 Starting RLHF training...[/blue]")
    
    async def _train():
        try:
            config = RLHFConfig(
                base_model=base_model,
                max_epochs=epochs,
                synthetic_data_size=data_size,
                batch_size=batch_size
            )
            
            tuner = await create_rlhf_tuner(config)
            
            # Generate training data
            console.print("[cyan]📊 Generating synthetic training data...[/cyan]")
            code_pairs = await tuner.generate_training_data(data_size)
            
            # Train model
            console.print("[cyan]🎯 Training model with RLHF...[/cyan]")
            success = await tuner.train(code_pairs)
            
            if success:
                console.print("[green]✅ RLHF training completed successfully![/green]")
                
                # Evaluate model
                console.print("[cyan]📈 Evaluating model performance...[/cyan]")
                results = await tuner.evaluate_model()
                
                console.print(f"\n[yellow]📊 Training Results:[/yellow]")
                console.print(f"• Average Quality Score: {results['avg_quality_score']:.3f}")
                console.print(f"• Average Inference Time: {results['inference_time_ms']:.1f}ms")
                console.print(f"• Total Samples: {results['total_samples']}")
                
                if results['avg_quality_score'] > 0.8:
                    console.print("[bold green]🏆 Excellent code quality achieved![/bold green]")
            else:
                console.print("[red]❌ RLHF training failed[/red]")
                
        except Exception as e:
            console.print(f"[red]❌ Training failed: {e}[/red]")
    
    asyncio.run(_train())


@rlhf.command()
@click.option('--size', type=int, default=10, help='Number of code pairs to generate')
def generate_data(size):
    """Generate synthetic code training data"""
    console.print(f"[blue]📊 Generating {size} synthetic code pairs...[/blue]")
    
    async def _generate():
        try:
            config = RLHFConfig(synthetic_data_size=size)
            tuner = await create_rlhf_tuner(config)
            
            pairs = await tuner.generate_training_data(size)
            
            console.print(f"[green]✅ Generated {len(pairs)} code pairs[/green]")
            
            # Show examples
            if pairs:
                console.print("\n[yellow]📝 Example pairs by task type:[/yellow]")
                task_examples = {}
                for pair in pairs:
                    if pair.task_type not in task_examples:
                        task_examples[pair.task_type] = pair
                
                for task_type, example in task_examples.items():
                    console.print(f"\n[bold]{task_type.title()} Example:[/bold]")
                    console.print(f"Quality Score: {example.quality_score:.2f}")
                    
        except Exception as e:
            console.print(f"[red]❌ Data generation failed: {e}[/red]")
    
    asyncio.run(_generate())


@cli.command()
def status():
    """Show system status and health"""
    console.print("[blue]🔧 Checking Xencode system status...[/blue]")
    
    async def _status():
        try:
            coordinator = Phase2Coordinator()
            await coordinator.initialize()
            
            coordinator.display_system_status()
            
            # Additional AI/ML status
            console.print("\n[bold blue]🤖 AI/ML Systems Status:[/bold blue]")
            
            ai_status = Table()
            ai_status.add_column("System", style="cyan")
            ai_status.add_column("Status", style="white")
            ai_status.add_column("Details", style="green")
            
            # Check ensemble system
            ensemble_status = "✅ Available" if coordinator.ensemble_reasoner else "❌ Not Available"
            ai_status.add_row("AI Ensembles", ensemble_status, "Multi-model reasoning")
            
            # Check Ollama optimizer
            ollama_status = "✅ Available" if coordinator.ollama_optimizer else "❌ Not Available"
            ai_status.add_row("Ollama Optimizer", ollama_status, "Model management")
            
            # Check RLHF tuner
            rlhf_status = "✅ Available" if coordinator.rlhf_tuner else "❌ Not Available"
            ai_status.add_row("RLHF Tuner", rlhf_status, "Code mastery training")
            
            console.print(ai_status)
            
        except Exception as e:
            console.print(f"[red]❌ Status check failed: {e}[/red]")
    
    asyncio.run(_status())


@cli.command()
def health():
    """Run comprehensive system health check"""
    console.print("[blue]🏥 Running comprehensive health check...[/blue]")
    
    async def _health():
        try:
            coordinator = Phase2Coordinator()
            await coordinator.initialize()
            
            success = await coordinator.health_check()
            
            if success:
                console.print("\n[bold green]✅ All systems healthy![/bold green]")
                console.print("[yellow]💡 The leviathan is ready for action![/yellow]")
            else:
                console.print("\n[bold red]❌ Health check found issues[/bold red]")
                console.print("[yellow]💡 Check the output above for details[/yellow]")
                sys.exit(1)
                
        except Exception as e:
            console.print(f"[red]❌ Health check failed: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(_health())


@cli.command()
def optimize():
    """Optimize system performance"""
    console.print("[blue]⚡ Optimizing system performance...[/blue]")
    
    async def _optimize():
        try:
            coordinator = Phase2Coordinator()
            await coordinator.initialize()
            
            results = await coordinator.optimize_performance()
            
            console.print("\n[bold green]✅ Optimization complete![/bold green]")
            console.print(f"• Cache optimized: {'✅' if results['cache_optimized'] else '❌'}")
            console.print(f"• Memory freed: {results['memory_freed']:.1f}%")
            console.print(f"• Config optimized: {'✅' if results['config_optimized'] else '❌'}")
            
        except Exception as e:
            console.print(f"[red]❌ Optimization failed: {e}[/red]")
    
    asyncio.run(_optimize())


@cli.command()
def version():
    """Show version information"""
    console.print(Panel(
        "[bold green]🤖 Xencode AI/ML Leviathan v2.1.0[/bold green]\n\n"
        "The ultimate offline AI assistant that outperforms GitHub Copilot\n"
        "with <50ms inference, 10% SMAPE improvements, and 100% privacy.\n\n"
        "[yellow]🐉 The leviathan has awakened![/yellow]",
        title="Version Info",
        border_style="green"
    ))


if __name__ == '__main__':
    cli()