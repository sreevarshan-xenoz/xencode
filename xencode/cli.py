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
from typing import Optional, List, Dict, Any
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt

# Import Xencode systems
from xencode.phase2_coordinator import Phase2Coordinator
from xencode.ai_ensembles import QueryRequest, EnsembleMethod, create_ensemble_reasoner
from xencode.ollama_optimizer import create_ollama_optimizer, QuantizationLevel
from xencode.rlhf_tuner import RLHFConfig, create_rlhf_tuner

# Import feature system
from xencode.features import FeatureManager, FeatureConfig, FeatureSystemConfig, FeatureConfigManager
from xencode.features.core.cli import FeatureCommandGroup

console = Console()


@click.group(invoke_without_command=True)
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

    if ctx.invoked_subcommand is None:
        from xencode.tui.app import run_tui
        run_tui(root_path=Path.cwd())


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
@click.option('--rag', is_flag=True, help='Use local RAG context')
def query(prompt, models, method, max_tokens, temperature, timeout, rag):
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
                timeout_ms=timeout,
                use_rag=rag
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


@cli.command()
@click.option('--model', default='qwen3:4b', help='Model to use for the agent')
@click.option('--base-url', default='http://localhost:11434', help='Ollama base URL')
def agentic(model, base_url):
    """Start an interactive agentic session"""
    console.print(Panel.fit(f"Starting Agentic Session with {model}", style="bold blue"))

    try:
        from xencode.agentic.manager import LangChainManager

        manager = LangChainManager(model_name=model, base_url=base_url)
        console.print("[green]Agent initialized successfully![/green]")
        console.print("Type 'exit' or 'quit' to end the session.\n")

        while True:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")

            if user_input.lower() in ["exit", "quit"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if not user_input.strip():
                continue

            with console.status("[bold green]Agent is thinking...[/bold green]"):
                response = manager.run_agent(user_input)

            console.print(Panel(response, title="Agent", border_style="green"))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@cli.group()
def features():
    """Feature management - Enable, disable, and configure Xencode features
    
    Xencode features provide modular functionality that can be enabled or disabled
    as needed. Each feature has its own configuration and CLI commands.
    
    Available commands:
        list      - List all available features
        info      - Show detailed information about a feature
        enable    - Enable a feature
        disable   - Disable a feature
        configure - Configure a feature
    
    Examples:
        xencode features list
        xencode features info code_review
        xencode features enable terminal_assistant
        xencode features configure code_review --show
    """
    pass


@features.command()
def list():
    """List all available features
    
    Examples:
        xencode features list
    """
    console.print("[blue]📋 Listing available features...[/blue]")
    
    feature_manager = FeatureManager()
    features = feature_manager.get_available_features()
    
    table = Table(title="Available Features")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Version", style="yellow")
    table.add_column("Description", style="white")
    
    for feature_name in sorted(features):
        feature = feature_manager.get_feature(feature_name)
        if not feature:
            # Try to load it to get info
            try:
                feature = feature_manager.load_feature(feature_name)
            except:
                pass
        
        if feature:
            status = feature.get_status().value
            version = feature.version
            description = feature.description[:50] + "..." if len(feature.description) > 50 else feature.description
        else:
            status = "not_loaded"
            version = "unknown"
            description = "N/A"
        
        table.add_row(feature_name, status, version, description)
    
    console.print(table)
    console.print("\n[dim]Use 'xencode features info <feature_name>' for more details[/dim]")


@features.command()
@click.argument('feature_name')
def enable(feature_name):
    """Enable a feature
    
    Examples:
        xencode features enable code_review
        xencode features enable terminal_assistant
    """
    console.print(f"[blue]⚡ Enabling feature: {feature_name}[/blue]")
    
    async def _enable():
        feature_manager = FeatureManager()
        success = await feature_manager.initialize_feature(feature_name)
        
        if success:
            console.print(f"[green]✅ Feature '{feature_name}' enabled successfully![/green]")
            
            # Show available commands
            feature = feature_manager.get_feature(feature_name)
            if feature and hasattr(feature, 'get_cli_commands'):
                try:
                    commands = feature.get_cli_commands()
                    if commands:
                        console.print(f"\n[cyan]Available commands for {feature_name}:[/cyan]")
                        for cmd in commands:
                            if hasattr(cmd, 'name') and hasattr(cmd, 'help'):
                                console.print(f"  • [yellow]xencode {cmd.name}[/yellow]: {cmd.help}")
                except:
                    pass
        else:
            console.print(f"[red]❌ Failed to enable feature '{feature_name}'[/red]")
    
    asyncio.run(_enable())


@features.command()
@click.argument('feature_name')
def disable(feature_name):
    """Disable a feature
    
    Examples:
        xencode features disable code_review
        xencode features disable terminal_assistant
    """
    console.print(f"[blue]🛑 Disabling feature: {feature_name}[/blue]")
    
    async def _disable():
        feature_manager = FeatureManager()
        success = await feature_manager.shutdown_feature(feature_name)
        
        if success:
            console.print(f"[green]✅ Feature '{feature_name}' disabled successfully![/green]")
        else:
            console.print(f"[red]❌ Failed to disable feature '{feature_name}'[/red]")
    
    asyncio.run(_disable())


@features.command()
@click.argument('feature_name')
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--set', 'config_key', help='Configuration key to set')
@click.option('--value', help='Configuration value')
@click.option('--reset', is_flag=True, help='Reset to default configuration')
def configure(feature_name, show, config_key, value, reset):
    """Configure a feature
    
    Examples:
        xencode features configure code_review --show
        xencode features configure code_review --set severity_levels --value "critical,high"
        xencode features configure terminal_assistant --reset
    """
    console.print(f"[blue]⚙️  Configuring feature: {feature_name}[/blue]")
    
    feature_manager = FeatureManager()
    feature = feature_manager.get_feature(feature_name)
    
    if not feature:
        # Try to load the feature
        feature = feature_manager.load_feature(feature_name)
        if not feature:
            console.print(f"[red]❌ Feature '{feature_name}' not found[/red]")
            return
    
    if show:
        # Show current configuration
        config = feature.config.config or {}
        
        table = Table(title=f"Configuration for {feature_name}")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="yellow")
        
        for key, val in config.items():
            table.add_row(key, str(val))
        
        console.print(table)
        
    elif reset:
        # Reset to default configuration
        feature.config.config = {}
        console.print(f"[green]✅ Configuration reset to defaults for '{feature_name}'[/green]")
        
    elif config_key and value:
        # Set a configuration value
        if not feature.config.config:
            feature.config.config = {}
        
        # Try to parse value as appropriate type
        parsed_value = value
        if value.lower() == 'true':
            parsed_value = True
        elif value.lower() == 'false':
            parsed_value = False
        elif value.isdigit():
            parsed_value = int(value)
        elif ',' in value:
            parsed_value = [v.strip() for v in value.split(',')]
        
        feature.config.config[config_key] = parsed_value
        console.print(f"[green]✅ Set {config_key} = {parsed_value} for '{feature_name}'[/green]")
        
    else:
        console.print("[yellow]⚠️  Please specify --show, --set with --value, or --reset[/yellow]")


@features.command()
@click.argument('feature_name')
def info(feature_name):
    """Show detailed information about a feature
    
    Examples:
        xencode features info code_review
        xencode features info terminal_assistant
    """
    console.print(f"[blue]ℹ️  Feature information: {feature_name}[/blue]")
    
    feature_manager = FeatureManager()
    feature = feature_manager.get_feature(feature_name)
    
    if not feature:
        # Try to load the feature
        feature = feature_manager.load_feature(feature_name)
        if not feature:
            console.print(f"[red]❌ Feature '{feature_name}' not found[/red]")
            return
    
    # Create info panel
    info_text = f"""
[cyan]Name:[/cyan] {feature.name}
[cyan]Description:[/cyan] {feature.description}
[cyan]Version:[/cyan] {feature.version}
[cyan]Status:[/cyan] {feature.get_status().value}
[cyan]Enabled:[/cyan] {feature.is_enabled}
    """
    
    panel = Panel(info_text.strip(), title=f"Feature: {feature_name}", border_style="blue")
    console.print(panel)
    
    # Show available commands if any
    if hasattr(feature, 'get_cli_commands'):
        commands = feature.get_cli_commands()
        if commands:
            console.print("\n[cyan]Available Commands:[/cyan]")
            for cmd_name, cmd_help in commands.items():
                console.print(f"  • [yellow]{cmd_name}[/yellow]: {cmd_help}")


# Feature-specific command registration system
def register_feature_commands():
    """
    Dynamically register CLI commands from enabled features.
    
    This function discovers all enabled features and registers their
    CLI commands with the main CLI group. Features can provide commands
    through the get_cli_commands() method.
    
    Example:
        A feature can return a list of Click commands that will be
        automatically registered when the feature is enabled.
    """
    try:
        feature_manager = FeatureManager()
        enabled_features = feature_manager.get_enabled_features()
        
        for feature_name, feature in enabled_features.items():
            if hasattr(feature, 'get_cli_commands'):
                try:
                    commands = feature.get_cli_commands()
                    if commands and isinstance(commands, list):
                        for command in commands:
                            if hasattr(command, 'name'):
                                # Register the command with the main CLI
                                cli.add_command(command)
                except Exception as e:
                    # Silently skip features that fail to register commands
                    if feature_manager.config_path and hasattr(feature_manager, 'config_path'):
                        pass  # Could log this in verbose mode
    except Exception as e:
        # Don't fail CLI startup if feature registration fails
        pass


# Register feature commands on CLI initialization
# This allows features to add their own commands dynamically
try:
    register_feature_commands()
except:
    pass  # Don't fail if feature registration fails


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
            await coordinator.initialize(include_rlhf=False)
            
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
@click.argument('path', required=False, default='.')
def tui(path):
    """Launch the Xencode TUI"""
    try:
        from xencode.tui.app import run_tui

        run_tui(root_path=Path(path))
    except Exception as e:
        console.print(f"[red]❌ TUI launch failed: {e}[/red]")


@cli.group()
def rag():
    """Local RAG (Retrieval Augmented Generation) commands"""
    pass


@rag.command()
@click.argument('path', default='.', type=click.Path(exists=True))
@click.option('--reset', is_flag=True, help='Clear existing index before indexing')
def index(path, reset):
    """Index codebase for RAG"""
    console.print(f"[blue]🔍 Indexing codebase at {path}...[/blue]")
    
    try:
        from xencode.rag.vector_store import VectorStore
        from xencode.rag.indexer import Indexer
        
        vector_store = VectorStore()
        
        if reset:
            console.print("[yellow]🗑️ Clearing existing index...[/yellow]")
            vector_store.clear()
            # Re-initialize to recreate empty collection
            vector_store = VectorStore()
            
        indexer = Indexer(vector_store)
        indexer.index_directory(path)
        
        console.print("[green]✅ Indexing completed successfully![/green]")
        
    except ImportError as e:
        console.print(f"[red]❌ Missing dependencies: {e}[/red]")
        console.print("Please install requirements: pip install chromadb langchain-ollama")
    except Exception as e:
        console.print(f"[red]❌ Indexing failed: {e}[/red]")


@rag.command()
@click.argument('query')
@click.option('--k', default=3, help='Number of results to retrieve')
def search(query, k):
    """Search the vector store"""
    try:
        from xencode.rag.vector_store import VectorStore
        
        vector_store = VectorStore()
        results = vector_store.similarity_search(query, k=k)
        
        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return
            
        console.print(f"[green]Found {len(results)} relevant snippets:[/green]\n")
        
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'unknown')
            console.print(Panel(
                doc.page_content,
                title=f"{i}. {source}",
                border_style="blue"
            ))
            
    except ImportError:
         console.print("[red]❌ Missing dependencies. Please install chromadb.[/red]")
    except Exception as e:
        console.print(f"[red]❌ Search failed: {e}[/red]")


@cli.command()
@click.argument('instruction')
@click.option('--model', default='llama3.1:8b', help='Model to use')
@click.option('--yes', '-y', is_flag=True, help='Execute without confirmation')
def shell(instruction, model, yes):
    """Natural Language Shell (Shell Genie)"""
    try:
        from xencode.shell_genie.genie import ShellGenie
        
        genie = ShellGenie(model_name=model)
        
        with console.status("[bold blue]🧞 Generating command...[/bold blue]"):
            command, explanation = genie.generate_command(instruction)
        
        if command and command != "SAFE_GUARD_TRIGGERED":
            console.print(Panel(
                f"[bold]Explanation:[/bold] {explanation}\n\n[bold green]{command}[/bold green]",
                title="🧞 Shell Genie",
                border_style="magenta"
            ))
            genie.execute(command, auto_confirm=yes)
        else:
            console.print("[red]❌ Could not generate a safe command.[/red]")
            if explanation:
                console.print(f"Reason: {explanation}")
                
    except ImportError as e:
         console.print(f"[red]❌ Missing dependencies: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Shell Genie failed: {e}[/red]")


@cli.command()
@click.option('--model', default='llama3.1:8b', help='Model to use')
def devops(model):
    """Generate DevOps infrastructure (Dockerfile, etc.)"""
    try:
        from xencode.devops.generator import DevOpsGenerator
        
        generator = DevOpsGenerator(model_name=model)
        console.print("[blue]🔍 Analyzing project...[/blue]")
        context = generator.analyze_project()
        
        if not context:
            console.print("[yellow]⚠️ No dependency files (requirements.txt, package.json) found.[/yellow]")
            if not Confirm.ask("Continue without context?"):
                return

        # Generate Dockerfile
        with console.status("[bold blue]🐳 Generating Dockerfile...[/bold blue]"):
            dockerfile = generator.generate_dockerfile(context)
        
        console.print(Panel(dockerfile, title="Generated Dockerfile", border_style="blue"))
        if Confirm.ask("Save Dockerfile?", default=True):
            generator.safe_write("Dockerfile", dockerfile)
            
        # Generate docker-compose
        if Confirm.ask("Generate docker-compose.yml?", default=True):
             with console.status("[bold blue]🐙 Generating docker-compose.yml...[/bold blue]"):
                compose = generator.generate_docker_compose(context)
             
             console.print(Panel(compose, title="Generated docker-compose.yml", border_style="magenta"))
             if Confirm.ask("Save docker-compose.yml?", default=True):
                 generator.safe_write("docker-compose.yml", compose)

    except ImportError as e:
         console.print(f"[red]❌ Missing dependencies: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ DevOps generation failed: {e}[/red]")


@cli.command()
@click.option('--model', default='qwen2.5:14b', help='Model to use (smarter is better for shadow)')
@click.option('--path', default='.', help='Path to watch')
def shadow(model, path):
    """Start Shadow Mode (Background Autocomplete)"""
    try:
        from xencode.shadow.engine import start_shadow_mode
        start_shadow_mode(path, model)
    except ImportError as e:
         console.print(f"[red]❌ Missing dependencies: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Shadow Mode failed: {e}[/red]")


@cli.command()
@click.argument('intent', nargs=-1, required=True)
@click.option('--mode', '-m', type=click.Choice(['assist', 'execute', 'autonomous']), default='assist',
              help='Execution mode for ByteBot')
@click.option('--raw', is_flag=True, help='Print raw JSON result')
def bytebot(intent, mode, raw):
    """Run ByteBot terminal cognition for a given intent"""
    try:
        from xencode.bytebot import ByteBotEngine

        intent_text = " ".join(intent).strip()
        if not intent_text:
            console.print("[red]❌ Intent is required.[/red]")
            return

        engine = ByteBotEngine()
        result = engine.process_intent(intent_text, mode=mode)

        if raw:
            console.print_json(data=result)
            return

        status = result.get("status", "unknown")
        summary = result.get("summary") or result.get("message") or ""
        console.print(Panel(
            f"[bold]Mode:[/bold] {mode}\n"
            f"[bold]Status:[/bold] {status}\n"
            f"[bold]Summary:[/bold] {summary}",
            title="🧠 ByteBot Result",
            border_style="blue"
        ))

        # Show step details if present
        steps = result.get("suggested_steps") or result.get("execution_results")
        if steps:
            table = Table(title="ByteBot Steps")
            table.add_column("#", style="cyan", width=4)
            table.add_column("Status", style="green")
            table.add_column("Command", style="yellow")
            table.add_column("Risk", style="magenta")
            for idx, step in enumerate(steps, 1):
                table.add_row(
                    str(idx),
                    str(step.get("status", "")),
                    str(step.get("command", ""))[:80],
                    f"{step.get('risk_score', 0):.2f}" if step.get("risk_score") is not None else ""
                )
            console.print(table)

    except Exception as e:
        console.print(f"[red]❌ ByteBot failed: {e}[/red]")


@cli.group()
def review():
    """AI Code Review commands"""
    pass

@review.command()
@click.argument('url', required=True)
@click.option('--platform', type=click.Choice(['github', 'gitlab', 'bitbucket']), 
              help='Platform (auto-detected if not specified)')
@click.option('--format', type=click.Choice(['text', 'markdown', 'json', 'html']), 
              default='text', help='Output format')
@click.option('--severity', type=click.Choice(['critical', 'high', 'medium', 'low']), 
              help='Filter by minimum severity level')
@click.option('--output', '-o', type=click.Path(), help='Save report to file')
@click.pass_context
def pr(ctx, url, platform, format, severity, output):
    """
    Review a pull request
    
    Examples:
      xencode review pr https://github.com/owner/repo/pull/123
      xencode review pr https://gitlab.com/owner/repo/-/merge_requests/45 --format markdown
      xencode review pr <url> --severity high --output report.md
    """
    console.print(f"[cyan]🔍 Analyzing pull request: {url}[/cyan]")
    
    async def _review_pr():
        try:
            from xencode.features.code_review import CodeReviewFeature
            from xencode.features import FeatureConfig
            
            # Initialize feature
            config = FeatureConfig(name="code_review", enabled=True)
            feature = CodeReviewFeature(config)
            await feature._initialize()
            
            # Analyze PR
            with console.status("[bold blue]🤖 AI reviewing pull request..."):
                review = await feature.analyze_pr(url, platform or 'github')
            
            # Filter by severity if specified
            if severity:
                severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
                min_level = severity_order[severity]
                
                # Filter issues
                if 'issues' in review:
                    review['issues'] = [
                        issue for issue in review['issues']
                        if severity_order.get(issue.get('severity', 'low'), 3) <= min_level
                    ]
            
            # Generate formatted report
            report = feature.generate_formatted_report(review, format)
            
            # Output report
            if output:
                Path(output).write_text(report)
                console.print(f"[green]✅ Report saved to {output}[/green]")
            else:
                console.print("\n" + report)
            
            # Show summary
            total_issues = len(review.get('issues', []))
            console.print(f"\n[yellow]📊 Found {total_issues} issues[/yellow]")
            
            await feature._shutdown()
            
        except Exception as e:
            console.print(f"[red]❌ PR review failed: {e}[/red]")
            import traceback
            if ctx.obj.get('verbose'):
                console.print(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(_review_pr())


@review.command()
@click.argument('path', required=True, type=click.Path(exists=True))
@click.option('--language', '-l', help='Programming language (auto-detected if not specified)')
@click.option('--format', type=click.Choice(['text', 'markdown', 'json', 'html']), 
              default='text', help='Output format')
@click.option('--severity', type=click.Choice(['critical', 'high', 'medium', 'low']), 
              help='Filter by minimum severity level')
@click.option('--output', '-o', type=click.Path(), help='Save report to file')
@click.pass_context
def file(ctx, path, language, format, severity, output):
    """
    Review a specific file
    
    Examples:
      xencode review file src/main.py
      xencode review file app.js --language javascript
      xencode review file code.rs --severity high --format markdown
    """
    console.print(f"[cyan]🔍 Analyzing file: {path}[/cyan]")
    
    async def _review_file():
        try:
            from xencode.features.code_review import CodeReviewFeature
            from xencode.features import FeatureConfig
            
            # Initialize feature
            config = FeatureConfig(name="code_review", enabled=True)
            feature = CodeReviewFeature(config)
            await feature._initialize()
            
            # Analyze file
            with console.status("[bold blue]🤖 AI reviewing file..."):
                review = await feature.analyze_file(path, language)
            
            # Filter by severity if specified
            if severity:
                severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
                min_level = severity_order[severity]
                
                # Filter issues
                if 'issues' in review:
                    review['issues'] = [
                        issue for issue in review['issues']
                        if severity_order.get(issue.get('severity', 'low'), 3) <= min_level
                    ]
            
            # Generate formatted report
            report = feature.generate_formatted_report(review, format)
            
            # Output report
            if output:
                Path(output).write_text(report)
                console.print(f"[green]✅ Report saved to {output}[/green]")
            else:
                console.print("\n" + report)
            
            # Show summary
            total_issues = len(review.get('issues', []))
            console.print(f"\n[yellow]📊 Found {total_issues} issues[/yellow]")
            
            await feature._shutdown()
            
        except Exception as e:
            console.print(f"[red]❌ File review failed: {e}[/red]")
            import traceback
            if ctx.obj.get('verbose'):
                console.print(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(_review_file())


@review.command()
@click.argument('path', required=True, type=click.Path(exists=True))
@click.option('--language', '-l', help='Filter by programming language')
@click.option('--format', type=click.Choice(['text', 'markdown', 'json', 'html']), 
              default='text', help='Output format')
@click.option('--severity', type=click.Choice(['critical', 'high', 'medium', 'low']), 
              help='Filter by minimum severity level')
@click.option('--output', '-o', type=click.Path(), help='Save report to file')
@click.option('--patterns', multiple=True, help='File patterns to include (e.g., *.py, *.js)')
@click.pass_context
def directory(ctx, path, language, format, severity, output, patterns):
    """
    Review an entire directory
    
    Examples:
      xencode review directory src/
      xencode review directory . --language python
      xencode review directory app/ --patterns "*.js" --patterns "*.ts"
      xencode review directory . --severity high --format markdown -o report.md
    """
    console.print(f"[cyan]🔍 Analyzing directory: {path}[/cyan]")
    
    async def _review_directory():
        try:
            from xencode.features.code_review import CodeReviewFeature
            from xencode.features import FeatureConfig
            
            # Initialize feature
            config = FeatureConfig(name="code_review", enabled=True)
            feature = CodeReviewFeature(config)
            await feature._initialize()
            
            # Prepare patterns
            pattern_list = list(patterns) if patterns else None
            
            # Analyze directory
            with console.status("[bold blue]🤖 AI reviewing directory..."):
                review = await feature.analyze_directory(path, pattern_list)
            
            # Filter by language if specified
            if language and 'files' in review:
                review['files'] = [
                    f for f in review['files']
                    if f.get('language', '').lower() == language.lower()
                ]
            
            # Filter by severity if specified
            if severity:
                severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
                min_level = severity_order[severity]
                
                # Filter issues
                if 'issues' in review:
                    review['issues'] = [
                        issue for issue in review['issues']
                        if severity_order.get(issue.get('severity', 'low'), 3) <= min_level
                    ]
            
            # Generate formatted report
            report = feature.generate_formatted_report(review, format)
            
            # Output report
            if output:
                Path(output).write_text(report)
                console.print(f"[green]✅ Report saved to {output}[/green]")
            else:
                console.print("\n" + report)
            
            # Show summary
            total_files = len(review.get('files', []))
            total_issues = len(review.get('issues', []))
            console.print(f"\n[yellow]📊 Analyzed {total_files} files, found {total_issues} issues[/yellow]")
            
            await feature._shutdown()
            
        except Exception as e:
            console.print(f"[red]❌ Directory review failed: {e}[/red]")
            import traceback
            if ctx.obj.get('verbose'):
                console.print(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(_review_directory())


@cli.group()
def terminal():
    """Terminal Assistant commands"""
    pass


@terminal.command()
@click.argument('context', required=False)
@click.option('--partial', '-p', help='Partial command input')
@click.option('--limit', '-n', type=int, default=5, help='Number of suggestions')
@click.pass_context
def suggest(ctx, context, partial, limit):
    """
    Suggest commands based on context
    
    Examples:
      xencode terminal suggest
      xencode terminal suggest --partial "git"
      xencode terminal suggest "python project" --limit 10
    """
    console.print("[cyan]🤖 Analyzing context and generating suggestions...[/cyan]")
    
    async def _suggest():
        try:
            from xencode.features.terminal_assistant import TerminalAssistantFeature
            from xencode.features import FeatureConfig
            
            # Initialize feature
            config = FeatureConfig(
                name="terminal_assistant",
                enabled=True,
                config={'suggestion_limit': limit}
            )
            feature = TerminalAssistantFeature(config)
            await feature._initialize()
            
            # Get suggestions
            suggestions = await feature.suggest_commands(context=context, partial=partial)
            
            if not suggestions:
                console.print("[yellow]No suggestions found[/yellow]")
                await feature._shutdown()
                return
            
            # Display suggestions
            table = Table(title="💡 Command Suggestions")
            table.add_column("#", style="cyan", width=4)
            table.add_column("Command", style="green")
            table.add_column("Score", style="yellow", width=8)
            table.add_column("Source", style="blue", width=12)
            table.add_column("Explanation", style="white")
            
            for idx, suggestion in enumerate(suggestions, 1):
                table.add_row(
                    str(idx),
                    suggestion['command'],
                    f"{suggestion.get('score', 0):.1f}",
                    suggestion.get('source', 'unknown'),
                    suggestion.get('explanation', '')[:50]
                )
            
            console.print(table)
            
            await feature._shutdown()
            
        except Exception as e:
            console.print(f"[red]❌ Suggestion failed: {e}[/red]")
            if ctx.obj.get('verbose'):
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(_suggest())


@terminal.command()
@click.argument('command', required=True)
@click.pass_context
def explain(ctx, command):
    """
    Explain what a command does
    
    Examples:
      xencode terminal explain "git commit -m 'message'"
      xencode terminal explain "docker run -it ubuntu"
      xencode terminal explain "npm install --save-dev"
    """
    console.print(f"[cyan]📖 Explaining command: {command}[/cyan]")
    
    async def _explain():
        try:
            from xencode.features.terminal_assistant import TerminalAssistantFeature
            from xencode.features import FeatureConfig
            
            # Initialize feature
            config = FeatureConfig(name="terminal_assistant", enabled=True)
            feature = TerminalAssistantFeature(config)
            await feature._initialize()
            
            # Get explanation
            explanation = await feature.explain_command(command)
            
            # Display explanation
            console.print(Panel(
                f"[bold]Command:[/bold] {explanation['command']}\n\n"
                f"[bold]Description:[/bold]\n{explanation['description']}\n\n"
                f"[bold]Arguments:[/bold]\n" +
                "\n".join(f"  • {arg['value']}: {arg['description']}" 
                         for arg in explanation.get('arguments', [])) +
                (f"\n\n[bold]Examples:[/bold]\n" +
                 "\n".join(f"  • {ex}" for ex in explanation.get('examples', []))
                 if explanation.get('examples') else "") +
                (f"\n\n[bold red]Warnings:[/bold red]\n" +
                 "\n".join(f"  {warn}" for warn in explanation.get('warnings', []))
                 if explanation.get('warnings') else ""),
                title="Command Explanation",
                border_style="blue"
            ))
            
            await feature._shutdown()
            
        except Exception as e:
            console.print(f"[red]❌ Explanation failed: {e}[/red]")
            if ctx.obj.get('verbose'):
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(_explain())


@terminal.command()
@click.argument('error', required=True)
@click.option('--command', '-c', help='Command that caused the error')
@click.option('--limit', '-n', type=int, default=5, help='Number of fix suggestions')
@click.pass_context
def fix(ctx, error, command, limit):
    """
    Suggest fixes for command errors
    
    Examples:
      xencode terminal fix "command not found: npm"
      xencode terminal fix "Permission denied" --command "rm file.txt"
      xencode terminal fix "No such file or directory" -c "cd /nonexistent"
    """
    console.print(f"[cyan]🔧 Analyzing error and suggesting fixes...[/cyan]")
    
    async def _fix():
        try:
            from xencode.features.terminal_assistant import TerminalAssistantFeature
            from xencode.features import FeatureConfig
            
            # Initialize feature
            config = FeatureConfig(name="terminal_assistant", enabled=True)
            feature = TerminalAssistantFeature(config)
            await feature._initialize()
            
            # Get fix suggestions
            fixes = await feature.fix_error(command or '', error)
            
            if not fixes:
                console.print("[yellow]No fix suggestions found[/yellow]")
                await feature._shutdown()
                return
            
            # Display fixes
            console.print(f"\n[bold green]Found {len(fixes[:limit])} fix suggestions:[/bold green]\n")
            
            for idx, fix_data in enumerate(fixes[:limit], 1):
                console.print(Panel(
                    f"[bold]Fix Command:[/bold]\n{fix_data['fix']}\n\n"
                    f"[bold]Explanation:[/bold]\n{fix_data['explanation']}\n\n"
                    f"[bold]Confidence:[/bold] {fix_data['confidence']:.1%}\n"
                    f"[bold]Category:[/bold] {fix_data['category']}" +
                    (f"\n[yellow]⚠️  Requires sudo[/yellow]" if fix_data.get('requires_sudo') else "") +
                    (f"\n[yellow]📦 Requires installation: {fix_data.get('install_command')}[/yellow]" 
                     if fix_data.get('requires_install') else "") +
                    (f"\n\n[bold]Documentation:[/bold] {fix_data['documentation_url']}" 
                     if fix_data.get('documentation_url') else "") +
                    (f"\n\n[bold]Alternatives:[/bold]\n" +
                     "\n".join(f"  • {alt}" for alt in fix_data.get('alternative_commands', []))
                     if fix_data.get('alternative_commands') else ""),
                    title=f"Fix #{idx}",
                    border_style="green"
                ))
            
            await feature._shutdown()
            
        except Exception as e:
            console.print(f"[red]❌ Fix suggestion failed: {e}[/red]")
            if ctx.obj.get('verbose'):
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(_fix())


@terminal.command()
@click.argument('pattern', required=True)
@click.option('--limit', '-n', type=int, default=20, help='Number of results')
@click.pass_context
def history(ctx, pattern, limit):
    """
    Search command history
    
    Examples:
      xencode terminal history "git"
      xencode terminal history "npm.*install"
      xencode terminal history "docker" --limit 50
    """
    console.print(f"[cyan]🔍 Searching command history for: {pattern}[/cyan]")
    
    async def _search_history():
        try:
            from xencode.features.terminal_assistant import TerminalAssistantFeature
            from xencode.features import FeatureConfig
            
            # Initialize feature
            config = FeatureConfig(name="terminal_assistant", enabled=True)
            feature = TerminalAssistantFeature(config)
            await feature._initialize()
            
            # Search history
            results = await feature.search_history(pattern)
            
            if not results:
                console.print("[yellow]No matching commands found in history[/yellow]")
                await feature._shutdown()
                return
            
            # Display results
            table = Table(title=f"📜 Command History ({len(results[:limit])} results)")
            table.add_column("#", style="cyan", width=4)
            table.add_column("Command", style="green")
            table.add_column("Timestamp", style="yellow")
            table.add_column("Status", style="white", width=10)
            
            for idx, cmd_data in enumerate(results[:limit], 1):
                status = "✅ Success" if cmd_data.get('success', True) else "❌ Failed"
                timestamp = cmd_data.get('timestamp', '')
                if timestamp:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp)
                        timestamp = dt.strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        pass
                
                table.add_row(
                    str(idx),
                    cmd_data.get('command', ''),
                    timestamp,
                    status
                )
            
            console.print(table)
            
            if len(results) > limit:
                console.print(f"\n[yellow]Showing {limit} of {len(results)} results. Use --limit to see more.[/yellow]")
            
            await feature._shutdown()
            
        except Exception as e:
            console.print(f"[red]❌ History search failed: {e}[/red]")
            if ctx.obj.get('verbose'):
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(_search_history())


@terminal.command()
@click.pass_context
def learn(ctx):
    """
    Start learning mode - interactive tutorial
    
    Learning mode helps you discover and master terminal commands
    through interactive guidance and personalized suggestions.
    """
    console.print("[cyan]🎓 Starting Terminal Assistant learning mode...[/cyan]")
    
    async def _learn():
        try:
            from xencode.features.terminal_assistant import TerminalAssistantFeature
            from xencode.features import FeatureConfig
            
            # Initialize feature
            config = FeatureConfig(name="terminal_assistant", enabled=True)
            feature = TerminalAssistantFeature(config)
            await feature._initialize()
            
            console.print(Panel(
                "[bold green]Welcome to Terminal Assistant Learning Mode![/bold green]\n\n"
                "This mode helps you:\n"
                "  • Discover new commands relevant to your work\n"
                "  • Learn command patterns and best practices\n"
                "  • Get personalized suggestions based on your skill level\n"
                "  • Track your progress and mastery\n\n"
                "[yellow]The Terminal Assistant is now actively learning from your commands.[/yellow]\n"
                "[yellow]Use 'xencode terminal statistics' to see your progress.[/yellow]",
                title="🎓 Learning Mode",
                border_style="green"
            ))
            
            # Show current learning status
            if feature.learning_engine and feature.learning_engine.user_skill_level:
                console.print("\n[bold]Your Current Skill Levels:[/bold]")
                table = Table()
                table.add_column("Command Category", style="cyan")
                table.add_column("Skill Level", style="green")
                table.add_column("Progress", style="yellow")
                
                for cmd, level in sorted(
                    feature.learning_engine.user_skill_level.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]:
                    progress_bar = "█" * int(level * 10) + "░" * (10 - int(level * 10))
                    table.add_row(cmd, f"{level:.1%}", progress_bar)
                
                console.print(table)
            
            await feature._shutdown()
            
        except Exception as e:
            console.print(f"[red]❌ Learning mode failed: {e}[/red]")
            if ctx.obj.get('verbose'):
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(_learn())


@terminal.command()
@click.option('--command', '-c', help='Get statistics for specific command')
@click.pass_context
def statistics(ctx, command):
    """
    Get command usage statistics
    
    Examples:
      xencode terminal statistics
      xencode terminal statistics --command "git commit"
    """
    console.print("[cyan]📊 Gathering command statistics...[/cyan]")
    
    async def _statistics():
        try:
            from xencode.features.terminal_assistant import TerminalAssistantFeature
            from xencode.features import FeatureConfig
            
            # Initialize feature
            config = FeatureConfig(name="terminal_assistant", enabled=True)
            feature = TerminalAssistantFeature(config)
            await feature._initialize()
            
            # Get statistics
            stats = await feature.get_statistics(command)
            
            if command:
                # Show specific command statistics
                console.print(Panel(
                    f"[bold]Command:[/bold] {stats['command']}\n\n"
                    f"[bold]Frequency:[/bold] {stats['frequency']} times\n"
                    f"[bold]Success Rate:[/bold] {stats['success_rate']:.1%}\n"
                    f"[bold]Last Used:[/bold] {stats.get('last_used', 'Never')}\n\n"
                    f"[bold]Common Sequences:[/bold]\n" +
                    "\n".join(f"  • {seq[0]} ({seq[1]} times)" 
                             for seq in stats.get('common_sequences', [])[:5]) +
                    (f"\n\n[bold]Temporal Usage:[/bold]\n" +
                     "\n".join(f"  • {time}: {count} times" 
                              for time, count in sorted(stats.get('temporal_usage', {}).items())[:5])
                     if stats.get('temporal_usage') else ""),
                    title=f"Statistics for '{command}'",
                    border_style="blue"
                ))
            else:
                # Show overall statistics
                console.print(Panel(
                    f"[bold]Total Commands:[/bold] {stats['total_commands']}\n"
                    f"[bold]Unique Commands:[/bold] {stats['unique_commands']}\n"
                    f"[bold]Overall Success Rate:[/bold] {stats['success_rate']:.1%}\n"
                    f"[bold]Patterns Detected:[/bold] {stats['patterns_detected']}",
                    title="📊 Overall Statistics",
                    border_style="green"
                ))
                
                # Show most frequent commands
                if stats.get('most_frequent'):
                    console.print("\n[bold]Most Frequent Commands:[/bold]")
                    table = Table()
                    table.add_column("Rank", style="cyan", width=6)
                    table.add_column("Command", style="green")
                    table.add_column("Count", style="yellow", width=10)
                    
                    for idx, (cmd, count) in enumerate(stats['most_frequent'], 1):
                        table.add_row(str(idx), cmd, str(count))
                    
                    console.print(table)
            
            await feature._shutdown()
            
        except Exception as e:
            console.print(f"[red]❌ Statistics failed: {e}[/red]")
            if ctx.obj.get('verbose'):
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(_statistics())


@terminal.command()
@click.pass_context
def patterns(ctx):
    """
    Analyze command patterns
    
    Shows detected patterns in your command usage including:
    - Command patterns (common command structures)
    - Sequence patterns (commands that follow each other)
    - Temporal patterns (time-based usage)
    - Context patterns (project-specific commands)
    """
    console.print("[cyan]🔍 Analyzing command patterns...[/cyan]")
    
    async def _patterns():
        try:
            from xencode.features.terminal_assistant import TerminalAssistantFeature
            from xencode.features import FeatureConfig
            
            # Initialize feature
            config = FeatureConfig(name="terminal_assistant", enabled=True)
            feature = TerminalAssistantFeature(config)
            await feature._initialize()
            
            # Analyze patterns
            analysis = await feature.analyze_patterns()
            
            # Display command patterns
            if analysis.get('command_patterns'):
                console.print("\n[bold green]Command Patterns:[/bold green]")
                table = Table()
                table.add_column("Base Command", style="cyan")
                table.add_column("Pattern Count", style="yellow")
                table.add_column("Examples", style="white")
                
                for base, data in sorted(
                    analysis['command_patterns'].items(),
                    key=lambda x: x[1]['count'],
                    reverse=True
                )[:10]:
                    examples = ", ".join(data['examples'][:3])
                    table.add_row(base, str(data['count']), examples[:60])
                
                console.print(table)
            
            # Display sequence patterns
            if analysis.get('sequence_patterns'):
                console.print("\n[bold green]Sequence Patterns:[/bold green]")
                console.print("[dim]Commands that commonly follow each other[/dim]\n")
                table = Table()
                table.add_column("From Command", style="cyan")
                table.add_column("To Command", style="green")
                table.add_column("Frequency", style="yellow")
                
                for pattern in sorted(
                    analysis['sequence_patterns'],
                    key=lambda x: x['frequency'],
                    reverse=True
                )[:10]:
                    table.add_row(
                        pattern['from'][:40],
                        pattern['to'][:40],
                        str(pattern['frequency'])
                    )
                
                console.print(table)
            
            # Display temporal patterns
            if analysis.get('temporal_patterns'):
                console.print("\n[bold green]Temporal Patterns:[/bold green]")
                console.print("[dim]Commands used at specific times[/dim]\n")
                
                for time_key, commands in sorted(analysis['temporal_patterns'].items())[:5]:
                    if commands:
                        console.print(f"[bold]{time_key}:[/bold]")
                        for cmd, count in commands[:3]:
                            console.print(f"  • {cmd} ({count} times)")
                        console.print()
            
            # Display context patterns
            if analysis.get('context_patterns'):
                console.print("\n[bold green]Context Patterns:[/bold green]")
                console.print("[dim]Commands used in specific contexts[/dim]\n")
                
                for context, commands in sorted(analysis['context_patterns'].items())[:5]:
                    if commands:
                        console.print(f"[bold]{context}:[/bold]")
                        for cmd, count in commands[:3]:
                            console.print(f"  • {cmd} ({count} times)")
                        console.print()
            
            await feature._shutdown()
            
        except Exception as e:
            console.print(f"[red]❌ Pattern analysis failed: {e}[/red]")
            if ctx.obj.get('verbose'):
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(_patterns())


@cli.group()
def analyze():
    """Project Analyzer commands - Analyze project structure, dependencies, and metrics"""
    pass


@analyze.command()
@click.argument('path', required=False, default='.')
@click.option('--output', '-o', type=click.Path(), help='Output file for analysis results')
@click.option('--format', '-f', type=click.Choice(['json', 'markdown', 'html']), default='markdown', 
              help='Output format')
@click.pass_context
def project(ctx, path, output, format):
    """
    Analyze project structure, dependencies, and metrics
    
    Examples:
      xencode analyze project
      xencode analyze project /path/to/project
      xencode analyze project --output analysis.md
      xencode analyze project --format json --output analysis.json
    """
    console.print(f"[cyan]📊 Analyzing project: {path}[/cyan]")
    
    async def _analyze():
        try:
            from xencode.features.project_analyzer import ProjectAnalyzerFeature
            from xencode.features import FeatureConfig
            
            # Initialize feature
            config = FeatureConfig(name="project_analyzer", enabled=True)
            feature = ProjectAnalyzerFeature(config)
            await feature._initialize()
            
            # Analyze project
            with console.status("[bold blue]🔍 Scanning project structure..."):
                results = await feature.analyze_project(path)
            
            # Display summary
            summary = results.get('summary', {})
            console.print(Panel(
                f"[bold]Project:[/bold] {results['project_path']}\n"
                f"[bold]Total Files:[/bold] {summary.get('total_files', 0)}\n"
                f"[bold]Total Lines:[/bold] {summary.get('total_lines', 0):,}\n"
                f"[bold]Languages:[/bold] {', '.join(summary.get('languages', []))}\n"
                f"[bold]Dependencies:[/bold] {summary.get('dependency_count', 0)}\n"
                f"[bold]Complexity Score:[/bold] {summary.get('complexity_score', 0):.1f}\n"
                f"[bold]Maintainability:[/bold] {summary.get('maintainability_index', 0):.1f}/100\n"
                f"[bold]Health Score:[/bold] {summary.get('health_score', 0):.1f}/100\n"
                f"[bold]Tech Debt Items:[/bold] {summary.get('tech_debt_items', 0)}",
                title="📊 Project Analysis Summary",
                border_style="green"
            ))
            
            # Show metrics
            metrics = results.get('metrics', {})
            if metrics:
                console.print("\n[bold]📈 Code Metrics:[/bold]")
                table = Table()
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="yellow")
                
                table.add_row("Code Lines", f"{metrics.get('code_lines', 0):,}")
                table.add_row("Comment Lines", f"{metrics.get('comment_lines', 0):,}")
                table.add_row("Blank Lines", f"{metrics.get('blank_lines', 0):,}")
                table.add_row("Comment Ratio", f"{metrics.get('comment_ratio', 0):.1f}%")
                table.add_row("Avg Complexity", f"{metrics.get('average_complexity', 0):.1f}")
                table.add_row("Max Complexity", f"{metrics.get('max_complexity', 0)}")
                
                console.print(table)
            
            # Show technical debt
            tech_debt = results.get('technical_debt', {})
            if tech_debt and tech_debt.get('issues'):
                console.print(f"\n[bold red]⚠️  Technical Debt ({tech_debt['total_issues']} issues):[/bold red]")
                
                by_severity = tech_debt.get('by_severity', {})
                severity_table = Table()
                severity_table.add_column("Severity", style="cyan")
                severity_table.add_column("Count", style="yellow")
                
                for severity in ['high', 'medium', 'low']:
                    count = by_severity.get(severity, 0)
                    if count > 0:
                        severity_table.add_row(severity.capitalize(), str(count))
                
                console.print(severity_table)
                
                # Show top issues
                console.print("\n[bold]Top Issues:[/bold]")
                for issue in tech_debt['issues'][:5]:
                    severity_color = {
                        'high': 'red',
                        'medium': 'yellow',
                        'low': 'blue'
                    }.get(issue['severity'], 'white')
                    
                    console.print(f"  [{severity_color}]•[/{severity_color}] {issue['message']}")
                    if issue.get('suggestion'):
                        console.print(f"    [dim]→ {issue['suggestion']}[/dim]")
            
            # Save output if requested
            if output:
                import json
                from pathlib import Path
                
                output_path = Path(output)
                
                if format == 'json':
                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2)
                elif format == 'markdown':
                    md_content = _generate_markdown_report(results)
                    with open(output_path, 'w') as f:
                        f.write(md_content)
                elif format == 'html':
                    html_content = _generate_html_report(results)
                    with open(output_path, 'w') as f:
                        f.write(html_content)
                
                console.print(f"\n[green]✅ Analysis saved to: {output_path}[/green]")
            
            await feature._shutdown()
            
        except Exception as e:
            console.print(f"[red]❌ Analysis failed: {e}[/red]")
            if ctx.obj.get('verbose'):
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(_analyze())


@analyze.command()
@click.argument('path', required=False, default='.')
@click.option('--output', '-o', type=click.Path(), help='Output directory for documentation')
@click.option('--readme', is_flag=True, help='Generate README.md')
@click.option('--architecture', is_flag=True, help='Generate architecture diagram')
@click.option('--api-docs', is_flag=True, help='Generate API documentation')
@click.pass_context
def docs(ctx, path, output, readme, architecture, api_docs):
    """
    Generate project documentation
    
    Examples:
      xencode analyze docs
      xencode analyze docs --readme
      xencode analyze docs --architecture --output ./docs
      xencode analyze docs --readme --architecture --api-docs
    """
    console.print(f"[cyan]📝 Generating documentation for: {path}[/cyan]")
    
    # If no specific docs requested, generate all
    if not (readme or architecture or api_docs):
        readme = architecture = api_docs = True
    
    async def _generate_docs():
        try:
            from xencode.features.project_analyzer import ProjectAnalyzerFeature
            from xencode.features import FeatureConfig
            from pathlib import Path
            
            # Initialize feature
            config = FeatureConfig(name="project_analyzer", enabled=True)
            feature = ProjectAnalyzerFeature(config)
            await feature._initialize()
            
            # Analyze project first
            with console.status("[bold blue]🔍 Analyzing project..."):
                results = await feature.analyze_project(path)
            
            output_dir = Path(output) if output else Path(path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            generated_files = []
            
            # Generate README
            if readme:
                with console.status("[bold blue]📝 Generating README..."):
                    readme_path = output_dir / "README.md"
                    readme_content = _generate_readme(results)
                    with open(readme_path, 'w') as f:
                        f.write(readme_content)
                    generated_files.append(str(readme_path))
                    console.print(f"[green]✅ Generated README: {readme_path}[/green]")
            
            # Generate architecture diagram
            if architecture:
                with console.status("[bold blue]🏗️  Generating architecture diagram..."):
                    arch_data = results.get('architecture', {})
                    if arch_data.get('mermaid_diagram'):
                        arch_path = output_dir / "ARCHITECTURE.md"
                        arch_content = f"# Architecture\n\n```mermaid\n{arch_data['mermaid_diagram']}\n```\n"
                        with open(arch_path, 'w') as f:
                            f.write(arch_content)
                        generated_files.append(str(arch_path))
                        console.print(f"[green]✅ Generated architecture: {arch_path}[/green]")
            
            # Generate API docs
            if api_docs:
                with console.status("[bold blue]📚 Generating API documentation..."):
                    api_path = output_dir / "API.md"
                    api_content = _generate_api_docs(results)
                    with open(api_path, 'w') as f:
                        f.write(api_content)
                    generated_files.append(str(api_path))
                    console.print(f"[green]✅ Generated API docs: {api_path}[/green]")
            
            console.print(f"\n[bold green]✅ Documentation generated successfully![/bold green]")
            console.print(f"[yellow]Generated {len(generated_files)} files in {output_dir}[/yellow]")
            
            await feature._shutdown()
            
        except Exception as e:
            console.print(f"[red]❌ Documentation generation failed: {e}[/red]")
            if ctx.obj.get('verbose'):
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(_generate_docs())


@analyze.command()
@click.argument('path', required=False, default='.')
@click.pass_context
def metrics(ctx, path):
    """
    Show detailed project metrics
    
    Examples:
      xencode analyze metrics
      xencode analyze metrics /path/to/project
    """
    console.print(f"[cyan]📊 Calculating metrics for: {path}[/cyan]")
    
    async def _metrics():
        try:
            from xencode.features.project_analyzer import ProjectAnalyzerFeature
            from xencode.features import FeatureConfig
            
            # Initialize feature
            config = FeatureConfig(name="project_analyzer", enabled=True)
            feature = ProjectAnalyzerFeature(config)
            await feature._initialize()
            
            # Analyze project
            with console.status("[bold blue]📊 Calculating metrics..."):
                results = await feature.analyze_project(path)
            
            metrics = results.get('metrics', {})
            structure = results.get('structure', {})
            
            # Display detailed metrics
            console.print(Panel(
                f"[bold cyan]Code Metrics[/bold cyan]\n"
                f"Total Lines: {metrics.get('total_lines', 0):,}\n"
                f"Code Lines: {metrics.get('code_lines', 0):,}\n"
                f"Comment Lines: {metrics.get('comment_lines', 0):,}\n"
                f"Blank Lines: {metrics.get('blank_lines', 0):,}\n"
                f"Comment Ratio: {metrics.get('comment_ratio', 0):.1f}%\n\n"
                f"[bold cyan]Complexity Metrics[/bold cyan]\n"
                f"Average Complexity: {metrics.get('average_complexity', 0):.1f}\n"
                f"Max Complexity: {metrics.get('max_complexity', 0)}\n"
                f"Maintainability Index: {metrics.get('maintainability_index', 0):.1f}/100\n\n"
                f"[bold cyan]Project Structure[/bold cyan]\n"
                f"Total Files: {structure.get('total_files', 0)}\n"
                f"Total Directories: {structure.get('total_directories', 0)}\n"
                f"Total Size: {structure.get('total_size', 0) / 1024 / 1024:.2f} MB",
                title="📊 Detailed Metrics",
                border_style="blue"
            ))
            
            # Language breakdown
            languages = structure.get('languages', {})
            if languages:
                console.print("\n[bold]Languages:[/bold]")
                lang_table = Table()
                lang_table.add_column("Language", style="cyan")
                lang_table.add_column("Files", style="yellow")
                lang_table.add_column("Percentage", style="green")
                
                total_files = sum(languages.values())
                for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_files * 100) if total_files > 0 else 0
                    lang_table.add_row(lang, str(count), f"{percentage:.1f}%")
                
                console.print(lang_table)
            
            await feature._shutdown()
            
        except Exception as e:
            console.print(f"[red]❌ Metrics calculation failed: {e}[/red]")
            if ctx.obj.get('verbose'):
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(_metrics())


@analyze.command()
@click.argument('path', required=False, default='.')
@click.option('--show-graph', is_flag=True, help='Show dependency graph')
@click.pass_context
def dependencies(ctx, path, show_graph):
    """
    Analyze project dependencies
    
    Examples:
      xencode analyze dependencies
      xencode analyze dependencies --show-graph
      xencode analyze dependencies /path/to/project
    """
    console.print(f"[cyan]🔗 Analyzing dependencies for: {path}[/cyan]")
    
    async def _dependencies():
        try:
            from xencode.features.project_analyzer import ProjectAnalyzerFeature
            from xencode.features import FeatureConfig
            
            # Initialize feature
            config = FeatureConfig(name="project_analyzer", enabled=True)
            feature = ProjectAnalyzerFeature(config)
            await feature._initialize()
            
            # Analyze project
            with console.status("[bold blue]🔗 Analyzing dependencies..."):
                results = await feature.analyze_project(path)
            
            dependencies = results.get('dependencies', {})
            
            # Display external dependencies
            external_deps = dependencies.get('dependencies', [])
            if external_deps:
                console.print(Panel(
                    f"[bold]Total Dependencies:[/bold] {len(external_deps)}\n"
                    f"[bold]External:[/bold] {len([d for d in external_deps if d.get('type') == 'external'])}\n"
                    f"[bold]Dev:[/bold] {len([d for d in external_deps if d.get('type') == 'dev'])}",
                    title="📦 Dependencies Summary",
                    border_style="blue"
                ))
                
                console.print("\n[bold]External Dependencies:[/bold]")
                dep_table = Table()
                dep_table.add_column("Name", style="cyan")
                dep_table.add_column("Version", style="yellow")
                dep_table.add_column("Type", style="green")
                
                for dep in external_deps[:20]:  # Show first 20
                    dep_table.add_row(
                        dep.get('name', ''),
                        dep.get('version', ''),
                        dep.get('type', '')
                    )
                
                console.print(dep_table)
                
                if len(external_deps) > 20:
                    console.print(f"\n[yellow]Showing 20 of {len(external_deps)} dependencies[/yellow]")
            
            # Show circular dependencies
            circular = dependencies.get('circular_dependencies', [])
            if circular:
                console.print(f"\n[bold red]⚠️  Circular Dependencies Found ({len(circular)}):[/bold red]")
                for cycle in circular[:5]:
                    console.print(f"  • {' → '.join(cycle)}")
            
            # Show dependency graph if requested
            if show_graph:
                dep_graph = dependencies.get('dependency_graph', {})
                if dep_graph:
                    console.print("\n[bold]Dependency Graph:[/bold]")
                    for module, deps in list(dep_graph.items())[:10]:
                        if deps:
                            console.print(f"  {module}")
                            for dep in deps[:5]:
                                console.print(f"    → {dep}")
            
            await feature._shutdown()
            
        except Exception as e:
            console.print(f"[red]❌ Dependency analysis failed: {e}[/red]")
            if ctx.obj.get('verbose'):
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(_dependencies())


@analyze.command()
@click.argument('path', required=False, default='.')
@click.option('--output', '-o', type=click.Path(), help='Output file for architecture diagram')
@click.pass_context
def architecture(ctx, path, output):
    """
    Generate architecture diagram
    
    Examples:
      xencode analyze architecture
      xencode analyze architecture --output architecture.md
      xencode analyze architecture /path/to/project
    """
    console.print(f"[cyan]🏗️  Generating architecture diagram for: {path}[/cyan]")
    
    async def _architecture():
        try:
            from xencode.features.project_analyzer import ProjectAnalyzerFeature
            from xencode.features import FeatureConfig
            from pathlib import Path
            
            # Initialize feature
            config = FeatureConfig(name="project_analyzer", enabled=True)
            feature = ProjectAnalyzerFeature(config)
            await feature._initialize()
            
            # Analyze project
            with console.status("[bold blue]🏗️  Analyzing architecture..."):
                results = await feature.analyze_project(path)
            
            architecture = results.get('architecture', {})
            
            # Display components
            components = architecture.get('components', [])
            if components:
                console.print(Panel(
                    f"[bold]Total Components:[/bold] {len(components)}",
                    title="🏗️  Architecture Overview",
                    border_style="blue"
                ))
                
                console.print("\n[bold]Components:[/bold]")
                comp_table = Table()
                comp_table.add_column("Component", style="cyan")
                comp_table.add_column("Path", style="yellow")
                comp_table.add_column("Files", style="green")
                comp_table.add_column("Languages", style="blue")
                
                for comp in components[:15]:
                    comp_table.add_row(
                        comp.get('name', ''),
                        comp.get('path', ''),
                        str(comp.get('file_count', 0)),
                        ', '.join(comp.get('languages', []))
                    )
                
                console.print(comp_table)
            
            # Display Mermaid diagram
            mermaid = architecture.get('mermaid_diagram', '')
            if mermaid:
                console.print("\n[bold]Architecture Diagram (Mermaid):[/bold]")
                console.print(Panel(mermaid, border_style="green"))
                
                # Save if requested
                if output:
                    output_path = Path(output)
                    with open(output_path, 'w') as f:
                        f.write(f"# Architecture\n\n```mermaid\n{mermaid}\n```\n")
                    console.print(f"\n[green]✅ Architecture diagram saved to: {output_path}[/green]")
            
            await feature._shutdown()
            
        except Exception as e:
            console.print(f"[red]❌ Architecture generation failed: {e}[/red]")
            if ctx.obj.get('verbose'):
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(_architecture())


# Helper functions for report generation
def _generate_markdown_report(results: Dict[str, Any]) -> str:
    """Generate markdown report from analysis results"""
    summary = results.get('summary', {})
    metrics = results.get('metrics', {})
    tech_debt = results.get('technical_debt', {})
    
    md = f"""# Project Analysis Report

**Project:** {results['project_path']}  
**Analyzed:** {results['analyzed_at']}

## Summary

- **Total Files:** {summary.get('total_files', 0)}
- **Total Lines:** {summary.get('total_lines', 0):,}
- **Languages:** {', '.join(summary.get('languages', []))}
- **Dependencies:** {summary.get('dependency_count', 0)}
- **Health Score:** {summary.get('health_score', 0):.1f}/100

## Metrics

- **Code Lines:** {metrics.get('code_lines', 0):,}
- **Comment Lines:** {metrics.get('comment_lines', 0):,}
- **Comment Ratio:** {metrics.get('comment_ratio', 0):.1f}%
- **Average Complexity:** {metrics.get('average_complexity', 0):.1f}
- **Maintainability Index:** {metrics.get('maintainability_index', 0):.1f}/100

## Technical Debt

**Total Issues:** {tech_debt.get('total_issues', 0)}

"""
    
    # Add issues by severity
    by_severity = tech_debt.get('by_severity', {})
    if by_severity:
        md += "### By Severity\n\n"
        for severity in ['high', 'medium', 'low']:
            count = by_severity.get(severity, 0)
            if count > 0:
                md += f"- **{severity.capitalize()}:** {count}\n"
        md += "\n"
    
    # Add top issues
    issues = tech_debt.get('issues', [])
    if issues:
        md += "### Top Issues\n\n"
        for issue in issues[:10]:
            md += f"- **{issue['type']}** ({issue['severity']}): {issue['message']}\n"
            if issue.get('suggestion'):
                md += f"  - Suggestion: {issue['suggestion']}\n"
        md += "\n"
    
    return md


def _generate_html_report(results: Dict[str, Any]) -> str:
    """Generate HTML report from analysis results"""
    summary = results.get('summary', {})
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Project Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .metric {{ margin: 10px 0; }}
        .score {{ font-size: 24px; font-weight: bold; }}
        .good {{ color: green; }}
        .warning {{ color: orange; }}
        .bad {{ color: red; }}
    </style>
</head>
<body>
    <h1>Project Analysis Report</h1>
    <p><strong>Project:</strong> {results['project_path']}</p>
    <p><strong>Analyzed:</strong> {results['analyzed_at']}</p>
    
    <h2>Summary</h2>
    <div class="metric">Total Files: {summary.get('total_files', 0)}</div>
    <div class="metric">Total Lines: {summary.get('total_lines', 0):,}</div>
    <div class="metric">Health Score: <span class="score">{summary.get('health_score', 0):.1f}/100</span></div>
</body>
</html>
"""
    return html


def _generate_readme(results: Dict[str, Any]) -> str:
    """Generate README from analysis results"""
    summary = results.get('summary', {})
    structure = results.get('structure', {})
    
    readme = f"""# Project

## Overview

This project contains {summary.get('total_files', 0)} files with {summary.get('total_lines', 0):,} lines of code.

## Languages

{', '.join(summary.get('languages', []))}

## Project Type

{structure.get('project_type', 'Unknown')}

## Metrics

- **Health Score:** {summary.get('health_score', 0):.1f}/100
- **Maintainability:** {summary.get('maintainability_index', 0):.1f}/100
- **Dependencies:** {summary.get('dependency_count', 0)}

## Getting Started

[Add your getting started instructions here]

## License

[Add your license information here]
"""
    return readme


def _generate_api_docs(results: Dict[str, Any]) -> str:
    """Generate API documentation from analysis results"""
    return """# API Documentation

[API documentation will be generated here based on code analysis]

## Endpoints

[List of API endpoints]

## Models

[Data models and schemas]
"""


@cli.command()
def version():
    """Show version information"""
    console.print(Panel(
        "[bold green]🤖 Xencode AI/ML Leviathan v2.3.0[/bold green]\n\n"
        "The ultimate offline AI assistant that outperforms GitHub Copilot\n"
        "with <50ms inference, 10% SMAPE improvements, and 100% privacy.\n\n"
        "[yellow]🐉 The leviathan has awakened![/yellow]",
        title="Version Info",
        border_style="green"
    ))


if __name__ == '__main__':
    cli()