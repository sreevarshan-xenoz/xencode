#!/usr/bin/env python3
"""
Phase 2 Feature Demonstration for Xencode

Interactive demo showcasing all Phase 2 capabilities:
- Intelligent Model Selection
- Advanced Caching System  
- Smart Configuration Management
- Error Handling & Recovery
- Performance Optimization
"""

import asyncio
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
import tempfile

# Import Phase 2 components
sys.path.insert(0, str(Path(__file__).parent))
from xencode.phase2_coordinator import Phase2Coordinator
from xencode.intelligent_model_selector import HardwareDetector, ModelRecommendationEngine
from xencode.advanced_cache_system import get_cache_manager
from xencode.smart_config_manager import get_config_manager
from xencode.advanced_error_handler import get_error_handler, ErrorSeverity, ErrorCategory, ErrorContext

console = Console()


class Phase2Demo:
    """Interactive demonstration of Phase 2 features"""
    
    def __init__(self):
        self.coordinator = Phase2Coordinator()
        self.temp_dir = None
        
    async def run_demo(self):
        """Run the complete Phase 2 demonstration"""
        
        # Welcome
        self.show_welcome()
        
        if not Confirm.ask("Ready to start the Phase 2 demo"):
            console.print("[yellow]Demo cancelled. See you next time![/yellow]")
            return
        
        # Setup temporary environment
        self.temp_dir = tempfile.mkdtemp(prefix="xencode_demo_")
        console.print(f"[blue]Using temporary directory: {self.temp_dir}[/blue]")
        
        try:
            # Demo sections
            await self.demo_hardware_detection()
            await self.demo_model_recommendations()
            await self.demo_configuration_system()
            await self.demo_caching_system()
            await self.demo_error_handling()
            await self.demo_integration()
            await self.demo_performance_optimization()
            
            # Final summary
            self.show_final_summary()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Demo interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"[red]Demo error: {e}[/red]")
        finally:
            # Cleanup
            if self.temp_dir:
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def show_welcome(self):
        """Show welcome message"""
        welcome_text = """
🚀 **Welcome to Xencode Phase 2 Demo!**

This interactive demonstration will showcase all the powerful new features in Phase 2:

**🧠 Intelligent Model Selection**
  • Hardware detection and analysis
  • Automatic model recommendations
  • Performance optimization

**⚡ Advanced Caching System**  
  • Hybrid memory + disk caching
  • Intelligent compression
  • Performance analytics

**🔧 Smart Configuration Management**
  • Multi-format support (YAML, TOML, JSON)
  • Environment variable overrides
  • Interactive setup wizards

**🛡️ Error Handling & Recovery**
  • Intelligent error classification
  • Automatic recovery strategies
  • User-friendly error reporting

**📊 Performance Optimization**
  • System health monitoring
  • Automatic optimization
  • Resource management

Let's explore these features together!
        """
        
        panel = Panel(
            welcome_text.strip(),
            title="[bold blue]Xencode Phase 2 Demo[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        )
        console.print(panel)
    
    async def demo_hardware_detection(self):
        """Demonstrate hardware detection capabilities"""
        console.print("\n[bold blue]🖥️  Hardware Detection Demo[/bold blue]")
        
        if not Confirm.ask("Run hardware detection"):
            return
        
        with console.status("[bold blue]Analyzing your system hardware..."):
            detector = HardwareDetector()
            specs = detector.detect_system_specs()
        
        # Display results in a nice table
        table = Table(title="🔍 System Analysis Results")
        table.add_column("Component", style="cyan", width=15)
        table.add_column("Details", style="white", width=40)
        table.add_column("Performance", style="green", width=15)
        
        # CPU
        cpu_perf = "⭐" * min(specs.cpu_cores // 2, 5)
        table.add_row(
            "CPU",
            f"{specs.cpu_cores} cores ({specs.cpu_architecture})",
            cpu_perf
        )
        
        # Memory
        memory_perf = "⭐" * min(int(specs.total_ram_gb // 4), 5)
        table.add_row(
            "Memory",
            f"{specs.total_ram_gb:.1f} GB total, {specs.available_ram_gb:.1f} GB available",
            memory_perf
        )
        
        # GPU
        gpu_desc = f"{specs.gpu_type.title()}" if specs.gpu_available else "No dedicated GPU"
        if specs.gpu_vram_gb > 0:
            gpu_desc += f" ({specs.gpu_vram_gb:.1f} GB VRAM)"
        gpu_perf = "⭐⭐⭐⭐⭐" if specs.gpu_available else "⭐"
        table.add_row("Graphics", gpu_desc, gpu_perf)
        
        # Storage
        storage_perf = "⭐⭐⭐⭐⭐" if specs.storage_type == "ssd" else "⭐⭐⭐"
        table.add_row(
            "Storage",
            f"{specs.storage_type.upper()}, {specs.available_storage_gb:.1f} GB free",
            storage_perf
        )
        
        # Overall score
        overall_perf = "⭐" * (specs.performance_score // 20)
        table.add_row(
            "[bold]Overall Score[/bold]",
            f"[bold]{specs.performance_score}/100[/bold]",
            f"[bold]{overall_perf}[/bold]"
        )
        
        console.print(table)
        
        Prompt.ask("\n[dim]Press Enter to continue...", default="")
    
    async def demo_model_recommendations(self):
        """Demonstrate model recommendation system"""
        console.print("\n[bold blue]🤖 Model Recommendation Demo[/bold blue]")
        
        if not Confirm.ask("Get AI model recommendations for your system"):
            return
        
        with console.status("[bold blue]Analyzing optimal models for your hardware..."):
            detector = HardwareDetector()
            specs = detector.detect_system_specs()
            
            recommender = ModelRecommendationEngine()
            primary, alternatives = recommender.get_recommendations(specs)
        
        # Show primary recommendation
        primary_panel = Panel(
            f"""[bold]{primary.name}[/bold] ({primary.size_gb:.1f} GB)
{primary.description}

• Speed: {primary.estimated_speed}
• Quality: {"⭐" * primary.quality_score}
• RAM Required: {primary.ram_required_gb:.1f} GB
• Ollama Tag: [cyan]{primary.ollama_tag}[/cyan]

[green]✅ RECOMMENDED - Best match for your system[/green]""",
            title="🚀 Primary Recommendation",
            border_style="green"
        )
        console.print(primary_panel)
        
        # Show alternatives
        if alternatives:
            console.print("\n[bold blue]Alternative Options:[/bold blue]")
            
            for i, alt in enumerate(alternatives, 1):
                style = "blue" if i == 1 else "yellow"
                icon = "⚡" if alt.performance_tier == "fast" else "🧠"
                
                alt_panel = Panel(
                    f"""[bold]{alt.name}[/bold] ({alt.size_gb:.1f} GB)
{alt.description}

• Speed: {alt.estimated_speed}
• Quality: {"⭐" * alt.quality_score}
• RAM Required: {alt.ram_required_gb:.1f} GB""",
                    title=f"{icon} Alternative {i}",
                    border_style=style
                )
                console.print(alt_panel)
        
        Prompt.ask("\n[dim]Press Enter to continue...", default="")
    
    async def demo_configuration_system(self):
        """Demonstrate smart configuration management"""
        console.print("\n[bold blue]🔧 Configuration Management Demo[/bold blue]")
        
        if not Confirm.ask("Test configuration system"):
            return
        
        # Create temporary config
        config_path = Path(self.temp_dir) / "demo_config.yaml"
        console.print(f"[blue]Creating demo configuration at: {config_path}[/blue]")
        
        # Initialize config manager
        config_manager = get_config_manager(config_path)
        config = config_manager.load_config()
        
        # Show current config
        console.print("\n[bold]Default Configuration:[/bold]")
        config_manager.show_config()
        
        # Test validation
        console.print("\n[blue]Testing configuration validation...[/blue]")
        validation_errors = config.validate()
        if validation_errors:
            console.print("[yellow]Found validation issues (this is expected for demo):[/yellow]")
            for section, errors in validation_errors.items():
                for error in errors:
                    console.print(f"  • {section}: {error}")
        else:
            console.print("[green]✅ Configuration is valid![/green]")
        
        # Test saving in different formats
        if Confirm.ask("Save configuration in multiple formats"):
            formats = [
                (Path(self.temp_dir) / "config.yaml", "YAML"),
                (Path(self.temp_dir) / "config.json", "JSON")
            ]
            
            for path, format_name in formats:
                try:
                    config_manager.save_config(path)
                    console.print(f"[green]✅ Saved {format_name} config to {path.name}[/green]")
                except Exception as e:
                    console.print(f"[red]❌ Failed to save {format_name}: {e}[/red]")
        
        Prompt.ask("\n[dim]Press Enter to continue...", default="")
    
    async def demo_caching_system(self):
        """Demonstrate advanced caching capabilities"""
        console.print("\n[bold blue]⚡ Advanced Caching Demo[/bold blue]")
        
        if not Confirm.ask("Test caching system performance"):
            return
        
        # Initialize cache with demo settings
        cache_dir = Path(self.temp_dir) / "cache"
        cache_manager = await get_cache_manager(memory_mb=64, disk_mb=128)
        
        console.print("[blue]Initializing cache system...[/blue]")
        
        # Demo cache operations
        test_prompts = [
            ("What is Python?", "A programming language..."),
            ("How to write a function?", "Use the def keyword..."),
            ("What is machine learning?", "A subset of AI..."),
            ("Explain recursion", "A function calling itself..."),
            ("What is Python?", "This should be cached!")  # Duplicate for cache hit
        ]
        
        console.print("\n[bold]Testing cache performance:[/bold]")
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing requests...", total=len(test_prompts))
            
            for i, (prompt, response) in enumerate(test_prompts):
                # Check cache first
                start_time = time.perf_counter()
                cached = await cache_manager.get_response(prompt, "demo-model")
                
                if cached:
                    end_time = time.perf_counter()
                    console.print(f"[green]✅ Cache HIT: {prompt[:30]}... ({(end_time-start_time)*1000:.1f}ms)[/green]")
                else:
                    # Simulate processing and store
                    await asyncio.sleep(0.1)  # Simulate work
                    await cache_manager.store_response(prompt, "demo-model", {"text": response})
                    end_time = time.perf_counter()
                    console.print(f"[yellow]⚠️  Cache MISS: {prompt[:30]}... ({(end_time-start_time)*1000:.1f}ms)[/yellow]")
                
                progress.update(task, advance=1)
        
        # Show performance statistics
        stats = cache_manager.get_performance_stats()
        
        stats_table = Table(title="📊 Cache Performance Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Requests", str(stats["total_requests"]))
        stats_table.add_row("Cache Hit Rate", f"{stats['total_hit_rate']:.1f}%")
        stats_table.add_row("Memory Cache Entries", str(stats["memory_cache"]["entries"]))
        stats_table.add_row("Memory Usage", f"{stats['memory_cache']['size_mb']:.1f} MB")
        stats_table.add_row("Efficiency Score", f"{stats['efficiency_score']:.1f}/100")
        
        console.print(stats_table)
        
        Prompt.ask("\n[dim]Press Enter to continue...", default="")
    
    async def demo_error_handling(self):
        """Demonstrate error handling and recovery"""
        console.print("\n[bold blue]🛡️  Error Handling & Recovery Demo[/bold blue]")
        
        if not Confirm.ask("Test error handling system"):
            return
        
        error_handler = await get_error_handler()
        
        # Simulate different types of errors
        test_errors = [
            (ConnectionError("Network timeout after 30 seconds"), "Network Error"),
            (FileNotFoundError("Configuration file not found"), "Config Error"),
            (MemoryError("Insufficient memory available"), "Performance Error"),
            (PermissionError("Access denied to file"), "Security Error")
        ]
        
        console.print("[blue]Simulating and handling different error types...[/blue]\n")
        
        for i, (error, error_type) in enumerate(test_errors, 1):
            context = ErrorContext(
                function_name="demo_function",
                user_action=f"Testing error handling - {error_type}"
            )
            
            console.print(f"[bold]{i}. Testing {error_type}:[/bold]")
            
            # Handle the error
            handled_error = await error_handler.handle_error(error, context)
            
            # Show brief info about handling
            console.print(f"   Severity: {handled_error.severity.value}")
            console.print(f"   Category: {handled_error.category.value}")
            console.print(f"   Recoverable: {'Yes' if handled_error.recoverable else 'No'}")
            console.print()
        
        # Show error summary
        console.print("[bold]Error Handling Summary:[/bold]")
        summary = error_handler.get_error_summary()
        
        summary_table = Table()
        summary_table.add_column("Category", style="cyan")
        summary_table.add_column("Count", style="green")
        
        for category, count in summary["categories"].items():
            summary_table.add_row(category.title(), str(count))
        
        console.print(summary_table)
        
        Prompt.ask("\n[dim]Press Enter to continue...", default="")
    
    async def demo_integration(self):
        """Demonstrate Phase 2 integration"""
        console.print("\n[bold blue]🔗 Phase 2 Integration Demo[/bold blue]")
        
        if not Confirm.ask("Test integrated Phase 2 systems"):
            return
        
        # Initialize coordinator
        config_path = Path(self.temp_dir) / "integration_config.yaml"
        console.print("[blue]Initializing Phase 2 coordinator...[/blue]")
        
        success = await self.coordinator.initialize(config_path)
        if not success:
            console.print("[red]❌ Failed to initialize coordinator[/red]")
            return
        
        console.print("[green]✅ Phase 2 systems initialized![/green]")
        
        # Show system status
        console.print("\n[bold]System Status:[/bold]")
        self.coordinator.display_system_status()
        
        # Run health check
        if Confirm.ask("\nRun system health check"):
            console.print("[blue]Running comprehensive health check...[/blue]")
            health_status = await self.coordinator.health_check()
            
            if health_status:
                console.print("[green]✅ All systems healthy![/green]")
            else:
                console.print("[yellow]⚠️  Some issues detected (check logs)[/yellow]")
        
        Prompt.ask("\n[dim]Press Enter to continue...", default="")
    
    async def demo_performance_optimization(self):
        """Demonstrate performance optimization"""
        console.print("\n[bold blue]📊 Performance Optimization Demo[/bold blue]")
        
        if not Confirm.ask("Run performance optimization"):
            return
        
        console.print("[blue]Running performance optimization...[/blue]")
        
        # Run optimization
        results = await self.coordinator.optimize_performance()
        
        # Show results
        console.print("\n[bold]Optimization Results:[/bold]")
        
        results_table = Table()
        results_table.add_column("Optimization", style="cyan")
        results_table.add_column("Status", style="green")
        results_table.add_column("Details", style="white")
        
        if results["cache_optimized"]:
            results_table.add_row("Cache System", "✅ Optimized", "Cache cleaned and optimized")
        else:
            results_table.add_row("Cache System", "⏭️  Skipped", "No optimization needed")
        
        if results["memory_freed"] > 0:
            results_table.add_row("Memory", "✅ Freed", f"{results['memory_freed']:.1f}% freed")
        else:
            results_table.add_row("Memory", "✅ Optimal", "Memory usage already optimal")
        
        if results["config_optimized"]:
            results_table.add_row("Configuration", "✅ Fixed", "Configuration issues resolved")
        else:
            results_table.add_row("Configuration", "✅ Valid", "No issues found")
        
        console.print(results_table)
        
        # Show final performance score
        final_status = self.coordinator.get_system_status()
        performance_score = final_status["performance_score"]
        
        score_panel = Panel(
            f"""**Final Performance Score: {performance_score}/100**

{
    "🚀 Excellent! Your system is optimized." if performance_score >= 75 else
    "⚡ Good performance with room for improvement." if performance_score >= 50 else
    "🔧 Consider upgrading hardware or adjusting settings." if performance_score >= 25 else
    "💡 System needs attention for optimal performance."
}""",
            title="Performance Assessment",
            border_style="green" if performance_score >= 75 else "yellow" if performance_score >= 50 else "red"
        )
        console.print(score_panel)
        
        Prompt.ask("\n[dim]Press Enter to continue...", default="")
    
    def show_final_summary(self):
        """Show final demo summary"""
        summary_text = """
🎉 **Phase 2 Demo Complete!**

You've experienced all the major Phase 2 features:

**✅ Hardware Detection** - Analyzed your system capabilities
**✅ Model Recommendations** - Found optimal AI models for your hardware  
**✅ Configuration Management** - Smart, multi-format configuration system
**✅ Advanced Caching** - High-performance hybrid caching with analytics
**✅ Error Handling** - Intelligent error classification and recovery
**✅ System Integration** - Coordinated operation of all components
**✅ Performance Optimization** - Automated system optimization

**What's Next?**
• Install Xencode Phase 2: `pip install -e .`
• Run setup wizard: `xencode init`
• Start coding with AI assistance!
• Explore advanced features in the documentation

Thank you for trying Xencode Phase 2! 🚀
        """
        
        panel = Panel(
            summary_text.strip(),
            title="[bold green]Demo Summary[/bold green]",
            border_style="green",
            padding=(1, 2)
        )
        console.print(panel)


async def main():
    """Run the Phase 2 demonstration"""
    demo = Phase2Demo()
    await demo.run_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted. Goodbye![/yellow]")
    except Exception as e:
        console.print(f"[red]Demo failed: {e}[/red]")
        sys.exit(1)