#!/usr/bin/env python3
"""
Phase 2 Integration System for Xencode

Coordinates all Phase 2 features: Intelligent Model Selection, Advanced Caching,
Smart Configuration, and Error Handling for optimal performance and reliability.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table
import time

# Import Phase 2 components
from xencode.intelligent_model_selector import FirstRunSetup, HardwareDetector
from xencode.advanced_cache_system import get_cache_manager, HybridCacheManager
from xencode.smart_config_manager import get_config_manager, ConfigurationManager, XencodeConfig
from xencode.advanced_error_handler import get_error_handler, ErrorHandler, ErrorSeverity, ErrorCategory

# Import vault for health checks
try:
    from xencode.auth.json_file_vault import JsonFileCredentialVault
    VAULT_AVAILABLE = True
except ImportError:
    JsonFileCredentialVault = None  # type: ignore
    VAULT_AVAILABLE = False

# Phase 6 AI/ML imports
try:
    from xencode.ai_ensembles import create_ensemble_reasoner, EnsembleReasoner
    from xencode.ollama_optimizer import create_ollama_optimizer, OllamaOptimizer
    from xencode.rlhf_tuner import create_rlhf_tuner, RLHFTuner
except ImportError:
    create_ensemble_reasoner = EnsembleReasoner = None
    create_ollama_optimizer = OllamaOptimizer = None
    create_rlhf_tuner = RLHFTuner = None

console = Console()


class Phase2Coordinator:
    """Main coordinator for Phase 2 features"""
    
    def __init__(self):
        self.config_manager: Optional[ConfigurationManager] = None
        self.cache_manager: Optional[HybridCacheManager] = None
        self.error_handler: Optional[ErrorHandler] = None
        self.config: Optional[XencodeConfig] = None
        self.ensemble_reasoner: Optional[EnsembleReasoner] = None
        self.ollama_optimizer: Optional[OllamaOptimizer] = None
        self.rlhf_tuner: Optional[RLHFTuner] = None
        self.initialized = False
        
    async def initialize(self, config_path: Optional[Path] = None, include_rlhf: bool = True) -> bool:
        """Initialize all Phase 2 systems"""

        async def _run_with_timeout(coro, timeout_seconds: float, name: str):
            try:
                return await asyncio.wait_for(coro, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                console.print(
                    f"[yellow]⚠️ {name} initialization timed out after {timeout_seconds:.0f}s; skipping[/yellow]"
                )
                return None
        
        with console.status("[bold blue]Initializing Xencode Phase 2 systems..."):
            try:
                # Initialize configuration manager
                self.config_manager = get_config_manager(config_path)
                self.config = self.config_manager.load_config()
                
                # Initialize error handler
                self.error_handler = await get_error_handler()
                
                # Initialize cache manager with config values
                self.cache_manager = await get_cache_manager(
                    memory_mb=self.config.cache.memory_cache_mb,
                    disk_mb=self.config.cache.disk_cache_mb
                )
                
                # Initialize AI ensemble system (Phase 6)
                if create_ensemble_reasoner:
                    self.ensemble_reasoner = await _run_with_timeout(
                        create_ensemble_reasoner(self.cache_manager),
                        8.0,
                        "AI Ensemble system"
                    )
                    if self.ensemble_reasoner:
                        console.print("[green]✅ AI Ensemble system initialized[/green]")
                
                # Initialize Ollama optimizer
                if create_ollama_optimizer:
                    self.ollama_optimizer = await _run_with_timeout(
                        create_ollama_optimizer(),
                        6.0,
                        "Ollama Optimizer"
                    )
                    if self.ollama_optimizer:
                        console.print("[green]✅ Ollama Optimizer initialized[/green]")
                
                # Initialize RLHF tuner (lightweight config)
                if include_rlhf and create_rlhf_tuner:
                    try:
                        from xencode.rlhf_tuner import RLHFConfig
                        rlhf_config = RLHFConfig(max_epochs=1, synthetic_data_size=10)
                        self.rlhf_tuner = await _run_with_timeout(
                            create_rlhf_tuner(rlhf_config),
                            10.0,
                            "RLHF Tuner"
                        )
                        if self.rlhf_tuner:
                            console.print("[green]✅ RLHF Tuner initialized[/green]")
                    except Exception as e:
                        console.print(f"[yellow]⚠️ RLHF Tuner initialization skipped: {e}[/yellow]")
                
                self.initialized = True
                console.print("[green]✅ Phase 2 systems initialized successfully[/green]")
                return True
                
            except Exception as e:
                console.print(f"[red]❌ Failed to initialize Phase 2 systems: {e}[/red]")
                return False
    
    async def run_first_time_setup(self) -> bool:
        """Run first-time setup wizard"""
        console.print("[bold blue]🚀 Welcome to Xencode Phase 2![/bold blue]\n")
        
        # Check if this is first run
        config_file = self.config_manager.find_config_file() if self.config_manager else None
        is_first_run = config_file is None
        
        if is_first_run:
            console.print("[yellow]First-time setup detected. Let's get you started![/yellow]\n")
            
            # Run intelligent model selection
            console.print("[bold]Step 1: AI Model Selection[/bold]")
            setup = FirstRunSetup()
            selected_model = setup.run_setup()
            
            if selected_model:
                # Update config with selected model
                if self.config:
                    self.config.model.name = selected_model.ollama_tag
                    self.config_manager.save_config()
            
            # Run configuration setup
            console.print("\n[bold]Step 2: Configuration Setup[/bold]")
            if self.config_manager:
                self.config = self.config_manager.interactive_setup()
            
            # Step 3: Offer vault migration
            console.print("\n[bold]Step 3: Credential Vault Migration[/bold]")
            console.print("[yellow]💡 Migrate plaintext API keys to an encrypted vault for security.[/yellow]")
            console.print("[yellow]   Keys like openai_api_key, anthropic_api_key will be stored[/yellow]")
            console.print("[yellow]   encrypted with Fernet (AES-128-CBC) instead of plaintext.[/yellow]")

            if Confirm.ask("Scan config for plaintext API keys and migrate to vault?", default=True):
                try:
                    from xencode.auth.json_file_vault import JsonFileCredentialVault

                    with console.status("[bold blue]🔍 Scanning config and migrating keys..."):
                        result = JsonFileCredentialVault.migrate_from_config()

                    migrated = result.get("migrated", 0)
                    skipped = result.get("skipped", 0)
                    errors = result.get("errors", [])

                    if migrated > 0:
                        console.print(f"[green]  ✅ Migrated {migrated} credential(s) to encrypted vault[/green]")
                    if skipped > 0:
                        console.print(f"[yellow]  ⏩ Skipped {skipped} env-var references (already secure)[/yellow]")
                    if errors:
                        for err in errors:
                            console.print(f"  [yellow]⚠️  {err}[/yellow]")
                    if migrated == 0 and not errors:
                        console.print("[yellow]  No plaintext API keys found to migrate[/yellow]")
                except ImportError:
                    console.print("[yellow]  Vault module not available; skipping migration[/yellow]")
                except Exception as e:
                    console.print(f"[yellow]  Vault migration skipped: {e}[/yellow]")

            console.print("\n[green]✅ First-time setup complete! Xencode is ready to use.[/green]")
            return True
        else:
            console.print("[green]✅ Configuration found. Xencode is ready![/green]")
            return True
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Run performance optimization"""
        console.print("[bold blue]⚡ Running Performance Optimization...[/bold blue]")
        
        optimization_results = {
            "cache_optimized": False,
            "memory_freed": 0,
            "config_optimized": False,
            "errors_cleared": 0
        }
        
        try:
            # Optimize cache
            if self.cache_manager:
                await self.cache_manager.optimize_cache()
                optimization_results["cache_optimized"] = True
            
            # Memory optimization
            import gc
            import psutil
            
            before_memory = psutil.virtual_memory().percent
            gc.collect()
            after_memory = psutil.virtual_memory().percent
            optimization_results["memory_freed"] = max(0, before_memory - after_memory)
            
            # Configuration optimization
            if self.config_manager and self.config:
                validation_errors = self.config.validate()
                if validation_errors:
                    self.config_manager._auto_fix_config()
                    self.config_manager.save_config()
                    optimization_results["config_optimized"] = True
            
            console.print("[green]✅ Performance optimization complete[/green]")
            
        except Exception as e:
            if self.error_handler:
                from xencode.advanced_error_handler import ErrorContext
                context = ErrorContext(function_name="optimize_performance")
                await self.error_handler.handle_error(e, context, 
                                                    severity=ErrorSeverity.WARNING,
                                                    category=ErrorCategory.PERFORMANCE)
        
        return optimization_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "phase2_initialized": self.initialized,
            "config_status": "unknown",
            "cache_status": "unknown",
            "model_status": "unknown",
            "performance_score": 0
        }
        
        try:
            # Configuration status
            if self.config:
                validation_errors = self.config.validate()
                status["config_status"] = "healthy" if not validation_errors else "needs_attention"
            
            # Cache status
            if self.cache_manager:
                cache_stats = self.cache_manager.get_performance_stats()
                status["cache_status"] = "active"
                status["cache_hit_rate"] = cache_stats["total_hit_rate"]
                status["cache_entries"] = cache_stats["memory_cache"]["entries"] + cache_stats["disk_cache"]["entries"]
            
            # Model status (basic check)
            if self.config and self.config.model.name:
                status["model_status"] = "configured"
                status["selected_model"] = self.config.model.name
            
            # Vault status
            status["vault_status"] = "not_available"
            if VAULT_AVAILABLE and JsonFileCredentialVault:
                try:
                    vault_health = JsonFileCredentialVault.health_check()
                    if vault_health.get("vault_exists"):
                        if vault_health.get("is_valid_json"):
                            status["vault_status"] = "healthy"
                            status["vault_credential_count"] = vault_health.get("credential_count", 0)
                            status["vault_encrypted"] = vault_health.get("encryption_available", False)
                        else:
                            status["vault_status"] = "corrupted"
                    else:
                        status["vault_status"] = "empty"
                except Exception:
                    status["vault_status"] = "error"
            
            # Performance score calculation
            score = 0
            if status["config_status"] == "healthy":
                score += 20
            if status["cache_status"] == "active":
                score += 20
            if status["model_status"] == "configured":
                score += 20
            if self.initialized:
                score += 20
            if status["vault_status"] == "healthy":
                score += 20
            
            status["performance_score"] = score
            
        except Exception as e:
            console.print(f"[red]Error getting system status: {e}[/red]")
        
        return status
    
    def display_system_status(self):
        """Display system status in a nice format"""
        status = self.get_system_status()
        
        # Create status table
        table = Table(title="🔧 Xencode Phase 2 System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Details", style="green")
        
        # Phase 2 Status
        phase2_status = "✅ Active" if status["phase2_initialized"] else "❌ Not Initialized"
        table.add_row("Phase 2 Systems", phase2_status, "")
        
        # Configuration
        config_icon = "✅" if status["config_status"] == "healthy" else "⚠️" if status["config_status"] == "needs_attention" else "❌"
        table.add_row("Configuration", f"{config_icon} {status['config_status'].title()}", "")
        
        # Cache
        cache_icon = "✅" if status["cache_status"] == "active" else "❌"
        cache_details = ""
        if "cache_hit_rate" in status:
            cache_details = f"{status['cache_hit_rate']:.1f}% hit rate, {status['cache_entries']} entries"
        table.add_row("Cache System", f"{cache_icon} {status['cache_status'].title()}", cache_details)
        
        # Model
        model_icon = "✅" if status["model_status"] == "configured" else "❌"
        model_details = status.get("selected_model", "")
        table.add_row("AI Model", f"{model_icon} {status['model_status'].title()}", model_details)
        
        # Credential Vault
        vault_status_str = status.get("vault_status", "not_available")
        if vault_status_str == "healthy":
            vault_icon = "✅"
            vault_details = f"{status.get('vault_credential_count', 0)} credentials, encrypted={'✅' if status.get('vault_encrypted') else '❌'}"
        elif vault_status_str == "empty":
            vault_icon = "⏳"
            vault_details = "No vault file yet — will be created on first use"
        elif vault_status_str == "corrupted":
            vault_icon = "❌"
            vault_details = "Vault file is corrupted"
        elif vault_status_str == "error":
            vault_icon = "❌"
            vault_details = "Health check error"
        else:
            vault_icon = "⚠️"
            vault_details = "Module not available"
        table.add_row("Credential Vault", f"{vault_icon} {vault_status_str.title()}", vault_details)
        
        # Performance Score
        score = status["performance_score"]
        score_icon = "🚀" if score >= 75 else "⚡" if score >= 50 else "🐌" if score >= 25 else "💔"
        table.add_row("Performance", f"{score_icon} {score}/100", f"{'Excellent' if score >= 75 else 'Good' if score >= 50 else 'Fair' if score >= 25 else 'Poor'}")
        
        console.print(table)
        
        # Show recommendations if needed
        recommendations = []
        if not status["phase2_initialized"]:
            recommendations.append("Run 'xencode init' to initialize Phase 2 systems")
        if status["config_status"] == "needs_attention":
            recommendations.append("Run 'xencode config --fix' to resolve configuration issues")  
        if status["cache_status"] != "active":
            recommendations.append("Enable caching in configuration for better performance")
        if status["model_status"] != "configured":
            recommendations.append("Run model selection wizard: 'xencode model select'")
        
        if recommendations:
            console.print("\n[bold yellow]💡 Recommendations:[/bold yellow]")
            for rec in recommendations:
                console.print(f"  • {rec}")
    
    async def health_check(self) -> bool:
        """Comprehensive health check"""
        console.print("[bold blue]🏥 Running System Health Check...[/bold blue]")
        
        health_status = True
        issues = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Check Phase 2 initialization
            progress.add_task("Checking Phase 2 systems...", total=None)
            if not self.initialized:
                issues.append("Phase 2 systems not initialized")
                health_status = False
            
            # Check configuration
            progress.add_task("Validating configuration...", total=None)
            if self.config:
                validation_errors = self.config.validate()
                if validation_errors:
                    issues.extend([f"Config: {error}" for section_errors in validation_errors.values() for error in section_errors])
                    health_status = False
            
            # Check cache system
            progress.add_task("Testing cache system...", total=None)
            if self.cache_manager:
                try:
                    # Test cache with a simple operation
                    test_key = "health_check_test"
                    test_value = {"timestamp": time.time(), "test": True}
                    await self.cache_manager.store_response("test", "test", test_value)
                    cached_value = await self.cache_manager.get_response("test", "test")
                    if cached_value != test_value:
                        issues.append("Cache system not functioning correctly")
                        health_status = False
                except Exception as e:
                    issues.append(f"Cache error: {e}")
                    health_status = False
            
            # Check system resources
            progress.add_task("Checking system resources...", total=None)
            try:
                import psutil
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('.')
                
                if memory.percent > 90:
                    issues.append(f"High memory usage: {memory.percent:.1f}%")
                    health_status = False
                
                if disk.percent > 90:
                    issues.append(f"Low disk space: {disk.percent:.1f}% used")
                    health_status = False
                    
            except Exception as e:
                issues.append(f"Resource check failed: {e}")
            
            # Check credential vault
            vault_health = {}
            if VAULT_AVAILABLE and JsonFileCredentialVault:
                progress.add_task("Checking credential vault...", total=None)
                try:
                    vault_health = JsonFileCredentialVault.health_check()
                    if vault_health.get("vault_exists"):
                        if vault_health.get("is_valid_json"):
                            count = vault_health.get("credential_count", 0)
                            console.print(
                                f"[green]  ✅ Vault: {count} credential(s), "
                                f"encryption={'✅' if vault_health.get('encryption_available') else '❌ base64'}"
                                f"[/green]"
                            )
                        else:
                            issues.append("Vault file exists but is corrupted (invalid JSON)")
                            health_status = False
                    # Non-existent vault is not an error — just unconfigured
                except Exception as e:
                    issues.append(f"Vault health check error: {e}")
                    health_status = False
        
        # Display results
        if health_status:
            console.print("[green]✅ System health check passed![/green]")
        else:
            console.print("[red]❌ System health check found issues:[/red]")
            for issue in issues:
                console.print(f"  • {issue}")
        
        return health_status
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.cache_manager:
            await self.cache_manager.optimize_cache()
        
        console.print("[green]✅ Cleanup complete[/green]")


# CLI Integration Functions
async def init_phase2(config_path: Optional[Path] = None) -> bool:
    """Initialize Phase 2 systems"""
    coordinator = Phase2Coordinator()
    
    if await coordinator.initialize(config_path):
        await coordinator.run_first_time_setup()
        return True
    return False


async def status_phase2() -> None:
    """Show Phase 2 status"""
    coordinator = Phase2Coordinator()
    await coordinator.initialize()
    coordinator.display_system_status()


async def optimize_phase2() -> None:
    """Optimize Phase 2 performance"""
    coordinator = Phase2Coordinator()
    await coordinator.initialize()
    results = await coordinator.optimize_performance()
    
    console.print("\n[bold]Optimization Results:[/bold]")
    if results["cache_optimized"]:
        console.print("  ✅ Cache optimized")
    if results["memory_freed"] > 0:
        console.print(f"  ✅ Memory freed: {results['memory_freed']:.1f}%")
    if results["config_optimized"]:
        console.print("  ✅ Configuration optimized")


async def health_check_phase2() -> bool:
    """Run health check"""
    coordinator = Phase2Coordinator()
    await coordinator.initialize()
    return await coordinator.health_check()


def main():
    """Main entry point for Phase 2 coordinator"""
    if len(sys.argv) < 2:
        console.print("[red]Usage: python phase2_coordinator.py <command>[/red]")
        console.print("Commands: init, status, optimize, health")
        return 1
    
    command = sys.argv[1].lower()
    
    try:
        if command == "init":
            asyncio.run(init_phase2())
        elif command == "status":
            asyncio.run(status_phase2())
        elif command == "optimize":
            asyncio.run(optimize_phase2())
        elif command == "health":
            success = asyncio.run(health_check_phase2())
            return 0 if success else 1
        else:
            console.print(f"[red]Unknown command: {command}[/red]")
            return 1
            
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())