#!/usr/bin/env python3
"""
Intelligent Model Selection System for Xencode Phase 2

Automatically detects system hardware and recommends optimal Ollama models
for the best performance based on available resources.
"""

import json
import os
import platform
import psutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class SystemSpecs:
    """System hardware specifications"""
    cpu_cores: int
    cpu_architecture: str
    total_ram_gb: float
    available_ram_gb: float
    gpu_available: bool
    gpu_type: str
    gpu_vram_gb: float
    storage_type: str
    available_storage_gb: float
    performance_score: int


@dataclass
class ModelRecommendation:
    """AI model recommendation with metadata"""
    name: str
    size_gb: float
    ram_required_gb: float
    performance_tier: str  # "fast", "balanced", "powerful"
    description: str
    ollama_tag: str
    download_url: str
    estimated_speed: str  # "Ultra-fast", "Fast", "Moderate", "Slow"
    quality_score: int  # 1-10 scale


class HardwareDetector:
    """Detects system hardware capabilities"""
    
    def __init__(self):
        self.console = Console()
    
    def detect_system_specs(self) -> SystemSpecs:
        """Perform comprehensive system hardware detection"""
        
        with self.console.status("[bold blue]Analyzing your system..."):
            # CPU Detection
            cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
            cpu_architecture = platform.machine().lower()
            
            # Memory Detection
            memory = psutil.virtual_memory()
            total_ram_gb = memory.total / (1024**3)
            available_ram_gb = memory.available / (1024**3)
            
            # GPU Detection
            gpu_info = self._detect_gpu()
            
            # Storage Detection
            storage_info = self._detect_storage()
            
            # Performance Score Calculation
            performance_score = self._calculate_performance_score(
                cpu_cores, total_ram_gb, gpu_info["available"], cpu_architecture
            )
        
        return SystemSpecs(
            cpu_cores=cpu_cores,
            cpu_architecture=cpu_architecture,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            gpu_available=gpu_info["available"],
            gpu_type=gpu_info["type"],
            gpu_vram_gb=gpu_info["vram_gb"],
            storage_type=storage_info["type"],
            available_storage_gb=storage_info["available_gb"],
            performance_score=performance_score
        )
    
    def _detect_gpu(self) -> Dict:
        """Detect GPU capabilities"""
        gpu_info = {
            "available": False,
            "type": "none",
            "vram_gb": 0.0
        }
        
        try:
            # Try NVIDIA first
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                vram_mb = int(result.stdout.strip())
                gpu_info = {
                    "available": True,
                    "type": "nvidia",
                    "vram_gb": vram_mb / 1024
                }
                return gpu_info
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        
        try:
            # Try AMD
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"], 
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and "Used" in result.stdout:
                gpu_info["available"] = True
                gpu_info["type"] = "amd"
                gpu_info["vram_gb"] = 8.0  # Default estimate
                return gpu_info
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Check for Apple Silicon
        if platform.system() == "Darwin" and "arm" in platform.machine().lower():
            gpu_info = {
                "available": True,
                "type": "apple_silicon",
                "vram_gb": 0.0  # Unified memory
            }
        
        return gpu_info
    
    def _detect_storage(self) -> Dict:
        """Detect storage type and available space"""
        try:
            # Get disk usage for current directory
            disk_usage = psutil.disk_usage('.')
            available_gb = disk_usage.free / (1024**3)
            
            # Try to detect SSD vs HDD (Linux)
            storage_type = "unknown"
            if platform.system() == "Linux":
                try:
                    with open("/sys/block/sda/queue/rotational", "r") as f:
                        rotational = f.read().strip()
                        storage_type = "hdd" if rotational == "1" else "ssd"
                except:
                    storage_type = "ssd"  # Default assumption for modern systems
            else:
                storage_type = "ssd"  # Default for macOS/Windows
            
            return {
                "type": storage_type,
                "available_gb": available_gb
            }
        except:
            return {
                "type": "unknown",
                "available_gb": 100.0  # Safe default
            }
    
    def _calculate_performance_score(self, cpu_cores: int, ram_gb: float, 
                                   has_gpu: bool, cpu_arch: str) -> int:
        """Calculate overall system performance score (0-100)"""
        score = 0
        
        # CPU Score (0-30)
        cpu_score = min(cpu_cores * 3, 30)
        if "apple" in cpu_arch or "arm" in cpu_arch:
            cpu_score *= 1.2  # Apple Silicon bonus
        
        # RAM Score (0-40)
        ram_score = min(ram_gb * 2, 40)
        
        # GPU Score (0-30)
        gpu_score = 30 if has_gpu else 0
        
        total_score = int(cpu_score + ram_score + gpu_score)
        return min(total_score, 100)


class ModelRecommendationEngine:
    """Recommends optimal models based on system specs"""
    
    def __init__(self):
        self.models = self._load_model_database()
    
    def _load_model_database(self) -> List[ModelRecommendation]:
        """Load database of available models with specifications"""
        return [
            # High-End Models (32GB+ RAM)
            ModelRecommendation(
                name="Qwen 2.5 72B",
                size_gb=45.0,
                ram_required_gb=48.0,
                performance_tier="powerful",
                description="Highest quality responses, slower inference",
                ollama_tag="qwen2.5:72b",
                download_url="https://ollama.com/library/qwen2.5:72b",
                estimated_speed="Slow",
                quality_score=10
            ),
            ModelRecommendation(
                name="Llama 3.1 70B",
                size_gb=40.0,
                ram_required_gb=42.0,
                performance_tier="powerful",
                description="Excellent for complex reasoning and coding",
                ollama_tag="llama3.1:70b",
                download_url="https://ollama.com/library/llama3.1:70b",
                estimated_speed="Slow",
                quality_score=9
            ),
            
            # Mid-Range Models (16GB+ RAM)
            ModelRecommendation(
                name="Qwen 2.5 14B",
                size_gb=8.5,
                ram_required_gb=12.0,
                performance_tier="balanced",
                description="Great balance of speed and quality",
                ollama_tag="qwen2.5:14b",
                download_url="https://ollama.com/library/qwen2.5:14b",
                estimated_speed="Moderate",
                quality_score=8
            ),
            ModelRecommendation(
                name="Llama 3.1 8B",
                size_gb=4.7,
                ram_required_gb=8.0,
                performance_tier="balanced",
                description="Recommended - Best overall performance",
                ollama_tag="llama3.1:8b",
                download_url="https://ollama.com/library/llama3.1:8b",
                estimated_speed="Fast",
                quality_score=8
            ),
            ModelRecommendation(
                name="Mistral 7B",
                size_gb=4.1,
                ram_required_gb=7.0,
                performance_tier="balanced",
                description="Fast and efficient for general use",
                ollama_tag="mistral:7b",
                download_url="https://ollama.com/library/mistral:7b",
                estimated_speed="Fast",
                quality_score=7
            ),
            
            # Low-End Models (8GB+ RAM)
            ModelRecommendation(
                name="Llama 3.2 3B",
                size_gb=2.0,
                ram_required_gb=4.0,
                performance_tier="fast",
                description="Quick responses, good for basic tasks",
                ollama_tag="llama3.2:3b",
                download_url="https://ollama.com/library/llama3.2:3b",
                estimated_speed="Fast",
                quality_score=6
            ),
            ModelRecommendation(
                name="Phi-3 Mini",
                size_gb=2.3,
                ram_required_gb=4.0,
                performance_tier="fast",
                description="Ultra-fast responses, Microsoft's efficient model",
                ollama_tag="phi3:mini",
                download_url="https://ollama.com/library/phi3:mini",
                estimated_speed="Ultra-fast",
                quality_score=6
            ),
            
            # Ultra Low-End Models (4GB+ RAM)
            ModelRecommendation(
                name="Gemma 2B",
                size_gb=1.4,
                ram_required_gb=3.0,
                performance_tier="fast",
                description="Lightweight model for resource-constrained systems",
                ollama_tag="gemma2:2b",
                download_url="https://ollama.com/library/gemma2:2b",
                estimated_speed="Ultra-fast",
                quality_score=5
            ),
            ModelRecommendation(
                name="TinyLlama 1.1B",
                size_gb=0.7,
                ram_required_gb=2.0,
                performance_tier="fast",
                description="Minimal resource usage, basic capabilities",
                ollama_tag="tinyllama:1.1b",
                download_url="https://ollama.com/library/tinyllama:1.1b",
                estimated_speed="Ultra-fast",
                quality_score=4
            ),
        ]
    
    def get_recommendations(self, specs: SystemSpecs) -> Tuple[ModelRecommendation, List[ModelRecommendation]]:
        """Get primary recommendation and alternatives based on system specs"""
        
        # Filter models that can run on this system
        compatible_models = [
            model for model in self.models 
            if model.ram_required_gb <= specs.available_ram_gb
        ]
        
        if not compatible_models:
            # Fallback to smallest model
            compatible_models = [self.models[-1]]
        
        # Sort by quality score (best first)
        compatible_models.sort(key=lambda m: m.quality_score, reverse=True)
        
        # Primary recommendation logic
        primary = self._select_primary_recommendation(specs, compatible_models)
        
        # Alternative recommendations (different tiers)
        alternatives = self._select_alternatives(specs, compatible_models, primary)
        
        return primary, alternatives
    
    def _select_primary_recommendation(self, specs: SystemSpecs, models: List[ModelRecommendation]) -> ModelRecommendation:
        """Select the best primary recommendation"""
        
        # High-end systems: Prefer powerful models
        if specs.total_ram_gb >= 32 and specs.performance_score >= 80:
            powerful_models = [m for m in models if m.performance_tier == "powerful"]
            if powerful_models:
                return powerful_models[0]
        
        # Mid-range systems: Prefer balanced models
        if specs.total_ram_gb >= 12 and specs.performance_score >= 50:
            balanced_models = [m for m in models if m.performance_tier == "balanced"]
            if balanced_models:
                # Prefer Llama 3.1 8B if available
                for model in balanced_models:
                    if "llama3.1:8b" in model.ollama_tag:
                        return model
                return balanced_models[0]
        
        # Low-end systems: Prefer fast models
        fast_models = [m for m in models if m.performance_tier == "fast"]
        if fast_models:
            return fast_models[0]
        
        # Fallback to any compatible model
        return models[0] if models else self.models[-1]
    
    def _select_alternatives(self, specs: SystemSpecs, models: List[ModelRecommendation], 
                           primary: ModelRecommendation) -> List[ModelRecommendation]:
        """Select 2-3 alternative recommendations"""
        alternatives = []
        
        # Remove primary from consideration
        remaining_models = [m for m in models if m.ollama_tag != primary.ollama_tag]
        
        # Add one faster option
        faster_models = [m for m in remaining_models if m.size_gb < primary.size_gb]
        if faster_models:
            alternatives.append(faster_models[0])
        
        # Add one more powerful option (if system can handle it)
        powerful_models = [m for m in remaining_models if m.size_gb > primary.size_gb]
        if powerful_models and specs.available_ram_gb >= powerful_models[0].ram_required_gb:
            alternatives.append(powerful_models[0])
        
        # Fill remaining slots
        for model in remaining_models:
            if len(alternatives) >= 2:
                break
            if model not in alternatives:
                alternatives.append(model)
        
        return alternatives[:2]  # Max 2 alternatives


class FirstRunSetup:
    """Interactive setup wizard for first-time users"""
    
    def __init__(self):
        self.console = Console()
        self.detector = HardwareDetector()
        self.recommender = ModelRecommendationEngine()
    
    def run_setup(self) -> Optional[ModelRecommendation]:
        """Run the complete first-run setup process"""
        
        # Welcome message
        self._show_welcome()
        
        # System detection
        specs = self.detector.detect_system_specs()
        self._show_system_specs(specs)
        
        # Get recommendations
        primary, alternatives = self.recommender.get_recommendations(specs)
        
        # Show recommendations and get user choice
        choice = self._show_recommendations(primary, alternatives, specs)
        
        if choice:
            # Download and setup chosen model
            success = self._download_model(choice)
            if success:
                self._save_model_config(choice)
                self._show_completion(choice)
                return choice
        
        return None
    
    def _show_welcome(self):
        """Display welcome message"""
        welcome_text = """
🤖 Welcome to Xencode's Intelligent Setup!

We'll analyze your system and recommend the perfect AI model 
for optimal performance on your hardware.

This process takes about 30 seconds and ensures you get the 
best possible experience from day one.
        """
        
        panel = Panel(
            welcome_text.strip(),
            title="[bold blue]Xencode Setup Wizard[/bold blue]",
            border_style="blue"
        )
        self.console.print(panel)
        self.console.print()
    
    def _show_system_specs(self, specs: SystemSpecs):
        """Display detected system specifications"""
        
        table = Table(title="🖥️  System Analysis Results")
        table.add_column("Component", style="cyan")
        table.add_column("Details", style="white")
        table.add_column("Score", style="green")
        
        # CPU
        cpu_desc = f"{specs.cpu_cores} cores ({specs.cpu_architecture})"
        cpu_score = "⭐" * min(specs.cpu_cores // 2, 5)
        table.add_row("CPU", cpu_desc, cpu_score)
        
        # RAM
        ram_desc = f"{specs.total_ram_gb:.1f} GB total, {specs.available_ram_gb:.1f} GB available"
        ram_score = "⭐" * min(int(specs.total_ram_gb // 4), 5)
        table.add_row("Memory", ram_desc, ram_score)
        
        # GPU
        gpu_desc = f"{specs.gpu_type.title()} GPU" if specs.gpu_available else "No dedicated GPU"
        if specs.gpu_vram_gb > 0:
            gpu_desc += f" ({specs.gpu_vram_gb:.1f} GB VRAM)"
        gpu_score = "⭐⭐⭐⭐⭐" if specs.gpu_available else "⭐"
        table.add_row("Graphics", gpu_desc, gpu_score)
        
        # Storage
        storage_desc = f"{specs.storage_type.upper()}, {specs.available_storage_gb:.1f} GB available"
        storage_score = "⭐⭐⭐⭐⭐" if specs.storage_type == "ssd" else "⭐⭐⭐"
        table.add_row("Storage", storage_desc, storage_score)
        
        # Overall score
        overall_score = "⭐" * (specs.performance_score // 20)
        table.add_row("[bold]Overall Performance[/bold]", 
                     f"[bold]{specs.performance_score}/100[/bold]", 
                     f"[bold]{overall_score}[/bold]")
        
        self.console.print(table)
        self.console.print()
    
    def _show_recommendations(self, primary: ModelRecommendation, 
                            alternatives: List[ModelRecommendation], 
                            specs: SystemSpecs) -> Optional[ModelRecommendation]:
        """Show model recommendations and get user choice"""
        
        self.console.print("[bold green]🎯 Recommended AI Models for Your System[/bold green]\n")
        
        # Primary recommendation
        primary_panel = Panel(
            f"""[bold]{primary.name}[/bold] ({primary.size_gb:.1f} GB)
{primary.description}

• Speed: {primary.estimated_speed}
• Quality: {"⭐" * primary.quality_score}
• RAM Required: {primary.ram_required_gb:.1f} GB

[green]✅ RECOMMENDED - Best match for your system[/green]""",
            title="🚀 Primary Recommendation",
            border_style="green"
        )
        self.console.print(primary_panel)
        
        # Alternatives
        if alternatives:
            self.console.print("\n[bold blue]Alternative Options:[/bold blue]")
            for i, alt in enumerate(alternatives, 1):
                style = "blue" if i == 1 else "yellow"
                icon = "⚡" if alt.performance_tier == "fast" else "🧠"
                
                alt_panel = Panel(
                    f"""[bold]{alt.name}[/bold] ({alt.size_gb:.1f} GB)
{alt.description}

• Speed: {alt.estimated_speed}
• Quality: {"⭐" * alt.quality_score}
• RAM Required: {alt.ram_required_gb:.1f} GB""",
                    title=f"{icon} Option {i + 1}",
                    border_style=style
                )
                self.console.print(alt_panel)
        
        # User choice
        self.console.print()
        choices = ["1"] + [str(i + 2) for i in range(len(alternatives))] + ["skip", "manual"]
        
        choice_text = f"""
What would you like to do?

[bold green]1[/bold green] - Install recommended model ({primary.name})
"""
        
        for i, alt in enumerate(alternatives, 2):
            choice_text += f"[bold blue]{i}[/bold blue] - Install {alt.name}\n"
        
        choice_text += """[bold yellow]skip[/bold yellow] - Skip setup (use defaults)
[bold red]manual[/bold red] - Browse all available models

"""
        
        self.console.print(choice_text)
        
        while True:
            choice = Prompt.ask("Your choice", choices=choices, default="1")
            
            if choice == "1":
                return primary
            elif choice in [str(i + 2) for i in range(len(alternatives))]:
                return alternatives[int(choice) - 2]
            elif choice == "skip":
                return None
            elif choice == "manual":
                return self._show_manual_selection()
            else:
                self.console.print("[red]Invalid choice. Please try again.[/red]")
    
    def _show_manual_selection(self) -> Optional[ModelRecommendation]:
        """Show all available models for manual selection"""
        
        table = Table(title="📋 All Available Models")
        table.add_column("#", style="cyan")
        table.add_column("Model", style="white")
        table.add_column("Size", style="yellow")
        table.add_column("RAM Req.", style="red")
        table.add_column("Speed", style="green")
        table.add_column("Quality", style="blue")
        
        for i, model in enumerate(self.recommender.models, 1):
            table.add_row(
                str(i),
                model.name,
                f"{model.size_gb:.1f} GB",
                f"{model.ram_required_gb:.1f} GB",
                model.estimated_speed,
                "⭐" * model.quality_score
            )
        
        self.console.print(table)
        self.console.print()
        
        choices = [str(i) for i in range(1, len(self.recommender.models) + 1)] + ["back"]
        choice = Prompt.ask(
            "Select a model number (or 'back' to return)", 
            choices=choices
        )
        
        if choice == "back":
            return None
        
        return self.recommender.models[int(choice) - 1]
    
    def _download_model(self, model: ModelRecommendation) -> bool:
        """Download and install the selected model"""
        
        self.console.print(f"\n[bold green]📥 Installing {model.name}...[/bold green]")
        
        # Check if Ollama is installed
        if not self._check_ollama_installed():
            self.console.print("[red]❌ Ollama is not installed. Please install Ollama first.[/red]")
            return False
        
        # Download with progress
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task(f"Downloading {model.name}...", total=None)
                
                # Run ollama pull command
                result = subprocess.run(
                    ["ollama", "pull", model.ollama_tag],
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minutes timeout
                )
                
                if result.returncode == 0:
                    progress.update(task, description=f"✅ {model.name} installed successfully!")
                    return True
                else:
                    progress.update(task, description=f"❌ Failed to install {model.name}")
                    self.console.print(f"[red]Error: {result.stderr}[/red]")
                    return False
                    
        except subprocess.TimeoutExpired:
            self.console.print("[red]❌ Download timed out. Please check your internet connection.[/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]❌ Error downloading model: {e}[/red]")
            return False
    
    def _check_ollama_installed(self) -> bool:
        """Check if Ollama is installed and accessible"""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def _save_model_config(self, model: ModelRecommendation):
        """Save model configuration for future use"""
        config_dir = Path.home() / ".xencode"
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / "model_config.json"
        config = {
            "selected_model": model.ollama_tag,
            "model_name": model.name,
            "setup_completed": True,
            "setup_date": str(subprocess.check_output(["date"], text=True).strip())
        }
        
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
    
    def _show_completion(self, model: ModelRecommendation):
        """Show setup completion message"""
        completion_text = f"""
🎉 Setup Complete!

Your AI model is ready to use:
• Model: {model.name}
• Performance: {model.estimated_speed} responses
• Quality: {"⭐" * model.quality_score}

You can now start using Xencode with optimized performance 
for your system. Happy coding! 🚀
        """
        
        panel = Panel(
            completion_text.strip(),
            title="[bold green]✅ Setup Successful[/bold green]",
            border_style="green"
        )
        self.console.print(panel)


def main():
    """Main entry point for the setup wizard"""
    try:
        setup = FirstRunSetup()
        model = setup.run_setup()
        
        if model:
            console.print(f"\n[green]Selected model: {model.name} ({model.ollama_tag})[/green]")
            return 0
        else:
            console.print("\n[yellow]Setup skipped. Using default configuration.[/yellow]")
            return 0
            
    except KeyboardInterrupt:
        console.print("\n[red]Setup cancelled by user.[/red]")
        return 1
    except Exception as e:
        console.print(f"\n[red]Setup failed: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())