#!/usr/bin/env python3
"""
Hardware-Aware Model Selection for Xencode

Intelligent model selection system that chooses the optimal AI models
based on current hardware capabilities and system resources.
"""

import asyncio
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import psutil
import ollama
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

console = Console()

# Try to import optional dependencies
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    console.print("[yellow]⚠️ GPUtil not available. GPU detection will be limited.[/yellow]")


class HardwareTier(Enum):
    """Hardware capability tiers"""
    LOW_END = "low_end"      # < 8GB RAM, older CPU
    MID_RANGE = "mid_range"  # 8-16GB RAM, modern CPU
    HIGH_END = "high_end"    # >16GB RAM, modern CPU, possible GPU
    SERVER = "server"       # High-end server hardware


class ModelSize(Enum):
    """Model size categories"""
    SMALL = "small"      # < 2GB
    MEDIUM = "medium"    # 2-8GB  
    LARGE = "large"      # 8-20GB
    XLARGE = "xlarge"    # >20GB


@dataclass
class HardwareSpecs:
    """Hardware specifications"""
    cpu_count: int
    cpu_freq_max: float  # GHz
    total_memory_gb: float
    available_memory_gb: float
    memory_percent: float
    disk_space_gb: float
    gpu_available: bool
    gpu_count: int
    gpu_memory_gb: float
    system_platform: str
    is_virtual_machine: bool
    cpu_architecture: str
    tier: HardwareTier = HardwareTier.LOW_END


@dataclass
class ModelSpecs:
    """Model specifications"""
    name: str
    size_gb: float
    required_memory_gb: float
    recommended_memory_gb: float
    inference_time_ms: float  # On reference hardware
    power_consumption: float  # Relative power consumption (0-1)
    accuracy_score: float     # Relative accuracy (0-1)
    size_category: ModelSize = ModelSize.SMALL
    quantization_support: List[str] = field(default_factory=list)


@dataclass
class ModelRecommendation:
    """Recommended model for current hardware"""
    model_name: str
    confidence_score: float  # 0-1
    estimated_performance: Dict[str, float]  # ms, memory usage, etc.
    reason: str
    alternatives: List[str] = field(default_factory=list)


class HardwareAnalyzer:
    """Analyzes system hardware capabilities"""

    def __init__(self):
        self.specs: Optional[HardwareSpecs] = None

    def analyze_hardware(self) -> HardwareSpecs:
        """Analyze current hardware and return specifications"""
        # CPU Information
        cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()
        
        # Get max CPU frequency if available
        cpu_freq_max = 0.0
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_freq_max = cpu_freq.max / 1000.0  # Convert MHz to GHz
        except:
            # Fallback: estimate from CPU count and system
            if cpu_count >= 8:
                cpu_freq_max = 3.0  # High-end estimate
            elif cpu_count >= 4:
                cpu_freq_max = 2.5  # Mid-range estimate
            else:
                cpu_freq_max = 2.0  # Low-end estimate

        # Memory Information
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        available_memory_gb = memory.available / (1024**3)
        memory_percent = memory.percent

        # Disk Space
        disk_usage = psutil.disk_usage('.')
        disk_space_gb = disk_usage.free / (1024**3)

        # GPU Detection
        gpu_available = False
        gpu_count = 0
        gpu_memory_gb = 0.0

        if GPU_AVAILABLE:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_available = True
                gpu_count = len(gpus)
                # Use the largest GPU memory
                gpu_memory_gb = max(gpu.memoryTotal / 1024.0 for gpu in gpus)  # Convert MB to GB

        # Check if running in VM
        is_virtual_machine = self._is_virtual_machine()

        # System platform and architecture
        system_platform = platform.system().lower()
        cpu_architecture = platform.machine().lower()

        # Determine hardware tier
        tier = self._determine_hardware_tier(
            total_memory_gb, cpu_count, cpu_freq_max, gpu_available, gpu_memory_gb
        )

        specs = HardwareSpecs(
            cpu_count=cpu_count,
            cpu_freq_max=cpu_freq_max,
            total_memory_gb=round(total_memory_gb, 2),
            available_memory_gb=round(available_memory_gb, 2),
            memory_percent=memory_percent,
            disk_space_gb=round(disk_space_gb, 2),
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            gpu_memory_gb=round(gpu_memory_gb, 2),
            system_platform=system_platform,
            is_virtual_machine=is_virtual_machine,
            cpu_architecture=cpu_architecture,
            tier=tier
        )

        self.specs = specs
        return specs

    def _determine_hardware_tier(self, memory_gb: float, cpu_count: int, 
                               cpu_freq: float, gpu_available: bool, 
                               gpu_memory_gb: float) -> HardwareTier:
        """Determine hardware tier based on specifications"""
        # Calculate a composite score
        memory_score = memory_gb / 32.0  # Normalize to 32GB = 1.0
        cpu_score = (cpu_count * cpu_freq) / 32.0  # Normalize to 8-core @ 4GHz = 1.0
        gpu_score = (gpu_memory_gb / 24.0) if gpu_available else 0.0  # Normalize to 24GB GPU = 1.0

        composite_score = (memory_score * 0.5) + (cpu_score * 0.4) + (gpu_score * 0.1)

        if composite_score >= 1.0:
            return HardwareTier.SERVER
        elif composite_score >= 0.5:
            return HardwareTier.HIGH_END
        elif composite_score >= 0.2:
            return HardwareTier.MID_RANGE
        else:
            return HardwareTier.LOW_END

    def _is_virtual_machine(self) -> bool:
        """Check if running in a virtual machine"""
        try:
            # Check for common VM indicators
            if platform.machine().lower() in ['x86_64', 'amd64'] and 'virtual' in platform.processor().lower():
                return True

            # Check hypervisor bit on Linux
            if platform.system() == 'Linux':
                try:
                    with open('/sys/hypervisor/type', 'r') as f:
                        return f.read().strip() != '0'
                except FileNotFoundError:
                    pass

            # Check for common VM vendor strings
            try:
                result = subprocess.run(['dmidecode', '-s', 'system-product-name'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    output = result.stdout.lower()
                    vm_indicators = ['virtual', 'vmware', 'qemu', 'kvm', 'xen', 'parallels', 'virtualbox']
                    return any(indicator in output for indicator in vm_indicators)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            return False
        except:
            return False


class ModelDatabase:
    """Database of model specifications and capabilities"""

    def __init__(self):
        self.models: List[ModelSpecs] = self._load_model_database()

    def _load_model_database(self) -> List[ModelSpecs]:
        """Load known model specifications"""
        return [
            # Small models (< 2GB)
            ModelSpecs(
                name="gemma2:2b",
                size_gb=1.4,
                required_memory_gb=2.0,
                recommended_memory_gb=3.0,
                inference_time_ms=45.0,
                power_consumption=0.3,
                accuracy_score=0.65,
                size_category=ModelSize.SMALL,
                quantization_support=["q4_0", "q4_1", "q5_0", "q8_0"]
            ),
            ModelSpecs(
                name="phi3:mini",
                size_gb=2.3,
                required_memory_gb=3.0,
                recommended_memory_gb=4.0,
                inference_time_ms=52.0,
                power_consumption=0.4,
                accuracy_score=0.70,
                size_category=ModelSize.SMALL,
                quantization_support=["q4_0", "q4_1", "q5_0"]
            ),
            ModelSpecs(
                name="llama3.2:3b",
                size_gb=2.0,
                required_memory_gb=3.0,
                recommended_memory_gb=4.0,
                inference_time_ms=65.0,
                power_consumption=0.45,
                accuracy_score=0.72,
                size_category=ModelSize.SMALL,
                quantization_support=["q4_0", "q4_1", "q5_0"]
            ),
            
            # Medium models (2-8GB)
            ModelSpecs(
                name="mistral:7b",
                size_gb=4.1,
                required_memory_gb=6.0,
                recommended_memory_gb=8.0,
                inference_time_ms=120.0,
                power_consumption=0.6,
                accuracy_score=0.78,
                size_category=ModelSize.MEDIUM,
                quantization_support=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"]
            ),
            ModelSpecs(
                name="llama3.1:8b",
                size_gb=4.7,
                required_memory_gb=8.0,
                recommended_memory_gb=12.0,
                inference_time_ms=140.0,
                power_consumption=0.7,
                accuracy_score=0.82,
                size_category=ModelSize.MEDIUM,
                quantization_support=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"]
            ),
            
            # Large models (8-20GB)
            ModelSpecs(
                name="qwen2.5:14b",
                size_gb=8.5,
                required_memory_gb=12.0,
                recommended_memory_gb=16.0,
                inference_time_ms=280.0,
                power_consumption=0.85,
                accuracy_score=0.88,
                size_category=ModelSize.LARGE,
                quantization_support=["q4_0", "q4_1", "q5_0"]
            ),
            
            # XLarge models (>20GB)
            ModelSpecs(
                name="llama3.1:70b",
                size_gb=40.0,
                required_memory_gb=48.0,
                recommended_memory_gb=64.0,
                inference_time_ms=800.0,
                power_consumption=0.95,
                accuracy_score=0.92,
                size_category=ModelSize.XLARGE,
                quantization_support=["q4_0", "q4_1"]
            ),
            ModelSpecs(
                name="qwen2.5:72b",
                size_gb=45.0,
                required_memory_gb=48.0,
                recommended_memory_gb=64.0,
                inference_time_ms=850.0,
                power_consumption=0.95,
                accuracy_score=0.93,
                size_category=ModelSize.XLARGE,
                quantization_support=["q4_0", "q4_1"]
            ),
        ]

    def get_model_by_name(self, name: str) -> Optional[ModelSpecs]:
        """Get model specification by name"""
        for model in self.models:
            if model.name.lower() == name.lower():
                return model
        return None

    def get_models_by_tier(self, tier: HardwareTier) -> List[ModelSpecs]:
        """Get models appropriate for hardware tier"""
        if tier == HardwareTier.LOW_END:
            return [m for m in self.models if m.size_category in [ModelSize.SMALL]]
        elif tier == HardwareTier.MID_RANGE:
            return [m for m in self.models if m.size_category in [ModelSize.SMALL, ModelSize.MEDIUM]]
        elif tier == HardwareTier.HIGH_END:
            return [m for m in self.models if m.size_category in [ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE]]
        else:  # SERVER
            return self.models  # All models


class HardwareAwareModelSelector:
    """Selects optimal models based on hardware capabilities"""

    def __init__(self):
        self.hardware_analyzer = HardwareAnalyzer()
        self.model_database = ModelDatabase()
        self.client = ollama.AsyncClient()

    async def get_hardware_specs(self) -> HardwareSpecs:
        """Get current hardware specifications"""
        return self.hardware_analyzer.analyze_hardware()

    async def get_available_models(self) -> List[str]:
        """Get list of available models in Ollama"""
        try:
            response = await self.client.list()
            return [model['name'] for model in response.get('models', [])]
        except Exception as e:
            console.print(f"[red]❌ Error getting available models: {e}[/red]")
            return []

    async def recommend_models(self, task_type: str = "general") -> List[ModelRecommendation]:
        """Recommend optimal models for current hardware and task type"""
        hardware = await self.get_hardware_specs()
        available_models = await self.get_available_models()
        
        # Get models appropriate for hardware tier
        tier_models = self.model_database.get_models_by_tier(hardware.tier)
        
        # Filter to only available models
        available_tier_models = [m for m in tier_models if m.name in available_models]
        
        recommendations = []
        
        for model in available_tier_models:
            # Calculate confidence score based on multiple factors
            confidence = self._calculate_model_confidence(model, hardware, task_type)
            
            if confidence > 0.1:  # Only include models with reasonable confidence
                estimated_performance = self._estimate_performance(model, hardware)
                
                # Determine reason for recommendation
                if model.size_category == ModelSize.SMALL and hardware.tier in [HardwareTier.LOW_END, HardwareTier.MID_RANGE]:
                    reason = "Small model suitable for limited hardware"
                elif model.size_category == ModelSize.MEDIUM and hardware.tier in [HardwareTier.MID_RANGE, HardwareTier.HIGH_END]:
                    reason = "Balanced performance and capability"
                elif model.size_category == ModelSize.LARGE and hardware.tier in [HardwareTier.HIGH_END, HardwareTier.SERVER]:
                    reason = "High capability model for powerful hardware"
                else:
                    reason = "Good fit for your hardware specifications"
                
                # Find alternatives of similar capability
                alternatives = self._find_alternatives(model, available_tier_models, hardware)
                
                recommendation = ModelRecommendation(
                    model_name=model.name,
                    confidence_score=round(confidence, 3),
                    estimated_performance=estimated_performance,
                    reason=reason,
                    alternatives=alternatives[:3]  # Top 3 alternatives
                )
                
                recommendations.append(recommendation)
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return recommendations

    def _calculate_model_confidence(self, model: ModelSpecs, hardware: HardwareSpecs, task_type: str) -> float:
        """Calculate confidence score for model on current hardware"""
        # Memory adequacy (0-1)
        memory_adequacy = min(1.0, hardware.available_memory_gb / model.required_memory_gb)
        
        # Tier compatibility (0-1)
        tier_match = self._calculate_tier_match(model.size_category, hardware.tier)
        
        # Task suitability (0-1)
        task_suitability = self._calculate_task_suitability(model, task_type)
        
        # Accuracy preference (0-1) - normalize accuracy score
        accuracy_factor = model.accuracy_score
        
        # Combine factors with weights
        confidence = (
            memory_adequacy * 0.4 +
            tier_match * 0.3 +
            task_suitability * 0.2 +
            accuracy_factor * 0.1
        )
        
        return min(1.0, confidence)

    def _calculate_tier_match(self, model_size: ModelSize, hardware_tier: HardwareTier) -> float:
        """Calculate how well model size matches hardware tier"""
        if hardware_tier == HardwareTier.LOW_END:
            return 1.0 if model_size == ModelSize.SMALL else 0.3
        elif hardware_tier == HardwareTier.MID_RANGE:
            if model_size in [ModelSize.SMALL, ModelSize.MEDIUM]:
                return 1.0
            else:
                return 0.5
        elif hardware_tier == HardwareTier.HIGH_END:
            if model_size in [ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE]:
                return 1.0
            else:
                return 0.7  # Can handle XL with limitations
        else:  # SERVER
            return 1.0  # Can handle any model

    def _calculate_task_suitability(self, model: ModelSpecs, task_type: str) -> float:
        """Calculate how suitable model is for specific task"""
        # Different tasks have different requirements
        if task_type == "general":
            return 1.0
        elif task_type == "creative":
            return model.accuracy_score  # Higher accuracy preferred
        elif task_type == "fast_response":
            return 1.0 - (model.inference_time_ms / 500.0)  # Faster is better, cap at 500ms
        elif task_type == "accuracy_critical":
            return model.accuracy_score * 1.2 if model.accuracy_score > 0.8 else model.accuracy_score
        else:
            return 0.8  # Default

    def _estimate_performance(self, model: ModelSpecs, hardware: HardwareSpecs) -> Dict[str, float]:
        """Estimate performance on current hardware"""
        # Adjust inference time based on hardware tier
        tier_multiplier = {
            HardwareTier.LOW_END: 1.5,    # Slower
            HardwareTier.MID_RANGE: 1.0,  # Reference
            HardwareTier.HIGH_END: 0.8,   # Faster
            HardwareTier.SERVER: 0.6      # Fastest
        }
        
        estimated_time = model.inference_time_ms * tier_multiplier[hardware.tier]
        
        # Memory usage estimate
        memory_usage_gb = min(hardware.total_memory_gb, model.recommended_memory_gb)
        
        return {
            "inference_time_ms": round(estimated_time, 2),
            "estimated_memory_usage_gb": round(memory_usage_gb, 2),
            "power_consumption_estimate": round(model.power_consumption, 2)
        }

    def _find_alternatives(self, current_model: ModelSpecs, all_models: List[ModelSpecs], 
                          hardware: HardwareSpecs) -> List[str]:
        """Find alternative models with similar capabilities"""
        alternatives = []
        
        for model in all_models:
            if model.name != current_model.name:
                # Similar size category or adjacent categories
                size_diff = abs(list(ModelSize).index(model.size_category) - 
                              list(ModelSize).index(current_model.size_category))
                
                # Similar accuracy range
                accuracy_diff = abs(model.accuracy_score - current_model.accuracy_score)
                
                if size_diff <= 1 and accuracy_diff <= 0.1:
                    alternatives.append(model.name)
        
        return alternatives

    async def get_top_recommendation(self, task_type: str = "general") -> Optional[ModelRecommendation]:
        """Get the top model recommendation"""
        recommendations = await self.recommend_models(task_type)
        return recommendations[0] if recommendations else None

    def display_hardware_report(self, specs: HardwareSpecs):
        """Display hardware analysis in a formatted table"""
        table = Table(title="Hardware Analysis")
        table.add_column("Component", style="cyan")
        table.add_column("Specification", style="magenta")
        table.add_column("Status", style="green")

        # CPU
        cpu_status = "🟢" if specs.cpu_count >= 4 else "🟡"
        table.add_row("CPU", f"{specs.cpu_count} cores @ {specs.cpu_freq_max}GHz", cpu_status)

        # Memory
        mem_status = "🟢" if specs.total_memory_gb >= 16 else "🟡" if specs.total_memory_gb >= 8 else "🔴"
        table.add_row("Memory", f"{specs.total_memory_gb}GB total, {specs.available_memory_gb}GB available", mem_status)

        # GPU
        gpu_status = "🟢" if specs.gpu_available else "🔴"
        gpu_info = f"{specs.gpu_count} GPU(s), {specs.gpu_memory_gb}GB" if specs.gpu_available else "None detected"
        table.add_row("GPU", gpu_info, gpu_status)

        # Storage
        storage_status = "🟢" if specs.disk_space_gb >= 50 else "🟡" if specs.disk_space_gb >= 20 else "🔴"
        table.add_row("Storage", f"{specs.disk_space_gb}GB available", storage_status)

        # System
        vm_status = "🔴 Virtual" if specs.is_virtual_machine else "🟢 Physical"
        table.add_row("System", f"{specs.system_platform} ({specs.cpu_architecture})", vm_status)

        # Tier
        tier_colors = {
            HardwareTier.LOW_END: "🔴 Low-End",
            HardwareTier.MID_RANGE: "🟡 Mid-Range", 
            HardwareTier.HIGH_END: "🟢 High-End",
            HardwareTier.SERVER: "🔵 Server"
        }
        table.add_row("Tier", "", tier_colors[specs.tier])

        console.print(table)

    def display_recommendations(self, recommendations: List[ModelRecommendation]):
        """Display model recommendations"""
        if not recommendations:
            console.print("[yellow]⚠️ No suitable models found for your hardware[/yellow]")
            return

        console.print(f"\n[bold green]🎯 Top Model Recommendations ({len(recommendations)} available):[/bold green]")
        
        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
            console.print(f"\n{i}. [bold]{rec.model_name}[/bold] (Confidence: {rec.confidence_score:.2f})")
            console.print(f"   Reason: {rec.reason}")
            console.print(f"   Est. Performance: {rec.estimated_performance['inference_time_ms']}ms response")
            
            if rec.alternatives:
                console.print(f"   💡 Alternatives: {', '.join(rec.alternatives[:3])}")


async def main():
    """Demo of hardware-aware model selection"""
    console.print(Panel(
        "[bold green]🤖 Hardware-Aware Model Selector[/bold green]\n"
        "Intelligent model selection based on your system capabilities",
        title="Xencode Model Selection",
        border_style="green"
    ))

    selector = HardwareAwareModelSelector()

    # Analyze hardware
    console.print("[blue]🔍 Analyzing your hardware...[/blue]")
    specs = await selector.get_hardware_specs()
    selector.display_hardware_report(specs)

    # Get recommendations
    console.print("[blue]💡 Getting model recommendations...[/blue]")
    recommendations = await selector.recommend_models(task_type="general")
    selector.display_recommendations(recommendations)

    # Show top recommendation details
    if recommendations:
        top_rec = recommendations[0]
        console.print(f"\n[bold green]🏆 Best Choice: {top_rec.model_name}[/bold green]")
        console.print(f"   This model is optimally matched to your {specs.tier.value} hardware configuration.")
        console.print(f"   Expected response time: ~{top_rec.estimated_performance['inference_time_ms']}ms")
        console.print(f"   Estimated memory usage: {top_rec.estimated_performance['estimated_memory_usage_gb']}GB")


if __name__ == "__main__":
    # Don't run by default since it requires Ollama
    # asyncio.run(main())
    pass