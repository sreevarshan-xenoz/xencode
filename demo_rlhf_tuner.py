#!/usr/bin/env python3
"""
Demo: RLHF Tuner System - Code Mastery Through Reinforcement Learning

Demonstrates the RLHF tuning system for achieving code mastery through
LoRA fine-tuning with synthetic data generation and human feedback loops.
"""

import asyncio
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from xencode.rlhf_tuner import (
    RLHFTuner, RLHFConfig, CodePair, SyntheticDataGenerator,
    create_rlhf_tuner, quick_tune_model
)

console = Console()


async def demo_synthetic_data_generation():
    """Demo synthetic data generation"""
    console.print("[bold green]📊 Synthetic Data Generation Demo[/bold green]\n")
    
    generator = SyntheticDataGenerator()
    
    # Generate sample data
    console.print("[cyan]Generating synthetic code pairs for training...[/cyan]")
    pairs = await generator.generate_pairs(10)
    
    # Display sample pairs
    console.print(f"\n[green]✅ Generated {len(pairs)} code pairs[/green]")
    
    # Show examples by task type
    task_examples = {}
    for pair in pairs:
        if pair.task_type not in task_examples:
            task_examples[pair.task_type] = pair
    
    for task_type, example in task_examples.items():
        console.print(f"\n[bold blue]{task_type.title()} Example:[/bold blue]")
        
        console.print("[yellow]Input Code:[/yellow]")
        console.print(Panel(example.input_code, border_style="yellow"))
        
        console.print("[green]Improved Code:[/green]")
        console.print(Panel(example.output_code, border_style="green"))
        
        console.print(f"[cyan]Quality Score: {example.quality_score:.2f}[/cyan]")
    
    return pairs


async def demo_model_initialization():
    """Demo model initialization"""
    console.print("\n[bold blue]🤖 Model Initialization Demo[/bold blue]\n")
    
    # Create lightweight config for demo
    config = RLHFConfig(
        base_model="microsoft/DialoGPT-small",  # Lightweight model
        lora_rank=8,  # Smaller rank for faster demo
        max_epochs=1,
        synthetic_data_size=20,
        batch_size=2
    )
    
    console.print("[cyan]Configuration:[/cyan]")
    config_table = Table()
    config_table.add_column("Parameter", style="yellow")
    config_table.add_column("Value", style="white")
    
    config_table.add_row("Base Model", config.base_model)
    config_table.add_row("LoRA Rank", str(config.lora_rank))
    config_table.add_row("LoRA Alpha", str(config.lora_alpha))
    config_table.add_row("Learning Rate", f"{config.learning_rate:.2e}")
    config_table.add_row("Max Epochs", str(config.max_epochs))
    config_table.add_row("Batch Size", str(config.batch_size))
    
    console.print(config_table)
    
    # Initialize tuner
    console.print("\n[yellow]Initializing RLHF tuner...[/yellow]")
    tuner = RLHFTuner(config)
    
    try:
        success = await tuner.initialize_model()
        if success:
            console.print("[green]✅ Model initialized successfully![/green]")
            
            # Show model info
            if tuner.peft_model:
                console.print("\n[bold]Model Information:[/bold]")
                console.print(f"• Base Model: {config.base_model}")
                console.print(f"• LoRA Configuration: Rank={config.lora_rank}, Alpha={config.lora_alpha}")
                console.print(f"• Training Mode: Enabled")
                
                return tuner
        else:
            console.print("[red]❌ Model initialization failed[/red]")
            return None
            
    except Exception as e:
        console.print(f"[red]❌ Error during initialization: {e}[/red]")
        console.print("[yellow]💡 This is expected in demo mode without actual model files[/yellow]")
        return None


async def demo_training_simulation():
    """Demo training process (simulated)"""
    console.print("\n[bold yellow]🎯 Training Process Demo (Simulated)[/bold yellow]\n")
    
    # Simulate training metrics
    console.print("[cyan]Simulating RLHF training process...[/cyan]")
    
    epochs = 3
    steps_per_epoch = 25
    
    training_table = Table(title="Training Progress")
    training_table.add_column("Epoch", style="cyan")
    training_table.add_column("Step", style="yellow")
    training_table.add_column("Loss", style="red")
    training_table.add_column("Perplexity", style="green")
    training_table.add_column("Learning Rate", style="blue")
    
    # Simulate training progress
    initial_loss = 2.5
    for epoch in range(1, epochs + 1):
        for step in range(1, steps_per_epoch + 1, 5):
            # Simulate decreasing loss
            loss = initial_loss * (0.95 ** (epoch * step / 5))
            perplexity = 2.71828 ** loss
            lr = 2e-4 * (0.98 ** (epoch * step / 5))
            
            training_table.add_row(
                str(epoch),
                str(step),
                f"{loss:.4f}",
                f"{perplexity:.2f}",
                f"{lr:.2e}"
            )
    
    console.print(training_table)
    
    # Training summary
    console.print("\n[bold]Training Summary:[/bold]")
    console.print(f"• Total Epochs: {epochs}")
    console.print(f"• Total Steps: {epochs * steps_per_epoch}")
    console.print(f"• Initial Loss: {initial_loss:.4f}")
    console.print(f"• Final Loss: {loss:.4f}")
    console.print(f"• Loss Improvement: {((initial_loss - loss) / initial_loss * 100):.1f}%")
    console.print(f"• Final Perplexity: {perplexity:.2f}")


async def demo_code_improvement_examples():
    """Demo code improvement examples"""
    console.print("\n[bold magenta]🔧 Code Improvement Examples[/bold magenta]\n")
    
    examples = [
        {
            "task": "Refactoring",
            "input": """def calculate_total(items):
    total = 0
    for item in items:
        total = total + item['price']
    return total""",
            "output": """def calculate_total(items: List[Dict[str, float]]) -> float:
    \"\"\"Calculate total price of items efficiently.\"\"\"
    return sum(item['price'] for item in items)""",
            "improvements": ["Type hints", "Docstring", "List comprehension", "Built-in sum()"]
        },
        {
            "task": "Debugging",
            "input": """def divide_numbers(a, b):
    return a / b""",
            "output": """def divide_numbers(a: float, b: float) -> float:
    \"\"\"Safely divide two numbers.\"\"\"
    if b == 0:
        raise ValueError(\"Cannot divide by zero\")
    return a / b""",
            "improvements": ["Zero division check", "Type hints", "Docstring", "Error handling"]
        },
        {
            "task": "Optimization",
            "input": """def find_duplicates(numbers):
    duplicates = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] == numbers[j] and numbers[i] not in duplicates:
                duplicates.append(numbers[i])
    return duplicates""",
            "output": """def find_duplicates(numbers: List[int]) -> List[int]:
    \"\"\"Find duplicate numbers efficiently using set operations.\"\"\"
    seen = set()
    duplicates = set()
    
    for num in numbers:
        if num in seen:
            duplicates.add(num)
        else:
            seen.add(num)
    
    return list(duplicates)""",
            "improvements": ["O(n) complexity", "Set operations", "Type hints", "Clear algorithm"]
        }
    ]
    
    for example in examples:
        console.print(f"[bold blue]{example['task']} Example:[/bold blue]")
        
        console.print("[red]Before (Original Code):[/red]")
        console.print(Panel(example['input'], border_style="red"))
        
        console.print("[green]After (RLHF Improved):[/green]")
        console.print(Panel(example['output'], border_style="green"))
        
        console.print("[yellow]Key Improvements:[/yellow]")
        for improvement in example['improvements']:
            console.print(f"  • {improvement}")
        
        console.print()


async def demo_evaluation_metrics():
    """Demo evaluation metrics"""
    console.print("[bold cyan]📈 Model Evaluation Demo[/bold cyan]\n")
    
    # Simulate evaluation results
    console.print("[cyan]Simulating model evaluation on test dataset...[/cyan]")
    
    # Simulate evaluation metrics
    eval_results = {
        "total_samples": 50,
        "avg_quality_score": 0.847,
        "inference_time_ms": 23.5,
        "task_type_performance": {
            "refactor": 0.892,
            "debug": 0.834,
            "optimize": 0.876,
            "explain": 0.789
        }
    }
    
    # Display overall metrics
    overall_table = Table(title="Overall Performance")
    overall_table.add_column("Metric", style="cyan")
    overall_table.add_column("Value", style="white")
    
    overall_table.add_row("Total Test Samples", str(eval_results["total_samples"]))
    overall_table.add_row("Average Quality Score", f"{eval_results['avg_quality_score']:.3f}")
    overall_table.add_row("Average Inference Time", f"{eval_results['inference_time_ms']:.1f}ms")
    
    console.print(overall_table)
    
    # Display task-specific performance
    console.print("\n[bold]Task-Specific Performance:[/bold]")
    task_table = Table()
    task_table.add_column("Task Type", style="yellow")
    task_table.add_column("Quality Score", style="green")
    task_table.add_column("Performance", style="blue")
    
    for task, score in eval_results["task_type_performance"].items():
        performance = "Excellent" if score > 0.85 else "Good" if score > 0.75 else "Fair"
        task_table.add_row(task.title(), f"{score:.3f}", performance)
    
    console.print(task_table)
    
    # Performance insights
    console.print("\n[bold]Performance Insights:[/bold]")
    best_task = max(eval_results["task_type_performance"].items(), key=lambda x: x[1])
    worst_task = min(eval_results["task_type_performance"].items(), key=lambda x: x[1])
    
    console.print(f"• Best Performance: {best_task[0].title()} ({best_task[1]:.3f})")
    console.print(f"• Needs Improvement: {worst_task[0].title()} ({worst_task[1]:.3f})")
    console.print(f"• Overall Quality: {'Excellent' if eval_results['avg_quality_score'] > 0.8 else 'Good'}")
    console.print(f"• Inference Speed: {'Fast' if eval_results['inference_time_ms'] < 50 else 'Moderate'}")


async def demo_human_feedback_loop():
    """Demo human feedback integration"""
    console.print("\n[bold green]👥 Human Feedback Loop Demo[/bold green]\n")
    
    console.print("[cyan]Demonstrating human feedback integration...[/cyan]")
    
    # Example feedback scenarios
    feedback_examples = [
        {
            "code_pair": "Function refactoring",
            "model_output": "Added type hints and docstring",
            "human_feedback": "Good improvements, but consider adding input validation",
            "feedback_score": 0.8,
            "action": "Incorporate validation patterns in future training"
        },
        {
            "code_pair": "Bug fix",
            "model_output": "Added try-catch block",
            "human_feedback": "Excellent error handling approach",
            "feedback_score": 0.95,
            "action": "Reinforce this pattern"
        },
        {
            "code_pair": "Code optimization",
            "model_output": "Replaced loop with list comprehension",
            "human_feedback": "Good optimization, but readability could be better",
            "feedback_score": 0.75,
            "action": "Balance optimization with readability"
        }
    ]
    
    feedback_table = Table(title="Human Feedback Examples")
    feedback_table.add_column("Task", style="cyan")
    feedback_table.add_column("Model Output", style="yellow")
    feedback_table.add_column("Human Feedback", style="green")
    feedback_table.add_column("Score", style="blue")
    
    for example in feedback_examples:
        feedback_table.add_row(
            example["code_pair"],
            example["model_output"],
            example["human_feedback"],
            f"{example['feedback_score']:.2f}"
        )
    
    console.print(feedback_table)
    
    # Feedback integration process
    console.print("\n[bold]Feedback Integration Process:[/bold]")
    console.print("1. 🤖 Model generates code improvements")
    console.print("2. 👨‍💻 Human reviewers provide feedback and scores")
    console.print("3. 📊 Feedback is weighted and integrated into training")
    console.print("4. 🔄 Model learns from human preferences")
    console.print("5. 📈 Continuous improvement through RLHF")
    
    # Show feedback statistics
    avg_feedback = sum(ex["feedback_score"] for ex in feedback_examples) / len(feedback_examples)
    console.print(f"\n[bold]Feedback Statistics:[/bold]")
    console.print(f"• Average Feedback Score: {avg_feedback:.3f}")
    console.print(f"• Feedback Weight in Training: {0.7:.1f}")
    console.print(f"• Code Quality Weight: {0.3:.1f}")


async def demo_quick_tuning():
    """Demo quick tuning API"""
    console.print("\n[bold blue]🚀 Quick Tuning API Demo[/bold blue]\n")
    
    console.print("[cyan]Demonstrating quick tuning for rapid prototyping...[/cyan]")
    
    # Create sample code pairs
    sample_pairs = [
        CodePair(
            input_code="def add(a, b): return a + b",
            output_code="def add(a: int, b: int) -> int:\n    \"\"\"Add two integers.\"\"\"\n    return a + b",
            task_type="refactor",
            quality_score=0.9
        ),
        CodePair(
            input_code="def divide(x, y): return x / y",
            output_code="def divide(x: float, y: float) -> float:\n    \"\"\"Safely divide two numbers.\"\"\"\n    if y == 0:\n        raise ValueError('Cannot divide by zero')\n    return x / y",
            task_type="debug",
            quality_score=0.95
        )
    ]
    
    console.print(f"[yellow]Sample training pairs: {len(sample_pairs)}[/yellow]")
    
    # Show what quick tuning would do
    console.print("\n[bold]Quick Tuning Process:[/bold]")
    console.print("1. 📝 Load sample code pairs")
    console.print("2. 🤖 Initialize lightweight model")
    console.print("3. ⚡ Run 1 epoch of training")
    console.print("4. 📊 Evaluate performance")
    console.print("5. 💾 Save tuned model")
    
    console.print("\n[green]✅ Quick tuning would complete in ~2-3 minutes[/green]")
    console.print("[yellow]💡 Use quick_tune_model() for rapid experimentation[/yellow]")


async def main():
    """Main demo function"""
    console.print(Panel(
        "[bold green]🎯 RLHF Tuner System Demo[/bold green]\n"
        "Demonstrating code mastery through reinforcement learning\n"
        "with LoRA fine-tuning and human feedback integration.",
        title="RLHF Tuner Demo",
        border_style="green"
    ))
    
    try:
        # Run all demo sections
        await demo_synthetic_data_generation()
        await demo_model_initialization()
        await demo_training_simulation()
        await demo_code_improvement_examples()
        await demo_evaluation_metrics()
        await demo_human_feedback_loop()
        await demo_quick_tuning()
        
        # Final summary
        console.print("\n" + "="*60)
        console.print("[bold green]🎯 RLHF Tuner Demo Summary[/bold green]")
        console.print("• Synthetic data generation: ✅")
        console.print("• LoRA model configuration: ✅")
        console.print("• Training process simulation: ✅")
        console.print("• Code improvement examples: ✅")
        console.print("• Evaluation metrics: ✅")
        console.print("• Human feedback integration: ✅")
        console.print("• Quick tuning API: ✅")
        
        console.print("\n[bold blue]🚀 RLHF system ready for code mastery training![/bold blue]")
        console.print("[yellow]💡 Run with actual models for real training experience[/yellow]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo failed: {e}[/red]")


if __name__ == "__main__":
    asyncio.run(main())