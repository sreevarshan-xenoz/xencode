#!/usr/bin/env python3
"""
RLHF Tuner System for Xencode Phase 6

Reinforcement Learning from Human Feedback (RLHF) tuning system for code mastery.
Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning of language models
on code-specific tasks with synthetic data generation and human feedback loops.
"""

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class RLHFConfig:
    """Configuration for RLHF tuning"""
    base_model: str = "microsoft/DialoGPT-small"  # Lightweight base model
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: float = 2e-4
    batch_size: int = 4
    max_epochs: int = 3
    max_length: int = 512
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 250
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    output_dir: str = ".xencode/rlhf_models"
    synthetic_data_size: int = 100
    human_feedback_weight: float = 0.7
    code_quality_weight: float = 0.3


@dataclass
class CodePair:
    """Code input-output pair for training"""
    input_code: str
    output_code: str
    task_type: str  # "refactor", "debug", "optimize", "explain"
    quality_score: float = 0.0
    human_feedback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    epoch: int
    loss: float
    perplexity: float
    learning_rate: float
    grad_norm: float
    step: int
    timestamp: datetime = field(default_factory=datetime.now)


class SyntheticDataGenerator:
    """Generates synthetic code pairs for training"""
    
    def __init__(self, base_model_name: str = "microsoft/DialoGPT-small"):
        self.base_model_name = base_model_name
        self.code_templates = self._load_code_templates()
        
    def _load_code_templates(self) -> Dict[str, List[str]]:
        """Load code templates for different tasks"""
        return {
            "refactor": [
                "def calculate_sum(numbers):\n    result = 0\n    for num in numbers:\n        result = result + num\n    return result",
                "class DataProcessor:\n    def __init__(self):\n        self.data = []\n    def add_item(self, item):\n        self.data.append(item)",
                "def find_max(arr):\n    max_val = arr[0]\n    for i in range(1, len(arr)):\n        if arr[i] > max_val:\n            max_val = arr[i]\n    return max_val"
            ],
            "debug": [
                "def divide_numbers(a, b):\n    return a / b  # Missing zero check",
                "def process_list(items):\n    for i in range(len(items) + 1):  # Off-by-one error\n        print(items[i])",
                "def get_user_age(user_dict):\n    return user_dict['age']  # Missing key check"
            ],
            "optimize": [
                "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "def search_list(items, target):\n    for i in range(len(items)):\n        if items[i] == target:\n            return i\n    return -1",
                "def sort_numbers(numbers):\n    for i in range(len(numbers)):\n        for j in range(len(numbers)-1):\n            if numbers[j] > numbers[j+1]:\n                numbers[j], numbers[j+1] = numbers[j+1], numbers[j]"
            ],
            "explain": [
                "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
                "class BinaryTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None",
                "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)"
            ]
        }
    
    async def generate_pairs(self, count: int = 100) -> List[CodePair]:
        """Generate synthetic code pairs"""
        pairs = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task("Generating synthetic code pairs...", total=count)
            
            for i in range(count):
                task_type = random.choice(list(self.code_templates.keys()))
                input_code = random.choice(self.code_templates[task_type])
                
                # Generate improved version based on task type
                output_code = await self._generate_improved_code(input_code, task_type)
                
                pair = CodePair(
                    input_code=input_code,
                    output_code=output_code,
                    task_type=task_type,
                    quality_score=random.uniform(0.7, 1.0),  # Synthetic quality score
                    metadata={"generated": True, "template_id": i}
                )
                
                pairs.append(pair)
                progress.update(task, advance=1)
        
        return pairs
    
    async def _generate_improved_code(self, input_code: str, task_type: str) -> str:
        """Generate improved version of code based on task type"""
        improvements = {
            "refactor": {
                "def calculate_sum(numbers):\n    result = 0\n    for num in numbers:\n        result = result + num\n    return result": 
                "def calculate_sum(numbers: List[float]) -> float:\n    \"\"\"Calculate sum of numbers efficiently.\"\"\"\n    return sum(numbers)",
                
                "class DataProcessor:\n    def __init__(self):\n        self.data = []\n    def add_item(self, item):\n        self.data.append(item)":
                "class DataProcessor:\n    \"\"\"Efficient data processor with type hints.\"\"\"\n    \n    def __init__(self) -> None:\n        self._data: List[Any] = []\n    \n    def add_item(self, item: Any) -> None:\n        \"\"\"Add item to processor.\"\"\"\n        self._data.append(item)",
                
                "def find_max(arr):\n    max_val = arr[0]\n    for i in range(1, len(arr)):\n        if arr[i] > max_val:\n            max_val = arr[i]\n    return max_val":
                "def find_max(arr: List[float]) -> float:\n    \"\"\"Find maximum value in array.\"\"\"\n    if not arr:\n        raise ValueError(\"Array cannot be empty\")\n    return max(arr)"
            },
            "debug": {
                "def divide_numbers(a, b):\n    return a / b  # Missing zero check":
                "def divide_numbers(a: float, b: float) -> float:\n    \"\"\"Safely divide two numbers.\"\"\"\n    if b == 0:\n        raise ValueError(\"Cannot divide by zero\")\n    return a / b",
                
                "def process_list(items):\n    for i in range(len(items) + 1):  # Off-by-one error\n        print(items[i])":
                "def process_list(items: List[Any]) -> None:\n    \"\"\"Process all items in list safely.\"\"\"\n    for item in items:\n        print(item)",
                
                "def get_user_age(user_dict):\n    return user_dict['age']  # Missing key check":
                "def get_user_age(user_dict: Dict[str, Any]) -> Optional[int]:\n    \"\"\"Safely get user age from dictionary.\"\"\"\n    return user_dict.get('age')"
            },
            "optimize": {
                "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)":
                "def fibonacci(n: int, memo: Dict[int, int] = None) -> int:\n    \"\"\"Optimized fibonacci with memoization.\"\"\"\n    if memo is None:\n        memo = {}\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)\n    return memo[n]",
                
                "def search_list(items, target):\n    for i in range(len(items)):\n        if items[i] == target:\n            return i\n    return -1":
                "def search_list(items: List[Any], target: Any) -> int:\n    \"\"\"Optimized search using built-in method.\"\"\"\n    try:\n        return items.index(target)\n    except ValueError:\n        return -1",
                
                "def sort_numbers(numbers):\n    for i in range(len(numbers)):\n        for j in range(len(numbers)-1):\n            if numbers[j] > numbers[j+1]:\n                numbers[j], numbers[j+1] = numbers[j+1], numbers[j]":
                "def sort_numbers(numbers: List[float]) -> List[float]:\n    \"\"\"Efficiently sort numbers.\"\"\"\n    return sorted(numbers)"
            },
            "explain": {
                # For explain tasks, add comprehensive comments
                "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)":
                "def quicksort(arr: List[int]) -> List[int]:\n    \"\"\"\n    Quicksort algorithm implementation.\n    \n    Time Complexity: O(n log n) average, O(n²) worst case\n    Space Complexity: O(log n) average\n    \n    Args:\n        arr: List of integers to sort\n    \n    Returns:\n        Sorted list of integers\n    \"\"\"\n    # Base case: arrays with 0 or 1 element are already sorted\n    if len(arr) <= 1:\n        return arr\n    \n    # Choose pivot element (middle element)\n    pivot = arr[len(arr) // 2]\n    \n    # Partition array into three parts\n    left = [x for x in arr if x < pivot]    # Elements less than pivot\n    middle = [x for x in arr if x == pivot]  # Elements equal to pivot\n    right = [x for x in arr if x > pivot]   # Elements greater than pivot\n    \n    # Recursively sort left and right partitions, combine results\n    return quicksort(left) + middle + quicksort(right)"
            }
        }
        
        # Return improved version if available, otherwise return input with basic improvements
        if input_code in improvements.get(task_type, {}):
            return improvements[task_type][input_code]
        
        # Fallback: add basic improvements
        return f"# Improved version of the code\n{input_code}"


class RLHFTuner:
    """Main RLHF tuning system"""
    
    def __init__(self, config: RLHFConfig = None):
        self.config = config or RLHFConfig()
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.training_history: List[TrainingMetrics] = []
        self.data_generator = SyntheticDataGenerator()
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
    
    async def initialize_model(self) -> bool:
        """Initialize base model and tokenizer"""
        try:
            console.print(f"[blue]🤖 Loading base model: {self.config.base_model}[/blue]")
            
            with console.status("[bold blue]Loading tokenizer and model..."):
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                # Setup LoRA configuration
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] if hasattr(self.model, 'q_proj') else ["c_attn"]
                )
                
                # Apply LoRA
                self.peft_model = get_peft_model(self.model, lora_config)
                self.peft_model.print_trainable_parameters()
            
            console.print("[green]✅ Model initialized successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to initialize model: {e}[/red]")
            logger.error(f"Model initialization failed: {e}")
            return False
    
    async def generate_training_data(self, size: int = None) -> List[CodePair]:
        """Generate synthetic training data"""
        size = size or self.config.synthetic_data_size
        
        console.print(f"[yellow]📊 Generating {size} synthetic code pairs...[/yellow]")
        
        pairs = await self.data_generator.generate_pairs(size)
        
        # Save generated data
        data_file = self.output_dir / "synthetic_data.json"
        with open(data_file, 'w') as f:
            json.dump([
                {
                    "input_code": pair.input_code,
                    "output_code": pair.output_code,
                    "task_type": pair.task_type,
                    "quality_score": pair.quality_score,
                    "metadata": pair.metadata
                }
                for pair in pairs
            ], f, indent=2)
        
        console.print(f"[green]✅ Generated {len(pairs)} code pairs[/green]")
        return pairs
    
    def _prepare_dataset(self, code_pairs: List[CodePair]) -> Dataset:
        """Prepare dataset for training"""
        def tokenize_function(examples):
            # Create input-output pairs for training
            inputs = []
            for i in range(len(examples['input_code'])):
                # Format as instruction-following task
                text = f"### Instruction: Improve this code\n### Input:\n{examples['input_code'][i]}\n### Output:\n{examples['output_code'][i]}"
                inputs.append(text)
            
            # Tokenize
            tokenized = self.tokenizer(
                inputs,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Convert to dataset format
        dataset_dict = {
            "input_code": [pair.input_code for pair in code_pairs],
            "output_code": [pair.output_code for pair in code_pairs],
            "task_type": [pair.task_type for pair in code_pairs],
            "quality_score": [pair.quality_score for pair in code_pairs]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    async def train(self, code_pairs: List[CodePair] = None) -> bool:
        """Train the model with RLHF"""
        if not self.peft_model:
            console.print("[red]❌ Model not initialized. Call initialize_model() first.[/red]")
            return False
        
        # Generate training data if not provided
        if not code_pairs:
            code_pairs = await self.generate_training_data()
        
        console.print(f"[blue]🎯 Starting RLHF training with {len(code_pairs)} pairs...[/blue]")
        
        try:
            # Prepare dataset
            train_dataset = self._prepare_dataset(code_pairs)
            
            # Split into train/eval
            train_size = int(0.8 * len(train_dataset))
            eval_size = len(train_dataset) - train_size
            train_dataset, eval_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, eval_size]
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(self.output_dir / "checkpoints"),
                num_train_epochs=self.config.max_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                warmup_steps=self.config.warmup_steps,
                learning_rate=self.config.learning_rate,
                fp16=self.config.fp16,
                logging_steps=50,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to=None,  # Disable wandb/tensorboard
                remove_unused_columns=False
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # Causal LM, not masked LM
            )
            
            # Custom trainer with metrics tracking
            class RLHFTrainer(Trainer):
                def __init__(self, rlhf_tuner, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.rlhf_tuner = rlhf_tuner
                
                def log(self, logs):
                    super().log(logs)
                    
                    # Track training metrics
                    if "train_loss" in logs:
                        metrics = TrainingMetrics(
                            epoch=int(logs.get("epoch", 0)),
                            loss=logs["train_loss"],
                            perplexity=np.exp(logs["train_loss"]),
                            learning_rate=logs.get("learning_rate", 0),
                            grad_norm=logs.get("grad_norm", 0),
                            step=logs.get("step", 0)
                        )
                        self.rlhf_tuner.training_history.append(metrics)
            
            # Initialize trainer
            trainer = RLHFTrainer(
                self,
                model=self.peft_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            
            # Start training
            console.print("[bold green]🚀 Starting training...[/bold green]")
            trainer.train()
            
            # Save final model
            final_model_path = self.output_dir / "final_model"
            trainer.save_model(str(final_model_path))
            
            console.print(f"[green]✅ Training completed! Model saved to {final_model_path}[/green]")
            
            # Display training summary
            self._display_training_summary()
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Training failed: {e}[/red]")
            logger.error(f"Training failed: {e}")
            return False
    
    def _display_training_summary(self):
        """Display training summary"""
        if not self.training_history:
            return
        
        console.print("\n[bold blue]📊 Training Summary[/bold blue]")
        
        # Create metrics table
        metrics_table = Table(title="Training Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Initial", style="yellow")
        metrics_table.add_column("Final", style="green")
        metrics_table.add_column("Best", style="blue")
        
        initial_metrics = self.training_history[0]
        final_metrics = self.training_history[-1]
        best_loss_metrics = min(self.training_history, key=lambda x: x.loss)
        
        metrics_table.add_row(
            "Loss",
            f"{initial_metrics.loss:.4f}",
            f"{final_metrics.loss:.4f}",
            f"{best_loss_metrics.loss:.4f}"
        )
        
        metrics_table.add_row(
            "Perplexity",
            f"{initial_metrics.perplexity:.2f}",
            f"{final_metrics.perplexity:.2f}",
            f"{best_loss_metrics.perplexity:.2f}"
        )
        
        metrics_table.add_row(
            "Learning Rate",
            f"{initial_metrics.learning_rate:.2e}",
            f"{final_metrics.learning_rate:.2e}",
            f"{best_loss_metrics.learning_rate:.2e}"
        )
        
        console.print(metrics_table)
        
        # Training progress
        console.print(f"\n[bold]Training Progress:[/bold]")
        console.print(f"• Total Steps: {final_metrics.step}")
        console.print(f"• Epochs Completed: {final_metrics.epoch}")
        console.print(f"• Loss Improvement: {((initial_metrics.loss - final_metrics.loss) / initial_metrics.loss * 100):.1f}%")
        console.print(f"• Best Perplexity: {best_loss_metrics.perplexity:.2f}")
    
    async def evaluate_model(self, test_pairs: List[CodePair] = None) -> Dict[str, float]:
        """Evaluate the trained model"""
        if not self.peft_model:
            console.print("[red]❌ No trained model available[/red]")
            return {}
        
        console.print("[blue]🔍 Evaluating model performance...[/blue]")
        
        # Generate test data if not provided
        if not test_pairs:
            test_pairs = await self.data_generator.generate_pairs(20)
        
        results = {
            "total_samples": len(test_pairs),
            "avg_quality_score": 0.0,
            "task_type_performance": {},
            "inference_time_ms": 0.0
        }
        
        total_quality = 0.0
        total_time = 0.0
        task_scores = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            eval_task = progress.add_task("Evaluating model...", total=len(test_pairs))
            
            for pair in test_pairs:
                start_time = time.perf_counter()
                
                # Generate response
                input_text = f"### Instruction: Improve this code\n### Input:\n{pair.input_code}\n### Output:\n"
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
                
                with torch.no_grad():
                    outputs = self.peft_model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                inference_time = (time.perf_counter() - start_time) * 1000
                total_time += inference_time
                
                # Calculate quality score (simplified)
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                quality_score = self._calculate_quality_score(pair.input_code, generated_text)
                
                total_quality += quality_score
                
                # Track by task type
                if pair.task_type not in task_scores:
                    task_scores[pair.task_type] = []
                task_scores[pair.task_type].append(quality_score)
                
                progress.update(eval_task, advance=1)
        
        # Calculate final metrics
        results["avg_quality_score"] = total_quality / len(test_pairs)
        results["inference_time_ms"] = total_time / len(test_pairs)
        results["task_type_performance"] = {
            task: sum(scores) / len(scores) 
            for task, scores in task_scores.items()
        }
        
        # Display results
        self._display_evaluation_results(results)
        
        return results
    
    def _calculate_quality_score(self, input_code: str, generated_text: str) -> float:
        """Calculate quality score for generated code (simplified)"""
        # Simple heuristics for code quality
        score = 0.5  # Base score
        
        # Check for improvements
        if "def " in generated_text and ":" in generated_text:
            score += 0.1  # Has function definition
        
        if '"""' in generated_text or "'''" in generated_text:
            score += 0.1  # Has docstring
        
        if "Type" in generated_text or "List" in generated_text:
            score += 0.1  # Has type hints
        
        if "try:" in generated_text or "except:" in generated_text:
            score += 0.1  # Has error handling
        
        if len(generated_text) > len(input_code):
            score += 0.1  # More comprehensive
        
        if "# " in generated_text:
            score += 0.1  # Has comments
        
        return min(1.0, score)
    
    def _display_evaluation_results(self, results: Dict[str, float]):
        """Display evaluation results"""
        console.print("\n[bold green]📈 Evaluation Results[/bold green]")
        
        eval_table = Table(title="Model Performance")
        eval_table.add_column("Metric", style="cyan")
        eval_table.add_column("Value", style="white")
        
        eval_table.add_row("Total Samples", str(results["total_samples"]))
        eval_table.add_row("Avg Quality Score", f"{results['avg_quality_score']:.3f}")
        eval_table.add_row("Avg Inference Time", f"{results['inference_time_ms']:.1f}ms")
        
        console.print(eval_table)
        
        # Task-specific performance
        if results["task_type_performance"]:
            console.print("\n[bold]Task-Specific Performance:[/bold]")
            for task, score in results["task_type_performance"].items():
                console.print(f"• {task.title()}: {score:.3f}")
    
    async def save_model(self, path: str = None) -> bool:
        """Save the trained model"""
        if not self.peft_model:
            console.print("[red]❌ No model to save[/red]")
            return False
        
        save_path = Path(path) if path else self.output_dir / "saved_model"
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save LoRA weights
            self.peft_model.save_pretrained(str(save_path))
            self.tokenizer.save_pretrained(str(save_path))
            
            # Save config
            config_path = save_path / "rlhf_config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    "base_model": self.config.base_model,
                    "lora_rank": self.config.lora_rank,
                    "lora_alpha": self.config.lora_alpha,
                    "lora_dropout": self.config.lora_dropout,
                    "max_length": self.config.max_length
                }, f, indent=2)
            
            console.print(f"[green]✅ Model saved to {save_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to save model: {e}[/red]")
            return False
    
    async def load_model(self, path: str) -> bool:
        """Load a trained model"""
        load_path = Path(path)
        
        if not load_path.exists():
            console.print(f"[red]❌ Model path does not exist: {path}[/red]")
            return False
        
        try:
            # Load config
            config_path = load_path / "rlhf_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    self.config.base_model = config_data.get("base_model", self.config.base_model)
            
            # Initialize base model
            await self.initialize_model()
            
            # Load LoRA weights
            self.peft_model = PeftModel.from_pretrained(self.model, str(load_path))
            
            console.print(f"[green]✅ Model loaded from {path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {e}[/red]")
            return False


# Convenience functions
async def create_rlhf_tuner(config: RLHFConfig = None) -> RLHFTuner:
    """Create and initialize RLHF tuner"""
    tuner = RLHFTuner(config)
    await tuner.initialize_model()
    return tuner


async def quick_tune_model(code_pairs: List[CodePair] = None, 
                          epochs: int = 1) -> RLHFTuner:
    """Quick model tuning for testing"""
    config = RLHFConfig(max_epochs=epochs, synthetic_data_size=50)
    tuner = await create_rlhf_tuner(config)
    await tuner.train(code_pairs)
    return tuner


if __name__ == "__main__":
    async def main():
        """Demo RLHF tuning"""
        console.print(Panel(
            "[bold green]🎯 RLHF Tuner Demo[/bold green]\n"
            "Demonstrating code mastery through reinforcement learning",
            title="RLHF Demo",
            border_style="green"
        ))
        
        # Create tuner with lightweight config
        config = RLHFConfig(
            max_epochs=1,
            synthetic_data_size=20,
            batch_size=2
        )
        
        tuner = RLHFTuner(config)
        
        # Initialize model
        if await tuner.initialize_model():
            # Generate training data
            code_pairs = await tuner.generate_training_data(20)
            
            # Train model
            if await tuner.train(code_pairs):
                # Evaluate model
                await tuner.evaluate_model()
                
                # Save model
                await tuner.save_model()
                
                console.print("\n[bold blue]🚀 RLHF tuning completed successfully![/bold blue]")
    
    asyncio.run(main())