#!/usr/bin/env python3
"""
Benchmark Suites

Pre-defined benchmark task definitions and dataset management.
Provides standardized tasks for evaluating model performance.

Features:
- Pre-defined benchmark tasks (code gen, chat, reasoning)
- Custom benchmark definition
- Dataset management for benchmarks
- Scoring and ranking system
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .benchmark_engine import BenchmarkTask, TaskType


@dataclass
class BenchmarkDataset:
    """Dataset for benchmark evaluation"""
    name: str
    description: str
    task_type: TaskType
    samples: List[Dict[str, Any]]
    scoring_method: str = "accuracy"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkSuites:
    """
    Pre-defined benchmark suites and task definitions

    Suites:
    - Code Generation: Programming tasks
    - Chat: Conversational quality
    - Reasoning: Logical problem solving
    - Summarization: Text condensation
    - Translation: Language translation
    - Question Answering: Knowledge retrieval
    """

    def __init__(self):
        """Initialize benchmark suites"""
        self._suites = self._initialize_suites()
        self._datasets: Dict[str, BenchmarkDataset] = {}

    def _initialize_suites(self) -> Dict[str, List[BenchmarkTask]]:
        """Initialize pre-defined benchmark suites"""
        return {
            "code_generation": self._code_generation_suite(),
            "chat": self._chat_suite(),
            "reasoning": self._reasoning_suite(),
            "summarization": self._summarization_suite(),
            "translation": self._translation_suite(),
            "question_answering": self._qa_suite(),
        }

    def _code_generation_suite(self) -> List[BenchmarkTask]:
        """Code generation benchmark tasks"""
        return [
            BenchmarkTask(
                name="simple_function",
                task_type=TaskType.CODE_GENERATION,
                prompt="Write a Python function that takes two numbers and returns their sum",
                expected_output="def add(a, b):",
                expected_pattern=r"def\s+\w+\s*\(.*\):",
                max_tokens=256,
                weight=1.0,
            ),
            BenchmarkTask(
                name="list_comprehension",
                task_type=TaskType.CODE_GENERATION,
                prompt="Write a Python list comprehension that filters even numbers from a list",
                expected_output="[x for x in",
                expected_pattern=r"\[.*for.*in.*\]",
                max_tokens=256,
                weight=1.0,
            ),
            BenchmarkTask(
                name="class_definition",
                task_type=TaskType.CODE_GENERATION,
                prompt="Write a Python class named 'Person' with __init__ method taking name and age",
                expected_output="class Person:",
                expected_pattern=r"class\s+\w+:",
                max_tokens=512,
                weight=1.5,
            ),
            BenchmarkTask(
                name="error_handling",
                task_type=TaskType.CODE_GENERATION,
                prompt="Write a Python function with try-except block to handle division by zero",
                expected_output="try:",
                expected_pattern=r"try:.*except.*:",
                max_tokens=512,
                weight=1.5,
            ),
            BenchmarkTask(
                name="async_function",
                task_type=TaskType.CODE_GENERATION,
                prompt="Write an async Python function that fetches data from a URL using aiohttp",
                expected_output="async def",
                expected_pattern=r"async\s+def\s+\w+.*await",
                max_tokens=512,
                weight=2.0,
            ),
        ]

    def _chat_suite(self) -> List[BenchmarkTask]:
        """Chat/conversational benchmark tasks"""
        return [
            BenchmarkTask(
                name="greeting",
                task_type=TaskType.CHAT,
                prompt="Hello! How are you today?",
                expected_pattern=r"(hello|hi|hey|good)",
                max_tokens=128,
                weight=0.5,
            ),
            BenchmarkTask(
                name="follow_up",
                task_type=TaskType.CHAT,
                prompt="I'm feeling a bit down today. Any suggestions?",
                expected_pattern=r"(sorry|understand|feel|suggest|recommend)",
                max_tokens=256,
                weight=1.0,
            ),
            BenchmarkTask(
                name="creative_writing",
                task_type=TaskType.CHAT,
                prompt="Write a short haiku about programming",
                expected_pattern=r".*\n.*\n.*",
                max_tokens=128,
                weight=1.5,
            ),
            BenchmarkTask(
                name="role_play",
                task_type=TaskType.CHAT,
                prompt="Act as a helpful coding assistant. Explain what a decorator is in Python.",
                expected_pattern=r"(decorator|function|wrapper|@)",
                max_tokens=512,
                weight=2.0,
            ),
        ]

    def _reasoning_suite(self) -> List[BenchmarkTask]:
        """Logical reasoning benchmark tasks"""
        return [
            BenchmarkTask(
                name="math_word_problem",
                task_type=TaskType.REASONING,
                prompt="If John has 5 apples and gives 2 to Mary, then buys 3 more, how many does he have?",
                expected_output="6",
                expected_pattern=r"6",
                max_tokens=256,
                weight=1.5,
            ),
            BenchmarkTask(
                name="logical_deduction",
                task_type=TaskType.REASONING,
                prompt="All cats are mammals. Fluffy is a cat. Is Fluffy a mammal?",
                expected_output="yes",
                expected_pattern=r"(yes|Yes|YES)",
                max_tokens=256,
                weight=1.5,
            ),
            BenchmarkTask(
                name="pattern_recognition",
                task_type=TaskType.REASONING,
                prompt="What comes next: 2, 4, 8, 16, ?",
                expected_output="32",
                expected_pattern=r"32",
                max_tokens=128,
                weight=1.5,
            ),
            BenchmarkTask(
                name="constraint_satisfaction",
                task_type=TaskType.REASONING,
                prompt="Arrange A, B, C in order where A comes before B, and C comes after B.",
                expected_output="ABC",
                expected_pattern=r"ABC|A.*B.*C",
                max_tokens=256,
                weight=2.0,
            ),
        ]

    def _summarization_suite(self) -> List[BenchmarkTask]:
        """Text summarization benchmark tasks"""
        long_text = """
        Artificial intelligence (AI) is intelligence demonstrated by machines,
        in contrast to the natural intelligence displayed by humans and animals.
        Leading AI textbooks define the field as the study of "intelligent agents":
        any device that perceives its environment and takes actions that maximize
        its chance of successfully achieving its goals. Colloquially, the term
        "artificial intelligence" is often used to describe machines that mimic
        "cognitive" functions that humans associate with the human mind, such as
        "learning" and "problem solving".
        """

        return [
            BenchmarkTask(
                name="short_summary",
                task_type=TaskType.SUMMARIZATION,
                prompt=f"Summarize this in one sentence:\n{long_text}",
                expected_pattern=r"(AI|artificial intelligence|machines|intelligence)",
                max_tokens=128,
                weight=1.0,
            ),
            BenchmarkTask(
                name="key_points",
                task_type=TaskType.SUMMARIZATION,
                prompt=f"Extract 3 key points from:\n{long_text}",
                expected_pattern=r"\d+\.|•|-",
                max_tokens=256,
                weight=1.5,
            ),
        ]

    def _translation_suite(self) -> List[BenchmarkTask]:
        """Translation benchmark tasks"""
        return [
            BenchmarkTask(
                name="en_to_es",
                task_type=TaskType.TRANSLATION,
                prompt="Translate to Spanish: 'Hello, how are you?'",
                expected_pattern=r"(Hola|¿Cómo estás?)",
                max_tokens=128,
                weight=1.0,
            ),
            BenchmarkTask(
                name="en_to_fr",
                task_type=TaskType.TRANSLATION,
                prompt="Translate to French: 'Good morning, nice to meet you'",
                expected_pattern=r"(Bonjour|matin)",
                max_tokens=128,
                weight=1.0,
            ),
        ]

    def _qa_suite(self) -> List[BenchmarkTask]:
        """Question answering benchmark tasks"""
        return [
            BenchmarkTask(
                name="factual_qa",
                task_type=TaskType.QUESTION_ANSWERING,
                prompt="What is the capital of France?",
                expected_output="Paris",
                expected_pattern=r"Paris",
                max_tokens=128,
                weight=1.0,
            ),
            BenchmarkTask(
                name="technical_qa",
                task_type=TaskType.QUESTION_ANSWERING,
                prompt="What does HTTP stand for?",
                expected_output="HyperText Transfer Protocol",
                expected_pattern=r"(HyperText|HTTP|Hypertext).*(Transfer|Protocol)",
                max_tokens=128,
                weight=1.0,
            ),
            BenchmarkTask(
                name="code_qa",
                task_type=TaskType.QUESTION_ANSWERING,
                prompt="What is the time complexity of binary search?",
                expected_output="O(log n)",
                expected_pattern=r"O\(log.*n\)|logarithmic",
                max_tokens=128,
                weight=1.5,
            ),
        ]

    def get_suite(self, suite_name: str) -> Optional[List[BenchmarkTask]]:
        """
        Get a pre-defined benchmark suite

        Args:
            suite_name: Name of the suite

        Returns:
            List of BenchmarkTask or None
        """
        return self._suites.get(suite_name)

    def get_all_suites(self) -> Dict[str, List[BenchmarkTask]]:
        """Get all pre-defined suites"""
        return self._suites.copy()

    def get_task_types(self) -> List[str]:
        """Get all available task types"""
        return list(self._suites.keys())

    def get_task_by_type(self, task_type: str) -> List[BenchmarkTask]:
        """
        Get tasks for a specific type

        Args:
            task_type: Task type name

        Returns:
            List of BenchmarkTask
        """
        suite = self._suites.get(task_type, [])
        return suite

    def create_custom_task(
        self,
        name: str,
        task_type: TaskType,
        prompt: str,
        expected_output: Optional[str] = None,
        expected_pattern: Optional[str] = None,
        max_tokens: int = 512,
        timeout_seconds: int = 60,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkTask:
        """
        Create a custom benchmark task

        Args:
            name: Task name
            task_type: Type of task
            prompt: Input prompt
            expected_output: Expected output for validation
            expected_pattern: Regex pattern for validation
            max_tokens: Maximum output tokens
            timeout_seconds: Request timeout
            weight: Task weight for scoring
            metadata: Additional metadata

        Returns:
            BenchmarkTask instance
        """
        return BenchmarkTask(
            name=name,
            task_type=task_type,
            prompt=prompt,
            expected_output=expected_output,
            expected_pattern=expected_pattern,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            weight=weight,
            metadata=metadata or {},
        )

    def create_custom_suite(
        self,
        name: str,
        tasks: List[BenchmarkTask],
    ) -> None:
        """
        Create a custom benchmark suite

        Args:
            name: Suite name
            tasks: List of BenchmarkTask
        """
        self._suites[name] = tasks

    def load_dataset(self, dataset: BenchmarkDataset) -> None:
        """
        Load a benchmark dataset

        Args:
            dataset: BenchmarkDataset to load
        """
        self._datasets[dataset.name] = dataset

    def get_dataset(self, name: str) -> Optional[BenchmarkDataset]:
        """Get a loaded dataset"""
        return self._datasets.get(name)

    def get_all_datasets(self) -> Dict[str, BenchmarkDataset]:
        """Get all loaded datasets"""
        return self._datasets.copy()

    def calculate_suite_score(
        self,
        results: List[Dict[str, Any]],
        suite_name: str,
    ) -> Dict[str, Any]:
        """
        Calculate overall score for a benchmark suite

        Args:
            results: List of benchmark results
            suite_name: Name of the suite

        Returns:
            Score breakdown
        """
        suite = self._suites.get(suite_name, [])
        if not suite:
            return {"error": f"Unknown suite: {suite_name}"}

        # Create task weight map
        task_weights = {task.name: task.weight for task in suite}

        # Filter results for this suite
        suite_results = [r for r in results if r.get("task_name") in task_weights]

        if not suite_results:
            return {"error": "No results found for suite"}

        # Calculate weighted scores
        total_weight = 0
        weighted_accuracy = 0
        weighted_latency = 0

        for result in suite_results:
            task_name = result.get("task_name")
            weight = task_weights.get(task_name, 1.0)

            total_weight += weight
            weighted_accuracy += result.get("accuracy_score", 0) * weight
            weighted_latency += result.get("latency_ms", 0) * weight

        return {
            "suite_name": suite_name,
            "total_tasks": len(suite_results),
            "weighted_accuracy": weighted_accuracy / total_weight if total_weight > 0 else 0,
            "weighted_avg_latency_ms": weighted_latency / total_weight if total_weight > 0 else 0,
            "total_weight": total_weight,
        }


# Global instance
_suites_instance: Optional[BenchmarkSuites] = None


def get_benchmark_suites() -> BenchmarkSuites:
    """Get global benchmark suites instance"""
    global _suites_instance
    if _suites_instance is None:
        _suites_instance = BenchmarkSuites()
    return _suites_instance
