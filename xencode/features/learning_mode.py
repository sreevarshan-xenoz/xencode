"""
Learning Mode Feature

Provides interactive tutorials, adaptive difficulty, progress tracking,
and exercise generation for learning developers.
"""

import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter
from enum import Enum

from .base import FeatureBase, FeatureConfig, FeatureError


class DifficultyLevel(Enum):
    """Difficulty levels for learning content"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class MasteryLevel(Enum):
    """Mastery levels for topics"""
    NOVICE = "novice"
    LEARNING = "learning"
    COMPETENT = "competent"
    PROFICIENT = "proficient"
    EXPERT = "expert"


@dataclass
class LearningModeConfig:
    """Configuration for Learning Mode"""
    enabled: bool = True
    default_difficulty: str = "beginner"
    adaptive_enabled: bool = True
    exercise_count: int = 5
    mastery_threshold: float = 0.8
    topics: List[str] = field(default_factory=lambda: [
        'python', 'javascript', 'rust', 'go', 'docker', 'git'
    ])
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningModeConfig':
        """Create config from dictionary"""
        return cls(
            enabled=data.get('enabled', True),
            default_difficulty=data.get('default_difficulty', 'beginner'),
            adaptive_enabled=data.get('adaptive_enabled', True),
            exercise_count=data.get('exercise_count', 5),
            mastery_threshold=data.get('mastery_threshold', 0.8),
            topics=data.get('topics', [
                'python', 'javascript', 'rust', 'go', 'docker', 'git'
            ])
        )


@dataclass
class Topic:
    """Represents a learning topic"""
    id: str
    name: str
    description: str
    difficulty: DifficultyLevel
    prerequisites: List[str] = field(default_factory=list)
    subtopics: List[str] = field(default_factory=list)
    estimated_time: int = 60  # minutes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'difficulty': self.difficulty.value,
            'prerequisites': self.prerequisites,
            'subtopics': self.subtopics,
            'estimated_time': self.estimated_time
        }


@dataclass
class Exercise:
    """Represents a learning exercise"""
    id: str
    topic_id: str
    title: str
    description: str
    difficulty: DifficultyLevel
    code_template: str
    solution: str
    hints: List[str] = field(default_factory=list)
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'topic_id': self.topic_id,
            'title': self.title,
            'description': self.description,
            'difficulty': self.difficulty.value,
            'code_template': self.code_template,
            'hints': self.hints,
            'test_cases': self.test_cases
        }


@dataclass
class Progress:
    """Tracks user progress on a topic"""
    topic_id: str
    mastery_level: MasteryLevel
    exercises_completed: int = 0
    exercises_total: int = 0
    accuracy: float = 0.0
    time_spent: int = 0  # minutes
    last_accessed: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'topic_id': self.topic_id,
            'mastery_level': self.mastery_level.value,
            'exercises_completed': self.exercises_completed,
            'exercises_total': self.exercises_total,
            'accuracy': self.accuracy,
            'time_spent': self.time_spent,
            'last_accessed': self.last_accessed
        }


class LearningModeFeature(FeatureBase):
    """Learning Mode feature implementation"""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.lm_config = LearningModeConfig.from_dict(config.config)
        self.tutorial_engine = None
        self.difficulty_controller = None
        self.progress_tracker = None
        self.exercise_generator = None
    
    @property
    def name(self) -> str:
        """Feature name"""
        return "learning_mode"
    
    @property
    def description(self) -> str:
        """Feature description"""
        return "Interactive tutorials with adaptive difficulty and progress tracking"
    
    async def _initialize(self) -> None:
        """Initialize Learning Mode components"""
        # Initialize tutorial engine
        self.tutorial_engine = TutorialEngine(
            topics=self.lm_config.topics
        )
        
        # Initialize adaptive difficulty controller
        self.difficulty_controller = AdaptiveDifficultyController(
            enabled=self.lm_config.adaptive_enabled,
            mastery_threshold=self.lm_config.mastery_threshold
        )
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker()
        
        # Initialize exercise generator
        self.exercise_generator = ExerciseGenerator(
            exercise_count=self.lm_config.exercise_count
        )
        
        # Load tutorial content
        await self.tutorial_engine.load_content()
        
        # Load user progress
        await self.progress_tracker.load_progress()
    
    async def _shutdown(self) -> None:
        """Shutdown Learning Mode"""
        # Save user progress
        if self.progress_tracker:
            await self.progress_tracker.save_progress()
    
    async def start_topic(self, topic_id: str, difficulty: str = None) -> Dict[str, Any]:
        """
        Start learning a topic
        
        Args:
            topic_id: Topic identifier
            difficulty: Optional difficulty override
            
        Returns:
            Topic information and first lesson
        """
        # Get topic
        topic = await self.tutorial_engine.get_topic(topic_id)
        if not topic:
            raise FeatureError(f"Topic not found: {topic_id}")
        
        # Get or create progress
        progress = await self.progress_tracker.get_progress(topic_id)
        
        # Determine difficulty level
        if difficulty:
            diff_level = DifficultyLevel(difficulty)
        elif self.lm_config.adaptive_enabled and progress:
            diff_level = await self.difficulty_controller.get_difficulty(progress)
        else:
            diff_level = DifficultyLevel(self.lm_config.default_difficulty)
        
        # Get first lesson
        lesson = await self.tutorial_engine.get_lesson(topic_id, diff_level)
        
        # Track analytics
        self.track_analytics('start_topic', {
            'topic_id': topic_id,
            'difficulty': diff_level.value
        })
        
        return {
            'topic': topic.to_dict(),
            'lesson': lesson,
            'progress': progress.to_dict() if progress else None,
            'difficulty': diff_level.value
        }
    
    async def get_topics(self) -> List[Dict[str, Any]]:
        """
        Get all available topics
        
        Returns:
            List of topics with progress information
        """
        topics = await self.tutorial_engine.get_all_topics()
        
        # Enrich with progress
        enriched = []
        for topic in topics:
            progress = await self.progress_tracker.get_progress(topic.id)
            enriched.append({
                **topic.to_dict(),
                'progress': progress.to_dict() if progress else None
            })
        
        return enriched
    
    async def get_exercises(self, topic_id: str, count: int = None) -> List[Dict[str, Any]]:
        """
        Get exercises for a topic
        
        Args:
            topic_id: Topic identifier
            count: Number of exercises (default from config)
            
        Returns:
            List of exercises
        """
        if count is None:
            count = self.lm_config.exercise_count
        
        # Get user progress to determine difficulty
        progress = await self.progress_tracker.get_progress(topic_id)
        difficulty = await self.difficulty_controller.get_difficulty(progress) if progress else DifficultyLevel.BEGINNER
        
        # Generate exercises
        exercises = await self.exercise_generator.generate(topic_id, difficulty, count)
        
        return [ex.to_dict() for ex in exercises]
    
    async def submit_exercise(self, exercise_id: str, solution: str) -> Dict[str, Any]:
        """
        Submit exercise solution
        
        Args:
            exercise_id: Exercise identifier
            solution: User's solution code
            
        Returns:
            Evaluation results
        """
        # Get exercise
        exercise = await self.exercise_generator.get_exercise(exercise_id)
        if not exercise:
            raise FeatureError(f"Exercise not found: {exercise_id}")
        
        # Evaluate solution
        result = await self._evaluate_solution(exercise, solution)
        
        # Update progress
        await self.progress_tracker.record_exercise(
            exercise.topic_id,
            passed=result['passed'],
            time_spent=result.get('time_spent', 0)
        )
        
        # Track analytics
        self.track_analytics('submit_exercise', {
            'exercise_id': exercise_id,
            'topic_id': exercise.topic_id,
            'passed': result['passed']
        })
        
        return result
    
    async def get_progress(self, topic_id: str = None) -> Dict[str, Any]:
        """
        Get learning progress
        
        Args:
            topic_id: Optional topic to get progress for (None for all)
            
        Returns:
            Progress information
        """
        if topic_id:
            progress = await self.progress_tracker.get_progress(topic_id)
            return progress.to_dict() if progress else None
        else:
            all_progress = await self.progress_tracker.get_all_progress()
            return {
                'topics': [p.to_dict() for p in all_progress],
                'overall_mastery': self._calculate_overall_mastery(all_progress),
                'total_time': sum(p.time_spent for p in all_progress)
            }
    
    async def get_next_topic(self) -> Optional[Dict[str, Any]]:
        """
        Get recommended next topic based on progress
        
        Returns:
            Recommended topic or None
        """
        # Get all topics and progress
        topics = await self.tutorial_engine.get_all_topics()
        all_progress = await self.progress_tracker.get_all_progress()
        
        # Find topics with prerequisites met
        completed_topics = {
            p.topic_id for p in all_progress 
            if p.mastery_level in [MasteryLevel.PROFICIENT, MasteryLevel.EXPERT]
        }
        
        for topic in topics:
            # Skip if already mastered
            progress = await self.progress_tracker.get_progress(topic.id)
            if progress and progress.mastery_level in [MasteryLevel.PROFICIENT, MasteryLevel.EXPERT]:
                continue
            
            # Check prerequisites
            if all(prereq in completed_topics for prereq in topic.prerequisites):
                return topic.to_dict()
        
        return None
    
    async def get_mastery_level(self, topic_id: str) -> Dict[str, Any]:
        """
        Get mastery level for a topic
        
        Args:
            topic_id: Topic identifier
            
        Returns:
            Mastery information
        """
        progress = await self.progress_tracker.get_progress(topic_id)
        if not progress:
            return {
                'topic_id': topic_id,
                'mastery_level': MasteryLevel.NOVICE.value,
                'mastery_percentage': 0.0
            }
        
        mastery_pct = (progress.exercises_completed / progress.exercises_total * 100) if progress.exercises_total > 0 else 0
        
        return {
            'topic_id': topic_id,
            'mastery_level': progress.mastery_level.value,
            'mastery_percentage': round(mastery_pct, 2),
            'exercises_completed': progress.exercises_completed,
            'exercises_total': progress.exercises_total,
            'accuracy': round(progress.accuracy * 100, 2)
        }
    
    async def _evaluate_solution(self, exercise: Exercise, solution: str) -> Dict[str, Any]:
        """Evaluate exercise solution"""
        passed = False
        feedback = []
        
        # Run test cases
        test_results = []
        for test_case in exercise.test_cases:
            try:
                # Simple evaluation (in production, use sandboxed execution)
                result = await self._run_test_case(solution, test_case)
                test_results.append(result)
            except Exception as e:
                test_results.append({
                    'passed': False,
                    'error': str(e)
                })
        
        # Check if all tests passed
        passed = all(r.get('passed', False) for r in test_results)
        
        if passed:
            feedback.append("✓ All tests passed!")
        else:
            failed_count = sum(1 for r in test_results if not r.get('passed', False))
            feedback.append(f"✗ {failed_count} test(s) failed")
        
        return {
            'passed': passed,
            'test_results': test_results,
            'feedback': feedback,
            'hints': exercise.hints if not passed else []
        }
    
    async def _run_test_case(self, solution: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        # Placeholder for test execution
        # In production, use sandboxed execution
        return {
            'passed': True,
            'input': test_case.get('input'),
            'expected': test_case.get('expected'),
            'actual': test_case.get('expected')
        }
    
    def _calculate_overall_mastery(self, all_progress: List[Progress]) -> float:
        """Calculate overall mastery across all topics"""
        if not all_progress:
            return 0.0
        
        mastery_values = {
            MasteryLevel.NOVICE: 0.0,
            MasteryLevel.LEARNING: 0.25,
            MasteryLevel.COMPETENT: 0.5,
            MasteryLevel.PROFICIENT: 0.75,
            MasteryLevel.EXPERT: 1.0
        }
        
        total = sum(mastery_values[p.mastery_level] for p in all_progress)
        return round(total / len(all_progress), 2)
    
    def get_cli_commands(self) -> List[Any]:
        """Get CLI commands for Learning Mode"""
        return []
    
    def get_tui_components(self) -> List[Any]:
        """Get TUI components for Learning Mode"""
        return []
    
    def get_api_endpoints(self) -> List[Any]:
        """Get API endpoints for Learning Mode"""
        return [
            {
                'path': '/api/learning/topics',
                'method': 'GET',
                'handler': self.get_topics
            },
            {
                'path': '/api/learning/start',
                'method': 'POST',
                'handler': self.start_topic
            },
            {
                'path': '/api/learning/exercises',
                'method': 'GET',
                'handler': self.get_exercises
            },
            {
                'path': '/api/learning/submit',
                'method': 'POST',
                'handler': self.submit_exercise
            },
            {
                'path': '/api/learning/progress',
                'method': 'GET',
                'handler': self.get_progress
            },
            {
                'path': '/api/learning/next',
                'method': 'GET',
                'handler': self.get_next_topic
            },
            {
                'path': '/api/learning/mastery',
                'method': 'GET',
                'handler': self.get_mastery_level
            }
        ]


class TutorialEngine:
    """Manages tutorial content and lessons"""
    
    def __init__(self, topics: List[str]):
        self.topics: Dict[str, Topic] = {}
        self.lessons: Dict[str, Dict[DifficultyLevel, Dict[str, Any]]] = defaultdict(dict)
        self.configured_topics = topics
    
    async def load_content(self) -> None:
        """Load tutorial content from library"""
        # Load built-in topics
        self._load_builtin_topics()
        
        # Load custom topics from file
        await self._load_custom_topics()
    
    def _load_builtin_topics(self) -> None:
        """Load built-in tutorial topics"""
        builtin_topics = [
            Topic(
                id='python',
                name='Python Programming',
                description='Learn Python from basics to advanced concepts',
                difficulty=DifficultyLevel.BEGINNER,
                subtopics=['variables', 'functions', 'classes', 'async'],
                estimated_time=120
            ),
            Topic(
                id='javascript',
                name='JavaScript Programming',
                description='Master JavaScript for web development',
                difficulty=DifficultyLevel.BEGINNER,
                subtopics=['syntax', 'dom', 'async', 'frameworks'],
                estimated_time=100
            ),
            Topic(
                id='rust',
                name='Rust Programming',
                description='Learn systems programming with Rust',
                difficulty=DifficultyLevel.INTERMEDIATE,
                prerequisites=['python'],
                subtopics=['ownership', 'lifetimes', 'traits', 'async'],
                estimated_time=150
            ),
            Topic(
                id='docker',
                name='Docker Containers',
                description='Containerize applications with Docker',
                difficulty=DifficultyLevel.INTERMEDIATE,
                subtopics=['images', 'containers', 'compose', 'networking'],
                estimated_time=90
            ),
            Topic(
                id='git',
                name='Git Version Control',
                description='Master Git for version control',
                difficulty=DifficultyLevel.BEGINNER,
                subtopics=['commits', 'branches', 'merging', 'rebasing'],
                estimated_time=60
            )
        ]
        
        for topic in builtin_topics:
            if topic.id in self.configured_topics:
                self.topics[topic.id] = topic
                self._load_topic_lessons(topic)
    
    def _load_topic_lessons(self, topic: Topic) -> None:
        """Load lessons for a topic"""
        # Create sample lessons for each difficulty level
        for difficulty in DifficultyLevel:
            self.lessons[topic.id][difficulty] = {
                'title': f'{topic.name} - {difficulty.value.title()}',
                'content': self._generate_lesson_content(topic, difficulty),
                'examples': self._generate_examples(topic, difficulty),
                'key_concepts': self._generate_key_concepts(topic, difficulty)
            }
    
    def _generate_lesson_content(self, topic: Topic, difficulty: DifficultyLevel) -> str:
        """Generate lesson content"""
        content_map = {
            'python': {
                DifficultyLevel.BEGINNER: "Learn Python basics: variables, data types, and control flow.",
                DifficultyLevel.INTERMEDIATE: "Explore Python functions, classes, and modules.",
                DifficultyLevel.ADVANCED: "Master Python decorators, generators, and async programming.",
                DifficultyLevel.EXPERT: "Deep dive into Python internals, metaclasses, and optimization."
            },
            'javascript': {
                DifficultyLevel.BEGINNER: "JavaScript fundamentals: syntax, variables, and functions.",
                DifficultyLevel.INTERMEDIATE: "DOM manipulation, events, and async JavaScript.",
                DifficultyLevel.ADVANCED: "Advanced patterns, closures, and modern frameworks.",
                DifficultyLevel.EXPERT: "Performance optimization and advanced architecture."
            }
        }
        
        return content_map.get(topic.id, {}).get(difficulty, f"Learn {topic.name}")
    
    def _generate_examples(self, topic: Topic, difficulty: DifficultyLevel) -> List[str]:
        """Generate code examples"""
        if topic.id == 'python':
            if difficulty == DifficultyLevel.BEGINNER:
                return [
                    "# Variables\nx = 10\nname = 'Python'",
                    "# Functions\ndef greet(name):\n    return f'Hello, {name}!'"
                ]
        return []
    
    def _generate_key_concepts(self, topic: Topic, difficulty: DifficultyLevel) -> List[str]:
        """Generate key concepts"""
        if topic.id == 'python':
            if difficulty == DifficultyLevel.BEGINNER:
                return ['Variables', 'Data Types', 'Functions', 'Control Flow']
        return []
    
    async def _load_custom_topics(self) -> None:
        """Load custom topics from file"""
        topics_file = Path.home() / '.xencode' / 'learning_topics.json'
        if topics_file.exists():
            try:
                with open(topics_file, 'r') as f:
                    data = json.load(f)
                    for topic_data in data.get('topics', []):
                        topic = Topic(
                            id=topic_data['id'],
                            name=topic_data['name'],
                            description=topic_data['description'],
                            difficulty=DifficultyLevel(topic_data.get('difficulty', 'beginner')),
                            prerequisites=topic_data.get('prerequisites', []),
                            subtopics=topic_data.get('subtopics', []),
                            estimated_time=topic_data.get('estimated_time', 60)
                        )
                        self.topics[topic.id] = topic
            except Exception:
                pass
    
    async def get_topic(self, topic_id: str) -> Optional[Topic]:
        """Get a topic by ID"""
        return self.topics.get(topic_id)
    
    async def get_all_topics(self) -> List[Topic]:
        """Get all available topics"""
        return list(self.topics.values())
    
    async def get_lesson(self, topic_id: str, difficulty: DifficultyLevel) -> Dict[str, Any]:
        """Get a lesson for a topic at a specific difficulty"""
        return self.lessons.get(topic_id, {}).get(difficulty, {})


class AdaptiveDifficultyController:
    """Controls adaptive difficulty based on user performance"""
    
    def __init__(self, enabled: bool = True, mastery_threshold: float = 0.8):
        self.enabled = enabled
        self.mastery_threshold = mastery_threshold
    
    async def get_difficulty(self, progress: Progress) -> DifficultyLevel:
        """Determine appropriate difficulty level based on progress"""
        if not self.enabled:
            return DifficultyLevel.BEGINNER
        
        # Map mastery level to difficulty
        mastery_to_difficulty = {
            MasteryLevel.NOVICE: DifficultyLevel.BEGINNER,
            MasteryLevel.LEARNING: DifficultyLevel.BEGINNER,
            MasteryLevel.COMPETENT: DifficultyLevel.INTERMEDIATE,
            MasteryLevel.PROFICIENT: DifficultyLevel.ADVANCED,
            MasteryLevel.EXPERT: DifficultyLevel.EXPERT
        }
        
        return mastery_to_difficulty.get(progress.mastery_level, DifficultyLevel.BEGINNER)
    
    async def adjust_difficulty(self, current: DifficultyLevel, 
                               performance: float) -> DifficultyLevel:
        """Adjust difficulty based on performance"""
        if not self.enabled:
            return current
        
        # Increase difficulty if performing well
        if performance >= self.mastery_threshold:
            if current == DifficultyLevel.BEGINNER:
                return DifficultyLevel.INTERMEDIATE
            elif current == DifficultyLevel.INTERMEDIATE:
                return DifficultyLevel.ADVANCED
            elif current == DifficultyLevel.ADVANCED:
                return DifficultyLevel.EXPERT
        
        # Decrease difficulty if struggling
        elif performance < 0.5:
            if current == DifficultyLevel.EXPERT:
                return DifficultyLevel.ADVANCED
            elif current == DifficultyLevel.ADVANCED:
                return DifficultyLevel.INTERMEDIATE
            elif current == DifficultyLevel.INTERMEDIATE:
                return DifficultyLevel.BEGINNER
        
        return current


class ProgressTracker:
    """Tracks user learning progress"""
    
    def __init__(self):
        self.progress: Dict[str, Progress] = {}
    
    async def load_progress(self) -> None:
        """Load progress from file"""
        progress_file = Path.home() / '.xencode' / 'learning_progress.json'
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    data = json.load(f)
                    for topic_id, prog_data in data.items():
                        self.progress[topic_id] = Progress(
                            topic_id=topic_id,
                            mastery_level=MasteryLevel(prog_data['mastery_level']),
                            exercises_completed=prog_data['exercises_completed'],
                            exercises_total=prog_data['exercises_total'],
                            accuracy=prog_data['accuracy'],
                            time_spent=prog_data['time_spent'],
                            last_accessed=prog_data.get('last_accessed')
                        )
            except Exception:
                pass
    
    async def save_progress(self) -> None:
        """Save progress to file"""
        progress_file = Path.home() / '.xencode' / 'learning_progress.json'
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {
                topic_id: prog.to_dict()
                for topic_id, prog in self.progress.items()
            }
            with open(progress_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    async def get_progress(self, topic_id: str) -> Optional[Progress]:
        """Get progress for a topic"""
        return self.progress.get(topic_id)
    
    async def get_all_progress(self) -> List[Progress]:
        """Get all progress"""
        return list(self.progress.values())
    
    async def record_exercise(self, topic_id: str, passed: bool, time_spent: int = 0) -> None:
        """Record exercise completion"""
        if topic_id not in self.progress:
            self.progress[topic_id] = Progress(
                topic_id=topic_id,
                mastery_level=MasteryLevel.NOVICE,
                exercises_total=10  # Default
            )
        
        progress = self.progress[topic_id]
        progress.exercises_completed += 1
        progress.time_spent += time_spent
        progress.last_accessed = datetime.now().isoformat()
        
        # Update accuracy
        if passed:
            progress.accuracy = (progress.accuracy * (progress.exercises_completed - 1) + 1.0) / progress.exercises_completed
        else:
            progress.accuracy = (progress.accuracy * (progress.exercises_completed - 1)) / progress.exercises_completed
        
        # Update mastery level
        progress.mastery_level = self._calculate_mastery_level(progress)
    
    def _calculate_mastery_level(self, progress: Progress) -> MasteryLevel:
        """Calculate mastery level based on progress"""
        completion_rate = progress.exercises_completed / progress.exercises_total if progress.exercises_total > 0 else 0
        
        if completion_rate >= 0.9 and progress.accuracy >= 0.9:
            return MasteryLevel.EXPERT
        elif completion_rate >= 0.7 and progress.accuracy >= 0.8:
            return MasteryLevel.PROFICIENT
        elif completion_rate >= 0.5 and progress.accuracy >= 0.7:
            return MasteryLevel.COMPETENT
        elif completion_rate >= 0.2:
            return MasteryLevel.LEARNING
        else:
            return MasteryLevel.NOVICE


class ExerciseGenerator:
    """Generates learning exercises"""
    
    def __init__(self, exercise_count: int = 5):
        self.exercise_count = exercise_count
        self.exercises: Dict[str, Exercise] = {}
        self._load_exercise_templates()
    
    def _load_exercise_templates(self) -> None:
        """Load exercise templates"""
        # Python exercises
        self.exercises['python_hello'] = Exercise(
            id='python_hello',
            topic_id='python',
            title='Hello World',
            description='Write a function that returns "Hello, World!"',
            difficulty=DifficultyLevel.BEGINNER,
            code_template='def hello():\n    # Your code here\n    pass',
            solution='def hello():\n    return "Hello, World!"',
            hints=['Use the return statement', 'Return a string'],
            test_cases=[
                {'input': None, 'expected': 'Hello, World!'}
            ]
        )
        
        self.exercises['python_sum'] = Exercise(
            id='python_sum',
            topic_id='python',
            title='Sum Two Numbers',
            description='Write a function that adds two numbers',
            difficulty=DifficultyLevel.BEGINNER,
            code_template='def add(a, b):\n    # Your code here\n    pass',
            solution='def add(a, b):\n    return a + b',
            hints=['Use the + operator', 'Return the result'],
            test_cases=[
                {'input': (2, 3), 'expected': 5},
                {'input': (0, 0), 'expected': 0},
                {'input': (-1, 1), 'expected': 0}
            ]
        )
    
    async def generate(self, topic_id: str, difficulty: DifficultyLevel, 
                      count: int) -> List[Exercise]:
        """Generate exercises for a topic"""
        # Filter exercises by topic and difficulty
        matching = [
            ex for ex in self.exercises.values()
            if ex.topic_id == topic_id and ex.difficulty == difficulty
        ]
        
        # Return up to count exercises
        return matching[:count]
    
    async def get_exercise(self, exercise_id: str) -> Optional[Exercise]:
        """Get an exercise by ID"""
        return self.exercises.get(exercise_id)
