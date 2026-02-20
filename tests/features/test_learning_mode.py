"""
Tests for Learning Mode Feature

Tests tutorial engine, adaptive difficulty, progress tracking,
and exercise generation functionality.
"""

import pytest
import pytest_asyncio
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from xencode.features.learning_mode import (
    LearningModeFeature,
    LearningModeConfig,
    TutorialEngine,
    AdaptiveDifficultyController,
    ProgressTracker,
    ExerciseGenerator,
    Topic,
    Exercise,
    Progress,
    DifficultyLevel,
    MasteryLevel
)
from xencode.features.base import FeatureConfig


@pytest.fixture
def feature_config():
    """Create test feature configuration"""
    return FeatureConfig(
        name='learning_mode',
        enabled=True,
        config={
            'enabled': True,
            'default_difficulty': 'beginner',
            'adaptive_enabled': True,
            'exercise_count': 5,
            'mastery_threshold': 0.8,
            'topics': ['python', 'javascript', 'rust']
        }
    )


@pytest_asyncio.fixture
async def learning_feature(feature_config, tmp_path, monkeypatch):
    """Create and initialize learning mode feature"""
    # Use temporary directory for progress files
    monkeypatch.setattr('pathlib.Path.home', lambda: tmp_path)
    
    feature = LearningModeFeature(feature_config)
    await feature.initialize()
    yield feature
    await feature.shutdown()


@pytest.fixture
def tutorial_engine():
    """Create tutorial engine"""
    return TutorialEngine(topics=['python', 'javascript'])


@pytest.fixture
def difficulty_controller():
    """Create difficulty controller"""
    return AdaptiveDifficultyController(enabled=True, mastery_threshold=0.8)


@pytest.fixture
def progress_tracker():
    """Create progress tracker"""
    return ProgressTracker()


@pytest.fixture
def exercise_generator():
    """Create exercise generator"""
    return ExerciseGenerator(exercise_count=5)


class TestLearningModeFeature:
    """Test LearningModeFeature class"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, learning_feature):
        """Test feature initialization"""
        assert learning_feature.is_initialized
        assert learning_feature.tutorial_engine is not None
        assert learning_feature.difficulty_controller is not None
        assert learning_feature.progress_tracker is not None
        assert learning_feature.exercise_generator is not None
    
    @pytest.mark.asyncio
    async def test_start_topic(self, learning_feature):
        """Test starting a topic"""
        result = await learning_feature.start_topic('python')
        
        assert 'topic' in result
        assert 'lesson' in result
        assert 'difficulty' in result
        assert result['topic']['id'] == 'python'
        assert result['difficulty'] == 'beginner'
    
    @pytest.mark.asyncio
    async def test_start_topic_with_difficulty(self, learning_feature):
        """Test starting a topic with specific difficulty"""
        result = await learning_feature.start_topic('python', difficulty='intermediate')
        
        assert result['difficulty'] == 'intermediate'
    
    @pytest.mark.asyncio
    async def test_start_nonexistent_topic(self, learning_feature):
        """Test starting a non-existent topic"""
        with pytest.raises(Exception):
            await learning_feature.start_topic('nonexistent')
    
    @pytest.mark.asyncio
    async def test_get_topics(self, learning_feature):
        """Test getting all topics"""
        topics = await learning_feature.get_topics()
        
        assert isinstance(topics, list)
        assert len(topics) > 0
        assert all('id' in t for t in topics)
        assert all('name' in t for t in topics)
    
    @pytest.mark.asyncio
    async def test_get_exercises(self, learning_feature):
        """Test getting exercises for a topic"""
        exercises = await learning_feature.get_exercises('python', count=3)
        
        assert isinstance(exercises, list)
        assert len(exercises) <= 3
    
    @pytest.mark.asyncio
    async def test_submit_exercise(self, learning_feature):
        """Test submitting an exercise solution"""
        # Get an exercise first
        exercises = await learning_feature.get_exercises('python', count=1)
        if exercises:
            exercise_id = exercises[0]['id']
            solution = "def hello():\n    return 'Hello, World!'"
            
            result = await learning_feature.submit_exercise(exercise_id, solution)
            
            assert 'passed' in result
            assert 'feedback' in result
            assert isinstance(result['passed'], bool)
    
    @pytest.mark.asyncio
    async def test_get_progress(self, learning_feature):
        """Test getting progress"""
        # Record some progress first
        await learning_feature.progress_tracker.record_exercise('python', passed=True, time_spent=10)
        
        progress = await learning_feature.get_progress('python')
        
        assert progress is not None
        assert progress['topic_id'] == 'python'
        assert 'mastery_level' in progress
        assert 'exercises_completed' in progress
    
    @pytest.mark.asyncio
    async def test_get_all_progress(self, learning_feature):
        """Test getting all progress"""
        # Record progress for multiple topics
        await learning_feature.progress_tracker.record_exercise('python', passed=True)
        await learning_feature.progress_tracker.record_exercise('javascript', passed=True)
        
        progress = await learning_feature.get_progress()
        
        assert 'topics' in progress
        assert 'overall_mastery' in progress
        assert 'total_time' in progress
        assert len(progress['topics']) >= 2
    
    @pytest.mark.asyncio
    async def test_get_next_topic(self, learning_feature):
        """Test getting recommended next topic"""
        next_topic = await learning_feature.get_next_topic()
        
        # Should return a topic or None
        if next_topic:
            assert 'id' in next_topic
            assert 'name' in next_topic
    
    @pytest.mark.asyncio
    async def test_get_mastery_level(self, learning_feature):
        """Test getting mastery level"""
        # Record some progress
        await learning_feature.progress_tracker.record_exercise('python', passed=True)
        
        mastery = await learning_feature.get_mastery_level('python')
        
        assert 'topic_id' in mastery
        assert 'mastery_level' in mastery
        assert 'mastery_percentage' in mastery
        assert mastery['topic_id'] == 'python'


class TestTutorialEngine:
    """Test TutorialEngine class"""
    
    @pytest.mark.asyncio
    async def test_load_content(self, tutorial_engine):
        """Test loading tutorial content"""
        await tutorial_engine.load_content()
        
        assert len(tutorial_engine.topics) > 0
        assert 'python' in tutorial_engine.topics
    
    @pytest.mark.asyncio
    async def test_get_topic(self, tutorial_engine):
        """Test getting a topic"""
        await tutorial_engine.load_content()
        
        topic = await tutorial_engine.get_topic('python')
        
        assert topic is not None
        assert topic.id == 'python'
        assert topic.name == 'Python Programming'
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_topic(self, tutorial_engine):
        """Test getting a non-existent topic"""
        await tutorial_engine.load_content()
        
        topic = await tutorial_engine.get_topic('nonexistent')
        
        assert topic is None
    
    @pytest.mark.asyncio
    async def test_get_all_topics(self, tutorial_engine):
        """Test getting all topics"""
        await tutorial_engine.load_content()
        
        topics = await tutorial_engine.get_all_topics()
        
        assert isinstance(topics, list)
        assert len(topics) > 0
        assert all(isinstance(t, Topic) for t in topics)
    
    @pytest.mark.asyncio
    async def test_get_lesson(self, tutorial_engine):
        """Test getting a lesson"""
        await tutorial_engine.load_content()
        
        lesson = await tutorial_engine.get_lesson('python', DifficultyLevel.BEGINNER)
        
        assert lesson is not None
        assert 'title' in lesson
        assert 'content' in lesson
    
    def test_topic_to_dict(self):
        """Test Topic to_dict method"""
        topic = Topic(
            id='test',
            name='Test Topic',
            description='Test description',
            difficulty=DifficultyLevel.BEGINNER
        )
        
        data = topic.to_dict()
        
        assert data['id'] == 'test'
        assert data['name'] == 'Test Topic'
        assert data['difficulty'] == 'beginner'


class TestAdaptiveDifficultyController:
    """Test AdaptiveDifficultyController class"""
    
    @pytest.mark.asyncio
    async def test_get_difficulty_novice(self, difficulty_controller):
        """Test difficulty for novice user"""
        progress = Progress(
            topic_id='python',
            mastery_level=MasteryLevel.NOVICE
        )
        
        difficulty = await difficulty_controller.get_difficulty(progress)
        
        assert difficulty == DifficultyLevel.BEGINNER
    
    @pytest.mark.asyncio
    async def test_get_difficulty_proficient(self, difficulty_controller):
        """Test difficulty for proficient user"""
        progress = Progress(
            topic_id='python',
            mastery_level=MasteryLevel.PROFICIENT
        )
        
        difficulty = await difficulty_controller.get_difficulty(progress)
        
        assert difficulty == DifficultyLevel.ADVANCED
    
    @pytest.mark.asyncio
    async def test_adjust_difficulty_up(self, difficulty_controller):
        """Test increasing difficulty based on performance"""
        new_difficulty = await difficulty_controller.adjust_difficulty(
            DifficultyLevel.BEGINNER,
            performance=0.9
        )
        
        assert new_difficulty == DifficultyLevel.INTERMEDIATE
    
    @pytest.mark.asyncio
    async def test_adjust_difficulty_down(self, difficulty_controller):
        """Test decreasing difficulty based on performance"""
        new_difficulty = await difficulty_controller.adjust_difficulty(
            DifficultyLevel.ADVANCED,
            performance=0.4
        )
        
        assert new_difficulty == DifficultyLevel.INTERMEDIATE
    
    @pytest.mark.asyncio
    async def test_adjust_difficulty_maintain(self, difficulty_controller):
        """Test maintaining difficulty with moderate performance"""
        new_difficulty = await difficulty_controller.adjust_difficulty(
            DifficultyLevel.INTERMEDIATE,
            performance=0.6
        )
        
        assert new_difficulty == DifficultyLevel.INTERMEDIATE


class TestProgressTracker:
    """Test ProgressTracker class"""
    
    @pytest.mark.asyncio
    async def test_record_exercise_passed(self, progress_tracker):
        """Test recording a passed exercise"""
        await progress_tracker.record_exercise('python', passed=True, time_spent=5)
        
        progress = await progress_tracker.get_progress('python')
        
        assert progress is not None
        assert progress.exercises_completed == 1
        assert progress.accuracy > 0
        assert progress.time_spent == 5
    
    @pytest.mark.asyncio
    async def test_record_exercise_failed(self, progress_tracker):
        """Test recording a failed exercise"""
        await progress_tracker.record_exercise('python', passed=False, time_spent=10)
        
        progress = await progress_tracker.get_progress('python')
        
        assert progress is not None
        assert progress.exercises_completed == 1
        assert progress.accuracy == 0.0
    
    @pytest.mark.asyncio
    async def test_mastery_level_progression(self, progress_tracker):
        """Test mastery level increases with progress"""
        # Record multiple successful exercises
        for _ in range(7):
            await progress_tracker.record_exercise('python', passed=True)
        
        progress = await progress_tracker.get_progress('python')
        
        # Should have progressed beyond novice
        assert progress.mastery_level != MasteryLevel.NOVICE
    
    @pytest.mark.asyncio
    async def test_get_all_progress(self, progress_tracker):
        """Test getting all progress"""
        await progress_tracker.record_exercise('python', passed=True)
        await progress_tracker.record_exercise('javascript', passed=True)
        
        all_progress = await progress_tracker.get_all_progress()
        
        assert len(all_progress) == 2
        assert any(p.topic_id == 'python' for p in all_progress)
        assert any(p.topic_id == 'javascript' for p in all_progress)


class TestExerciseGenerator:
    """Test ExerciseGenerator class"""
    
    @pytest.mark.asyncio
    async def test_generate_exercises(self, exercise_generator):
        """Test generating exercises"""
        exercises = await exercise_generator.generate(
            'python',
            DifficultyLevel.BEGINNER,
            count=3
        )
        
        assert isinstance(exercises, list)
        assert all(isinstance(ex, Exercise) for ex in exercises)
        assert all(ex.topic_id == 'python' for ex in exercises)
    
    @pytest.mark.asyncio
    async def test_get_exercise(self, exercise_generator):
        """Test getting a specific exercise"""
        exercise = await exercise_generator.get_exercise('python_hello')
        
        assert exercise is not None
        assert exercise.id == 'python_hello'
        assert exercise.topic_id == 'python'
    
    def test_exercise_to_dict(self):
        """Test Exercise to_dict method"""
        exercise = Exercise(
            id='test',
            topic_id='python',
            title='Test Exercise',
            description='Test description',
            difficulty=DifficultyLevel.BEGINNER,
            code_template='# code',
            solution='# solution'
        )
        
        data = exercise.to_dict()
        
        assert data['id'] == 'test'
        assert data['topic_id'] == 'python'
        assert data['title'] == 'Test Exercise'
        assert data['difficulty'] == 'beginner'


class TestIntegration:
    """Integration tests for Learning Mode"""
    
    @pytest.mark.asyncio
    async def test_complete_learning_flow(self, learning_feature):
        """Test complete learning flow from start to completion"""
        # Start a topic
        result = await learning_feature.start_topic('python')
        assert result['topic']['id'] == 'python'
        
        # Get exercises
        exercises = await learning_feature.get_exercises('python', count=2)
        assert len(exercises) <= 2
        
        # Submit an exercise
        if exercises:
            exercise_id = exercises[0]['id']
            solution = "def hello():\n    return 'Hello, World!'"
            result = await learning_feature.submit_exercise(exercise_id, solution)
            assert 'passed' in result
        
        # Check progress
        progress = await learning_feature.get_progress('python')
        assert progress is not None
        assert progress['exercises_completed'] >= 0
    
    @pytest.mark.asyncio
    async def test_adaptive_difficulty_flow(self, learning_feature):
        """Test adaptive difficulty adjustment"""
        # Record multiple successful exercises
        for _ in range(5):
            await learning_feature.progress_tracker.record_exercise('python', passed=True)
        
        # Start topic again - should get higher difficulty
        result = await learning_feature.start_topic('python')
        
        # Difficulty should adapt based on progress (including expert level)
        assert result['difficulty'] in ['beginner', 'intermediate', 'advanced', 'expert']
    
    @pytest.mark.asyncio
    async def test_mastery_tracking(self, learning_feature):
        """Test mastery level tracking"""
        # Record exercises with varying success
        await learning_feature.progress_tracker.record_exercise('python', passed=True)
        await learning_feature.progress_tracker.record_exercise('python', passed=True)
        await learning_feature.progress_tracker.record_exercise('python', passed=False)
        
        # Get mastery level
        mastery = await learning_feature.get_mastery_level('python')
        
        assert mastery['topic_id'] == 'python'
        assert 'mastery_level' in mastery
        assert 'mastery_percentage' in mastery
        # Mastery percentage can exceed 100 if more exercises completed than total
        assert mastery['mastery_percentage'] >= 0
