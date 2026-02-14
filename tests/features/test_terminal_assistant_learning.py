"""
Tests for Terminal Assistant Learning Engine

Tests the advanced learning capabilities including:
- User preference learning and adaptation
- Command pattern recognition
- Personalized suggestion scoring
- Adaptive difficulty adjustment
- Learning progress tracking
- Preference persistence
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from collections import Counter

from xencode.features.terminal_assistant import LearningEngine


@pytest.fixture
def temp_prefs_dir(tmp_path):
    """Create temporary preferences directory"""
    prefs_dir = tmp_path / '.xencode'
    prefs_dir.mkdir(parents=True, exist_ok=True)
    return prefs_dir


@pytest.fixture
def learning_engine():
    """Create a LearningEngine instance"""
    return LearningEngine(enabled=True)


@pytest.fixture
def learning_engine_with_data(learning_engine):
    """Create a LearningEngine with sample data"""
    # Add some patterns
    learning_engine.patterns = Counter({
        'git status': 15,
        'git commit': 10,
        'npm install': 8,
        'python test.py': 5,
        'ls -la': 20
    })
    
    # Add command contexts
    now = datetime.now()
    learning_engine.command_contexts = {
        'git status': [
            {
                'timestamp': (now - timedelta(hours=1)).isoformat(),
                'success': True,
                'current_directory': '/project',
                'project_type': 'python'
            },
            {
                'timestamp': (now - timedelta(hours=2)).isoformat(),
                'success': True,
                'current_directory': '/project',
                'project_type': 'python'
            }
        ],
        'npm install': [
            {
                'timestamp': (now - timedelta(days=1)).isoformat(),
                'success': True,
                'current_directory': '/webapp',
                'project_type': 'javascript'
            }
        ]
    }
    
    # Add skill levels
    learning_engine.user_skill_level = {
        'git': 0.7,
        'npm': 0.4,
        'python': 0.6
    }
    
    # Add learning progress
    learning_engine.learning_progress = {
        'git': {
            'total_uses': 25,
            'successful_uses': 23,
            'first_used': (now - timedelta(days=30)).isoformat(),
            'mastery_level': 0.85
        },
        'npm': {
            'total_uses': 8,
            'successful_uses': 6,
            'first_used': (now - timedelta(days=10)).isoformat(),
            'mastery_level': 0.3
        }
    }
    
    return learning_engine


class TestLearningEngineInitialization:
    """Test LearningEngine initialization"""
    
    def test_init_default(self):
        """Test default initialization"""
        engine = LearningEngine()
        assert engine.enabled is True
        assert engine.preferences == {}
        assert isinstance(engine.patterns, Counter)
        assert engine.command_contexts == {}
        assert engine.user_skill_level == {}
        assert engine.learning_progress == {}
        assert 'frequency' in engine.preference_weights
        assert 'recency' in engine.preference_weights
        assert 'success_rate' in engine.preference_weights
        assert 'context_match' in engine.preference_weights
    
    def test_init_disabled(self):
        """Test initialization with disabled state"""
        engine = LearningEngine(enabled=False)
        assert engine.enabled is False


class TestPreferencePersistence:
    """Test preference loading and saving"""
    
    @pytest.mark.asyncio
    async def test_save_preferences(self, learning_engine_with_data, temp_prefs_dir):
        """Test saving preferences to file"""
        prefs_file = temp_prefs_dir / 'terminal_preferences.json'
        
        with patch('pathlib.Path.home', return_value=temp_prefs_dir.parent):
            await learning_engine_with_data.save_preferences()
        
        assert prefs_file.exists()
        
        with open(prefs_file, 'r') as f:
            data = json.load(f)
        
        assert 'preferences' in data
        assert 'patterns' in data
        assert 'command_contexts' in data
        assert 'user_skill_level' in data
        assert 'learning_progress' in data
        assert 'preference_weights' in data
        
        assert data['patterns']['git status'] == 15
        assert data['user_skill_level']['git'] == 0.7
    
    @pytest.mark.asyncio
    async def test_load_preferences(self, learning_engine, temp_prefs_dir):
        """Test loading preferences from file"""
        prefs_file = temp_prefs_dir / 'terminal_preferences.json'
        
        # Create sample preferences file
        sample_data = {
            'preferences': {'theme': 'dark'},
            'patterns': {'git status': 10, 'ls': 5},
            'command_contexts': {
                'git status': [{
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'current_directory': '/test',
                    'project_type': 'python'
                }]
            },
            'user_skill_level': {'git': 0.5},
            'learning_progress': {
                'git': {
                    'total_uses': 10,
                    'successful_uses': 9,
                    'first_used': datetime.now().isoformat(),
                    'mastery_level': 0.45
                }
            },
            'preference_weights': {
                'frequency': 0.4,
                'recency': 0.2,
                'success_rate': 0.2,
                'context_match': 0.2
            }
        }
        
        with open(prefs_file, 'w') as f:
            json.dump(sample_data, f)
        
        with patch('pathlib.Path.home', return_value=temp_prefs_dir.parent):
            await learning_engine.load_preferences()
        
        assert learning_engine.preferences == {'theme': 'dark'}
        assert learning_engine.patterns['git status'] == 10
        assert learning_engine.patterns['ls'] == 5
        assert 'git status' in learning_engine.command_contexts
        assert learning_engine.user_skill_level['git'] == 0.5
        assert learning_engine.learning_progress['git']['mastery_level'] == 0.45
        assert learning_engine.preference_weights['frequency'] == 0.4
    
    @pytest.mark.asyncio
    async def test_load_preferences_missing_file(self, learning_engine, temp_prefs_dir):
        """Test loading preferences when file doesn't exist"""
        with patch('pathlib.Path.home', return_value=temp_prefs_dir.parent):
            await learning_engine.load_preferences()
        
        # Should not raise error, just keep defaults
        assert learning_engine.preferences == {}
        assert len(learning_engine.patterns) == 0


class TestPersonalizedScoring:
    """Test personalized suggestion scoring"""
    
    def test_calculate_personalized_score(self, learning_engine_with_data):
        """Test personalized score calculation"""
        context = {
            'current_directory': '/project',
            'project_type': 'python'
        }
        
        score = learning_engine_with_data._calculate_personalized_score('git status', context)
        
        # Should have positive score due to frequency, recency, success rate, and context match
        assert score > 0
    
    def test_calculate_recency_score_recent(self, learning_engine_with_data):
        """Test recency score for recently used command"""
        score = learning_engine_with_data._calculate_recency_score('git status')
        
        # Recently used (1-2 hours ago), should have high score
        assert score > 0.9
    
    def test_calculate_recency_score_old(self, learning_engine_with_data):
        """Test recency score for old command"""
        score = learning_engine_with_data._calculate_recency_score('npm install')
        
        # Used 1 day ago, should have lower score
        assert 0 < score < 0.9
    
    def test_calculate_recency_score_unknown(self, learning_engine_with_data):
        """Test recency score for unknown command"""
        score = learning_engine_with_data._calculate_recency_score('unknown command')
        
        assert score == 0.0
    
    def test_calculate_success_score_high(self, learning_engine_with_data):
        """Test success score for high success rate command"""
        score = learning_engine_with_data._calculate_success_score('git status')
        
        # 2 successes out of 2, should be 1.0
        assert score == 1.0
    
    def test_calculate_success_score_unknown(self, learning_engine_with_data):
        """Test success score for unknown command"""
        score = learning_engine_with_data._calculate_success_score('unknown command')
        
        # Unknown commands get neutral score
        assert score == 0.5
    
    def test_calculate_context_match_exact(self, learning_engine_with_data):
        """Test context match for exact match"""
        context = {
            'current_directory': '/project',
            'project_type': 'python'
        }
        
        score = learning_engine_with_data._calculate_context_match('git status', context)
        
        # Both directory and project type match
        assert score == 1.0
    
    def test_calculate_context_match_partial(self, learning_engine_with_data):
        """Test context match for partial match"""
        context = {
            'current_directory': '/project',
            'project_type': 'javascript'
        }
        
        score = learning_engine_with_data._calculate_context_match('git status', context)
        
        # Only directory matches
        assert 0 < score < 1.0
    
    def test_calculate_context_match_no_match(self, learning_engine_with_data):
        """Test context match for no match"""
        context = {
            'current_directory': '/other',
            'project_type': 'rust'
        }
        
        score = learning_engine_with_data._calculate_context_match('git status', context)
        
        # No match
        assert score == 0.0


class TestDifficultyEstimation:
    """Test command difficulty estimation"""
    
    def test_estimate_difficulty_beginner(self, learning_engine):
        """Test difficulty estimation for beginner commands"""
        assert learning_engine._estimate_difficulty('ls') == 'beginner'
        assert learning_engine._estimate_difficulty('cd /home') == 'beginner'
        assert learning_engine._estimate_difficulty('pwd') == 'beginner'
        assert learning_engine._estimate_difficulty('mkdir test') == 'beginner'
    
    def test_estimate_difficulty_intermediate(self, learning_engine):
        """Test difficulty estimation for intermediate commands"""
        assert learning_engine._estimate_difficulty('git status') == 'intermediate'
        assert learning_engine._estimate_difficulty('npm install') == 'intermediate'
        assert learning_engine._estimate_difficulty('grep pattern file') == 'intermediate'
        assert learning_engine._estimate_difficulty('find . -name "*.py"') == 'intermediate'
    
    def test_estimate_difficulty_advanced_pipes(self, learning_engine):
        """Test difficulty estimation for commands with pipes"""
        assert learning_engine._estimate_difficulty('cat file | grep pattern') == 'advanced'
        assert learning_engine._estimate_difficulty('ps aux | grep python') == 'advanced'
    
    def test_estimate_difficulty_advanced_redirects(self, learning_engine):
        """Test difficulty estimation for commands with redirects"""
        assert learning_engine._estimate_difficulty('echo test > file.txt') == 'advanced'
        assert learning_engine._estimate_difficulty('cat < input.txt') == 'advanced'
    
    def test_estimate_difficulty_advanced_flags(self, learning_engine):
        """Test difficulty estimation for commands with many flags"""
        assert learning_engine._estimate_difficulty('ls -la -h -R -t') == 'advanced'


class TestLearningHints:
    """Test learning hint generation"""
    
    def test_is_learning_command_low_usage(self, learning_engine):
        """Test learning detection for low usage commands"""
        learning_engine.patterns = Counter({'git status': 3})
        
        assert learning_engine._is_learning_command('git status') is True
    
    def test_is_learning_command_high_usage(self, learning_engine):
        """Test learning detection for high usage commands"""
        learning_engine.patterns = Counter({'git status': 10})
        learning_engine.user_skill_level = {'git': 0.8}
        
        assert learning_engine._is_learning_command('git status') is False
    
    def test_is_learning_command_low_skill(self, learning_engine):
        """Test learning detection for low skill level"""
        learning_engine.patterns = Counter({'git status': 10})
        learning_engine.user_skill_level = {'git': 0.3}
        
        assert learning_engine._is_learning_command('git status') is True
    
    def test_generate_learning_hint_known(self, learning_engine):
        """Test learning hint generation for known commands"""
        hint = learning_engine._generate_learning_hint('git status')
        
        assert 'git status' in hint
        assert 'Tip:' in hint
    
    def test_generate_learning_hint_unknown(self, learning_engine):
        """Test learning hint generation for unknown commands"""
        hint = learning_engine._generate_learning_hint('unknown command')
        
        assert 'Practice' in hint


class TestExplanationGeneration:
    """Test command explanation generation"""
    
    def test_generate_explanation_git(self, learning_engine):
        """Test explanation for git commands"""
        context = {'project_type': 'python'}
        explanation = learning_engine._generate_explanation('git status', context)
        
        assert 'Version control' in explanation
        assert 'python' in explanation
    
    def test_generate_explanation_with_frequency(self, learning_engine):
        """Test explanation includes frequency information"""
        learning_engine.patterns = Counter({'npm install': 15})
        
        explanation = learning_engine._generate_explanation('npm install', {})
        
        assert 'frequently used' in explanation
        assert '15 times' in explanation
    
    def test_generate_explanation_unknown(self, learning_engine):
        """Test explanation for unknown commands"""
        explanation = learning_engine._generate_explanation('unknown command', {})
        
        assert 'Execute' in explanation


class TestEnhanceSuggestions:
    """Test suggestion enhancement"""
    
    @pytest.mark.asyncio
    async def test_enhance_suggestions_basic(self, learning_engine_with_data):
        """Test basic suggestion enhancement"""
        suggestions = [
            {'command': 'git status', 'score': 0.5},
            {'command': 'npm install', 'score': 0.3}
        ]
        
        context = {
            'current_directory': '/project',
            'project_type': 'python'
        }
        
        enhanced = await learning_engine_with_data.enhance_suggestions(suggestions, context)
        
        assert len(enhanced) == 2
        assert all('explanation' in s for s in enhanced)
        assert all('difficulty' in s for s in enhanced)
        
        # git status should score higher due to recency and context match
        assert enhanced[0]['command'] == 'git status'
    
    @pytest.mark.asyncio
    async def test_enhance_suggestions_with_learning_hints(self, learning_engine):
        """Test suggestion enhancement includes learning hints"""
        learning_engine.patterns = Counter({'git status': 2})  # Low usage
        
        suggestions = [{'command': 'git status', 'score': 0.5}]
        context = {}
        
        enhanced = await learning_engine.enhance_suggestions(suggestions, context)
        
        assert 'learning_hint' in enhanced[0]
    
    @pytest.mark.asyncio
    async def test_enhance_suggestions_disabled(self, learning_engine):
        """Test suggestion enhancement when disabled"""
        learning_engine.enabled = False
        
        suggestions = [{'command': 'git status', 'score': 0.5}]
        enhanced = await learning_engine.enhance_suggestions(suggestions, {})
        
        assert enhanced == suggestions


class TestLearning:
    """Test learning from command execution"""
    
    @pytest.mark.asyncio
    async def test_learn_success(self, learning_engine):
        """Test learning from successful command"""
        context = {
            'current_directory': '/project',
            'project_type': 'python'
        }
        
        await learning_engine.learn('git status', True, context)
        
        assert learning_engine.patterns['git status'] == 1
        assert 'git status' in learning_engine.command_contexts
        assert learning_engine.command_contexts['git status'][0]['success'] is True
        assert 'git' in learning_engine.user_skill_level
        assert learning_engine.user_skill_level['git'] > 0
    
    @pytest.mark.asyncio
    async def test_learn_failure(self, learning_engine):
        """Test learning from failed command"""
        await learning_engine.learn('git status', False, {})
        
        # Failed commands don't increment pattern count
        assert learning_engine.patterns['git status'] == 0
        
        # But they are recorded in contexts
        assert 'git status' in learning_engine.command_contexts
        assert learning_engine.command_contexts['git status'][0]['success'] is False
    
    @pytest.mark.asyncio
    async def test_learn_context_limit(self, learning_engine):
        """Test context history is limited to 50 entries"""
        for i in range(60):
            await learning_engine.learn('git status', True, {})
        
        # Should keep only last 50
        assert len(learning_engine.command_contexts['git status']) == 50
    
    @pytest.mark.asyncio
    async def test_learn_disabled(self, learning_engine):
        """Test learning when disabled"""
        learning_engine.enabled = False
        
        await learning_engine.learn('git status', True, {})
        
        assert len(learning_engine.patterns) == 0
        assert len(learning_engine.command_contexts) == 0


class TestSkillLevelTracking:
    """Test skill level tracking"""
    
    @pytest.mark.asyncio
    async def test_update_skill_level_success(self, learning_engine):
        """Test skill level increases on success"""
        await learning_engine._update_skill_level('git status', True)
        
        assert learning_engine.user_skill_level['git'] == 0.05
        
        # Multiple successes
        for _ in range(10):
            await learning_engine._update_skill_level('git status', True)
        
        assert learning_engine.user_skill_level['git'] > 0.5
    
    @pytest.mark.asyncio
    async def test_update_skill_level_failure(self, learning_engine):
        """Test skill level decreases on failure"""
        learning_engine.user_skill_level['git'] = 0.5
        
        await learning_engine._update_skill_level('git status', False)
        
        assert learning_engine.user_skill_level['git'] == 0.48
    
    @pytest.mark.asyncio
    async def test_update_skill_level_bounds(self, learning_engine):
        """Test skill level stays within bounds"""
        # Test upper bound
        learning_engine.user_skill_level['git'] = 0.99
        await learning_engine._update_skill_level('git status', True)
        assert learning_engine.user_skill_level['git'] <= 1.0
        
        # Test lower bound
        learning_engine.user_skill_level['npm'] = 0.01
        await learning_engine._update_skill_level('npm install', False)
        assert learning_engine.user_skill_level['npm'] >= 0.0


class TestLearningProgress:
    """Test learning progress tracking"""
    
    @pytest.mark.asyncio
    async def test_update_learning_progress_new(self, learning_engine):
        """Test learning progress for new command"""
        await learning_engine._update_learning_progress('git status', True)
        
        assert 'git' in learning_engine.learning_progress
        progress = learning_engine.learning_progress['git']
        assert progress['total_uses'] == 1
        assert progress['successful_uses'] == 1
        assert 'first_used' in progress
        assert progress['mastery_level'] > 0
    
    @pytest.mark.asyncio
    async def test_update_learning_progress_mastery(self, learning_engine):
        """Test mastery level calculation"""
        # Simulate 20 successful uses
        for _ in range(20):
            await learning_engine._update_learning_progress('git status', True)
        
        progress = learning_engine.learning_progress['git']
        
        # Should reach high mastery with 100% success rate and 20 uses
        assert progress['mastery_level'] > 0.9
    
    @pytest.mark.asyncio
    async def test_update_learning_progress_mixed(self, learning_engine):
        """Test mastery with mixed success/failure"""
        # 15 successes, 5 failures
        for _ in range(15):
            await learning_engine._update_learning_progress('git status', True)
        for _ in range(5):
            await learning_engine._update_learning_progress('git status', False)
        
        progress = learning_engine.learning_progress['git']
        
        # 75% success rate, full usage factor
        assert 0.7 < progress['mastery_level'] < 0.8


class TestPreferenceWeightAdaptation:
    """Test adaptive preference weight adjustment"""
    
    @pytest.mark.asyncio
    async def test_adapt_preference_weights_high_diversity(self, learning_engine):
        """Test weight adaptation for diverse command usage"""
        # Create high diversity pattern (many different commands)
        for i in range(100):
            learning_engine.patterns[f'command_{i}'] = 1
        
        await learning_engine._adapt_preference_weights()
        
        # High diversity should favor context over frequency
        assert learning_engine.preference_weights['context_match'] > learning_engine.preference_weights['frequency']
    
    @pytest.mark.asyncio
    async def test_adapt_preference_weights_low_diversity(self, learning_engine):
        """Test weight adaptation for repetitive command usage"""
        # Create low diversity pattern (few commands, high frequency)
        learning_engine.patterns = Counter({
            'git status': 50,
            'git commit': 30,
            'git push': 20
        })
        
        await learning_engine._adapt_preference_weights()
        
        # Low diversity should favor frequency over context
        assert learning_engine.preference_weights['frequency'] > learning_engine.preference_weights['context_match']
    
    @pytest.mark.asyncio
    async def test_adapt_preference_weights_insufficient_data(self, learning_engine):
        """Test weight adaptation with insufficient data"""
        learning_engine.patterns = Counter({'git status': 10})
        original_weights = learning_engine.preference_weights.copy()
        
        await learning_engine._adapt_preference_weights()
        
        # Weights should not change with insufficient data
        assert learning_engine.preference_weights == original_weights


class TestLearningStats:
    """Test learning statistics"""
    
    @pytest.mark.asyncio
    async def test_get_learning_stats(self, learning_engine_with_data):
        """Test getting learning statistics"""
        stats = await learning_engine_with_data.get_learning_stats()
        
        assert 'total_commands_learned' in stats
        assert 'total_executions' in stats
        assert 'skill_levels' in stats
        assert 'learning_progress' in stats
        assert 'preference_weights' in stats
        assert 'mastered_commands' in stats
        
        assert stats['total_commands_learned'] == 5
        assert stats['total_executions'] == 58  # Sum of all patterns
        assert 'git' in stats['mastered_commands']  # mastery_level > 0.8


class TestDifficultyAdjustment:
    """Test adaptive difficulty adjustment"""
    
    @pytest.mark.asyncio
    async def test_adjust_difficulty_beginner_to_intermediate(self, learning_engine):
        """Test difficulty adjustment from beginner to intermediate"""
        # High mastery
        learning_engine.learning_progress = {
            'git': {'mastery_level': 0.9},
            'npm': {'mastery_level': 0.85}
        }
        
        new_level = await learning_engine.adjust_difficulty('beginner')
        
        assert new_level == 'intermediate'
    
    @pytest.mark.asyncio
    async def test_adjust_difficulty_intermediate_to_advanced(self, learning_engine):
        """Test difficulty adjustment from intermediate to advanced"""
        learning_engine.learning_progress = {
            'git': {'mastery_level': 0.9},
            'docker': {'mastery_level': 0.85}
        }
        
        new_level = await learning_engine.adjust_difficulty('intermediate')
        
        assert new_level == 'advanced'
    
    @pytest.mark.asyncio
    async def test_adjust_difficulty_advanced_to_intermediate(self, learning_engine):
        """Test difficulty adjustment from advanced to intermediate"""
        learning_engine.learning_progress = {
            'git': {'mastery_level': 0.2},
            'docker': {'mastery_level': 0.25}
        }
        
        new_level = await learning_engine.adjust_difficulty('advanced')
        
        assert new_level == 'intermediate'
    
    @pytest.mark.asyncio
    async def test_adjust_difficulty_no_change(self, learning_engine):
        """Test difficulty stays same with moderate mastery"""
        learning_engine.learning_progress = {
            'git': {'mastery_level': 0.5},
            'npm': {'mastery_level': 0.6}
        }
        
        new_level = await learning_engine.adjust_difficulty('intermediate')
        
        assert new_level == 'intermediate'
    
    @pytest.mark.asyncio
    async def test_adjust_difficulty_no_progress(self, learning_engine):
        """Test difficulty adjustment with no learning progress"""
        new_level = await learning_engine.adjust_difficulty('beginner')
        
        assert new_level == 'beginner'
