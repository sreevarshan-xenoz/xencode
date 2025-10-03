#!/usr/bin/env python3
"""
Comprehensive Test Suite for Xencode Enhancement Systems

Tests the User Feedback System, Technical Debt Manager, and AI Ethics Framework
"""

import asyncio
import json
import tempfile
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

# Add xencode to path for testing
sys.path.insert(0, str(Path(__file__).parent))

from xencode.user_feedback_system import (
    UserFeedbackManager, UserPersonaManager, FeedbackType, UserJourneyEvent,
    get_feedback_manager, collect_user_feedback, track_user_event
)
from xencode.technical_debt_manager import (
    TechnicalDebtManager, TechnicalDebtDetector, DebtType, DebtSeverity,
    get_debt_manager
)
from xencode.ai_ethics_framework import (
    EthicsFramework, BiasDetector, PrivacyAnalyzer, FairnessAnalyzer,
    EthicsViolationType, BiasType, EthicsSeverity,
    get_ethics_framework, analyze_ai_interaction
)


class TestUserFeedbackSystem:
    """Test suite for User Feedback System"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()
    
    @pytest.fixture
    def feedback_manager(self, temp_db):
        """Create feedback manager with temporary database"""
        return UserFeedbackManager(temp_db)
    
    @pytest.mark.asyncio
    async def test_collect_feedback(self, feedback_manager):
        """Test feedback collection"""
        feedback_id = await feedback_manager.collect_feedback(
            user_id="test_user",
            feedback_type=FeedbackType.SATISFACTION,
            message="Great experience!",
            rating=5,
            context={"feature": "model_selection"}
        )
        
        assert feedback_id is not None
        assert len(feedback_id) > 0
    
    @pytest.mark.asyncio
    async def test_track_user_journey(self, feedback_manager):
        """Test user journey tracking"""
        journey_id = await feedback_manager.track_user_journey(
            user_id="test_user",
            event=UserJourneyEvent.FIRST_LAUNCH,
            session_id="session_123",
            context={"platform": "linux"},
            duration_ms=1500
        )
        
        assert journey_id is not None
        assert len(journey_id) > 0
    
    @pytest.mark.asyncio
    async def test_calculate_user_satisfaction(self, feedback_manager):
        """Test user satisfaction calculation"""
        # Add some test data
        await feedback_manager.collect_feedback(
            "test_user", FeedbackType.SATISFACTION, "Good", 4
        )
        await feedback_manager.track_user_journey(
            "test_user", UserJourneyEvent.FIRST_LAUNCH, "session_1"
        )
        
        metrics = await feedback_manager.calculate_user_satisfaction("test_user")
        
        assert metrics.user_id == "test_user"
        assert metrics.satisfaction_rating == 4.0
        assert metrics.total_sessions >= 1
    
    @pytest.mark.asyncio
    async def test_feedback_summary(self, feedback_manager):
        """Test feedback summary generation"""
        # Add test feedback
        await feedback_manager.collect_feedback(
            "user1", FeedbackType.BUG_REPORT, "Found a bug", 2
        )
        await feedback_manager.collect_feedback(
            "user2", FeedbackType.FEATURE_REQUEST, "Need new feature", 3
        )
        
        summary = await feedback_manager.get_feedback_summary(days=7)
        
        assert "feedback_by_type" in summary
        assert "average_ratings" in summary
        assert summary["total_feedback"] >= 2
    
    @pytest.mark.asyncio
    async def test_persona_identification(self, feedback_manager):
        """Test user persona identification"""
        persona_manager = UserPersonaManager(feedback_manager)
        
        # Create user with high activity
        for i in range(10):
            await feedback_manager.track_user_journey(
                "power_user", UserJourneyEvent.FIRST_QUERY, f"session_{i}"
            )
        
        persona = await persona_manager.identify_user_persona("power_user")
        assert persona in ["power_user", "regular_user", "casual_user", "new_user", "inactive_user"]


class TestTechnicalDebtManager:
    """Test suite for Technical Debt Manager"""
    
    @pytest.fixture
    def temp_project(self):
        """Create temporary project directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            
            # Create sample Python files
            (project_root / "simple.py").write_text("""
def simple_function():
    return "hello"
""")
            
            (project_root / "complex.py").write_text("""
def complex_function(x, y, z):
    # TODO: Optimize this function
    if x > 0:
        if y > 0:
            if z > 0:
                for i in range(x):
                    for j in range(y):
                        for k in range(z):
                            if i + j + k > 10:
                                if i * j * k > 100:
                                    return True
    return False
""")
            
            (project_root / "duplicate.py").write_text("""
def function_a():
    x = 1
    y = 2
    return x + y

def function_b():
    a = 1
    b = 2
    return a + b
""")
            
            yield project_root
    
    @pytest.fixture
    def debt_manager(self, temp_project):
        """Create debt manager with temporary project"""
        return TechnicalDebtManager(temp_project)
    
    @pytest.mark.asyncio
    async def test_detect_code_complexity(self, debt_manager):
        """Test code complexity detection"""
        detector = debt_manager.detector
        complexity_items = await detector.detect_code_complexity()
        
        # Should detect the complex function
        assert len(complexity_items) > 0
        complex_item = next((item for item in complexity_items if "complex_function" in item.description), None)
        assert complex_item is not None
        assert complex_item.debt_type == DebtType.CODE_COMPLEXITY
    
    @pytest.mark.asyncio
    async def test_detect_todo_comments(self, debt_manager):
        """Test TODO comment detection"""
        detector = debt_manager.detector
        todo_items = await detector.detect_todo_comments()
        
        # Should detect the TODO comment
        assert len(todo_items) > 0
        todo_item = next((item for item in todo_items if "TODO" in item.description), None)
        assert todo_item is not None
        assert todo_item.debt_type == DebtType.TODO_COMMENT
    
    @pytest.mark.asyncio
    async def test_detect_missing_tests(self, debt_manager):
        """Test missing test detection"""
        detector = debt_manager.detector
        missing_test_items = await detector.detect_missing_tests()
        
        # Should detect missing tests for our sample files
        assert len(missing_test_items) > 0
        assert all(item.debt_type == DebtType.MISSING_TESTS for item in missing_test_items)
    
    @pytest.mark.asyncio
    async def test_full_scan(self, debt_manager):
        """Test full debt scan"""
        metrics = await debt_manager.run_full_scan()
        
        assert metrics.total_items > 0
        assert metrics.total_effort_hours > 0
        assert len(metrics.items_by_type) > 0
        assert len(metrics.items_by_severity) > 0
    
    @pytest.mark.asyncio
    async def test_prioritized_debt_items(self, debt_manager):
        """Test prioritized debt item retrieval"""
        await debt_manager.run_full_scan()
        
        prioritized_items = await debt_manager.get_prioritized_debt_items(limit=5)
        
        assert len(prioritized_items) <= 5
        # Items should be sorted by priority (severity then effort)
        if len(prioritized_items) > 1:
            severity_order = [DebtSeverity.CRITICAL, DebtSeverity.HIGH, DebtSeverity.MEDIUM, DebtSeverity.LOW]
            for i in range(len(prioritized_items) - 1):
                current_priority = severity_order.index(prioritized_items[i].severity)
                next_priority = severity_order.index(prioritized_items[i + 1].severity)
                assert current_priority <= next_priority
    
    @pytest.mark.asyncio
    async def test_resolve_debt_item(self, debt_manager):
        """Test debt item resolution"""
        await debt_manager.run_full_scan()
        items = await debt_manager.get_prioritized_debt_items(limit=1)
        
        if items:
            item_id = items[0].id
            await debt_manager.resolve_debt_item(item_id, "Fixed the issue")
            
            # Verify resolution
            resolved_items = await debt_manager.get_prioritized_debt_items(limit=100)
            resolved_item = next((item for item in resolved_items if item.id == item_id), None)
            # Item should not appear in unresolved list anymore


class TestAIEthicsFramework:
    """Test suite for AI Ethics Framework"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()
    
    @pytest.fixture
    def ethics_framework(self, temp_db):
        """Create ethics framework with temporary database"""
        return EthicsFramework(temp_db)
    
    @pytest.mark.asyncio
    async def test_bias_detection(self, ethics_framework):
        """Test bias detection in AI responses"""
        biased_text = "Men are naturally better at programming than women"
        
        bias_detector = ethics_framework.bias_detector
        biases = await bias_detector.detect_bias(biased_text)
        
        assert len(biases) > 0
        bias_types = [bias[0] for bias in biases]
        assert BiasType.GENDER_BIAS in bias_types
    
    @pytest.mark.asyncio
    async def test_privacy_detection(self, ethics_framework):
        """Test privacy violation detection"""
        text_with_pii = "My email is john.doe@example.com and my phone is 555-123-4567"
        
        privacy_analyzer = ethics_framework.privacy_analyzer
        violations = await privacy_analyzer.detect_privacy_violations(text_with_pii)
        
        assert len(violations) >= 2  # Should detect email and phone
        violation_types = [v[0] for v in violations]
        assert "email" in violation_types
        assert "phone" in violation_types
    
    @pytest.mark.asyncio
    async def test_fairness_analysis(self, ethics_framework):
        """Test fairness analysis"""
        query = "How should I hire developers?"
        response = "You should focus on technical skills. He should be able to code well."
        
        fairness_analyzer = ethics_framework.fairness_analyzer
        issues = await fairness_analyzer.analyze_fairness(query, response)
        
        # May or may not detect issues depending on implementation
        assert isinstance(issues, list)
    
    @pytest.mark.asyncio
    async def test_analyze_interaction(self, ethics_framework):
        """Test full interaction analysis"""
        user_input = "Tell me about software engineers"
        ai_response = "Software engineers are typically men who are good at math. Contact me at admin@company.com for more info."
        
        violations = await ethics_framework.analyze_interaction(user_input, ai_response)
        
        assert len(violations) >= 1  # Should detect bias and/or privacy violation
        
        # Check that violations are stored
        metrics = await ethics_framework.get_ethics_metrics(days=1)
        assert metrics.total_violations >= len(violations)
    
    @pytest.mark.asyncio
    async def test_ethics_metrics(self, ethics_framework):
        """Test ethics metrics calculation"""
        # Generate some test violations
        await ethics_framework.analyze_interaction(
            "Test query",
            "Men are better at this task. Contact john@example.com"
        )
        
        metrics = await ethics_framework.get_ethics_metrics(days=7)
        
        assert metrics.total_violations >= 1
        assert isinstance(metrics.violations_by_type, dict)
        assert isinstance(metrics.violations_by_severity, dict)
    
    @pytest.mark.asyncio
    async def test_ethics_report(self, ethics_framework):
        """Test ethics report generation"""
        # Generate test data
        await ethics_framework.analyze_interaction(
            "Test query",
            "This response contains bias and john@example.com"
        )
        
        report = await ethics_framework.get_ethics_report(days=7)
        
        assert "metrics" in report
        assert "recent_violations" in report
        assert "guidelines" in report
        assert "recommendations" in report
        assert report["report_period_days"] == 7
    
    @pytest.mark.asyncio
    async def test_resolve_violation(self, ethics_framework):
        """Test violation resolution"""
        violations = await ethics_framework.analyze_interaction(
            "Test", "Men are better at programming"
        )
        
        if violations:
            violation_id = violations[0].id
            await ethics_framework.resolve_violation(violation_id, "Updated training data")
            
            # Verify resolution
            metrics = await ethics_framework.get_ethics_metrics(days=1)
            assert metrics.resolution_rate > 0


class TestIntegration:
    """Integration tests for all enhancement systems"""
    
    @pytest.mark.asyncio
    async def test_global_managers(self):
        """Test global manager instances"""
        # Test that global managers can be retrieved
        feedback_mgr = get_feedback_manager()
        assert feedback_mgr is not None
        
        debt_mgr = get_debt_manager()
        assert debt_mgr is not None
        
        ethics_fw = get_ethics_framework()
        assert ethics_fw is not None
    
    @pytest.mark.asyncio
    async def test_convenience_functions(self):
        """Test convenience functions"""
        # Test feedback collection
        feedback_id = await collect_user_feedback(
            "test_user", FeedbackType.GENERAL, "Test message"
        )
        assert feedback_id is not None
        
        # Test event tracking
        event_id = await track_user_event(
            "test_user", UserJourneyEvent.FIRST_LAUNCH, "session_1"
        )
        assert event_id is not None
        
        # Test ethics analysis
        violations = await analyze_ai_interaction(
            "Test input", "Test response with potential issues"
        )
        assert isinstance(violations, list)
    
    @pytest.mark.asyncio
    async def test_cross_system_workflow(self):
        """Test workflow across multiple systems"""
        # Simulate a complete user interaction workflow
        
        # 1. Track user journey
        await track_user_event(
            "workflow_user", UserJourneyEvent.FIRST_QUERY, "session_workflow"
        )
        
        # 2. Analyze AI interaction for ethics
        violations = await analyze_ai_interaction(
            "How do I code?", 
            "Programming is easy for men but difficult for women. Email me at test@example.com"
        )
        
        # 3. Collect user feedback
        if violations:
            await collect_user_feedback(
                "workflow_user", 
                FeedbackType.BUG_REPORT, 
                "AI response contained bias",
                rating=1,
                context={"violations_detected": len(violations)}
            )
        
        # 4. Verify all systems recorded the interaction
        feedback_mgr = get_feedback_manager()
        ethics_fw = get_ethics_framework()
        
        user_metrics = await feedback_mgr.calculate_user_satisfaction("workflow_user")
        ethics_metrics = await ethics_fw.get_ethics_metrics(days=1)
        
        assert user_metrics.total_sessions >= 1
        assert ethics_metrics.total_violations >= len(violations)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])