#!/usr/bin/env python3
"""
Tests for the enhanced test infrastructure
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock

from xencode.test_infrastructure import (
    TestInfrastructure,
    DependencyResolver,
    EnhancedTestRunner,
    CoverageTracker,
    TestResults,
    IntegrationIssue
)


class TestDependencyResolver:
    """Test dependency resolution functionality"""
    
    def setup_method(self):
        self.resolver = DependencyResolver()
    
    @pytest.mark.asyncio
    async def test_resolve_dependencies_success(self):
        """Test successful dependency resolution"""
        with patch.object(self.resolver, '_detect_conflicts', return_value=[]), \
             patch.object(self.resolver, '_detect_missing_dependencies', return_value=[]):
            
            result = await self.resolver.resolve_dependencies()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_resolve_dependencies_with_conflicts(self):
        """Test dependency resolution with conflicts"""
        conflicts = ['pytest version conflict']
        
        with patch.object(self.resolver, '_detect_conflicts', return_value=conflicts), \
             patch.object(self.resolver, '_resolve_conflicts') as mock_resolve, \
             patch.object(self.resolver, '_detect_missing_dependencies', return_value=[]):
            
            await self.resolver.resolve_dependencies()
            mock_resolve.assert_called_once_with(conflicts)
    
    @pytest.mark.asyncio
    async def test_detect_missing_dependencies(self):
        """Test detection of missing dependencies"""
        missing = await self.resolver._detect_missing_dependencies()
        assert isinstance(missing, list)


class TestCoverageTracker:
    """Test coverage tracking functionality"""
    
    def setup_method(self):
        self.tracker = CoverageTracker(['xencode'])
    
    def test_start_stop_coverage(self):
        """Test starting and stopping coverage"""
        self.tracker.start_coverage()
        assert self.tracker.coverage_data is not None
        
        # Stop coverage (may return 0 if no code executed)
        percentage = self.tracker.stop_coverage()
        assert isinstance(percentage, (int, float))
        assert percentage >= 0


class TestEnhancedTestRunner:
    """Test enhanced test runner functionality"""
    
    def setup_method(self):
        self.runner = EnhancedTestRunner(['tests'])
    
    def test_initialization(self):
        """Test test runner initialization"""
        assert self.runner.test_dirs == ['tests']
        assert isinstance(self.runner.coverage_tracker, CoverageTracker)
        assert isinstance(self.runner.mock_registry, dict)
    
    def test_setup_teardown_isolation(self):
        """Test test isolation setup and teardown"""
        self.runner.setup_test_isolation()
        assert len(self.runner.mock_registry) > 0
        
        self.runner.teardown_test_isolation()
        # Mock registry should be cleared
        assert len(self.runner.mock_registry) == 0
    
    def test_parse_pytest_output(self):
        """Test parsing pytest output"""
        stdout = """
        ========================= test session starts =========================
        collected 10 items
        
        test_file.py::test_one PASSED
        test_file.py::test_two FAILED
        
        ========================= FAILURES =========================
        FAILED test_file.py::test_two - AssertionError
        
        =================== 1 failed, 9 passed in 2.5s ===================
        """
        
        results = self.runner._parse_pytest_output(stdout, "")
        assert results.passed == 9
        assert results.failed == 1
        assert results.total_tests == 10


class TestTestResults:
    """Test TestResults data class"""
    
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        results = TestResults(total_tests=10, passed=8, failed=2)
        assert results.success_rate == 80.0
        
        # Test zero division
        empty_results = TestResults()
        assert empty_results.success_rate == 0.0
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        results = TestResults(total_tests=5, passed=4, failed=1)
        data = results.to_dict()
        
        assert isinstance(data, dict)
        assert data['total_tests'] == 5
        assert data['passed'] == 4
        assert data['failed'] == 1
        assert 'success_rate' in data


class TestTestInfrastructure:
    """Test main test infrastructure"""
    
    def setup_method(self):
        self.infrastructure = TestInfrastructure()
    
    def test_initialization(self):
        """Test infrastructure initialization"""
        assert isinstance(self.infrastructure.dependency_resolver, DependencyResolver)
        assert isinstance(self.infrastructure.test_runner, EnhancedTestRunner)
        assert isinstance(self.infrastructure.coverage_tracker, CoverageTracker)
    
    @pytest.mark.asyncio
    async def test_validate_integration_points(self):
        """Test integration point validation"""
        issues = await self.infrastructure.validate_integration_points()
        assert isinstance(issues, list)
        
        # All issues should be IntegrationIssue instances
        for issue in issues:
            assert isinstance(issue, IntegrationIssue)
            assert hasattr(issue, 'component')
            assert hasattr(issue, 'issue_type')
            assert hasattr(issue, 'severity')
    
    @pytest.mark.asyncio
    async def test_run_comprehensive_tests_with_dependency_failure(self):
        """Test comprehensive test run with dependency failure"""
        with patch.object(self.infrastructure, 'resolve_dependencies', return_value=False):
            results = await self.infrastructure.run_comprehensive_tests()
            
            assert isinstance(results, TestResults)
            assert len(results.errors) > 0
            assert "Failed to resolve dependencies" in results.errors[0]
    
    def test_generate_test_report(self):
        """Test test report generation"""
        results = TestResults(total_tests=5, passed=4, failed=1)
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            report_file = f.name
        
        try:
            self.infrastructure.generate_test_report(results, report_file)
            
            # Check if file was created
            assert os.path.exists(report_file)
            
            # Check file content
            import json
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            assert 'timestamp' in report_data
            assert 'results' in report_data
            assert 'dependency_status' in report_data
            
        finally:
            # Clean up
            if os.path.exists(report_file):
                os.unlink(report_file)


class TestIntegrationIssue:
    """Test IntegrationIssue data class"""
    
    def test_creation(self):
        """Test creating integration issue"""
        issue = IntegrationIssue(
            component='test_component',
            issue_type='import_error',
            description='Test description',
            severity='high',
            suggested_fix='Test fix'
        )
        
        assert issue.component == 'test_component'
        assert issue.issue_type == 'import_error'
        assert issue.description == 'Test description'
        assert issue.severity == 'high'
        assert issue.suggested_fix == 'Test fix'


# Integration test
@pytest.mark.asyncio
async def test_full_infrastructure_workflow():
    """Test the complete infrastructure workflow"""
    infrastructure = TestInfrastructure()
    
    # Test dependency resolution
    deps_resolved = await infrastructure.resolve_dependencies()
    assert isinstance(deps_resolved, bool)
    
    # Test integration validation
    issues = await infrastructure.validate_integration_points()
    assert isinstance(issues, list)
    
    # Note: We don't run comprehensive tests here as it would be recursive
    # (this test file would try to test itself)