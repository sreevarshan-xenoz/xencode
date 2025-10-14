#!/usr/bin/env python3
"""
Integration tests for core integration infrastructure components

Tests the enhanced CLI system, resource monitor, feature detector,
and cold start optimization working together.
"""

import time
import unittest
from unittest.mock import patch

# Import components to test
from xencode.enhanced_cli_system import (
    CommandRouter,
    EnhancedXencodeCLI,
    FeatureAvailability,
    FeatureDetector,
)
from xencode.resource_monitor import (
    FeatureLevel,
    HardwareProfiler,
    ResourceMonitor,
    ResourcePressure,
    ResourceUsage,
    ScanStrategy,
    SystemProfile,
)
from xencode.security_manager import SecurityManager


class TestFeatureDetector(unittest.TestCase):
    """Test feature detection with timeout handling"""

    def setUp(self):
        self.detector = FeatureDetector(timeout_seconds=1.0)

    def test_feature_detection_basic(self):
        """Test basic feature detection"""
        features = self.detector.detect_features()

        # Should always return FeatureAvailability object
        self.assertIsInstance(features, FeatureAvailability)

        # Core features should be available
        self.assertTrue(features.security_manager)
        self.assertTrue(features.context_cache)
        self.assertTrue(features.model_stability)

    def test_feature_detection_caching(self):
        """Test feature detection caching"""
        # First call
        start_time = time.time()
        features1 = self.detector.detect_features()
        first_call_time = time.time() - start_time

        # Second call (should use cache)
        start_time = time.time()
        features2 = self.detector.detect_features()
        second_call_time = time.time() - start_time

        # Second call should be much faster (cached)
        self.assertLess(second_call_time, first_call_time / 2)

        # Results should be identical
        self.assertEqual(features1.feature_level, features2.feature_level)

    def test_feature_detection_timeout(self):
        """Test timeout protection in feature detection"""
        # Mock a slow feature test
        original_test = self.detector._test_multi_model

        def slow_test():
            time.sleep(2.0)  # Longer than timeout
            return True

        self.detector._test_multi_model = slow_test

        # Should complete within timeout + small buffer
        start_time = time.time()
        features = self.detector.detect_features(force_refresh=True)
        elapsed = time.time() - start_time

        self.assertLess(elapsed, 2.0)  # Should timeout before 2 seconds
        self.assertFalse(features.multi_model)  # Should be False due to timeout

        # Restore original method
        self.detector._test_multi_model = original_test

    def test_force_refresh(self):
        """Test force refresh bypasses cache"""
        # Set up cache
        self.detector.detect_features()
        original_cache = self.detector.detection_cache

        # Force refresh should bypass cache
        with patch.object(self.detector, '_test_multi_model', return_value=True):
            features = self.detector.detect_features(force_refresh=True)

        # Cache should be updated
        self.assertNotEqual(original_cache, self.detector.detection_cache)


class TestCommandRouter(unittest.TestCase):
    """Test command routing logic"""

    def setUp(self):
        # Create feature availability scenarios
        self.full_features = FeatureAvailability(
            multi_model=True, smart_context=True, code_analysis=True
        )

        self.partial_features = FeatureAvailability(
            multi_model=True, smart_context=False, code_analysis=True
        )

        self.basic_features = FeatureAvailability(
            multi_model=False, smart_context=False, code_analysis=False
        )

    def test_full_features_routing(self):
        """Test routing with all features available"""
        router = CommandRouter(self.full_features)

        # All enhanced commands should be available
        self.assertTrue(router.can_handle_enhanced_command('analyze'))
        self.assertTrue(router.can_handle_enhanced_command('models'))
        self.assertTrue(router.can_handle_enhanced_command('context'))
        self.assertTrue(router.can_handle_enhanced_command('smart'))
        self.assertTrue(router.can_handle_enhanced_command('git-commit'))

    def test_partial_features_routing(self):
        """Test routing with partial features available"""
        router = CommandRouter(self.partial_features)

        # Only some commands should be available
        self.assertTrue(router.can_handle_enhanced_command('analyze'))
        self.assertTrue(router.can_handle_enhanced_command('models'))
        self.assertFalse(router.can_handle_enhanced_command('context'))
        self.assertFalse(router.can_handle_enhanced_command('smart'))  # Requires both
        self.assertTrue(router.can_handle_enhanced_command('git-commit'))

    def test_basic_features_routing(self):
        """Test routing with no enhanced features"""
        router = CommandRouter(self.basic_features)

        # No enhanced commands should be available
        self.assertFalse(router.can_handle_enhanced_command('analyze'))
        self.assertFalse(router.can_handle_enhanced_command('models'))
        self.assertFalse(router.can_handle_enhanced_command('context'))
        self.assertFalse(router.can_handle_enhanced_command('smart'))
        self.assertFalse(router.can_handle_enhanced_command('git-commit'))

    def test_unknown_command(self):
        """Test handling of unknown commands"""
        router = CommandRouter(self.full_features)

        # Unknown commands should return False
        self.assertFalse(router.can_handle_enhanced_command('unknown'))
        self.assertFalse(router.can_handle_enhanced_command(''))


class TestHardwareProfiler(unittest.TestCase):
    """Test hardware profiling functionality"""

    def setUp(self):
        self.profiler = HardwareProfiler()

    def test_system_profiling(self):
        """Test basic system profiling"""
        profile = self.profiler.get_system_profile()

        # Should return valid SystemProfile
        self.assertIsInstance(profile, SystemProfile)

        # Should have reasonable values
        self.assertGreater(profile.ram_gb, 0)
        self.assertGreater(profile.cpu_cores, 0)
        self.assertGreater(profile.storage_gb, 0)
        self.assertIn(
            profile.feature_level,
            [FeatureLevel.BASIC, FeatureLevel.STANDARD, FeatureLevel.ADVANCED],
        )

    def test_profile_caching(self):
        """Test profile caching mechanism"""
        # First call
        start_time = time.time()
        profile1 = self.profiler.get_system_profile()
        first_call_time = time.time() - start_time

        # Second call (should use cache)
        start_time = time.time()
        profile2 = self.profiler.get_system_profile()
        second_call_time = time.time() - start_time

        # Second call should be much faster
        self.assertLess(second_call_time, first_call_time / 2)

        # Profiles should be identical
        self.assertEqual(profile1.ram_gb, profile2.ram_gb)
        self.assertEqual(profile1.cpu_cores, profile2.cpu_cores)

    def test_feature_level_determination(self):
        """Test feature level determination logic"""
        # Test advanced level
        advanced_level = self.profiler._determine_feature_level(16.0, 8)
        self.assertEqual(advanced_level, FeatureLevel.ADVANCED)

        # Test standard level
        standard_level = self.profiler._determine_feature_level(8.0, 4)
        self.assertEqual(standard_level, FeatureLevel.STANDARD)

        # Test basic level
        basic_level = self.profiler._determine_feature_level(4.0, 2)
        self.assertEqual(basic_level, FeatureLevel.BASIC)

        # Test edge cases
        edge_basic = self.profiler._determine_feature_level(2.0, 1)
        self.assertEqual(edge_basic, FeatureLevel.BASIC)


class TestResourceMonitor(unittest.TestCase):
    """Test resource monitoring functionality"""

    def setUp(self):
        self.monitor = ResourceMonitor()

    def test_resource_usage_monitoring(self):
        """Test resource usage monitoring"""
        usage = self.monitor.monitor_resource_usage()

        # Should return valid ResourceUsage
        self.assertIsInstance(usage, ResourceUsage)

        # Should have reasonable values
        self.assertGreaterEqual(usage.ram_percent, 0)
        self.assertLessEqual(usage.ram_percent, 100)
        self.assertGreaterEqual(usage.cpu_percent, 0)
        self.assertIn(
            usage.pressure_level,
            [
                ResourcePressure.LOW,
                ResourcePressure.MEDIUM,
                ResourcePressure.HIGH,
                ResourcePressure.CRITICAL,
            ],
        )

    def test_scan_strategy_adaptation(self):
        """Test scan strategy adaptation based on system profile"""
        strategy = self.monitor.get_scan_strategy()

        # Should return valid ScanStrategy
        self.assertIsInstance(strategy, ScanStrategy)

        # Should have reasonable values
        self.assertGreater(strategy.depth, 0)
        self.assertGreater(strategy.batch_size, 0)
        self.assertGreater(strategy.max_file_size_kb, 0)
        self.assertIn(strategy.compression, ["none", "semantic", "advanced"])

    def test_feature_recommendations(self):
        """Test feature recommendations based on system state"""
        recommendations = self.monitor.get_recommended_feature_set()

        # Should return dict with expected keys
        expected_keys = [
            'context_scanning',
            'semantic_compression',
            'parallel_processing',
            'background_monitoring',
            'detailed_progress',
            'model_stability_testing',
        ]

        for key in expected_keys:
            self.assertIn(key, recommendations)
            self.assertIsInstance(recommendations[key], bool)

    def test_throttling_detection(self):
        """Test throttling detection logic"""
        # Should not throw exceptions
        should_throttle = self.monitor.should_throttle_features()
        self.assertIsInstance(should_throttle, bool)

    def test_progress_reporting(self):
        """Test progress reporting functionality"""
        strategy = self.monitor.get_scan_strategy()

        report = self.monitor.generate_progress_report(
            current=50,
            total=100,
            current_size_mb=25.0,
            total_size_mb=50.0,
            file_types={'py': 30, 'js': 15, 'md': 5},
            strategy=strategy,
        )

        # Should return valid ProgressReport
        self.assertEqual(report.current_files, 50)
        self.assertEqual(report.total_files, 100)
        self.assertEqual(report.progress_percent, 50.0)
        self.assertGreater(report.elapsed_seconds, 0)


class TestEnhancedCLI(unittest.TestCase):
    """Test enhanced CLI system integration"""

    def setUp(self):
        # Mock Phase 1 systems to avoid import errors
        with patch('xencode.enhanced_cli_system.MULTI_MODEL_AVAILABLE', False), patch(
            'xencode.enhanced_cli_system.SMART_CONTEXT_AVAILABLE', False
        ), patch('xencode.enhanced_cli_system.CODE_ANALYSIS_AVAILABLE', False):
            self.cli = EnhancedXencodeCLI()

    def test_cli_initialization(self):
        """Test CLI initialization with cold start optimization"""
        # Should initialize without errors
        self.assertIsNotNone(self.cli.security_manager)
        self.assertIsNotNone(self.cli.context_cache)
        self.assertIsNotNone(self.cli.model_stability)
        self.assertIsNotNone(self.cli.feature_detector)
        self.assertIsNotNone(self.cli.features)

    def test_argument_parser_creation(self):
        """Test argument parser creation"""
        parser = self.cli.create_parser()

        # Should create valid parser
        self.assertIsNotNone(parser)

        # Test parsing various arguments
        args = parser.parse_args(['--feature-status'])
        self.assertTrue(args.feature_status)

        args = parser.parse_args(['--refresh-features'])
        self.assertTrue(args.refresh_features)

        args = parser.parse_args(['test query'])
        self.assertEqual(args.query, 'test query')

    def test_feature_status_command(self):
        """Test feature status command"""
        result = self.cli._handle_feature_status()

        # Should return string with status information
        self.assertIsInstance(result, str)
        self.assertIn('Feature Status', result)
        self.assertIn('Core Systems', result)
        self.assertIn('Enhanced Systems', result)

    def test_refresh_features_command(self):
        """Test refresh features command"""
        result = self.cli._handle_refresh_features()

        # Should return success message
        self.assertIsInstance(result, str)
        self.assertIn('refreshed', result.lower())

    def test_enhanced_command_fallback(self):
        """Test enhanced command fallback when features unavailable"""
        # Test analyze command without code analysis
        result = self.cli.handle_analyze_command('/tmp')
        self.assertIn('not available', result)

        # Test models command without multi-model system
        result = self.cli.handle_models_command()
        self.assertIn('not available', result)

        # Test context command without smart context
        result = self.cli.handle_context_command()
        self.assertIn('not available', result)

    def test_process_enhanced_args_legacy_fallback(self):
        """Test processing args with legacy fallback"""
        parser = self.cli.create_parser()

        # Test legacy query (should return None for legacy handling)
        args = parser.parse_args(['test query'])
        result = self.cli.process_enhanced_args(args)
        self.assertIsNone(result)

        # Test feature status (should return result)
        args = parser.parse_args(['--feature-status'])
        result = self.cli.process_enhanced_args(args)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)


class TestColdStartOptimization(unittest.TestCase):
    """Test cold start optimization and progressive warm-up"""

    def test_initialization_timing(self):
        """Test that initialization completes within reasonable time"""
        start_time = time.time()

        with patch('xencode.enhanced_cli_system.MULTI_MODEL_AVAILABLE', False), patch(
            'xencode.enhanced_cli_system.SMART_CONTEXT_AVAILABLE', False
        ), patch('xencode.enhanced_cli_system.CODE_ANALYSIS_AVAILABLE', False):
            cli = EnhancedXencodeCLI()

        initialization_time = time.time() - start_time

        # Should complete within 5 seconds
        self.assertLess(initialization_time, 5.0)

        # Core components should be ready immediately
        self.assertIsNotNone(cli.security_manager)
        self.assertIsNotNone(cli.context_cache)
        self.assertIsNotNone(cli.model_stability)

    def test_background_initialization(self):
        """Test background initialization of enhanced features"""
        with patch('xencode.enhanced_cli_system.MULTI_MODEL_AVAILABLE', True), patch(
            'xencode.enhanced_cli_system.SMART_CONTEXT_AVAILABLE', True
        ), patch('xencode.enhanced_cli_system.CODE_ANALYSIS_AVAILABLE', True):
            # Mock the Phase 1 systems
            with patch('xencode.enhanced_cli_system.MultiModelManager') as mock_mm, patch(
                'xencode.enhanced_cli_system.SmartContextManager'
            ) as mock_sc, patch('xencode.enhanced_cli_system.CodeAnalyzer') as mock_ca:
                cli = EnhancedXencodeCLI()

                # Wait for background initialization
                if hasattr(cli, '_init_thread'):
                    cli._init_thread.join(timeout=5)

                # Enhanced features should be initialized
                self.assertIsNotNone(getattr(cli, 'multi_model', None))
                self.assertIsNotNone(getattr(cli, 'smart_context', None))
                self.assertIsNotNone(getattr(cli, 'code_analyzer', None))


class TestSecurityIntegration(unittest.TestCase):
    """Test security integration with other components"""

    def setUp(self):
        self.security_manager = SecurityManager()

    def test_path_validation_integration(self):
        """Test path validation integration"""
        # Valid paths
        self.assertTrue(self.security_manager.validate_project_path('/tmp'))
        self.assertTrue(self.security_manager.validate_project_path('.'))

        # Invalid paths
        self.assertFalse(self.security_manager.validate_project_path('../../../etc'))
        self.assertFalse(self.security_manager.validate_project_path('/nonexistent'))

    def test_commit_message_sanitization(self):
        """Test commit message sanitization"""
        # Safe message
        safe_msg = "Fix bug in authentication module"
        sanitized = self.security_manager.sanitize_commit_message(safe_msg)
        self.assertEqual(safe_msg, sanitized)

        # Dangerous message
        dangerous_msg = "Fix bug $(rm -rf /)"
        sanitized = self.security_manager.sanitize_commit_message(dangerous_msg)
        self.assertNotIn('$(rm -rf /)', sanitized)
        self.assertIn('$(SANITIZED)', sanitized)


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios"""

    def test_full_system_integration(self):
        """Test full system working together"""
        # Initialize all components
        with patch('xencode.enhanced_cli_system.MULTI_MODEL_AVAILABLE', False), patch(
            'xencode.enhanced_cli_system.SMART_CONTEXT_AVAILABLE', False
        ), patch('xencode.enhanced_cli_system.CODE_ANALYSIS_AVAILABLE', False):
            cli = EnhancedXencodeCLI()
            monitor = ResourceMonitor()

            # Test feature detection
            features = cli.features
            self.assertIsInstance(features, FeatureAvailability)

            # Test resource monitoring
            usage = monitor.monitor_resource_usage()
            self.assertIsInstance(usage, ResourceUsage)

            # Test CLI argument processing
            parser = cli.create_parser()
            args = parser.parse_args(['--feature-status'])
            result = cli.process_enhanced_args(args)
            self.assertIsInstance(result, str)

    def test_graceful_degradation_scenario(self):
        """Test graceful degradation when features fail"""
        with patch('xencode.enhanced_cli_system.MULTI_MODEL_AVAILABLE', True), patch(
            'xencode.enhanced_cli_system.MultiModelManager',
            side_effect=Exception("Import failed"),
        ):
            # Should still initialize successfully
            cli = EnhancedXencodeCLI()

            # Features should be detected as unavailable
            self.assertFalse(cli.features.multi_model)

            # Commands should gracefully fail
            result = cli.handle_models_command()
            self.assertIn('not available', result)

    def test_resource_pressure_adaptation(self):
        """Test system adaptation under resource pressure"""
        monitor = ResourceMonitor()

        # Get base strategy
        base_strategy = monitor.get_scan_strategy()

        # Simulate high resource pressure
        with patch.object(monitor, 'monitor_resource_usage') as mock_usage:
            mock_usage.return_value = ResourceUsage(
                ram_mb=8000,
                ram_percent=95.0,
                cpu_percent=90.0,
                storage_mb=1000000,
                storage_percent=85.0,
                pressure_level=ResourcePressure.HIGH,
                is_throttled=True,
            )

            # Strategy should be adjusted for pressure
            adjusted_strategy = monitor.adjust_strategy_for_pressure(base_strategy)

            # Should have reduced batch size and increased pauses
            self.assertLessEqual(adjusted_strategy.batch_size, base_strategy.batch_size)
            self.assertGreaterEqual(
                adjusted_strategy.pause_between_batches_ms,
                base_strategy.pause_between_batches_ms,
            )


def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running Integration Infrastructure Tests")
    print("=" * 50)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestFeatureDetector,
        TestCommandRouter,
        TestHardwareProfiler,
        TestResourceMonitor,
        TestEnhancedCLI,
        TestColdStartOptimization,
        TestSecurityIntegration,
        TestIntegrationScenarios,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n📊 Test Results:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")

    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")

    return success


if __name__ == "__main__":
    run_integration_tests()
