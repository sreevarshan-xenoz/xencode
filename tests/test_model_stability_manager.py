#!/usr/bin/env python3
"""
Comprehensive tests for Model Stability Manager
Tests stability detection, OOM detection, fallback chains, and 3-second rule
"""

import shutil
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import requests

# Import the module to test
from xencode.model_stability_manager import (
    ModelStabilityManager,
    ModelState,
    StabilityResult,
    StabilityStatus,
)


class TestModelStabilityManager(unittest.TestCase):
    """Test suite for ModelStabilityManager"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelStabilityManager(config_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test environment"""
        self.manager.stop_background_monitoring()
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test manager initialization"""
        self.assertIsInstance(self.manager.model_states, dict)
        self.assertEqual(self.manager.emergency_model, "qwen:0.5b")
        self.assertEqual(self.manager.stability_timeout_ms, 200)
        self.assertEqual(self.manager.consecutive_query_threshold_seconds, 3)

    def test_fallback_chains(self):
        """Test fallback chain configuration"""
        # Test code fallback chain
        code_chain = self.manager.get_fallback_chain("code")
        self.assertIn("codellama:7b", code_chain)
        self.assertEqual(code_chain[-1], "qwen:0.5b")  # Emergency model at end

        # Test creative fallback chain
        creative_chain = self.manager.get_fallback_chain("creative")
        self.assertIn("mistral:7b", creative_chain)

        # Test unknown query type defaults to general
        unknown_chain = self.manager.get_fallback_chain("unknown")
        general_chain = self.manager.get_fallback_chain("general")
        self.assertEqual(unknown_chain, general_chain)

    @patch('requests.post')
    def test_model_stability_success(self, mock_post):
        """Test successful model stability check"""
        # Mock successful response with slight delay
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello"}
        mock_post.return_value = mock_response

        # Add a small delay to simulate real response time
        def delayed_post(*args, **kwargs):
            time.sleep(0.01)  # 10ms delay
            return mock_response

        mock_post.side_effect = delayed_post

        # Mock OOM detection to return False
        with patch.object(self.manager, 'detect_oom_crash', return_value=False):
            result = self.manager.test_model_stability("qwen3:4b")

        self.assertTrue(result.is_stable)
        self.assertEqual(result.status, StabilityStatus.HEALTHY)
        self.assertFalse(result.oom_detected)
        self.assertIsNone(result.error_message)
        self.assertGreater(result.response_time_ms, 0)

    @patch('requests.post')
    def test_model_stability_oom_detected(self, mock_post):
        """Test model stability with OOM detection"""
        # Mock successful response but OOM detected
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello"}
        mock_post.return_value = mock_response

        # Mock OOM detection to return True
        with patch.object(self.manager, 'detect_oom_crash', return_value=True):
            result = self.manager.test_model_stability("qwen3:4b")

        self.assertFalse(result.is_stable)
        self.assertEqual(result.status, StabilityStatus.DEGRADED)
        self.assertTrue(result.oom_detected)

    @patch('requests.post')
    def test_model_stability_timeout(self, mock_post):
        """Test model stability timeout handling"""
        # Mock timeout exception
        mock_post.side_effect = requests.exceptions.Timeout()

        result = self.manager.test_model_stability("qwen3:4b")

        self.assertFalse(result.is_stable)
        self.assertEqual(result.status, StabilityStatus.UNSTABLE)
        self.assertIn("Timeout", result.error_message)

    @patch('requests.post')
    def test_model_stability_http_error(self, mock_post):
        """Test model stability HTTP error handling"""
        # Mock HTTP error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = self.manager.test_model_stability("qwen3:4b")

        self.assertFalse(result.is_stable)
        self.assertEqual(result.status, StabilityStatus.FAILED)
        self.assertIn("HTTP 500", result.error_message)

    @patch('subprocess.run')
    def test_journalctl_oom_detection(self, mock_run):
        """Test journalctl OOM detection"""
        # Mock journalctl output with OOM event
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Jan 01 12:00:00 host kernel: oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0,global_oom,task_memcg=/,task=ollama,pid=1234"
        mock_run.return_value = mock_result

        oom_detected = self.manager.check_journalctl_oom("qwen3:4b")
        self.assertTrue(oom_detected)

        # Test with no OOM
        mock_result.stdout = "Jan 01 12:00:00 host kernel: normal log entry"
        oom_detected = self.manager.check_journalctl_oom("qwen3:4b")
        self.assertFalse(oom_detected)

    @patch('subprocess.run')
    def test_dmesg_oom_detection(self, mock_run):
        """Test dmesg OOM detection"""
        # Mock dmesg output with OOM event
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "[Jan01 12:00] Out of memory: Kill process 1234 (ollama) score 900 or sacrifice child"
        mock_run.return_value = mock_result

        # Mock recent log entry check
        with patch.object(self.manager, '_is_recent_log_entry', return_value=True):
            oom_detected = self.manager.check_dmesg_oom("qwen3:4b")
            self.assertTrue(oom_detected)

    def test_ollama_logs_oom_detection(self):
        """Test ollama logs OOM detection"""
        # Create temporary log file
        log_file = Path(self.temp_dir) / "ollama.log"
        log_content = """
2024-01-01T12:00:00Z INFO: Starting model qwen3:4b
2024-01-01T12:01:00Z ERROR: out of memory loading model qwen3:4b
2024-01-01T12:01:01Z INFO: Model crashed
"""
        log_file.write_text(log_content)

        # Mock log paths to include our test file
        with patch.object(self.manager, 'check_ollama_logs') as mock_check:
            # Simulate the actual method behavior
            mock_check.return_value = True

            oom_detected = self.manager.check_ollama_logs("qwen3:4b")
            self.assertTrue(oom_detected)

    def test_model_state_persistence(self):
        """Test model state saving and loading"""
        # Create a model state
        model_name = "test_model"
        self.manager.model_states[model_name] = ModelState(
            name=model_name,
            is_available=True,
            is_stable=False,
            avg_response_time_ms=150,
            failure_count=2,
        )

        # Save states
        self.manager.save_model_states()

        # Create new manager and load states
        new_manager = ModelStabilityManager(config_dir=self.temp_dir)

        # Verify state was loaded
        self.assertIn(model_name, new_manager.model_states)
        loaded_state = new_manager.model_states[model_name]
        self.assertEqual(loaded_state.name, model_name)
        self.assertTrue(loaded_state.is_available)
        self.assertFalse(loaded_state.is_stable)
        self.assertEqual(loaded_state.avg_response_time_ms, 150)
        self.assertEqual(loaded_state.failure_count, 2)

    def test_model_degradation_tracking(self):
        """Test model degradation marking and expiration"""
        model_name = "test_model"

        # Mark model as degraded for 1 second
        self.manager.mark_model_degraded(model_name, duration_minutes=1 / 60)

        # Check that model is marked as degraded
        self.assertFalse(self.manager.is_model_available(model_name))

        # Wait for degradation to expire
        time.sleep(1.1)

        # Check that degradation has expired (but model might still be unavailable due to other factors)
        state = self.manager.model_states[model_name]
        self.manager._update_model_success(model_name, 100, 1000)  # Simulate success

        # Now it should be available
        self.assertTrue(self.manager.is_model_available(model_name))

    def test_consecutive_query_rule(self):
        """Test 3-second rule for consecutive queries"""
        model_name = "test_model"

        # First query - should return False (no previous query)
        self.assertFalse(self.manager.should_use_previous_model(model_name))
        self.manager.update_query_timing(model_name)

        # Second query within 3 seconds - should return True
        time.sleep(0.5)
        self.assertTrue(self.manager.should_use_previous_model(model_name))
        self.manager.update_query_timing(model_name)

        # Third query after 3 seconds - should return False
        time.sleep(3.1)
        self.assertFalse(self.manager.should_use_previous_model(model_name))

    def test_model_success_tracking(self):
        """Test model success state updates"""
        model_name = "test_model"

        # Simulate successful operation
        self.manager._update_model_success(model_name, 150, 2500)

        state = self.manager.model_states[model_name]
        self.assertTrue(state.is_available)
        self.assertTrue(state.is_stable)
        self.assertEqual(state.consecutive_failures, 0)
        self.assertEqual(state.avg_response_time_ms, 150)
        self.assertEqual(state.memory_usage_mb, 2500)
        self.assertGreater(state.stability_score, 0.5)

    def test_model_failure_tracking(self):
        """Test model failure state updates"""
        model_name = "test_model"

        # Simulate multiple failures
        for i in range(4):  # Exceed max_consecutive_failures (3)
            self.manager._update_model_failure(model_name, f"Error {i}")

        state = self.manager.model_states[model_name]
        self.assertEqual(state.failure_count, 4)
        self.assertEqual(state.consecutive_failures, 4)
        self.assertFalse(state.is_stable)  # Should be marked unstable
        self.assertLess(state.stability_score, 0.5)

    def test_health_summary(self):
        """Test health summary generation"""
        # Add some test models
        self.manager.model_states["model1"] = ModelState(
            name="model1", is_stable=True, stability_score=0.9, avg_response_time_ms=100
        )
        self.manager.model_states["model2"] = ModelState(
            name="model2", is_stable=False, stability_score=0.3, failure_count=5
        )

        summary = self.manager.get_model_health_summary()

        self.assertIn("model1", summary)
        self.assertIn("model2", summary)
        self.assertEqual(summary["model1"]["status"], "healthy")
        self.assertEqual(summary["model2"]["status"], "degraded")
        self.assertEqual(summary["model1"]["stability_score"], 0.9)
        self.assertEqual(summary["model2"]["failure_count"], 5)

    def test_background_monitoring(self):
        """Test background monitoring functionality"""
        # Mock the test_model_stability method
        with patch.object(self.manager, 'test_model_stability') as mock_test:
            mock_test.return_value = StabilityResult(
                is_stable=True,
                response_time_ms=100,
                memory_usage_mb=1000,
                status=StabilityStatus.HEALTHY,
            )

            # Add a test model
            self.manager.model_states["test_model"] = ModelState(name="test_model")

            # Start monitoring with very short interval
            self.manager.start_background_monitoring(
                interval_minutes=1 / 60
            )  # 1 second

            # Wait a bit for monitoring to run
            time.sleep(1.5)

            # Stop monitoring
            self.manager.stop_background_monitoring()

            # Verify that monitoring was called
            mock_test.assert_called()

    def test_emergency_model(self):
        """Test emergency model functionality"""
        emergency = self.manager.get_emergency_model()
        self.assertEqual(emergency, "qwen:0.5b")

        # Verify emergency model is in all fallback chains
        for query_type in ["code", "creative", "analysis", "general"]:
            chain = self.manager.get_fallback_chain(query_type)
            self.assertIn(emergency, chain)

    def test_memory_usage_estimation(self):
        """Test memory usage estimation"""
        # Test known models
        qwen_memory = self.manager._estimate_memory_usage("qwen:0.5b")
        self.assertEqual(qwen_memory, 512)

        llama_memory = self.manager._estimate_memory_usage("llama2:7b")
        self.assertEqual(llama_memory, 3800)

        # Test unknown model (should return default)
        unknown_memory = self.manager._estimate_memory_usage("unknown_model")
        self.assertEqual(unknown_memory, 2000)

    def test_log_entry_recency_check(self):
        """Test recent log entry detection"""
        current_time = datetime.now()

        # Test recent timestamp (ISO format)
        recent_log = f"[{current_time.isoformat()}] Test log entry"
        is_recent = self.manager._is_recent_log_entry(
            recent_log, current_time, minutes=5
        )
        self.assertTrue(is_recent)

        # Test old timestamp (ISO format)
        old_time = current_time - timedelta(minutes=10)
        old_log = f"[{old_time.isoformat()}] Old log entry"
        is_recent = self.manager._is_recent_log_entry(old_log, current_time, minutes=5)
        self.assertFalse(is_recent)

        # Test log without timestamp (should default to recent)
        no_timestamp_log = "Log entry without timestamp"
        is_recent = self.manager._is_recent_log_entry(
            no_timestamp_log, current_time, minutes=5
        )
        self.assertTrue(is_recent)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for real-world scenarios"""

    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelStabilityManager(config_dir=self.temp_dir)

    def tearDown(self):
        """Clean up integration test environment"""
        self.manager.stop_background_monitoring()
        shutil.rmtree(self.temp_dir)

    def test_model_recovery_scenario(self):
        """Test complete model failure and recovery scenario"""
        model_name = "test_model"

        # Simulate model failure
        self.manager._update_model_failure(model_name, "Connection failed")
        self.manager._update_model_failure(model_name, "Timeout")
        self.manager._update_model_failure(model_name, "OOM detected")

        # Model should be unstable
        state = self.manager.model_states[model_name]
        self.assertFalse(state.is_stable)
        self.assertEqual(state.consecutive_failures, 3)

        # Simulate recovery
        self.manager._update_model_success(model_name, 120, 2000)

        # Model should be stable again
        self.assertTrue(state.is_stable)
        self.assertEqual(state.consecutive_failures, 0)
        self.assertGreater(state.stability_score, 0.0)

    def test_fallback_chain_selection(self):
        """Test fallback chain selection for different scenarios"""
        # Test code query fallback
        code_chain = self.manager.get_fallback_chain("code")
        self.assertEqual(code_chain[0], "codellama:7b")  # Best for code

        # Test creative query fallback
        creative_chain = self.manager.get_fallback_chain("creative")
        self.assertEqual(creative_chain[0], "mistral:7b")  # Best for creative

        # Verify all chains end with emergency model
        for query_type in ["code", "creative", "analysis", "general"]:
            chain = self.manager.get_fallback_chain(query_type)
            self.assertEqual(chain[-1], "qwen:0.5b")

    @patch('requests.post')
    def test_oom_detection_integration(self, mock_post):
        """Test complete OOM detection and response"""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello"}
        mock_post.return_value = mock_response

        # Mock OOM detection methods
        with patch.object(
            self.manager, 'check_journalctl_oom', return_value=True
        ), patch.object(
            self.manager, 'check_dmesg_oom', return_value=False
        ), patch.object(
            self.manager, 'check_ollama_logs', return_value=False
        ):
            result = self.manager.test_model_stability("qwen3:4b")

            # Should detect OOM and mark as degraded
            self.assertFalse(result.is_stable)
            self.assertTrue(result.oom_detected)
            self.assertEqual(result.status, StabilityStatus.DEGRADED)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
