#!/usr/bin/env python3
"""
Model Stability Manager for Xencode Phase 2
Handles model health monitoring, OOM detection, and fallback chains
"""

import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import requests


class StabilityStatus(Enum):
    """Model stability status levels"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNSTABLE = "unstable"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class StabilityResult:
    """Result of model stability test"""

    is_stable: bool
    response_time_ms: int
    memory_usage_mb: int
    error_message: Optional[str] = None
    oom_detected: bool = False
    status: StabilityStatus = StabilityStatus.UNKNOWN


@dataclass
class ModelState:
    """Tracks model performance and health over time"""

    name: str
    is_available: bool = True
    is_stable: bool = True
    last_tested: Optional[datetime] = None

    # Performance metrics
    avg_response_time_ms: int = 0
    memory_usage_mb: int = 0
    stability_score: float = 1.0  # 0.0 - 1.0

    # Degradation tracking
    is_degraded: bool = False
    degraded_until: Optional[datetime] = None
    failure_count: int = 0
    last_failure: Optional[str] = None
    consecutive_failures: int = 0

    # Conversational flow tracking
    last_query_time: Optional[datetime] = None
    consecutive_query_count: int = 0


@dataclass
class QueryContext:
    """Context for query type and model selection"""

    query_type: str  # "code", "creative", "analysis", "explanation", "general"
    confidence: float
    keywords: List[str]
    suggested_model: str
    reasoning: str


class ModelStabilityManager:
    """
    Manages model health, stability testing, and fallback chains
    Implements cross-platform OOM detection and degradation tracking
    """

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or Path.home() / ".xencode" / "stability")
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Model state tracking
        self.model_states: Dict[str, ModelState] = {}
        self.state_file = self.config_dir / "model_states.json"

        # Configuration
        self.stability_timeout_ms = 200  # Warm-up query timeout
        self.degradation_duration_minutes = 5  # How long to mark model as degraded
        self.max_consecutive_failures = 3  # Failures before marking unstable
        self.consecutive_query_threshold_seconds = 3  # 3-second rule

        # Fallback chains by query type
        self.fallback_chains = {
            "code": [
                "codellama:7b",
                "qwen3:4b",
                "llama2:7b",
                "mistral:7b",
                "qwen:0.5b",
            ],
            "creative": ["mistral:7b", "llama2:7b", "qwen3:4b", "qwen:0.5b"],
            "analysis": ["llama2:7b", "mistral:7b", "qwen3:4b", "qwen:0.5b"],
            "explanation": ["qwen3:4b", "llama2:7b", "mistral:7b", "qwen:0.5b"],
            "general": ["qwen3:4b", "llama2:7b", "mistral:7b", "qwen:0.5b"],
        }

        # Emergency lightweight model
        self.emergency_model = "qwen:0.5b"

        # Load existing state
        self.load_model_states()

        # Background monitoring
        self._monitoring_active = False
        self._monitor_thread = None

    def load_model_states(self) -> None:
        """Load model states from persistent storage"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)

                for model_name, state_data in data.items():
                    state = ModelState(name=model_name)

                    # Load basic fields
                    state.is_available = state_data.get('is_available', True)
                    state.is_stable = state_data.get('is_stable', True)
                    state.avg_response_time_ms = state_data.get(
                        'avg_response_time_ms', 0
                    )
                    state.memory_usage_mb = state_data.get('memory_usage_mb', 0)
                    state.stability_score = state_data.get('stability_score', 1.0)
                    state.is_degraded = state_data.get('is_degraded', False)
                    state.failure_count = state_data.get('failure_count', 0)
                    state.consecutive_failures = state_data.get(
                        'consecutive_failures', 0
                    )
                    state.last_failure = state_data.get('last_failure')
                    state.consecutive_query_count = state_data.get(
                        'consecutive_query_count', 0
                    )

                    # Parse datetime fields
                    if state_data.get('last_tested'):
                        state.last_tested = datetime.fromisoformat(
                            state_data['last_tested']
                        )
                    if state_data.get('degraded_until'):
                        state.degraded_until = datetime.fromisoformat(
                            state_data['degraded_until']
                        )
                    if state_data.get('last_query_time'):
                        state.last_query_time = datetime.fromisoformat(
                            state_data['last_query_time']
                        )

                    self.model_states[model_name] = state

        except Exception as e:
            print(f"Warning: Could not load model states: {e}")

    def save_model_states(self) -> None:
        """Save model states to persistent storage"""
        try:
            data = {}
            for model_name, state in self.model_states.items():
                state_data = {
                    'is_available': state.is_available,
                    'is_stable': state.is_stable,
                    'avg_response_time_ms': state.avg_response_time_ms,
                    'memory_usage_mb': state.memory_usage_mb,
                    'stability_score': state.stability_score,
                    'is_degraded': state.is_degraded,
                    'failure_count': state.failure_count,
                    'consecutive_failures': state.consecutive_failures,
                    'last_failure': state.last_failure,
                    'consecutive_query_count': state.consecutive_query_count,
                }

                # Serialize datetime fields
                if state.last_tested:
                    state_data['last_tested'] = state.last_tested.isoformat()
                if state.degraded_until:
                    state_data['degraded_until'] = state.degraded_until.isoformat()
                if state.last_query_time:
                    state_data['last_query_time'] = state.last_query_time.isoformat()

                data[model_name] = state_data

            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save model states: {e}")

    def test_model_stability(self, model_name: str) -> StabilityResult:
        """
        Test model stability with warm-up query and OOM detection
        Returns comprehensive stability assessment
        """
        start_time = time.time()

        try:
            # Warm-up query with timeout
            test_payload = {"model": model_name, "prompt": "Hello", "stream": False}

            response = requests.post(
                "http://localhost:11434/api/generate",
                json=test_payload,
                timeout=self.stability_timeout_ms / 1000.0,
            )

            response_time_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                # Check for OOM after successful response
                oom_detected = self.detect_oom_crash(model_name)

                # Estimate memory usage (simplified)
                memory_usage_mb = self._estimate_memory_usage(model_name)

                # Update model state
                self._update_model_success(
                    model_name, response_time_ms, memory_usage_mb
                )

                return StabilityResult(
                    is_stable=not oom_detected,
                    response_time_ms=response_time_ms,
                    memory_usage_mb=memory_usage_mb,
                    oom_detected=oom_detected,
                    status=(
                        StabilityStatus.DEGRADED
                        if oom_detected
                        else StabilityStatus.HEALTHY
                    ),
                )
            else:
                # HTTP error
                error_msg = f"HTTP {response.status_code}"
                self._update_model_failure(model_name, error_msg)

                return StabilityResult(
                    is_stable=False,
                    response_time_ms=response_time_ms,
                    memory_usage_mb=0,
                    error_message=error_msg,
                    status=StabilityStatus.FAILED,
                )

        except requests.exceptions.Timeout:
            error_msg = f"Timeout after {self.stability_timeout_ms}ms"
            self._update_model_failure(model_name, error_msg)

            return StabilityResult(
                is_stable=False,
                response_time_ms=self.stability_timeout_ms,
                memory_usage_mb=0,
                error_message=error_msg,
                status=StabilityStatus.UNSTABLE,
            )

        except Exception as e:
            error_msg = str(e)
            self._update_model_failure(model_name, error_msg)

            return StabilityResult(
                is_stable=False,
                response_time_ms=int((time.time() - start_time) * 1000),
                memory_usage_mb=0,
                error_message=error_msg,
                status=StabilityStatus.FAILED,
            )

    def detect_oom_crash(self, model_name: str) -> bool:
        """
        Cross-platform OOM detection using multiple methods
        Returns True if OOM is detected for the model
        """
        # Method 1: Check journalctl (systemd systems)
        if self.check_journalctl_oom(model_name):
            return True

        # Method 2: Check dmesg (older/non-systemd systems)
        if self.check_dmesg_oom(model_name):
            return True

        # Method 3: Check ollama logs (universal)
        if self.check_ollama_logs(model_name):
            return True

        return False

    def check_journalctl_oom(self, model_name: str) -> bool:
        """Check journalctl for OOM events related to the model"""
        try:
            # Look for OOM events in the last 5 minutes
            cmd = [
                "journalctl",
                "--since",
                "5 minutes ago",
                "--grep",
                "oom-kill",
                "--no-pager",
                "-q",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            if result.returncode == 0 and result.stdout:
                # Check if ollama or model name appears in OOM logs
                log_content = result.stdout.lower()
                return (
                    "ollama" in log_content
                    or model_name.lower() in log_content
                    or "out of memory" in log_content
                )

        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            pass

        return False

    def check_dmesg_oom(self, model_name: str) -> bool:
        """Check dmesg for OOM events (fallback for non-systemd systems)"""
        try:
            result = subprocess.run(
                ["dmesg", "-T"], capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                lines = result.stdout.split('\n')

                # Look for recent OOM events (last 5 minutes)
                current_time = datetime.now()
                for line in lines:
                    if "oom-kill" in line.lower() or "out of memory" in line.lower():
                        # Try to parse timestamp and check if recent
                        if self._is_recent_log_entry(line, current_time, minutes=5):
                            # Check if related to ollama or model
                            if (
                                "ollama" in line.lower()
                                or model_name.lower() in line.lower()
                            ):
                                return True

        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            pass

        return False

    def check_ollama_logs(self, model_name: str) -> bool:
        """Check ollama logs for OOM or crash indicators"""
        try:
            # Common ollama log locations
            log_paths = [
                "/var/log/ollama.log",
                "/tmp/ollama.log",
                Path.home() / ".ollama" / "logs" / "server.log",
                "/usr/local/var/log/ollama.log",
            ]

            current_time = datetime.now()

            for log_path in log_paths:
                if isinstance(log_path, str):
                    log_path = Path(log_path)

                if log_path.exists():
                    try:
                        with open(log_path, 'r') as f:
                            # Read last 100 lines for performance
                            lines = f.readlines()[-100:]

                        for line in lines:
                            line_lower = line.lower()

                            # Look for OOM indicators
                            oom_indicators = [
                                "out of memory",
                                "oom",
                                "memory allocation failed",
                                "cannot allocate memory",
                                "killed",
                                "segmentation fault",
                            ]

                            if any(
                                indicator in line_lower for indicator in oom_indicators
                            ):
                                if (
                                    model_name.lower() in line_lower
                                    and self._is_recent_log_entry(
                                        line, current_time, minutes=5
                                    )
                                ):
                                    return True

                    except (IOError, PermissionError):
                        continue

        except Exception:
            pass

        return False

    def _is_recent_log_entry(
        self, log_line: str, current_time: datetime, minutes: int = 5
    ) -> bool:
        """Check if log entry is within the specified time window"""
        try:
            # Try to extract timestamp from log line (various formats)
            timestamp_patterns = [
                r'\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]',  # [2024-01-01T12:00:00] or [2024-01-01T12:00:00.123]
                r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]',  # [2024-01-01 12:00:00]
                r'(\w{3} \d{2} \d{2}:\d{2}:\d{2})',  # Jan 01 12:00:00
                r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)',  # ISO format without brackets
            ]

            for pattern in timestamp_patterns:
                match = re.search(pattern, log_line)
                if match:
                    timestamp_str = match.group(1)

                    # Parse timestamp
                    try:
                        if 'T' in timestamp_str:
                            # Handle ISO format with optional microseconds
                            if '.' in timestamp_str:
                                timestamp_str = timestamp_str.split('.')[
                                    0
                                ]  # Remove microseconds
                            log_time = datetime.fromisoformat(timestamp_str)
                        else:
                            # Handle other formats as needed - for now skip
                            continue

                        time_diff = current_time - log_time
                        return time_diff.total_seconds() <= (minutes * 60)

                    except ValueError:
                        continue

            # If no timestamp found, assume recent
            return True

        except Exception:
            return True

    def _estimate_memory_usage(self, model_name: str) -> int:
        """Estimate memory usage for model (simplified implementation)"""
        # Basic size estimates (in MB)
        size_estimates = {
            "qwen:0.5b": 512,
            "qwen3:4b": 2500,
            "llama2:7b": 3800,
            "codellama:7b": 3800,
            "mistral:7b": 4100,
        }

        return size_estimates.get(model_name, 2000)  # Default estimate

    def _update_model_success(
        self, model_name: str, response_time_ms: int, memory_usage_mb: int
    ) -> None:
        """Update model state after successful operation"""
        if model_name not in self.model_states:
            self.model_states[model_name] = ModelState(name=model_name)

        state = self.model_states[model_name]
        state.last_tested = datetime.now()
        state.is_available = True
        state.is_stable = True
        state.consecutive_failures = 0

        # Update performance metrics (exponential moving average)
        if state.avg_response_time_ms == 0:
            state.avg_response_time_ms = response_time_ms
        else:
            state.avg_response_time_ms = int(
                0.7 * state.avg_response_time_ms + 0.3 * response_time_ms
            )

        state.memory_usage_mb = memory_usage_mb

        # Update stability score
        state.stability_score = min(1.0, state.stability_score + 0.1)

        # Check if degradation period has expired
        if state.is_degraded and state.degraded_until:
            if datetime.now() > state.degraded_until:
                state.is_degraded = False
                state.degraded_until = None

        self.save_model_states()

    def _update_model_failure(self, model_name: str, error_message: str) -> None:
        """Update model state after failure"""
        if model_name not in self.model_states:
            self.model_states[model_name] = ModelState(name=model_name)

        state = self.model_states[model_name]
        state.last_tested = datetime.now()
        state.failure_count += 1
        state.consecutive_failures += 1
        state.last_failure = error_message

        # Decrease stability score
        state.stability_score = max(0.0, state.stability_score - 0.2)

        # Mark as unstable if too many consecutive failures
        if state.consecutive_failures >= self.max_consecutive_failures:
            state.is_stable = False

        self.save_model_states()

    def get_fallback_chain(self, query_type: str) -> List[str]:
        """Get ordered fallback chain for query type"""
        return self.fallback_chains.get(
            query_type, self.fallback_chains["general"]
        ).copy()

    def mark_model_degraded(self, model_name: str, duration_minutes: int = 5) -> None:
        """Mark model as degraded for specified duration"""
        if model_name not in self.model_states:
            self.model_states[model_name] = ModelState(name=model_name)

        state = self.model_states[model_name]
        state.is_degraded = True
        state.degraded_until = datetime.now() + timedelta(minutes=duration_minutes)

        self.save_model_states()

    def is_model_available(self, model_name: str) -> bool:
        """Check if model is available and not degraded"""
        if model_name not in self.model_states:
            # Test model if not in state
            result = self.test_model_stability(model_name)
            return result.is_stable

        state = self.model_states[model_name]

        # Check if degradation period has expired
        if state.is_degraded and state.degraded_until:
            if datetime.now() > state.degraded_until:
                state.is_degraded = False
                state.degraded_until = None
                self.save_model_states()

        return state.is_available and state.is_stable and not state.is_degraded

    def get_emergency_model(self) -> str:
        """Get emergency lightweight model"""
        return self.emergency_model

    def should_use_previous_model(
        self, model_name: str, query_context: Optional[str] = None
    ) -> bool:
        """
        Implement 3-second rule for consecutive queries
        Returns True if should use previous model to maintain conversational flow
        """
        if model_name not in self.model_states:
            return False

        state = self.model_states[model_name]

        if state.last_query_time is None:
            return False

        time_since_last = datetime.now() - state.last_query_time

        # If within 3 seconds and context hasn't changed significantly
        if time_since_last.total_seconds() <= self.consecutive_query_threshold_seconds:
            state.consecutive_query_count += 1
            return True
        else:
            state.consecutive_query_count = 0
            return False

    def update_query_timing(self, model_name: str) -> None:
        """Update timing for query to track consecutive queries"""
        if model_name not in self.model_states:
            self.model_states[model_name] = ModelState(name=model_name)

        state = self.model_states[model_name]
        state.last_query_time = datetime.now()
        self.save_model_states()

    def get_model_health_summary(self) -> Dict[str, Dict]:
        """Get comprehensive health summary for all tracked models"""
        summary = {}

        for model_name, state in self.model_states.items():
            summary[model_name] = {
                'status': (
                    'healthy'
                    if state.is_stable and not state.is_degraded
                    else 'degraded'
                ),
                'availability': state.is_available,
                'stability_score': state.stability_score,
                'avg_response_time_ms': state.avg_response_time_ms,
                'memory_usage_mb': state.memory_usage_mb,
                'failure_count': state.failure_count,
                'consecutive_failures': state.consecutive_failures,
                'last_tested': (
                    state.last_tested.isoformat() if state.last_tested else None
                ),
                'is_degraded': state.is_degraded,
                'degraded_until': (
                    state.degraded_until.isoformat() if state.degraded_until else None
                ),
            }

        return summary

    def start_background_monitoring(self, interval_minutes: int = 5) -> None:
        """Start background monitoring of model health"""
        if self._monitoring_active:
            return

        self._monitoring_active = True

        def monitor_loop():
            while self._monitoring_active:
                try:
                    # Test all known models
                    for model_name in list(self.model_states.keys()):
                        if self._monitoring_active:
                            self.test_model_stability(model_name)

                    # Sleep for interval
                    sleep_seconds = int(interval_minutes * 60)
                    for _ in range(sleep_seconds):
                        if not self._monitoring_active:
                            break
                        time.sleep(1)

                except Exception as e:
                    print(f"Background monitoring error: {e}")
                    time.sleep(60)  # Wait before retrying

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_background_monitoring(self) -> None:
        """Stop background monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)


# Example usage and testing
if __name__ == "__main__":
    manager = ModelStabilityManager()

    print("🔍 Model Stability Manager Demo")
    print("=" * 40)

    # Test a model
    test_model = "qwen3:4b"
    print(f"\n🧪 Testing {test_model}...")

    result = manager.test_model_stability(test_model)
    print(f"Stable: {result.is_stable}")
    print(f"Response Time: {result.response_time_ms}ms")
    print(f"Memory Usage: {result.memory_usage_mb}MB")
    print(f"OOM Detected: {result.oom_detected}")
    print(f"Status: {result.status.value}")

    if result.error_message:
        print(f"Error: {result.error_message}")

    # Show fallback chains
    print("\n🔄 Fallback Chains:")
    for query_type in ["code", "creative", "analysis", "general"]:
        chain = manager.get_fallback_chain(query_type)
        print(f"  {query_type}: {' → '.join(chain)}")

    # Show health summary
    print("\n📊 Health Summary:")
    summary = manager.get_model_health_summary()
    for model, health in summary.items():
        print(f"  {model}: {health['status']} (score: {health['stability_score']:.2f})")
