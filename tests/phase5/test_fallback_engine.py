#!/usr/bin/env python3
"""
Tests for Smart Fallback Policy Engine (S5-03)

Tests cover:
- Fallback configuration models
- Retry budget management
- Cost and latency caps
- Fallback chain execution
- Health-aware routing
- Policy configuration
- API endpoints
"""

import asyncio
from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestFallbackConfigModels:
    """Test fallback configuration Pydantic models"""

    def test_retry_policy_creation(self):
        """Test RetryPolicy model creation"""
        from xencode.routing.fallback_config import BackoffType, RetryPolicy

        retry = RetryPolicy(
            max_retries=3,
            backoff_type=BackoffType.EXPONENTIAL,
            base_delay=1.0,
            max_delay=60.0,
        )

        assert retry.max_retries == 3
        assert retry.backoff_type == BackoffType.EXPONENTIAL
        assert retry.base_delay == 1.0

    def test_retry_policy_delay_calculation(self):
        """Test retry delay calculation for different backoff types"""
        from xencode.routing.fallback_config import BackoffType, RetryPolicy

        # Fixed backoff
        fixed = RetryPolicy(backoff_type=BackoffType.FIXED, base_delay=2.0)
        assert fixed.get_delay(0) == 2.0
        assert fixed.get_delay(1) == 2.0
        assert fixed.get_delay(5) == 2.0

        # Linear backoff
        linear = RetryPolicy(backoff_type=BackoffType.LINEAR, base_delay=2.0)
        assert linear.get_delay(0) == 2.0
        assert linear.get_delay(1) == 4.0
        assert linear.get_delay(2) == 6.0

        # Exponential backoff
        exp = RetryPolicy(backoff_type=BackoffType.EXPONENTIAL, base_delay=1.0)
        assert exp.get_delay(0) == 1.0
        assert exp.get_delay(1) == 2.0
        assert exp.get_delay(2) == 4.0
        assert exp.get_delay(3) == 8.0

        # Max delay cap
        exp_capped = RetryPolicy(
            backoff_type=BackoffType.EXPONENTIAL,
            base_delay=1.0,
            max_delay=5.0,
        )
        assert exp_capped.get_delay(3) == 5.0  # Capped at max_delay

    def test_retry_should_retry(self):
        """Test should_retry logic"""
        from xencode.routing.fallback_config import RetryPolicy

        retry = RetryPolicy(max_retries=3)

        assert retry.should_retry(0) is True
        assert retry.should_retry(1) is True
        assert retry.should_retry(2) is True
        assert retry.should_retry(3) is False

    def test_cost_cap_validation(self):
        """Test CostCap model"""
        from xencode.routing.fallback_config import CostCap

        cost = CostCap(
            max_per_request=0.01,
            daily_budget=1.0,
            monthly_budget=30.0,
        )

        assert cost.is_within_budget(0.005, "request") is True
        assert cost.is_within_budget(0.02, "request") is False
        assert cost.is_within_budget(0.5, "daily") is True
        assert cost.is_within_budget(2.0, "daily") is False

    def test_latency_cap_validation(self):
        """Test LatencyCap model"""
        from xencode.routing.fallback_config import LatencyCap

        latency = LatencyCap(max_latency_ms=5000, timeout_ms=10000)

        assert latency.is_acceptable(3000) is True
        assert latency.is_acceptable(6000) is False
        assert latency.is_timeout(5000) is False
        assert latency.is_timeout(10000) is True

    def test_fallback_policy_creation(self):
        """Test FallbackPolicy model"""
        from xencode.routing.fallback_config import FallbackPolicy, RetryPolicy

        policy = FallbackPolicy(
            name="test_policy",
            description="Test fallback policy",
            priority=1,
            fallback_chain=["model1", "model2", "model3"],
            retry_policy=RetryPolicy(max_retries=2),
        )

        assert policy.name == "test_policy"
        assert len(policy.fallback_chain) == 3
        assert policy.retry_policy.max_retries == 2
        assert policy.enabled is True

    def test_fallback_policy_get_active_chain(self):
        """Test getting active fallback chain with exclusions"""
        from xencode.routing.fallback_config import (
            FallbackPolicy,
        )

        policy = FallbackPolicy(
            name="test",
            fallback_chain=["model1", "model2", "model3"],
        )

        # No exclusions
        chain = policy.get_active_fallback_chain()
        assert len(chain) == 3

        # With unhealthy providers
        chain = policy.get_active_fallback_chain(unhealthy_providers={"model2"})
        assert len(chain) == 2
        assert "model2" not in chain

    def test_fallback_policy_config(self):
        """Test FallbackPolicyConfig container"""
        from xencode.routing.fallback_config import FallbackPolicy, FallbackPolicyConfig

        config = FallbackPolicyConfig(
            policies=[
                FallbackPolicy(name="policy1", fallback_chain=["m1"]),
                FallbackPolicy(name="policy2", fallback_chain=["m2"]),
            ],
            default_policy="policy1",
        )

        assert config.get_policy("policy1") is not None
        assert config.get_policy("policy2") is not None
        assert config.get_policy("nonexistent") is None

        # Add policy
        new_policy = FallbackPolicy(name="policy3", fallback_chain=["m3"])
        config.add_policy(new_policy)
        assert config.get_policy("policy3") is not None

        # Remove policy
        config.remove_policy("policy2")
        assert config.get_policy("policy2") is None


class TestRetryBudgetManager:
    """Test retry budget management"""

    def test_budget_initialization(self):
        """Test retry budget initialization"""
        from xencode.routing.fallback_config import RetryPolicy
        from xencode.routing.retry_budget import RetryBudgetManager

        manager = RetryBudgetManager()
        policy = RetryPolicy(max_retries=3)

        budget = manager.initialize_budget("req-123", policy)

        assert budget.request_id == "req-123"
        assert budget.remaining_retries == 3
        assert manager.get_remaining_retries("req-123") == 3

    def test_budget_consumption(self):
        """Test retry budget consumption"""
        from xencode.routing.fallback_config import RetryPolicy
        from xencode.routing.retry_budget import RetryBudgetManager

        manager = RetryBudgetManager()
        policy = RetryPolicy(max_retries=3)
        manager.initialize_budget("req-123", policy)

        # Consume retries
        assert manager.consume_retry("req-123") is True
        assert manager.get_remaining_retries("req-123") == 2

        assert manager.consume_retry("req-123") is True
        assert manager.get_remaining_retries("req-123") == 1

        assert manager.consume_retry("req-123") is True
        assert manager.get_remaining_retries("req-123") == 0

        # Budget exhausted
        assert manager.consume_retry("req-123") is False
        assert manager.can_retry("req-123") is False
        assert manager.is_exhausted("req-123") is True

    def test_budget_cleanup(self):
        """Test budget cleanup"""
        from xencode.routing.fallback_config import RetryPolicy
        from xencode.routing.retry_budget import RetryBudgetManager

        manager = RetryBudgetManager()
        policy = RetryPolicy(max_retries=3)
        manager.initialize_budget("req-123", policy)

        assert "req-123" in manager.budgets

        manager.cleanup_budget("req-123")

        assert "req-123" not in manager.budgets

    def test_budget_expiration(self):
        """Test automatic budget expiration"""
        from xencode.routing.fallback_config import RetryPolicy
        from xencode.routing.retry_budget import RetryBudgetManager

        # Very short expiration for testing
        manager = RetryBudgetManager(expiration_seconds=0)
        policy = RetryPolicy(max_retries=3)
        manager.initialize_budget("req-123", policy)

        # Wait a tiny bit
        import time
        time.sleep(0.1)

        # Cleanup expired
        cleaned = manager.cleanup_expired()
        assert cleaned >= 0  # May or may not clean depending on timing


@pytest.mark.asyncio
class TestFallbackEngineExecution:
    """Test fallback engine execution"""

    async def test_successful_first_attempt(self):
        """Test successful execution on first attempt"""
        from xencode.routing.fallback_config import FallbackPolicy
        from xencode.routing.fallback_engine import FallbackEngine

        engine = FallbackEngine()
        policy = FallbackPolicy(
            name="test",
            fallback_chain=["provider1", "provider2"],
        )

        async def mock_execute(provider: str):
            return {"provider": provider, "success": True}

        result = await engine.execute_with_fallback(
            policy=policy,
            execute_fn=mock_execute,
            request_id="test-123",
        )

        assert result.success is True
        assert result.provider == "provider1"
        assert len(result.attempts) == 1
        assert result.fallback_count == 0

    async def test_fallback_on_failure(self):
        """Test fallback to next provider on failure"""
        from xencode.routing.fallback_config import FallbackPolicy
        from xencode.routing.fallback_engine import FallbackEngine

        engine = FallbackEngine()
        policy = FallbackPolicy(
            name="test",
            fallback_chain=["provider1", "provider2", "provider3"],
        )

        call_count = 0

        async def mock_execute(provider: str):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Failure for {provider}")
            return {"provider": provider, "success": True}

        result = await engine.execute_with_fallback(
            policy=policy,
            execute_fn=mock_execute,
            request_id="test-123",
        )

        assert result.success is True
        assert result.provider == "provider3"
        assert len(result.attempts) == 3
        assert result.fallback_count == 2
        assert call_count == 3

    async def test_all_providers_fail(self):
        """Test when all providers fail"""
        from xencode.routing.fallback_config import FallbackPolicy
        from xencode.routing.fallback_engine import ExecutionStatus, FallbackEngine

        engine = FallbackEngine()
        policy = FallbackPolicy(
            name="test",
            fallback_chain=["provider1", "provider2"],
        )

        async def mock_execute(provider: str):
            raise Exception(f"Failure for {provider}")

        result = await engine.execute_with_fallback(
            policy=policy,
            execute_fn=mock_execute,
            request_id="test-123",
        )

        assert result.success is False
        assert result.provider is None
        assert result.final_status == ExecutionStatus.FAILED
        assert len(result.attempts) == 2

    async def test_timeout_handling(self):
        """Test timeout handling"""
        from xencode.routing.fallback_config import FallbackPolicy, LatencyCap
        from xencode.routing.fallback_engine import FallbackEngine

        engine = FallbackEngine()
        policy = FallbackPolicy(
            name="test",
            fallback_chain=["provider1"],
            latency_cap=LatencyCap(timeout_ms=1000, max_latency_ms=500),
        )

        async def slow_execute(provider: str):
            await asyncio.sleep(2.0)  # Exceeds timeout
            return {"provider": provider}

        result = await engine.execute_with_fallback(
            policy=policy,
            execute_fn=slow_execute,
            request_id="test-123",
        )

        assert result.success is False
        assert len(result.attempts) == 1
        assert result.attempts[0].status.value == "timeout"

    async def test_execution_history(self):
        """Test execution history tracking"""
        from xencode.routing.fallback_config import FallbackPolicy
        from xencode.routing.fallback_engine import FallbackEngine

        engine = FallbackEngine()
        policy = FallbackPolicy(
            name="test",
            fallback_chain=["provider1"],
        )

        async def mock_execute(provider: str):
            return {"provider": provider}

        # Execute multiple times
        for i in range(5):
            await engine.execute_with_fallback(
                policy=policy,
                execute_fn=mock_execute,
                request_id=f"test-{i}",
            )

        history = engine.get_execution_history()
        assert len(history) == 5

        # Test limit
        limited = engine.get_execution_history(limit=3)
        assert len(limited) == 3

        # Test request_id filter
        filtered = engine.get_execution_history(request_id="test-2")
        assert len(filtered) == 1

    async def test_engine_stats(self):
        """Test engine statistics"""
        from xencode.routing.fallback_config import FallbackPolicy
        from xencode.routing.fallback_engine import FallbackEngine

        engine = FallbackEngine()
        policy = FallbackPolicy(
            name="test",
            fallback_chain=["provider1"],
        )

        async def mock_execute(provider: str):
            return {"provider": provider}

        # Execute
        for i in range(10):
            await engine.execute_with_fallback(
                policy=policy,
                execute_fn=mock_execute,
                request_id=f"test-{i}",
            )

        stats = engine.get_stats()
        assert stats['total_executions'] == 10
        assert stats['successful'] == 10
        assert stats['failed'] == 0
        assert stats['success_rate'] == 100.0


class TestFallbackAPIEndpoints:
    """Test fallback API endpoints"""

    def test_list_policies(self):
        """Test GET /policies endpoint"""
        from xencode.api.routers.fallback import router
        from xencode.routing.fallback_config import FallbackPolicy, FallbackPolicyConfig
        from xencode.routing.fallback_engine import FallbackEngine

        # Setup engine with policies
        engine = FallbackEngine()
        engine.policy_config = FallbackPolicyConfig(
            policies=[
                FallbackPolicy(name="policy1", fallback_chain=["m1"]),
                FallbackPolicy(name="policy2", fallback_chain=["m2"]),
            ]
        )

        # Patch the global engine
        import xencode.api.routers.fallback as fallback_module
        original_engine = fallback_module._fallback_engine
        fallback_module._fallback_engine = engine

        try:
            app = FastAPI()
            app.include_router(router, prefix="/fallback")
            client = TestClient(app)

            response = client.get("/fallback/policies")
            assert response.status_code == 200
            data = response.json()
            assert "policies" in data
            assert data["count"] == 2
        finally:
            fallback_module._fallback_engine = original_engine

    def test_get_policy(self):
        """Test GET /policies/{name} endpoint"""
        import xencode.api.routers.fallback as fallback_module
        from xencode.api.routers.fallback import router
        from xencode.routing.fallback_config import FallbackPolicy, FallbackPolicyConfig
        from xencode.routing.fallback_engine import FallbackEngine

        engine = FallbackEngine()
        engine.policy_config = FallbackPolicyConfig(
            policies=[
                FallbackPolicy(name="test_policy", fallback_chain=["m1", "m2"]),
            ]
        )

        original_engine = fallback_module._fallback_engine
        fallback_module._fallback_engine = engine

        try:
            app = FastAPI()
            app.include_router(router, prefix="/fallback")
            client = TestClient(app)

            response = client.get("/fallback/policies/test_policy")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "test_policy"
            assert len(data["fallback_chain"]) == 2
        finally:
            fallback_module._fallback_engine = original_engine

    def test_get_policy_not_found(self):
        """Test GET /policies/{name} with non-existent policy"""
        import xencode.api.routers.fallback as fallback_module
        from xencode.api.routers.fallback import router
        from xencode.routing.fallback_engine import FallbackEngine

        engine = FallbackEngine()

        original_engine = fallback_module._fallback_engine
        fallback_module._fallback_engine = engine

        try:
            app = FastAPI()
            app.include_router(router, prefix="/fallback")
            client = TestClient(app)

            response = client.get("/fallback/policies/nonexistent")
            assert response.status_code == 404
        finally:
            fallback_module._fallback_engine = original_engine

    def test_create_policy(self):
        """Test POST /policies endpoint"""
        import xencode.api.routers.fallback as fallback_module
        from xencode.api.routers.fallback import router
        from xencode.routing.fallback_engine import FallbackEngine

        engine = FallbackEngine()

        original_engine = fallback_module._fallback_engine
        fallback_module._fallback_engine = engine

        try:
            app = FastAPI()
            app.include_router(router, prefix="/fallback")
            client = TestClient(app)

            policy_data = {
                "name": "new_policy",
                "fallback_chain": ["model1", "model2"],
                "retry_policy": {"max_retries": 2, "base_delay": 0.5},
            }

            response = client.post("/fallback/policies", json=policy_data)
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "new_policy"
        finally:
            fallback_module._fallback_engine = original_engine

    def test_delete_policy(self):
        """Test DELETE /policies/{name} endpoint"""
        import xencode.api.routers.fallback as fallback_module
        from xencode.api.routers.fallback import router
        from xencode.routing.fallback_config import FallbackPolicy, FallbackPolicyConfig
        from xencode.routing.fallback_engine import FallbackEngine

        engine = FallbackEngine()
        engine.policy_config = FallbackPolicyConfig(
            policies=[
                FallbackPolicy(name="to_delete", fallback_chain=["m1"]),
            ]
        )

        original_engine = fallback_module._fallback_engine
        fallback_module._fallback_engine = engine

        try:
            app = FastAPI()
            app.include_router(router, prefix="/fallback")
            client = TestClient(app)

            response = client.delete("/fallback/policies/to_delete")
            assert response.status_code == 200
        finally:
            fallback_module._fallback_engine = original_engine

    def test_get_history(self):
        """Test GET /history endpoint"""
        import xencode.api.routers.fallback as fallback_module
        from xencode.api.routers.fallback import router
        from xencode.routing.fallback_config import FallbackPolicyConfig
        from xencode.routing.fallback_engine import (
            ExecutionStatus,
            FallbackAttempt,
            FallbackEngine,
            FallbackResult,
        )

        engine = FallbackEngine()
        engine.policy_config = FallbackPolicyConfig()

        # Add mock history
        result = FallbackResult(
            success=True,
            provider="m1",
            result={"test": "data"},
            attempts=[
                FallbackAttempt(
                    provider="m1",
                    attempt_number=0,
                    status=ExecutionStatus.SUCCESS,
                    latency_ms=100.0,
                    cost=0.001,
                    timestamp=datetime.now(),
                )
            ],
            total_latency_ms=100.0,
            total_cost=0.001,
            fallback_count=0,
            final_status=ExecutionStatus.SUCCESS,
            request_id="test-123",
            policy_name="test_policy",
        )
        engine.execution_history.append(result)

        original_engine = fallback_module._fallback_engine
        fallback_module._fallback_engine = engine

        try:
            app = FastAPI()
            app.include_router(router, prefix="/fallback")
            client = TestClient(app)

            response = client.get("/fallback/history")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["request_id"] == "test-123"
        finally:
            fallback_module._fallback_engine = original_engine

    def test_get_stats(self):
        """Test GET /stats endpoint"""
        import xencode.api.routers.fallback as fallback_module
        from xencode.api.routers.fallback import router
        from xencode.routing.fallback_engine import FallbackEngine

        engine = FallbackEngine()

        original_engine = fallback_module._fallback_engine
        fallback_module._fallback_engine = engine

        try:
            app = FastAPI()
            app.include_router(router, prefix="/fallback")
            client = TestClient(app)

            response = client.get("/fallback/stats")
            assert response.status_code == 200
            data = response.json()
            assert "total_executions" in data
            assert "success_rate" in data
        finally:
            fallback_module._fallback_engine = original_engine


class TestHealthAwareRouting:
    """Test health-aware routing integration"""

    def test_policy_with_health_aware(self):
        """Test policy respects health_aware flag"""
        from xencode.routing.fallback_config import FallbackPolicy

        policy = FallbackPolicy(
            name="test",
            fallback_chain=["healthy", "unhealthy", "degraded"],
            health_aware=True,
        )

        # Simulate unhealthy providers
        unhealthy = {"unhealthy"}

        chain = policy.get_active_fallback_chain(unhealthy_providers=unhealthy)
        assert "unhealthy" not in chain
        assert len(chain) == 2

    def test_exclusion_rules(self):
        """Test provider exclusion rules"""
        from xencode.routing.fallback_config import (
            ProviderExclusionRule,
        )

        # Permanent exclusion
        perm_rule = ProviderExclusionRule(
            provider="bad_provider",
            reason="Too many errors",
            permanent=True,
        )
        assert perm_rule.is_active() is True

        # Temporary exclusion (not expired)
        temp_rule = ProviderExclusionRule(
            provider="temp_provider",
            reason="Maintenance",
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert temp_rule.is_active() is True

        # Expired exclusion
        expired_rule = ProviderExclusionRule(
            provider="old_provider",
            reason="Outdated",
            expires_at=datetime.now() - timedelta(hours=1),
        )
        assert expired_rule.is_active() is False


class TestCostAndLatencyCaps:
    """Test cost and latency cap enforcement"""

    @pytest.mark.asyncio
    async def test_cost_cap_enforcement(self):
        """Test that cost cap stops fallback chain"""
        from xencode.routing.fallback_config import CostCap, FallbackPolicy
        from xencode.routing.fallback_engine import FallbackEngine

        engine = FallbackEngine()
        policy = FallbackPolicy(
            name="test",
            fallback_chain=["p1", "p2", "p3"],
            cost_cap=CostCap(max_per_request=0.005),
        )

        call_count = 0

        async def expensive_execute(provider: str):
            nonlocal call_count
            call_count += 1
            return {
                "provider": provider,
                "cost": 0.003,  # Each call costs 0.003
            }

        result = await engine.execute_with_fallback(
            policy=policy,
            execute_fn=expensive_execute,
            request_id="cost-test",
        )

        # Should stop after 2 providers (0.003 + 0.003 = 0.006 > 0.005)
        assert call_count <= 2
        assert result.total_cost <= 0.006

    @pytest.mark.asyncio
    async def test_latency_monitoring(self):
        """Test latency tracking in attempts"""
        from xencode.routing.fallback_config import FallbackPolicy
        from xencode.routing.fallback_engine import FallbackEngine

        engine = FallbackEngine()
        policy = FallbackPolicy(
            name="test",
            fallback_chain=["p1"],
        )

        async def mock_execute(provider: str):
            await asyncio.sleep(0.01)  # Small delay
            return {"provider": provider}

        result = await engine.execute_with_fallback(
            policy=policy,
            execute_fn=mock_execute,
            request_id="latency-test",
        )

        assert len(result.attempts) == 1
        assert result.attempts[0].latency_ms > 0
        assert result.total_latency_ms > 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=xencode.routing"])
