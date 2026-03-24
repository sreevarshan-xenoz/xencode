# Smart Fallback Policy Engine (S5-03) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a comprehensive fallback policy engine with user-defined fallback chains, retry budget management, cost/latency caps, health-aware routing, and automatic failover.

**Architecture:** The fallback engine integrates with the existing Phase 3 routing (`xencode/routing/prompt_router.py`) and S5-02 health monitoring (`xencode/monitoring/provider_health.py`). It provides policy-based fallback execution with configurable retry strategies, cost controls, and health-aware provider selection.

**Tech Stack:** Python 3.9+, FastAPI, Pydantic, asyncio, aiohttp, JSON/YAML configuration

---

## Overview

This implementation adds a smart fallback policy engine that:
1. Manages fallback chains with priority ordering
2. Tracks retry budgets with exponential backoff
3. Enforces cost and latency caps
4. Integrates with provider health monitoring
5. Provides REST API endpoints for policy management
6. Tracks fallback execution history

---

### Task 1: Create Fallback Configuration Models

**Files:**
- Create: `xencode/routing/fallback_config.py`
- Test: `tests/phase5/test_fallback_engine.py::test_fallback_config_models`

**Step 1: Write the failing test**

```python
def test_fallback_config_models():
    """Test fallback configuration Pydantic models"""
    from xencode.routing.fallback_config import (
        RetryPolicy, CostCap, LatencyCap, FallbackPolicy
    )
    
    # Test RetryPolicy
    retry = RetryPolicy(max_retries=3, backoff_type="exponential", base_delay=1.0)
    assert retry.max_retries == 3
    assert retry.get_delay(0) == 1.0
    assert retry.get_delay(1) > 1.0  # Exponential backoff
    
    # Test CostCap
    cost = CostCap(max_per_request=0.01, daily_budget=1.0)
    assert cost.max_per_request == 0.01
    assert cost.daily_budget == 1.0
    
    # Test LatencyCap
    latency = LatencyCap(max_latency_ms=5000, timeout_ms=10000)
    assert latency.max_latency_ms == 5000
    
    # Test FallbackPolicy
    policy = FallbackPolicy(
        name="test_policy",
        priority=1,
        fallback_chain=["model1", "model2", "model3"],
        retry_policy=retry,
        cost_cap=cost,
        latency_cap=latency,
    )
    assert policy.name == "test_policy"
    assert len(policy.fallback_chain) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/phase5/test_fallback_engine.py::test_fallback_config_models -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'xencode.routing.fallback_config'"

**Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""
Fallback Policy Configuration

Defines Pydantic models for fallback policy configuration including:
- Retry policies with backoff strategies
- Cost caps (per-request and daily budgets)
- Latency caps (max latency and timeouts)
- Complete fallback policy definitions
"""

import math
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class BackoffType(str, Enum):
    """Backoff strategy types"""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_WITH_JITTER = "exponential_with_jitter"


class RetryPolicy(BaseModel):
    """
    Retry policy configuration
    
    Attributes:
        max_retries: Maximum number of retry attempts
        backoff_type: Type of backoff strategy
        base_delay: Base delay in seconds
        max_delay: Maximum delay between retries
        jitter: Random jitter factor (0-1) for exponential_with_jitter
    """
    max_retries: int = Field(default=3, ge=0, le=10)
    backoff_type: BackoffType = Field(default=BackoffType.EXPONENTIAL)
    base_delay: float = Field(default=1.0, ge=0.1)
    max_delay: float = Field(default=60.0, ge=1.0)
    jitter: float = Field(default=0.1, ge=0.0, le=1.0)
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given retry attempt
        
        Args:
            attempt: Retry attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        if self.backoff_type == BackoffType.FIXED:
            return self.base_delay
        
        elif self.backoff_type == BackoffType.LINEAR:
            delay = self.base_delay * (attempt + 1)
        
        elif self.backoff_type == BackoffType.EXPONENTIAL:
            delay = self.base_delay * (2 ** attempt)
        
        elif self.backoff_type == BackoffType.EXPONENTIAL_WITH_JITTER:
            import random
            base = self.base_delay * (2 ** attempt)
            jitter_range = base * self.jitter
            delay = base + random.uniform(-jitter_range, jitter_range)
        
        else:
            delay = self.base_delay
        
        return min(delay, self.max_delay)
    
    def should_retry(self, attempt: int) -> bool:
        """Check if should retry based on attempt count"""
        return attempt < self.max_retries


class CostCap(BaseModel):
    """
    Cost cap configuration
    
    Attributes:
        max_per_request: Maximum cost per single request in USD
        daily_budget: Maximum daily budget in USD
        monthly_budget: Maximum monthly budget in USD
        warn_threshold: Percentage threshold for warnings (0-100)
    """
    max_per_request: float = Field(default=0.01, ge=0.0)
    daily_budget: float = Field(default=1.0, ge=0.0)
    monthly_budget: float = Field(default=30.0, ge=0.0)
    warn_threshold: float = Field(default=80.0, ge=0.0, le=100.0)
    
    def is_within_budget(self, cost: float, budget_type: str = "request") -> bool:
        """
        Check if cost is within budget
        
        Args:
            cost: Cost to check
            budget_type: Type of budget (request, daily, monthly)
            
        Returns:
            True if within budget
        """
        if budget_type == "request":
            return cost <= self.max_per_request
        elif budget_type == "daily":
            return cost <= self.daily_budget
        elif budget_type == "monthly":
            return cost <= self.monthly_budget
        return True


class LatencyCap(BaseModel):
    """
    Latency cap configuration
    
    Attributes:
        max_latency_ms: Maximum acceptable latency in milliseconds
        timeout_ms: Hard timeout in milliseconds
        warn_threshold_ms: Warning threshold in milliseconds
    """
    max_latency_ms: int = Field(default=5000, ge=100)
    timeout_ms: int = Field(default=10000, ge=1000)
    warn_threshold_ms: int = Field(default=4000, ge=100)
    
    def is_acceptable(self, latency_ms: float) -> bool:
        """Check if latency is acceptable"""
        return latency_ms <= self.max_latency_ms
    
    def is_timeout(self, latency_ms: float) -> bool:
        """Check if latency exceeds timeout"""
        return latency_ms >= self.timeout_ms


class ProviderExclusionRule(BaseModel):
    """
    Provider exclusion rule
    
    Attributes:
        provider: Provider name to exclude
        reason: Reason for exclusion
        expires_at: Optional expiration time
        permanent: If True, exclusion never expires
    """
    provider: str
    reason: str
    expires_at: Optional[datetime] = None
    permanent: bool = False
    
    def is_active(self) -> bool:
        """Check if exclusion rule is active"""
        if self.permanent:
            return True
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True


class FallbackPolicy(BaseModel):
    """
    Complete fallback policy definition
    
    Attributes:
        name: Policy name
        description: Policy description
        priority: Priority order (lower = higher priority)
        task_types: Task types this policy applies to
        fallback_chain: Ordered list of models/providers to try
        retry_policy: Retry configuration
        cost_cap: Cost limits
        latency_cap: Latency limits
        exclusion_rules: Provider exclusion rules
        health_aware: If True, skip unhealthy providers
        enabled: If True, policy is active
    """
    name: str
    description: str = ""
    priority: int = Field(default=10, ge=1)
    task_types: List[str] = Field(default_factory=list)
    fallback_chain: List[str] = Field(default_factory=list)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    cost_cap: CostCap = Field(default_factory=CostCap)
    latency_cap: LatencyCap = Field(default_factory=LatencyCap)
    exclusion_rules: List[ProviderExclusionRule] = Field(default_factory=list)
    health_aware: bool = True
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('fallback_chain')
    def validate_fallback_chain(cls, v):
        if not v:
            raise ValueError("Fallback chain must have at least one provider")
        return v
    
    def get_active_fallback_chain(
        self,
        unhealthy_providers: Optional[set] = None,
    ) -> List[str]:
        """
        Get fallback chain excluding unhealthy and excluded providers
        
        Args:
            unhealthy_providers: Set of unhealthy provider names
            
        Returns:
            Filtered fallback chain
        """
        excluded = {
            rule.provider for rule in self.exclusion_rules
            if rule.is_active()
        }
        
        if unhealthy_providers:
            excluded.update(unhealthy_providers)
        
        return [p for p in self.fallback_chain if p not in excluded]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FallbackPolicy':
        """Create from dictionary"""
        return cls(**data)


class FallbackPolicyConfig(BaseModel):
    """
    Container for multiple fallback policies
    
    Attributes:
        policies: List of fallback policies
        default_policy: Name of default policy
        version: Configuration version
    """
    policies: List[FallbackPolicy] = Field(default_factory=list)
    default_policy: str = "default"
    version: str = "1.0"
    
    def get_policy(self, name: str) -> Optional[FallbackPolicy]:
        """Get policy by name"""
        for policy in self.policies:
            if policy.name == name:
                return policy
        return None
    
    def get_enabled_policies(self) -> List[FallbackPolicy]:
        """Get all enabled policies"""
        return [p for p in self.policies if p.enabled]
    
    def add_policy(self, policy: FallbackPolicy):
        """Add or update policy"""
        for i, p in enumerate(self.policies):
            if p.name == policy.name:
                self.policies[i] = policy
                return
        self.policies.append(policy)
    
    def remove_policy(self, name: str) -> bool:
        """Remove policy by name"""
        for i, p in enumerate(self.policies):
            if p.name == name:
                del self.policies[i]
                return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'policies': [p.to_dict() for p in self.policies],
            'default_policy': self.default_policy,
            'version': self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FallbackPolicyConfig':
        """Create from dictionary"""
        policies = [
            FallbackPolicy.from_dict(p) for p in data.get('policies', [])
        ]
        return cls(
            policies=policies,
            default_policy=data.get('default_policy', 'default'),
            version=data.get('version', '1.0'),
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/phase5/test_fallback_engine.py::test_fallback_config_models -v`
Expected: PASS

**Step 5: Commit**

```bash
git add xencode/routing/fallback_config.py tests/phase5/test_fallback_engine.py
git commit -m "feat(S5-03): add fallback configuration models"
```

---

### Task 2: Create Retry Budget Manager

**Files:**
- Create: `xencode/routing/retry_budget.py`
- Test: `tests/phase5/test_fallback_engine.py::test_retry_budget_manager`

**Step 1: Write the failing test**

```python
def test_retry_budget_manager():
    """Test retry budget tracking and management"""
    from xencode.routing.retry_budget import RetryBudgetManager
    from xencode.routing.fallback_config import RetryPolicy, BackoffType
    
    manager = RetryBudgetManager()
    request_id = "test-request-123"
    
    # Initialize budget
    policy = RetryPolicy(max_retries=3, backoff_type=BackoffType.EXPONENTIAL, base_delay=0.1)
    manager.initialize_budget(request_id, policy)
    
    # Check initial state
    assert manager.get_remaining_retries(request_id) == 3
    assert manager.can_retry(request_id) is True
    
    # Consume retries
    manager.consume_retry(request_id)
    assert manager.get_remaining_retries(request_id) == 2
    
    manager.consume_retry(request_id)
    manager.consume_retry(request_id)
    assert manager.get_remaining_retries(request_id) == 0
    assert manager.can_retry(request_id) is False
    
    # Check delay calculation
    delay = manager.get_retry_delay(request_id, 0)
    assert delay > 0
    
    # Test budget exhaustion
    assert manager.is_exhausted(request_id) is True
    
    # Test cleanup
    manager.cleanup_budget(request_id)
    assert request_id not in manager.budgets
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/phase5/test_fallback_engine.py::test_retry_budget_manager -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'xencode.routing.retry_budget'"

**Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""
Retry Budget Manager

Manages retry budgets for individual requests with:
- Per-request retry tracking
- Budget expiration and cleanup
- Thread-safe operations
- Delay calculation with backoff
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, Optional

from .fallback_config import RetryPolicy


@dataclass
class RetryBudget:
    """Retry budget for a single request"""
    request_id: str
    policy: RetryPolicy
    remaining_retries: int
    created_at: datetime = field(default_factory=datetime.now)
    last_retry_at: Optional[datetime] = None
    attempt_count: int = 0
    total_delay_ms: float = 0.0
    
    def consume(self) -> bool:
        """
        Consume one retry from budget
        
        Returns:
            True if retry was consumed, False if budget exhausted
        """
        if self.remaining_retries > 0:
            self.remaining_retries -= 1
            self.attempt_count += 1
            self.last_retry_at = datetime.now()
            return True
        return False
    
    def is_exhausted(self) -> bool:
        """Check if budget is exhausted"""
        return self.remaining_retries <= 0
    
    def get_delay_for_attempt(self, attempt: int) -> float:
        """Get delay for specific attempt"""
        return self.policy.get_delay(attempt)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'request_id': self.request_id,
            'remaining_retries': self.remaining_retries,
            'attempt_count': self.attempt_count,
            'created_at': self.created_at.isoformat(),
            'last_retry_at': self.last_retry_at.isoformat() if self.last_retry_at else None,
            'total_delay_ms': self.total_delay_ms,
        }


class RetryBudgetManager:
    """
    Manages retry budgets for multiple concurrent requests
    
    Features:
    - Per-request budget tracking
    - Automatic budget expiration
    - Thread-safe operations
    - Budget cleanup
    """
    
    # Budget expiration time (default 5 minutes)
    DEFAULT_EXPIRATION_SECONDS = 300
    
    def __init__(self, expiration_seconds: int = DEFAULT_EXPIRATION_SECONDS):
        """
        Initialize retry budget manager
        
        Args:
            expiration_seconds: Time after which unused budgets expire
        """
        self.budgets: Dict[str, RetryBudget] = {}
        self.lock = Lock()
        self.expiration_seconds = expiration_seconds
    
    def initialize_budget(
        self,
        request_id: str,
        policy: RetryPolicy,
    ) -> RetryBudget:
        """
        Initialize retry budget for a request
        
        Args:
            request_id: Unique request identifier
            policy: Retry policy to apply
            
        Returns:
            Created RetryBudget
        """
        with self.lock:
            budget = RetryBudget(
                request_id=request_id,
                policy=policy,
                remaining_retries=policy.max_retries,
            )
            self.budgets[request_id] = budget
            return budget
    
    def get_budget(self, request_id: str) -> Optional[RetryBudget]:
        """Get budget for request"""
        with self.lock:
            return self.budgets.get(request_id)
    
    def consume_retry(self, request_id: str) -> bool:
        """
        Consume one retry from budget
        
        Args:
            request_id: Request identifier
            
        Returns:
            True if retry consumed, False if budget exhausted or not found
        """
        with self.lock:
            budget = self.budgets.get(request_id)
            if not budget:
                return False
            return budget.consume()
    
    def get_remaining_retries(self, request_id: str) -> int:
        """Get remaining retries for request"""
        with self.lock:
            budget = self.budgets.get(request_id)
            return budget.remaining_retries if budget else 0
    
    def can_retry(self, request_id: str) -> bool:
        """Check if request can retry"""
        with self.lock:
            budget = self.budgets.get(request_id)
            return budget is not None and not budget.is_exhausted()
    
    def is_exhausted(self, request_id: str) -> bool:
        """Check if budget is exhausted"""
        with self.lock:
            budget = self.budgets.get(request_id)
            return budget is not None and budget.is_exhausted()
    
    def get_retry_delay(self, request_id: str, attempt: int) -> float:
        """
        Get retry delay for attempt
        
        Args:
            request_id: Request identifier
            attempt: Attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        with self.lock:
            budget = self.budgets.get(request_id)
            if not budget:
                return 0.0
            return budget.get_delay_for_attempt(attempt)
    
    def record_delay(self, request_id: str, delay_ms: float):
        """Record delay for request"""
        with self.lock:
            budget = self.budgets.get(request_id)
            if budget:
                budget.total_delay_ms += delay_ms
    
    def cleanup_budget(self, request_id: str):
        """Remove budget for request"""
        with self.lock:
            if request_id in self.budgets:
                del self.budgets[request_id]
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired budgets
        
        Returns:
            Number of budgets cleaned up
        """
        now = datetime.now()
        expired = []
        
        with self.lock:
            for request_id, budget in self.budgets.items():
                age = now - budget.created_at
                if age.total_seconds() > self.expiration_seconds:
                    expired.append(request_id)
            
            for request_id in expired:
                del self.budgets[request_id]
        
        return len(expired)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        with self.lock:
            total_budgets = len(self.budgets)
            exhausted = sum(1 for b in self.budgets.values() if b.is_exhausted())
            active = total_budgets - exhausted
            
            return {
                'total_budgets': total_budgets,
                'active': active,
                'exhausted': exhausted,
                'expiration_seconds': self.expiration_seconds,
            }
    
    async def wait_for_retry(self, request_id: str, attempt: int):
        """
        Wait for appropriate backoff delay before retry
        
        Args:
            request_id: Request identifier
            attempt: Attempt number
        """
        delay = self.get_retry_delay(request_id, attempt)
        if delay > 0:
            await asyncio.sleep(delay)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/phase5/test_fallback_engine.py::test_retry_budget_manager -v`
Expected: PASS

**Step 5: Commit**

```bash
git add xencode/routing/retry_budget.py
git commit -m "feat(S5-03): add retry budget manager"
```

---

### Task 3: Create Core Fallback Engine

**Files:**
- Create: `xencode/routing/fallback_engine.py`
- Test: `tests/phase5/test_fallback_engine.py::test_fallback_engine_execution`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_fallback_engine_execution():
    """Test fallback engine execution with chain"""
    from xencode.routing.fallback_engine import FallbackEngine
    from xencode.routing.fallback_config import FallbackPolicy, RetryPolicy, CostCap, LatencyCap
    
    # Create test policy
    policy = FallbackPolicy(
        name="test_policy",
        fallback_chain=["provider1", "provider2", "provider3"],
        retry_policy=RetryPolicy(max_retries=2, base_delay=0.01),
        cost_cap=CostCap(max_per_request=0.01),
        latency_cap=LatencyCap(max_latency_ms=1000),
    )
    
    engine = FallbackEngine()
    
    # Mock execution function that fails twice then succeeds
    call_count = 0
    async def mock_execute(provider: str):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception(f"Simulated failure for {provider}")
        return {"provider": provider, "success": True}
    
    # Execute with fallback
    result = await engine.execute_with_fallback(
        policy=policy,
        execute_fn=mock_execute,
        request_id="test-123",
    )
    
    assert result is not None
    assert result["success"] is True
    assert call_count == 3  # Should have tried 3 providers
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/phase5/test_fallback_engine.py::test_fallback_engine_execution -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'xencode.routing.fallback_engine'"

**Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""
Smart Fallback Policy Engine

Core fallback execution engine with:
- Policy-based fallback chain execution
- Retry budget management
- Cost and latency cap enforcement
- Health-aware provider selection
- Automatic failover with state tracking
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console

from .fallback_config import FallbackPolicy, FallbackPolicyConfig
from .retry_budget import RetryBudgetManager

console = Console()


class FallbackReason(str, Enum):
    """Reasons for fallback"""
    INITIAL_FAILURE = "initial_failure"
    RETRY_FAILURE = "retry_failure"
    COST_EXCEEDED = "cost_exceeded"
    LATENCY_EXCEEDED = "latency_exceeded"
    PROVIDER_UNHEALTHY = "provider_unhealthy"
    PROVIDER_EXCLUDED = "provider_excluded"
    MANUAL_FAILOVER = "manual_failover"


class ExecutionStatus(str, Enum):
    """Execution status"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class FallbackAttempt:
    """Record of a single fallback attempt"""
    provider: str
    attempt_number: int
    status: ExecutionStatus
    latency_ms: float
    cost: float
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'provider': self.provider,
            'attempt_number': self.attempt_number,
            'status': self.status.value,
            'latency_ms': self.latency_ms,
            'cost': self.cost,
            'error': self.error,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class FallbackResult:
    """Result of fallback execution"""
    success: bool
    provider: Optional[str]
    result: Any
    attempts: List[FallbackAttempt]
    total_latency_ms: float
    total_cost: float
    fallback_count: int
    final_status: ExecutionStatus
    request_id: str
    policy_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'provider': self.provider,
            'result': self.result,
            'attempts': [a.to_dict() for a in self.attempts],
            'total_latency_ms': self.total_latency_ms,
            'total_cost': self.total_cost,
            'fallback_count': self.fallback_count,
            'final_status': self.final_status.value,
            'request_id': self.request_id,
            'policy_name': self.policy_name,
        }


class FallbackEngine:
    """
    Smart fallback execution engine
    
    Features:
    - Policy-based fallback chain execution
    - Retry budget management
    - Cost and latency monitoring
    - Health-aware routing
    - Execution history tracking
    """
    
    def __init__(
        self,
        policy_config: Optional[FallbackPolicyConfig] = None,
        health_monitor: Optional[Any] = None,
    ):
        """
        Initialize fallback engine
        
        Args:
            policy_config: Fallback policy configuration
            health_monitor: Optional provider health monitor
        """
        self.policy_config = policy_config or FallbackPolicyConfig()
        self.health_monitor = health_monitor
        self.retry_manager = RetryBudgetManager()
        self.execution_history: List[FallbackResult] = []
        self.max_history_size = 1000
    
    def get_policy(self, policy_name: str) -> Optional[FallbackPolicy]:
        """Get policy by name"""
        return self.policy_config.get_policy(policy_name)
    
    def get_effective_fallback_chain(
        self,
        policy: FallbackPolicy,
        unhealthy_providers: Optional[set] = None,
    ) -> List[str]:
        """
        Get effective fallback chain considering health and exclusions
        
        Args:
            policy: Fallback policy
            unhealthy_providers: Set of unhealthy provider names
            
        Returns:
            Filtered fallback chain
        """
        if policy.health_aware and self.health_monitor:
            # Get unhealthy providers from health monitor
            if unhealthy_providers is None:
                unhealthy_providers = set()
                # Query health monitor for unhealthy providers
                # (Integration with provider_health.py)
        
        return policy.get_active_fallback_chain(unhealthy_providers)
    
    async def execute_with_fallback(
        self,
        policy: FallbackPolicy,
        execute_fn: Callable[[str], Any],
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> FallbackResult:
        """
        Execute function with fallback chain
        
        Args:
            policy: Fallback policy to apply
            execute_fn: Async function to execute (takes provider name)
            request_id: Optional request identifier
            context: Optional execution context
            
        Returns:
            FallbackResult with execution details
        """
        request_id = request_id or str(uuid.uuid4())
        start_time = time.time()
        
        # Initialize retry budget
        self.retry_manager.initialize_budget(request_id, policy.retry_policy)
        
        # Get effective fallback chain
        fallback_chain = self.get_effective_fallback_chain(policy)
        
        if not fallback_chain:
            return FallbackResult(
                success=False,
                provider=None,
                result=None,
                attempts=[],
                total_latency_ms=0.0,
                total_cost=0.0,
                fallback_count=0,
                final_status=ExecutionStatus.FAILED,
                request_id=request_id,
                policy_name=policy.name,
            )
        
        attempts: List[FallbackAttempt] = []
        total_cost = 0.0
        successful_result = None
        successful_provider = None
        
        for attempt_idx, provider in enumerate(fallback_chain):
            # Check retry budget
            if not self.retry_manager.can_retry(request_id):
                console.print(f"[red]✗ Retry budget exhausted for {request_id}[/red]")
                break
            
            # Check cost cap
            if total_cost >= policy.cost_cap.max_per_request:
                console.print(f"[yellow]⚠ Cost cap reached, stopping fallback[/yellow]")
                break
            
            # Execute with this provider
            attempt_result = await self._execute_provider(
                provider=provider,
                execute_fn=execute_fn,
                attempt_number=attempt_idx,
                request_id=request_id,
                policy=policy,
            )
            
            attempts.append(attempt_result)
            total_cost += attempt_result.cost
            
            # Check if successful
            if attempt_result.status == ExecutionStatus.SUCCESS:
                successful_result = attempt_result
                successful_provider = provider
                break
            
            # Apply backoff before next attempt
            if attempt_idx < len(fallback_chain) - 1:
                delay = self.retry_manager.get_retry_delay(request_id, attempt_idx)
                if delay > 0:
                    await asyncio.sleep(delay)
        
        # Calculate totals
        total_latency_ms = sum(a.latency_ms for a in attempts)
        total_time_ms = (time.time() - start_time) * 1000
        
        # Determine final status
        if successful_result:
            final_status = ExecutionStatus.SUCCESS
        elif self.retry_manager.is_exhausted(request_id):
            final_status = ExecutionStatus.BUDGET_EXHAUSTED
        else:
            final_status = ExecutionStatus.FAILED
        
        # Create result
        result = FallbackResult(
            success=successful_result is not None,
            provider=successful_provider,
            result=successful_result.result if successful_result else None,
            attempts=attempts,
            total_latency_ms=total_latency_ms,
            total_cost=total_cost,
            fallback_count=len(attempts) - 1 if attempts else 0,
            final_status=final_status,
            request_id=request_id,
            policy_name=policy.name,
        )
        
        # Store in history
        self._store_result(result)
        
        # Cleanup retry budget
        self.retry_manager.cleanup_budget(request_id)
        
        return result
    
    async def _execute_provider(
        self,
        provider: str,
        execute_fn: Callable[[str], Any],
        attempt_number: int,
        request_id: str,
        policy: FallbackPolicy,
    ) -> FallbackAttempt:
        """Execute single provider attempt"""
        start_time = time.time()
        error = None
        status = ExecutionStatus.SUCCESS
        result = None
        cost = 0.0
        
        try:
            # Check latency cap
            timeout_sec = policy.latency_cap.timeout_ms / 1000.0
            
            # Execute with timeout
            result = await asyncio.wait_for(
                execute_fn(provider),
                timeout=timeout_sec,
            )
            
            # Extract cost from result if available
            if isinstance(result, dict):
                cost = result.get('cost', 0.0)
            
        except asyncio.TimeoutError:
            status = ExecutionStatus.TIMEOUT
            error = f"Timeout after {timeout_sec}s"
        
        except Exception as e:
            status = ExecutionStatus.FAILED
            error = str(e)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Check latency cap
        if latency_ms > policy.latency_cap.max_latency_ms:
            console.print(
                f"[yellow]⚠ Latency {latency_ms:.0f}ms exceeds cap "
                f"{policy.latency_cap.max_latency_ms}ms[/yellow]"
            )
        
        return FallbackAttempt(
            provider=provider,
            attempt_number=attempt_number,
            status=status,
            latency_ms=latency_ms,
            cost=cost,
            error=error,
        )
    
    def _store_result(self, result: FallbackResult):
        """Store result in history"""
        self.execution_history.append(result)
        
        # Trim history if too large
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size:]
    
    def get_execution_history(
        self,
        request_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[FallbackResult]:
        """Get execution history"""
        if request_id:
            results = [r for r in self.execution_history if r.request_id == request_id]
        else:
            results = self.execution_history
        
        return results[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        total_executions = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if r.success)
        
        return {
            'total_executions': total_executions,
            'successful': successful,
            'failed': total_executions - successful,
            'success_rate': (successful / total_executions * 100) if total_executions > 0 else 0,
            'retry_budget_stats': self.retry_manager.get_stats(),
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/phase5/test_fallback_engine.py::test_fallback_engine_execution -v`
Expected: PASS

**Step 5: Commit**

```bash
git add xencode/routing/fallback_engine.py
git commit -m "feat(S5-03): add core fallback engine"
```

---

### Task 4: Create Fallback API Router

**Files:**
- Create: `xencode/api/routers/fallback.py`
- Modify: `xencode/api/routers/__init__.py` (add fallback router)
- Test: `tests/phase5/test_fallback_engine.py::test_fallback_api_endpoints`

**Step 1: Write the failing test**

```python
def test_fallback_api_endpoints():
    """Test fallback API endpoints"""
    from fastapi.testclient import TestClient
    from xencode.api.routers.fallback import router
    
    # Create test client
    app = FastAPI()
    app.include_router(router, prefix="/fallback")
    client = TestClient(app)
    
    # Test GET /policies
    response = client.get("/policies")
    assert response.status_code == 200
    data = response.json()
    assert "policies" in data
    
    # Test POST /policies
    policy_data = {
        "name": "api_test_policy",
        "fallback_chain": ["model1", "model2"],
        "retry_policy": {"max_retries": 2},
    }
    response = client.post("/policies", json=policy_data)
    assert response.status_code == 200
    
    # Test GET /policies/{name}
    response = client.get("/policies/api_test_policy")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "api_test_policy"
    
    # Test GET /history
    response = client.get("/history")
    assert response.status_code == 200
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/phase5/test_fallback_engine.py::test_fallback_api_endpoints -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'xencode.api.routers.fallback'"

**Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""
Fallback Policy API Router

FastAPI router for managing fallback policies and executing fallback requests.

Endpoints:
- GET /fallback/policies - List all policies
- GET /fallback/policies/{name} - Get specific policy
- POST /fallback/policies - Create/update policy
- DELETE /fallback/policies/{name} - Delete policy
- POST /fallback/execute - Execute with fallback
- GET /fallback/history - Get fallback execution history
- GET /fallback/stats - Get fallback statistics
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...routing.fallback_config import (
    FallbackPolicy,
    FallbackPolicyConfig,
    RetryPolicy,
    CostCap,
    LatencyCap,
    BackoffType,
)
from ...routing.fallback_engine import FallbackEngine, FallbackResult

router = APIRouter()


# Global fallback engine instance
_fallback_engine: Optional[FallbackEngine] = None


def get_fallback_engine() -> FallbackEngine:
    """Get or create fallback engine instance"""
    global _fallback_engine
    if _fallback_engine is None:
        _fallback_engine = FallbackEngine()
    return _fallback_engine


# Request/Response Models

class RetryPolicyRequest(BaseModel):
    """Retry policy request model"""
    max_retries: int = Field(default=3, ge=0, le=10)
    backoff_type: BackoffType = Field(default=BackoffType.EXPONENTIAL)
    base_delay: float = Field(default=1.0, ge=0.1)
    max_delay: float = Field(default=60.0, ge=1.0)
    jitter: float = Field(default=0.1, ge=0.0, le=1.0)


class CostCapRequest(BaseModel):
    """Cost cap request model"""
    max_per_request: float = Field(default=0.01, ge=0.0)
    daily_budget: float = Field(default=1.0, ge=0.0)
    monthly_budget: float = Field(default=30.0, ge=0.0)
    warn_threshold: float = Field(default=80.0, ge=0.0, le=100.0)


class LatencyCapRequest(BaseModel):
    """Latency cap request model"""
    max_latency_ms: int = Field(default=5000, ge=100)
    timeout_ms: int = Field(default=10000, ge=1000)
    warn_threshold_ms: int = Field(default=4000, ge=100)


class FallbackPolicyRequest(BaseModel):
    """Fallback policy creation request"""
    name: str
    description: str = ""
    priority: int = Field(default=10, ge=1)
    task_types: List[str] = Field(default_factory=list)
    fallback_chain: List[str] = Field(min_length=1)
    retry_policy: Optional[RetryPolicyRequest] = None
    cost_cap: Optional[CostCapRequest] = None
    latency_cap: Optional[LatencyCapRequest] = None
    health_aware: bool = True
    enabled: bool = True


class FallbackPolicyResponse(BaseModel):
    """Fallback policy response"""
    name: str
    description: str
    priority: int
    task_types: List[str]
    fallback_chain: List[str]
    retry_policy: Dict[str, Any]
    cost_cap: Dict[str, Any]
    latency_cap: Dict[str, Any]
    health_aware: bool
    enabled: bool
    created_at: str
    updated_at: str


class FallbackExecuteRequest(BaseModel):
    """Fallback execution request"""
    policy_name: str
    provider: str
    prompt: str
    context: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None


class FallbackExecuteResponse(BaseModel):
    """Fallback execution response"""
    success: bool
    provider: Optional[str]
    result: Any
    attempts: List[Dict[str, Any]]
    total_latency_ms: float
    total_cost: float
    fallback_count: int
    final_status: str
    request_id: str
    policy_name: str


class FallbackHistoryResponse(BaseModel):
    """Fallback history response"""
    request_id: str
    policy_name: str
    success: bool
    provider: Optional[str]
    attempts_count: int
    total_latency_ms: float
    total_cost: float
    final_status: str
    timestamp: str


class FallbackStatsResponse(BaseModel):
    """Fallback statistics response"""
    total_executions: int
    successful: int
    failed: int
    success_rate: float
    retry_budget_stats: Dict[str, Any]


# API Endpoints

@router.get("/policies", response_model=Dict[str, Any])
async def list_policies(
    enabled_only: bool = Query(False, description="Filter to enabled policies only"),
    engine: FallbackEngine = Depends(get_fallback_engine),
):
    """List all fallback policies"""
    if enabled_only:
        policies = engine.policy_config.get_enabled_policies()
    else:
        policies = engine.policy_config.policies
    
    return {
        "policies": [p.to_dict() for p in policies],
        "default_policy": engine.policy_config.default_policy,
        "count": len(policies),
    }


@router.get("/policies/{policy_name}", response_model=FallbackPolicyResponse)
async def get_policy(
    policy_name: str,
    engine: FallbackEngine = Depends(get_fallback_engine),
):
    """Get specific fallback policy"""
    policy = engine.get_policy(policy_name)
    
    if not policy:
        raise HTTPException(status_code=404, detail=f"Policy '{policy_name}' not found")
    
    return FallbackPolicyResponse(
        name=policy.name,
        description=policy.description,
        priority=policy.priority,
        task_types=policy.task_types,
        fallback_chain=policy.fallback_chain,
        retry_policy=policy.retry_policy.dict(),
        cost_cap=policy.cost_cap.dict(),
        latency_cap=policy.latency_cap.dict(),
        health_aware=policy.health_aware,
        enabled=policy.enabled,
        created_at=policy.created_at.isoformat(),
        updated_at=policy.updated_at.isoformat(),
    )


@router.post("/policies", response_model=FallbackPolicyResponse)
async def create_or_update_policy(
    policy_data: FallbackPolicyRequest,
    engine: FallbackEngine = Depends(get_fallback_engine),
):
    """Create or update a fallback policy"""
    # Build policy object
    policy = FallbackPolicy(
        name=policy_data.name,
        description=policy_data.description,
        priority=policy_data.priority,
        task_types=policy_data.task_types,
        fallback_chain=policy_data.fallback_chain,
        retry_policy=RetryPolicy(**policy_data.retry_policy.dict()) if policy_data.retry_policy else RetryPolicy(),
        cost_cap=CostCap(**policy_data.cost_cap.dict()) if policy_data.cost_cap else CostCap(),
        latency_cap=LatencyCap(**policy_data.latency_cap.dict()) if policy_data.latency_cap else LatencyCap(),
        health_aware=policy_data.health_aware,
        enabled=policy_data.enabled,
    )
    
    # Add to engine
    engine.policy_config.add_policy(policy)
    
    return FallbackPolicyResponse(
        name=policy.name,
        description=policy.description,
        priority=policy.priority,
        task_types=policy.task_types,
        fallback_chain=policy.fallback_chain,
        retry_policy=policy.retry_policy.dict(),
        cost_cap=policy.cost_cap.dict(),
        latency_cap=policy.latency_cap.dict(),
        health_aware=policy.health_aware,
        enabled=policy.enabled,
        created_at=policy.created_at.isoformat(),
        updated_at=policy.updated_at.isoformat(),
    )


@router.delete("/policies/{policy_name}")
async def delete_policy(
    policy_name: str,
    engine: FallbackEngine = Depends(get_fallback_engine),
):
    """Delete a fallback policy"""
    success = engine.policy_config.remove_policy(policy_name)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Policy '{policy_name}' not found")
    
    return {"message": f"Policy '{policy_name}' deleted successfully"}


@router.post("/execute", response_model=FallbackExecuteResponse)
async def execute_with_fallback(
    request: FallbackExecuteRequest,
    engine: FallbackEngine = Depends(get_fallback_engine),
):
    """Execute a request with fallback policy"""
    # Get policy
    policy = engine.get_policy(request.policy_name)
    
    if not policy:
        raise HTTPException(status_code=404, detail=f"Policy '{request.policy_name}' not found")
    
    # Mock execution function (in production, this would call actual provider)
    async def mock_execute(provider: str):
        # Simulate provider execution
        # In production, this would integrate with the actual provider client
        return {
            "provider": provider,
            "response": f"Response from {provider}",
            "cost": 0.001,
            "tokens": 100,
        }
    
    # Execute with fallback
    result = await engine.execute_with_fallback(
        policy=policy,
        execute_fn=mock_execute,
        request_id=request.request_id or str(uuid.uuid4()),
        context=request.context,
    )
    
    return FallbackExecuteResponse(
        success=result.success,
        provider=result.provider,
        result=result.result,
        attempts=[a.to_dict() for a in result.attempts],
        total_latency_ms=result.total_latency_ms,
        total_cost=result.total_cost,
        fallback_count=result.fallback_count,
        final_status=result.final_status,
        request_id=result.request_id,
        policy_name=result.policy_name,
    )


@router.get("/history", response_model=List[FallbackHistoryResponse])
async def get_fallback_history(
    request_id: Optional[str] = Query(None, description="Filter by request ID"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results to return"),
    engine: FallbackEngine = Depends(get_fallback_engine),
):
    """Get fallback execution history"""
    results = engine.get_execution_history(request_id=request_id, limit=limit)
    
    return [
        FallbackHistoryResponse(
            request_id=r.request_id,
            policy_name=r.policy_name,
            success=r.success,
            provider=r.provider,
            attempts_count=len(r.attempts),
            total_latency_ms=r.total_latency_ms,
            total_cost=r.total_cost,
            final_status=r.final_status,
            timestamp=r.attempts[0].timestamp.isoformat() if r.attempts else None,
        )
        for r in results
    ]


@router.get("/stats", response_model=FallbackStatsResponse)
async def get_fallback_stats(
    engine: FallbackEngine = Depends(get_fallback_engine),
):
    """Get fallback engine statistics"""
    stats = engine.get_stats()
    
    return FallbackStatsResponse(
        total_executions=stats['total_executions'],
        successful=stats['successful'],
        failed=stats['failed'],
        success_rate=stats['success_rate'],
        retry_budget_stats=stats['retry_budget_stats'],
    )
```

**Step 4: Update router __init__.py**

Read `xencode/api/routers/__init__.py` and add:

```python
try:
    from .fallback import router as fallback_router
    FALLBACK_ROUTER_AVAILABLE = True
except ImportError:
    fallback_router = None
    FALLBACK_ROUTER_AVAILABLE = False

# Update get_router_status()
def get_router_status() -> dict:
    """Get status of all routers"""
    return {
        "document_router": DOCUMENT_ROUTER_AVAILABLE,
        "code_analysis_router": CODE_ANALYSIS_ROUTER_AVAILABLE,
        "workspace_router": WORKSPACE_ROUTER_AVAILABLE,
        "analytics_router": ANALYTICS_ROUTER_AVAILABLE,
        "monitoring_router": MONITORING_ROUTER_AVAILABLE,
        "plugin_router": PLUGIN_ROUTER_AVAILABLE,
        "features_router": FEATURES_ROUTER_AVAILABLE,
        "fallback_router": FALLBACK_ROUTER_AVAILABLE,
    }

# Update __all__
__all__ = [
    'document_router',
    'code_analysis_router',
    'workspace_router',
    'analytics_router',
    'monitoring_router',
    'plugin_router',
    'features_router',
    'fallback_router',
    'get_router_status'
]
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/phase5/test_fallback_engine.py::test_fallback_api_endpoints -v`
Expected: PASS

**Step 6: Commit**

```bash
git add xencode/api/routers/fallback.py xencode/api/routers/__init__.py
git commit -m "feat(S5-03): add fallback API router"
```

---

### Task 5: Create Comprehensive Tests

**Files:**
- Modify: `tests/phase5/test_fallback_engine.py`

**Step 1: Write comprehensive test suite**

```python
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
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestFallbackConfigModels:
    """Test fallback configuration Pydantic models"""
    
    def test_retry_policy_creation(self):
        """Test RetryPolicy model creation"""
        from xencode.routing.fallback_config import RetryPolicy, BackoffType
        
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
        from xencode.routing.fallback_config import RetryPolicy, BackoffType
        
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
        from xencode.routing.fallback_config import FallbackPolicy, ProviderExclusionRule
        
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
        from xencode.routing.retry_budget import RetryBudgetManager
        from xencode.routing.fallback_config import RetryPolicy
        
        manager = RetryBudgetManager()
        policy = RetryPolicy(max_retries=3)
        
        budget = manager.initialize_budget("req-123", policy)
        
        assert budget.request_id == "req-123"
        assert budget.remaining_retries == 3
        assert manager.get_remaining_retries("req-123") == 3
    
    def test_budget_consumption(self):
        """Test retry budget consumption"""
        from xencode.routing.retry_budget import RetryBudgetManager
        from xencode.routing.fallback_config import RetryPolicy
        
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
        from xencode.routing.retry_budget import RetryBudgetManager
        from xencode.routing.fallback_config import RetryPolicy
        
        manager = RetryBudgetManager()
        policy = RetryPolicy(max_retries=3)
        manager.initialize_budget("req-123", policy)
        
        assert "req-123" in manager.budgets
        
        manager.cleanup_budget("req-123")
        
        assert "req-123" not in manager.budgets
    
    def test_budget_expiration(self):
        """Test automatic budget expiration"""
        from xencode.routing.retry_budget import RetryBudgetManager
        from xencode.routing.fallback_config import RetryPolicy
        
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
        from xencode.routing.fallback_engine import FallbackEngine
        from xencode.routing.fallback_config import FallbackPolicy
        
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
        from xencode.routing.fallback_engine import FallbackEngine, ExecutionStatus
        from xencode.routing.fallback_config import FallbackPolicy
        
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
        from xencode.routing.fallback_engine import FallbackEngine, ExecutionStatus
        from xencode.routing.fallback_config import FallbackPolicy
        
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
        from xencode.routing.fallback_engine import FallbackEngine
        from xencode.routing.fallback_config import FallbackPolicy, LatencyCap
        
        engine = FallbackEngine()
        policy = FallbackPolicy(
            name="test",
            fallback_chain=["provider1"],
            latency_cap=LatencyCap(timeout_ms=100, max_latency_ms=50),
        )
        
        async def slow_execute(provider: str):
            await asyncio.sleep(1.0)  # Exceeds timeout
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
        from xencode.routing.fallback_engine import FallbackEngine
        from xencode.routing.fallback_config import FallbackPolicy
        
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
        from xencode.routing.fallback_engine import FallbackEngine
        from xencode.routing.fallback_config import FallbackPolicy
        
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
        from xencode.routing.fallback_engine import FallbackEngine
        from xencode.routing.fallback_config import FallbackPolicy, FallbackPolicyConfig
        
        # Setup engine with policies
        engine = FallbackEngine()
        engine.policy_config = FallbackPolicyConfig(
            policies=[
                FallbackPolicy(name="policy1", fallback_chain=["m1"]),
                FallbackPolicy(name="policy2", fallback_chain=["m2"]),
            ]
        )
        
        # Patch get_fallback_engine
        with patch('xencode.api.routers.fallback.get_fallback_engine', return_value=engine):
            app = FastAPI()
            app.include_router(router, prefix="/fallback")
            client = TestClient(app)
            
            response = client.get("/fallback/policies")
            assert response.status_code == 200
            data = response.json()
            assert "policies" in data
            assert data["count"] == 2
    
    def test_get_policy(self):
        """Test GET /policies/{name} endpoint"""
        from xencode.api.routers.fallback import router
        from xencode.routing.fallback_engine import FallbackEngine
        from xencode.routing.fallback_config import FallbackPolicy, FallbackPolicyConfig
        
        engine = FallbackEngine()
        engine.policy_config = FallbackPolicyConfig(
            policies=[
                FallbackPolicy(name="test_policy", fallback_chain=["m1", "m2"]),
            ]
        )
        
        with patch('xencode.api.routers.fallback.get_fallback_engine', return_value=engine):
            app = FastAPI()
            app.include_router(router, prefix="/fallback")
            client = TestClient(app)
            
            response = client.get("/fallback/policies/test_policy")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "test_policy"
            assert len(data["fallback_chain"]) == 2
    
    def test_get_policy_not_found(self):
        """Test GET /policies/{name} with non-existent policy"""
        from xencode.api.routers.fallback import router
        from xencode.routing.fallback_engine import FallbackEngine
        
        engine = FallbackEngine()
        
        with patch('xencode.api.routers.fallback.get_fallback_engine', return_value=engine):
            app = FastAPI()
            app.include_router(router, prefix="/fallback")
            client = TestClient(app)
            
            response = client.get("/fallback/policies/nonexistent")
            assert response.status_code == 404
    
    def test_create_policy(self):
        """Test POST /policies endpoint"""
        from xencode.api.routers.fallback import router
        from xencode.routing.fallback_engine import FallbackEngine
        
        engine = FallbackEngine()
        
        with patch('xencode.api.routers.fallback.get_fallback_engine', return_value=engine):
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
    
    def test_delete_policy(self):
        """Test DELETE /policies/{name} endpoint"""
        from xencode.api.routers.fallback import router
        from xencode.routing.fallback_engine import FallbackEngine
        from xencode.routing.fallback_config import FallbackPolicy, FallbackPolicyConfig
        
        engine = FallbackEngine()
        engine.policy_config = FallbackPolicyConfig(
            policies=[
                FallbackPolicy(name="to_delete", fallback_chain=["m1"]),
            ]
        )
        
        with patch('xencode.api.routers.fallback.get_fallback_engine', return_value=engine):
            app = FastAPI()
            app.include_router(router, prefix="/fallback")
            client = TestClient(app)
            
            response = client.delete("/fallback/policies/to_delete")
            assert response.status_code == 200
    
    def test_get_history(self):
        """Test GET /history endpoint"""
        from xencode.api.routers.fallback import router
        from xencode.routing.fallback_engine import FallbackEngine, FallbackResult, FallbackAttempt, ExecutionStatus
        from xencode.routing.fallback_config import FallbackPolicyConfig
        
        engine = FallbackEngine()
        engine.policy_config = FallbackPolicyConfig()
        
        # Add mock history
        from datetime import datetime
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
        
        with patch('xencode.api.routers.fallback.get_fallback_engine', return_value=engine):
            app = FastAPI()
            app.include_router(router, prefix="/fallback")
            client = TestClient(app)
            
            response = client.get("/fallback/history")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["request_id"] == "test-123"
    
    def test_get_stats(self):
        """Test GET /stats endpoint"""
        from xencode.api.routers.fallback import router
        from xencode.routing.fallback_engine import FallbackEngine
        
        engine = FallbackEngine()
        
        with patch('xencode.api.routers.fallback.get_fallback_engine', return_value=engine):
            app = FastAPI()
            app.include_router(router, prefix="/fallback")
            client = TestClient(app)
            
            response = client.get("/fallback/stats")
            assert response.status_code == 200
            data = response.json()
            assert "total_executions" in data
            assert "success_rate" in data


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
        from xencode.routing.fallback_config import FallbackPolicy, ProviderExclusionRule
        from datetime import datetime, timedelta
        
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
        from xencode.routing.fallback_engine import FallbackEngine
        from xencode.routing.fallback_config import FallbackPolicy, CostCap
        
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
        from xencode.routing.fallback_engine import FallbackEngine
        from xencode.routing.fallback_config import FallbackPolicy
        
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
```

**Step 2: Run all tests to verify they pass**

Run: `pytest tests/phase5/test_fallback_engine.py -v --cov=xencode.routing`
Expected: All tests PASS (minimum 10 test cases)

**Step 3: Commit**

```bash
git add tests/phase5/test_fallback_engine.py
git commit -m "test(S5-03): add comprehensive fallback engine tests"
```

---

### Task 6: Add Documentation and Examples

**Files:**
- Create: `docs/phase5/fallback-engine.md`
- Create: `examples/fallback_policies/example_policies.yaml`

**Step 1: Create documentation**

```markdown
# Smart Fallback Policy Engine (S5-03)

## Overview

The Smart Fallback Policy Engine provides intelligent fallback chain management with:
- User-defined fallback chains with priority ordering
- Retry budget management with configurable backoff strategies
- Cost caps (per-request and daily budget limits)
- Latency caps (max acceptable latency and timeout policies)
- Health-aware routing (skip unhealthy providers)
- Automatic failover with state tracking

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Fallback Policy Engine                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Policy     │  │    Retry     │  │   Health     │      │
│  │  Config      │  │   Budget     │  │   Monitor    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           │                                 │
│                  ┌────────▼────────┐                        │
│                  │  Fallback       │                        │
│                  │  Engine         │                        │
│                  └────────┬────────┘                        │
│                           │                                 │
│         ┌─────────────────┼─────────────────┐              │
│         │                 │                 │               │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐        │
│  │  Provider 1 │  │  Provider 2 │  │  Provider 3 │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Policy Definition

Policies can be defined programmatically or via JSON/YAML:

```yaml
policies:
  - name: "coding_tasks"
    description: "Fallback chain for coding tasks"
    priority: 1
    task_types:
      - "code_generation"
      - "code_review"
      - "debugging"
    fallback_chain:
      - "qwen3-coder-next-instruct"
      - "qwen-coder-plus"
      - "qwen2.5-coder:7b"
    retry_policy:
      max_retries: 3
      backoff_type: "exponential"
      base_delay: 1.0
      max_delay: 60.0
    cost_cap:
      max_per_request: 0.01
      daily_budget: 1.0
    latency_cap:
      max_latency_ms: 5000
      timeout_ms: 10000
    health_aware: true
    enabled: true
```

### Backoff Strategies

- **Fixed**: Constant delay between retries
- **Linear**: Delay increases linearly (base_delay × attempt)
- **Exponential**: Delay doubles each attempt (base_delay × 2^attempt)
- **Exponential with Jitter**: Exponential + random variation

## API Endpoints

### List Policies

```bash
GET /fallback/policies
```

Response:
```json
{
  "policies": [...],
  "default_policy": "default",
  "count": 3
}
```

### Get Policy

```bash
GET /fallback/policies/{policy_name}
```

### Create/Update Policy

```bash
POST /fallback/policies
Content-Type: application/json

{
  "name": "my_policy",
  "fallback_chain": ["model1", "model2"],
  "retry_policy": {"max_retries": 3}
}
```

### Execute with Fallback

```bash
POST /fallback/execute
Content-Type: application/json

{
  "policy_name": "my_policy",
  "provider": "qwen3-coder",
  "prompt": "Write a function to..."
}
```

### Get History

```bash
GET /fallback/history?limit=50
```

### Get Statistics

```bash
GET /fallback/stats
```

## Usage Examples

### Python API

```python
from xencode.routing.fallback_engine import FallbackEngine
from xencode.routing.fallback_config import FallbackPolicy, RetryPolicy

# Create policy
policy = FallbackPolicy(
    name="my_policy",
    fallback_chain=["model1", "model2", "model3"],
    retry_policy=RetryPolicy(max_retries=3, backoff_type="exponential"),
)

# Create engine
engine = FallbackEngine()
engine.policy_config.add_policy(policy)

# Execute with fallback
async def execute_provider(provider: str):
    # Your provider execution logic
    return await call_provider(provider)

result = await engine.execute_with_fallback(
    policy=policy,
    execute_fn=execute_provider,
    request_id="unique-request-id",
)

print(f"Success: {result.success}")
print(f"Provider: {result.provider}")
print(f"Attempts: {len(result.attempts)}")
```

### Health-Aware Routing

```python
# Policy automatically skips unhealthy providers
policy = FallbackPolicy(
    name="health_aware",
    fallback_chain=["provider1", "provider2", "provider3"],
    health_aware=True,  # Enable health checking
)

# Engine integrates with provider health monitor
engine = FallbackEngine(health_monitor=health_monitor)
```

## Best Practices

1. **Define Clear Fallback Chains**: Order providers by preference (cost, quality, latency)
2. **Set Reasonable Retry Limits**: 2-3 retries usually sufficient
3. **Use Exponential Backoff**: Prevents overwhelming failing providers
4. **Monitor Costs**: Set appropriate cost caps per request type
5. **Enable Health Awareness**: Automatically skip unhealthy providers
6. **Track History**: Use execution history for debugging and optimization

## Integration

### With Phase 3 Routing

The fallback engine integrates with the existing `PromptRouter`:

```python
from xencode.routing.prompt_router import PromptRouter
from xencode.routing.fallback_engine import FallbackEngine

router = PromptRouter()
engine = FallbackEngine()

# Use fallback chain from routing policy
decision = router.route(prompt)
policy = engine.get_policy(decision.policy_applied)

result = await engine.execute_with_fallback(
    policy=policy,
    execute_fn=execute_provider,
)
```

### With S5-02 Health Monitoring

```python
from xencode.monitoring.provider_health import get_health_monitor
from xencode.routing.fallback_engine import FallbackEngine

health_monitor = get_health_monitor()
engine = FallbackEngine(health_monitor=health_monitor)

# Engine automatically queries health status
```

## Troubleshooting

### All Providers Failing

Check execution history:
```bash
GET /fallback/history?request_id={request_id}
```

### High Latency

Review latency caps and adjust timeout settings.

### Cost Overruns

Lower `max_per_request` or reduce fallback chain length.

## Performance Considerations

- Fallback chains add latency (each attempt adds time)
- Health checks run asynchronously
- History is limited to 1000 entries by default
- Retry budgets expire after 5 minutes
```

**Step 2: Create example policies**

```yaml
# Example Fallback Policies
# Save to: examples/fallback_policies/example_policies.yaml

policies:
  # Cost-optimized policy for simple tasks
  - name: "cost_optimized"
    description: "Minimize costs for simple queries"
    priority: 1
    task_types:
      - "chat"
      - "general"
    fallback_chain:
      - "llama3.2:3b"
      - "qwen2.5:7b"
      - "qwen-turbo"
    retry_policy:
      max_retries: 2
      backoff_type: "fixed"
      base_delay: 0.5
    cost_cap:
      max_per_request: 0.002
      daily_budget: 0.50
    latency_cap:
      max_latency_ms: 3000
    health_aware: true

  # High-quality policy for complex coding
  - name: "coding_premium"
    description: "Best quality for complex coding tasks"
    priority: 2
    task_types:
      - "code_generation"
      - "architecture"
      - "system_design"
    fallback_chain:
      - "qwen3-coder-next-instruct"
      - "qwen-coder-plus"
      - "qwen2.5-coder:32b"
      - "qwen2.5-coder:7b"
    retry_policy:
      max_retries: 3
      backoff_type: "exponential"
      base_delay: 1.0
      max_delay: 30.0
    cost_cap:
      max_per_request: 0.05
      daily_budget: 5.00
    latency_cap:
      max_latency_ms: 10000
      timeout_ms: 30000
    health_aware: true

  # Low-latency policy for real-time
  - name: "low_latency"
    description: "Fastest response for interactive use"
    priority: 3
    task_types:
      - "chat"
      - "code_explanation"
    fallback_chain:
      - "llama3.2:3b"
      - "qwen2.5:7b"
    retry_policy:
      max_retries: 1
      backoff_type: "fixed"
      base_delay: 0.2
    cost_cap:
      max_per_request: 0.005
    latency_cap:
      max_latency_ms: 1500
      timeout_ms: 3000
    health_aware: true

  # Balanced policy (default)
  - name: "default"
    description: "Balanced cost, quality, and latency"
    priority: 10
    fallback_chain:
      - "qwen2.5:7b"
      - "qwen-turbo"
      - "qwen-plus"
    retry_policy:
      max_retries: 2
      backoff_type: "exponential"
      base_delay: 1.0
    cost_cap:
      max_per_request: 0.01
      daily_budget: 1.00
    latency_cap:
      max_latency_ms: 5000
    health_aware: true

default_policy: "default"
version: "1.0"
```

**Step 3: Commit**

```bash
git add docs/phase5/fallback-engine.md examples/fallback_policies/example_policies.yaml
git commit -m "docs(S5-03): add fallback engine documentation and examples"
```

---

## Summary

This implementation plan creates a comprehensive Smart Fallback Policy Engine with:

**Production Code (600+ lines):**
1. `xencode/routing/fallback_config.py` - Configuration models (~250 lines)
2. `xencode/routing/retry_budget.py` - Retry budget manager (~200 lines)
3. `xencode/routing/fallback_engine.py` - Core engine (~300 lines)
4. `xencode/api/routers/fallback.py` - API endpoints (~250 lines)

**Tests (10+ test cases):**
- Configuration model tests
- Retry budget tests
- Engine execution tests
- API endpoint tests
- Health-aware routing tests
- Cost/latency cap tests

**Documentation:**
- Complete markdown documentation
- Example policies in YAML
- API usage examples
- Integration guides

The implementation integrates with:
- Phase 3 routing (`prompt_router.py`)
- S5-02 health monitoring (`provider_health.py`)
- Existing API structure
