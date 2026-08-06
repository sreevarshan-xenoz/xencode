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
from typing import Any, Callable, Dict, List, Optional

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
    result: Optional[Any] = None  # Store successful result

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
                console.print("[yellow]⚠ Cost cap reached, stopping fallback[/yellow]")
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
        (time.time() - start_time) * 1000

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
            result=result,
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
