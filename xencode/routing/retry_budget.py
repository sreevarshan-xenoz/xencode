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
from dataclasses import dataclass, field
from datetime import datetime
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
