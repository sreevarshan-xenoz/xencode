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
import random
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
            return min(self.base_delay, self.max_delay)
        
        elif self.backoff_type == BackoffType.LINEAR:
            delay = self.base_delay * (attempt + 1)
        
        elif self.backoff_type == BackoffType.EXPONENTIAL:
            delay = self.base_delay * (2 ** attempt)
        
        elif self.backoff_type == BackoffType.EXPONENTIAL_WITH_JITTER:
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
