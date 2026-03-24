#!/usr/bin/env python3
"""
Prompt Routing Layer

Routes prompts to the best provider/model based on task classification and policy.
"""

from .prompt_router import (
    PromptRouter,
    TaskClassifier,
    TaskType,
    ProviderType,
    RoutingPolicy,
    TaskClassification,
    RoutingDecision,
    get_router,
    route_prompt,
)

from .fallback_config import (
    FallbackPolicy,
    FallbackPolicyConfig,
    RetryPolicy,
    CostCap,
    LatencyCap,
    ProviderExclusionRule,
    BackoffType,
)

from .fallback_engine import (
    FallbackEngine,
    FallbackAttempt,
    FallbackResult,
    FallbackReason,
    ExecutionStatus,
)

from .retry_budget import (
    RetryBudgetManager,
    RetryBudget,
)

__all__ = [
    # Prompt Router
    'PromptRouter',
    'TaskClassifier',
    'TaskType',
    'ProviderType',
    'RoutingPolicy',
    'TaskClassification',
    'RoutingDecision',
    'get_router',
    'route_prompt',
    # Fallback Config
    'FallbackPolicy',
    'FallbackPolicyConfig',
    'RetryPolicy',
    'CostCap',
    'LatencyCap',
    'ProviderExclusionRule',
    'BackoffType',
    # Fallback Engine
    'FallbackEngine',
    'FallbackAttempt',
    'FallbackResult',
    'FallbackReason',
    'ExecutionStatus',
    # Retry Budget
    'RetryBudgetManager',
    'RetryBudget',
]
