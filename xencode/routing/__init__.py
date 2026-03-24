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

__all__ = [
    'PromptRouter',
    'TaskClassifier',
    'TaskType',
    'ProviderType',
    'RoutingPolicy',
    'TaskClassification',
    'RoutingDecision',
    'get_router',
    'route_prompt',
    # Fallback engine exports
    'FallbackPolicy',
    'FallbackPolicyConfig',
    'RetryPolicy',
    'CostCap',
    'LatencyCap',
    'ProviderExclusionRule',
    'BackoffType',
]
