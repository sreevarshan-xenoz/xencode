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
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...routing.fallback_config import (
    BackoffType,
    CostCap,
    FallbackPolicy,
    LatencyCap,
    RetryPolicy,
)
from ...routing.fallback_engine import FallbackEngine

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
