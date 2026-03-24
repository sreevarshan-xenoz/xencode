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
