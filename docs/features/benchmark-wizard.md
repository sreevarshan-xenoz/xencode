# Model Benchmark Wizard (S5-04)

## Overview

The Model Benchmark Wizard provides automated benchmarking and comparison of AI providers and models. It measures performance, quality, and cost metrics to generate data-driven recommendations.

## Features

- **Automated Benchmarking**: Run standardized tests against multiple providers
- **Performance Metrics**: Latency, throughput, tokens/second
- **Quality Metrics**: Accuracy, consistency scores
- **Cost Analysis**: Cost per request, cost efficiency rankings
- **Recommendations**: Use-case specific model recommendations
- **Historical Tracking**: Performance trends over time

## Quick Start

### Run a Benchmark Suite

```bash
curl -X POST http://localhost:8000/api/v1/monitoring/benchmarks/run \
  -H "Content-Type: application/json" \
  -d '{
    "suite_name": "code_generation",
    "providers": [
      {"provider": "ollama", "model": "llama3.2"},
      {"provider": "openai", "model": "gpt-3.5-turbo"}
    ]
  }'
```

### Get Recommendations

```bash
curl "http://localhost:8000/api/v1/monitoring/benchmarks/recommendations?task_type=code_generation&use_case=production"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Benchmark Wizard System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │  Benchmark       │  │  Benchmark       │  │  Benchmark    │ │
│  │  Engine          │  │  Suites          │  │  Store        │ │
│  │                  │  │                  │  │               │ │
│  │ - Execute tasks  │  │ - Pre-defined    │  │ - SQLite      │ │
│  │ - Measure metrics│  │   suites         │  │ - Persistence │ │
│  │ - Calculate      │  │ - Custom tasks   │  │ - Queries     │ │
│  │   scores         │  │ - Datasets       │  │ - Aggregation │ │
│  └────────┬─────────┘  └──────────────────┘  └───────┬───────┘ │
│           │                                          │           │
│           │         ┌──────────────────┐             │           │
│           └────────▶│  Recommendations │◀────────────┘           │
│                     │  Engine          │                         │
│                     │                  │                         │
│                     │ - Model scoring  │                         │
│                     │ - Recommendations│                         │
│                     │ - Cost/quality   │                         │
│                     │   analysis       │                         │
│                     └────────┬─────────┘                         │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  API Endpoints   │
                    │                  │
                    │ - POST /run      │
                    │ - GET /results   │
                    │ - GET /comparison│
                    │ - GET /recommend │
                    └──────────────────┘
```

## API Endpoints

### POST /benchmarks/run

Run a benchmark suite against specified providers.

**Request:**
```json
{
  "suite_name": "code_generation",
  "providers": [{"provider": "ollama", "model": "llama3.2"}],
  "concurrent": true
}
```

**Response:**
```json
{
  "run_id": "abc-123",
  "status": "completed",
  "total_tasks": 10,
  "successful": 9,
  "avg_latency_ms": 245.5,
  "avg_accuracy_score": 0.87
}
```

### GET /benchmarks/results

Get benchmark results with optional filters.

**Query Parameters:**
- `run_id`: Filter by specific run
- `provider`: Filter by provider
- `model`: Filter by model
- `task_type`: Filter by task type
- `limit`: Maximum results (default: 100)

**Response:**
```json
[
  {
    "run_id": "abc-123",
    "task_name": "simple_function",
    "task_type": "code_generation",
    "provider": "ollama",
    "model": "llama3.2",
    "latency_ms": 245.5,
    "tokens_per_sec": 45.2,
    "accuracy_score": 0.87,
    "cost_per_request": 0.0,
    "success": true,
    "timestamp": "2026-03-24T10:30:00"
  }
]
```

### GET /benchmarks/comparison

Compare provider performance.

**Query Parameters:**
- `task_type`: Filter by task type

**Response:**
```json
{
  "task_type": "code_generation",
  "providers": [
    {
      "provider": "ollama",
      "model": "llama3.2",
      "overall_score": 85.5,
      "performance_score": 90.0,
      "quality_score": 82.0,
      "cost_score": 100.0,
      "avg_latency_ms": 245.5,
      "avg_accuracy": 0.87,
      "avg_cost": 0.0
    }
  ],
  "best_provider": "ollama",
  "best_model": "llama3.2",
  "pareto_optimal_count": 2
}
```

### GET /benchmarks/recommendations

Get model recommendations.

**Query Parameters:**
- `task_type`: Task type (required)
- `use_case`: production, development, realtime, high_quality
- `max_budget`: Maximum cost per request
- `max_latency_ms`: Maximum acceptable latency
- `min_accuracy`: Minimum accuracy score

**Response:**
```json
{
  "task_type": "code_generation",
  "use_case": "production",
  "recommended_provider": "ollama",
  "recommended_model": "llama3.2",
  "confidence": 0.85,
  "reasons": [
    "Excellent performance (245ms avg latency)",
    "High accuracy (0.87 avg)",
    "Free to use"
  ],
  "alternatives": [
    {
      "provider": "openai",
      "model": "gpt-4",
      "overall_score": 78.5,
      "avg_latency_ms": 500,
      "avg_accuracy": 0.95,
      "avg_cost": 0.05
    }
  ],
  "tradeoffs": {
    "accuracy": "openai/gpt-4 has 0.08 higher accuracy"
  }
}
```

### GET /benchmarks/suites

List available benchmark suites.

**Response:**
```json
{
  "suites": {
    "code_generation": {
      "task_count": 5,
      "task_types": ["code_generation"]
    },
    "chat": {
      "task_count": 4,
      "task_types": ["chat"]
    }
  },
  "task_types": ["code_generation", "chat", "reasoning", ...]
}
```

## Pre-defined Suites

### Code Generation
- `simple_function`: Write basic functions
- `list_comprehension`: List comprehensions
- `class_definition`: Class definitions
- `error_handling`: Try-except blocks
- `async_function`: Async/await patterns

### Chat
- `greeting`: Basic greetings
- `follow_up`: Conversational follow-ups
- `creative_writing`: Creative content
- `role_play`: Role-playing scenarios

### Reasoning
- `math_word_problem`: Math problems
- `logical_deduction`: Logical reasoning
- `pattern_recognition`: Pattern completion
- `constraint_satisfaction`: Constraint problems

### Summarization
- `short_summary`: One-sentence summaries
- `key_points`: Key point extraction

### Translation
- `en_to_es`: English to Spanish
- `en_to_fr`: English to French

### Question Answering
- `factual_qa`: Factual questions
- `technical_qa`: Technical questions
- `code_qa`: Code-related questions

## Custom Benchmarks

Create custom benchmark tasks:

```python
from xencode.monitoring.benchmark_suites import BenchmarkSuites, TaskType

suites = BenchmarkSuites()

task = suites.create_custom_task(
    name="my_custom_task",
    task_type=TaskType.CODE_GENERATION,
    prompt="Write a function to...",
    expected_output="def my_function",
    weight=2.0,
)
```

Create custom suites:

```python
custom_tasks = [task1, task2, task3]
suites.create_custom_suite("my_suite", custom_tasks)
```

## Scoring System

### Performance Score (0-100)
- Based on latency and throughput
- Lower latency = higher score
- Higher tokens/sec = bonus
- Formula: `max(0, 100 - (avg_latency / 20)) + throughput_bonus`

### Quality Score (0-100)
- Based on accuracy and consistency
- Higher accuracy = higher score
- Low variance = consistency bonus
- Formula: `accuracy * 100 + consistency_bonus`

### Cost Score (0-100)
- Based on cost per request
- Free = 100
- Higher cost = lower score
- Formula: `max(0, 100 - (avg_cost * 1000))`

### Overall Score
Weighted combination based on use case:

| Use Case | Performance | Quality | Cost |
|----------|-------------|---------|------|
| production | 40% | 50% | 10% |
| development | 30% | 30% | 40% |
| realtime | 60% | 30% | 10% |
| high_quality | 20% | 70% | 10% |
| experimental | 20% | 30% | 50% |

## Integration with S5-02

The benchmark system integrates with provider health monitoring:

- Uses health status to filter available providers
- Correlates benchmark results with health metrics
- Provides comprehensive provider evaluation
- Shares SQLite storage for historical data

## Usage Examples

### Python API

```python
from xencode.monitoring.benchmark_engine import BenchmarkEngine
from xencode.monitoring.benchmark_suites import get_benchmark_suites
from xencode.monitoring.benchmark_recommendations import get_recommendations_engine

# Run benchmarks
engine = BenchmarkEngine()
suites = get_benchmark_suites()

tasks = suites.get_suite("code_generation")
summary = await engine.run_benchmark_suite(
    tasks=tasks,
    providers=[("ollama", "llama3.2")],
    suite_name="my_benchmark",
)

# Get recommendations
rec_engine = get_recommendations_engine()
rec = rec_engine.get_recommendations(
    task_type="code_generation",
    use_case="production",
)

print(f"Recommended: {rec.recommended_provider}/{rec.recommended_model}")
print(f"Reasons: {rec.reasons}")

await engine.close()
```

### CLI Example

```bash
# Run code generation benchmark
curl -X POST http://localhost:8000/api/v1/monitoring/benchmarks/run \
  -H "Content-Type: application/json" \
  -d '{"suite_name": "code_generation", "providers": [{"provider": "ollama", "model": "llama3.2"}]}'

# Get recommendations for production use
curl "http://localhost:8000/api/v1/monitoring/benchmarks/recommendations?task_type=code_generation&use_case=production"

# Compare providers
curl "http://localhost:8000/api/v1/monitoring/benchmarks/comparison?task_type=code_generation"
```

## Best Practices

1. **Run benchmarks regularly**: Provider performance can change over time
2. **Use appropriate use cases**: Select use case matching your needs (production vs development)
3. **Consider tradeoffs**: Cheapest isn't always best; balance cost, quality, and performance
4. **Check sample counts**: Higher sample counts = more confidence in recommendations
5. **Monitor trends**: Look for improving or declining performance over time
6. **Use filters**: Apply budget, latency, and accuracy filters for targeted recommendations
7. **Compare Pareto optimal**: Focus on Pareto-optimal models for best cost/quality balance

## Performance Considerations

- **Concurrent execution**: Benchmarks run concurrently by default for speed
- **SQLite indexing**: Results indexed by provider, model, task_type for fast queries
- **In-memory caching**: Recommendations engine caches model scores
- **Configurable limits**: Set limits on result queries to prevent slow operations

## Troubleshooting

### Benchmark module not available
Ensure all dependencies are installed:
```bash
pip install aiohttp
```

### No benchmark data available
Run a benchmark suite first:
```bash
curl -X POST http://localhost:8000/api/v1/monitoring/benchmarks/run \
  -H "Content-Type: application/json" \
  -d '{"suite_name": "code_generation", "providers": [{"provider": "ollama", "model": "llama3.2"}]}'
```

### Provider connection errors
Check that providers are accessible:
- Ollama: `http://localhost:11434`
- OpenAI: Requires API key
- Anthropic: Requires API key

## Testing

Run the test suite:
```bash
pytest tests/phase5/test_benchmark_wizard.py -v
```

Test coverage includes:
- Benchmark store CRUD operations
- Benchmark engine execution
- Benchmark suites
- Recommendations engine
- API endpoints
- Integration tests

## Files

```
xencode/monitoring/
├── benchmark_store.py           # SQLite persistence layer
├── benchmark_engine.py          # Core execution engine
├── benchmark_suites.py          # Task definitions
├── benchmark_recommendations.py # Recommendations engine
└── __init__.py                  # Package exports

xencode/api/routers/
└── monitoring.py                # API endpoints (extended)

tests/phase5/
└── test_benchmark_wizard.py     # Test suite
```

## Related

- **S5-02**: Provider Health Monitoring - Integrated health checks
- **S5-01**: Performance Monitoring - Shared metrics infrastructure
- **Model Providers**: Provider transport and resolution
