# Xencode Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Core Features](#core-features)
4. [Performance Optimizations](#performance-optimizations)
5. [Visual Workflow Builder](#visual-workflow-builder)
6. [Multi-Agent Collaboration](#multi-agent-collaboration)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

## Overview

Xencode is an AI-powered development assistant platform that combines multiple AI models for superior reasoning through ensemble methods. It features advanced multi-agent collaboration, visual workflow building, and comprehensive monitoring capabilities.

## Installation

### Prerequisites
- Python 3.8+
- Ollama (for local AI models)
- Git

### Quick Start
```bash
pip install xencode
```

### Development Setup
```bash
git clone <repository-url>
cd xencode
pip install -e .
```

## Core Features

### AI Ensemble Reasoning
The core of Xencode is its multi-model ensemble system that combines responses from multiple AI models for better accuracy and reliability.

#### Supported Methods
- **VOTE**: Simple majority voting across model responses
- **WEIGHTED**: Weighted voting based on model confidence and performance
- **SEMANTIC**: Semantic-aware fusion using embeddings
- **CONSENSUS**: Consensus-based selection with fallback
- **HYBRID**: Adaptive method selection based on response characteristics

### Caching System
Xencode features a hybrid caching system with:
- In-memory caching for fast access (<1ms response times)
- Disk-based caching with SQLite for persistence
- LZMA compression for efficient storage
- LRU eviction policies

## Performance Optimizations

### Ensemble System Optimizations

#### 1. Parallel Model Availability Checking
The `_get_available_models` method now performs availability checks in parallel, reducing startup time when multiple models are requested.

**Before:**
- Sequential model checks (O(n) time complexity)
- 1-second timeout per model check

**After:**
- Parallel model checks (O(1) effective time complexity)
- Reduced timeout to 0.8 seconds
- Fallback model checking in parallel

#### 2. Efficient Consensus Calculation
The `calculate_consensus` method has been optimized to reduce the computational complexity of pairwise similarity calculations.

**Improvements:**
- More efficient set operations
- Early termination conditions
- Reduced memory allocations

#### 3. Optimized Parallel Inference
The `_parallel_inference` method now includes:
- Named asyncio tasks for better debugging
- More efficient response processing
- Improved error handling

#### 4. Streamlined Confidence Calculation
The `_calculate_confidence` method now calculates all metrics in a single pass, reducing iteration overhead.

### Multi-Agent Collaboration Optimizations

#### 1. Resource Management
- Market-based resource allocation system
- Negotiation protocols between agents
- Swarm intelligence behaviors
- Cross-domain expertise combination

#### 2. Communication Protocol
- Efficient message queuing system
- Subscription-based communication model
- Asynchronous message handling

## Visual Workflow Builder

The Visual Workflow Builder allows users to create and modify AI workflows through a drag-and-drop interface.

### Core Concepts

#### Node Types
- **INPUT**: Data input nodes
- **PROCESSING**: Data processing nodes
- **AI_MODEL**: AI model execution nodes
- **CONDITIONAL**: Conditional logic nodes
- **OUTPUT**: Result output nodes
- **DATA_SOURCE**: External data source nodes
- **TRANSFORMATION**: Data transformation nodes

#### Connection Types
- **DATA_FLOW**: Standard data flow between nodes
- **CONTROL_FLOW**: Control flow for conditional execution
- **TRIGGER**: Trigger-based connections

### Usage Example

```python
from xencode.visual_workflow_builder import WorkflowBuilder, NodeType, ConnectionType

builder = WorkflowBuilder()
builder.workflow_name = "Code Review Workflow"

# Add nodes
input_node = builder.add_node(NodeType.INPUT, "Code Input", 0, 0)
analysis_node = builder.add_node(NodeType.PROCESSING, "Code Analysis", 20, 0)
ai_node = builder.add_node(NodeType.AI_MODEL, "AI Review", 40, 0)
output_node = builder.add_node(NodeType.OUTPUT, "Review Output", 60, 0)

# Connect nodes
builder.connect_nodes(input_node, analysis_node)
builder.connect_nodes(analysis_node, ai_node)
builder.connect_nodes(ai_node, output_node)

# Validate and execute
errors = builder.validate_workflow()
if not errors:
    result = builder.execute_workflow()
    print(f"Execution result: {result}")
```

### Templates
The workflow builder supports templates for common workflow patterns:

```python
# Create a template
template_id = builder.create_template(
    "Code Review Template",
    "Standard template for code review workflows",
    ["ai", "development", "review"]
)

# Apply a template
builder.apply_template(template_id)
```

## Multi-Agent Collaboration

Xencode features an advanced multi-agent collaboration system with several coordination strategies.

### Agent Roles
- **COORDINATOR**: Manages overall task coordination
- **SPECIALIST**: Domain-specific expertise
- **GENERALIST**: General-purpose tasks
- **MONITOR**: Monitoring and reporting
- **VALIDATOR**: Quality assurance and validation
- **RESOURCE_MANAGER**: Resource allocation and management

### Coordination Strategies

#### 1. Market-Based Allocation
Resources are allocated using a market-based auction system where agents bid for resources based on their capabilities and current workload.

#### 2. Negotiation Protocols
Agents can negotiate with each other to resolve conflicts, share resources, or coordinate complex tasks.

#### 3. Swarm Intelligence
Behavior-based coordination inspired by natural systems, including:
- Foraging behavior for resource discovery
- Consensus behavior for agreement
- Task allocation based on fitness

#### 4. Human-in-the-Loop Supervision
Critical decisions can be escalated to human supervisors for approval.

#### 5. Cross-Domain Expertise
Combines expertise from multiple domains to solve complex problems.

### Usage Example

```python
from xencode.multi_agent_collaboration import CollaborationOrchestrator, Agent, AgentRole, AgentCapabilities

# Create orchestrator
orchestrator = CollaborationOrchestrator()

# Create agents
coding_specialist_capabilities = AgentCapabilities(
    skills=["python", "javascript", "code_review"],
    processing_power=7,
    available_resources={"memory": "medium", "cpu": "high"},
    specialization="software_development",
    max_concurrent_tasks=3
)
coding_specialist = Agent("coder_001", AgentRole.SPECIALIST, coding_specialist_capabilities)
orchestrator.register_agent(coding_specialist)

# Create team
team_id = orchestrator.create_team(
    ["coder_001"],
    "Software Development Team"
)

# Allocate resources using market-based system
allocated_agent = await orchestrator.allocate_resource_market_based(
    "computing_power", 
    "coder_001", 
    priority=9
)

# Execute swarm intelligence behavior
swarm_result = await orchestrator.execute_swarm_behavior(
    "consensus", 
    ["coder_001"],
    {"topic": "feature_priority", "options": ["high", "medium", "low"]}
)
```

## API Reference

### Ensemble Reasoner API

#### `EnsembleReasoner`
Main class for ensemble reasoning operations.

**Methods:**
- `reason(query: QueryRequest) -> QueryResponse`: Main reasoning method
- `benchmark_models(test_prompts: List[str]) -> Dict[str, Any]`: Benchmark model performance

#### `QueryRequest`
Request object for ensemble queries.

**Fields:**
- `prompt: str`: Input prompt for reasoning
- `models: List[str]`: Models to use in ensemble
- `method: EnsembleMethod`: Ensemble fusion method
- `max_tokens: int`: Maximum tokens per response
- `temperature: float`: Sampling temperature
- `timeout_ms: int`: Per-model timeout in milliseconds
- `require_consensus: bool`: Require model agreement
- `use_rag: bool`: Use Local RAG for context

### Visual Workflow Builder API

#### `WorkflowBuilder`
Main class for building visual workflows.

**Methods:**
- `add_node(...) -> str`: Add a node to the workflow
- `remove_node(node_id: str) -> bool`: Remove a node
- `connect_nodes(...) -> str`: Connect two nodes
- `move_node(...) -> bool`: Move a node to new position
- `validate_workflow() -> List[str]`: Validate workflow for errors
- `execute_workflow() -> Dict[str, Any]`: Execute the workflow
- `save_workflow(filename: str) -> bool`: Save workflow to file
- `load_workflow(filename: str) -> bool`: Load workflow from file

## Troubleshooting

### Common Issues

#### Model Availability
If models are not available, ensure Ollama is running and models are pulled:
```bash
ollama serve  # Start Ollama server
ollama pull llama3.1:8b  # Pull required models
```

#### Performance Issues
- Check system resources (CPU, memory, disk)
- Verify model availability
- Review cache configuration

#### Multi-Agent Communication
- Ensure all agents are properly registered
- Check communication protocol connectivity
- Verify shared memory access

### Performance Tuning

#### Caching
- Adjust cache sizes based on available memory
- Monitor cache hit rates
- Tune eviction policies for your use case

#### Model Selection
- Choose models appropriate for your hardware
- Consider response time vs. quality trade-offs
- Use fallback models for reliability

## Contributing

We welcome contributions to Xencode! Please see our contributing guidelines for more information.

## License

Xencode is released under the MIT License.