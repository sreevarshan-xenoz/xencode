# Xencode Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Core Features](#core-features)
    - [Credential Vault](#credential-vault)
    - [Real-Time Health Monitoring (WebSocket)](#real-time-health-monitoring-websocket)
    - [AI Ensemble Reasoning](#ai-ensemble-reasoning)
    - [Caching System](#caching-system)
    - [Vector Store](#vector-store)
4. [Performance Optimizations](#performance-optimizations)
5. [Visual Workflow Builder](#visual-workflow-builder)
6. [Multi-Agent Collaboration](#multi-agent-collaboration)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

## Overview

Xencode is an AI-powered development assistant platform that combines multiple AI models for superior reasoning through ensemble methods. It features advanced multi-agent collaboration, visual workflow building, and comprehensive monitoring capabilities.

The project is a **dual-stack architecture** — Python feature system + Rust core runtime. The Rust workspace (**12 crates, 65 tests, zero warnings**) handles workspace scanning, config management, caching, conversation memory, model health, provider streaming, code analysis, security scanning, HTTP/WebSocket server, collaboration sync, plugin lifecycle, and a ratatui-based TUI with 14 interactive feature panels. The Python stack (~180+ files) covers the full feature system, Textual-based TUI widget library, agentic workflows, analytics, security, and the FastAPI server.

The **Rust migration is complete** — all 8 phases implemented across 5 execution batches. The Rust binary (`xencode`) is the primary entry point with `server`, `analyze`, and `plugin` subcommands, and a rich TUI with 14 panels.

## Installation

### Option A: Rust binary (recommended)
```bash
cd rust && cargo build --release -p xencode-cli
./target/release/xencode --help
```

### Option B: Python package
```bash
pip install xencode
```

### Prerequisites
- Rust 1.75+ (for building from source)
- Python 3.8+ (for Python stack)
- Ollama (for local AI models)
- Git

### Development Setup
```bash
git clone <repository-url>
cd xencode
pip install -e .
cd rust && cargo build --release -p xencode-cli
```

## Core Features

### Credential Vault
Xencode includes an encrypted credential vault (`JsonFileCredentialVault`) that securely stores API keys and secrets in a local JSON file using Fernet encryption (AES-128-CBC with HMAC SHA256).

#### CLI Commands

```bash
# Initialize a new vault (creates ~/.xencode/vault.json)
xencode vault init

# Initialize with a custom path
xencode vault init --vault-path /custom/path/vault.json

# Check vault status
xencode vault status
```

#### Migration from Plaintext Config

```bash
# Scan config and migrate plaintext API keys into the vault
xencode vault migrate --config-path ~/.xencode/config.json

# Migrate and replace migrated keys with env-var references
xencode vault migrate --delete-after
```

#### First-Time Setup Integration
During first-time setup (`xencode health` or `xencode status` on a fresh install), Xencode automatically prompts you to migrate plaintext API keys into the vault.

#### Vault Encryption
- **Algorithm**: Fernet (AES-128-CBC with HMAC SHA256)
- **Key derivation**: PBKDF2 with machine-specific seed
- **Fallback**: Base64 obfuscation when the `cryptography` package is not installed
- **Storage**: Single JSON file at `~/.xencode/vault.json` (permissions: `0600`)

#### Programmatic Usage

```python
from xencode.auth.json_file_vault import JsonFileCredentialVault

# Create or open vault
vault = JsonFileCredentialVault()

# Store a credential
vault.set(Credential(
    service="openai",
    username="api_key",
    secret="sk-...",
    description="OpenAI API key",
))

# Retrieve a credential
cred = vault.get("openai", "api_key")
print(cred.secret)  # 'sk-...'

# Bulk operations
info = vault.get_storage_info()       # vault metadata
vault.clear_vault()                    # remove all credentials
export = vault.export_vault()          # export (secrets still encrypted)
vault.import_vault(Path("backup.json"))  # import from another vault
```

#### Migration Tutorial: Plaintext Config to Vault

This tutorial walks through migrating API keys from a plaintext config file into the encrypted credential vault.

---

##### Before You Start

Your config file likely looks like this:

```json
{
  "features": {
    "code_review": {
      "openai_api_key": "sk-your-openai-key-here",
      "model": "gpt-4"
    },
    "learning_mode": {
      "anthropic_api_key": "sk-ant-your-anthropic-key-here",
      "model": "claude-3"
    }
  }
}
```

Keys are stored as **plaintext** — any process that can read the file can steal them.

---

##### Step 1: Initialize the Vault

```bash
xencode vault init
```

This creates an encrypted vault at `~/.xencode/vault.json`.

* Vault directory created
* Empty vault file written with `0600` permissions
* Safe to run multiple times -- won't overwrite an existing vault

To check it exists:

```bash
ls -la ~/.xencode/vault.json
```

---

##### Step 2: Check Vault Status

```bash
xencode vault status
```

Expected output (before migration):

```
Vault Path:    C:\Users\you\.xencode\vault.json
Credential Count:  0
Encryption:    Fernet (AES-128-CBC)
```

---

##### Step 3: Migrate Plaintext Keys into the Vault

```bash
xencode vault migrate --config-path ~/.xencode/config.json
```

The migrator scans your config for known API key field names (`openai_api_key`, `anthropic_api_key`, `google_gemini_api_key`, etc.), encrypts each value, and stores it in the vault.

Expected output:

```
OK: Stored credential for openai (encrypted)
OK: Stored credential for anthropic (encrypted)
Migrated 2 credentials from config
```

---

##### Step 4: Verify the Migration

```bash
xencode vault status
```

Expected output (after migration):

```
Vault Path:    C:\Users\you\.xencode\vault.json
Credential Count:  2
Encryption:    Fernet (AES-128-CBC)
```

You can also check via the REST API:

```bash
curl http://localhost:8000/api/v1/vault/health
```

---

##### Step 5 (Optional): Delete Plaintext Keys

Pass `--delete-after` to replace plaintext keys with environment-variable references:

```bash
xencode vault migrate --config-path ~/.xencode/config.json --delete-after
```

After this, your config file is marked as migrated:

```json
{
  "_migrated_to_vault": true,
  "_vault_migrated_at": "2026-05-27T14:30:00+00:00",
  "features": {
    "code_review": {
      "openai_api_key": "${OPENAI_API_KEY}",
      "model": "gpt-4"
    }
  }
}
```

Original plaintext values are removed from the config they exist only in the encrypted vault.

---

##### Step 6: Use the Vault in Code

```python
from xencode.auth.json_file_vault import JsonFileCredentialVault

vault = JsonFileCredentialVault()

# Retrieve the key at runtime
cred = vault.get("openai", "api_key")
api_key = cred.secret  # "sk-your-openai-key-here"

# Pass it to your AI client
client = OpenAI(api_key=api_key)
```

---

##### One-Command Migration (If Eligible)

If both `~/.xencode/config.json` and default paths apply, you can combine init + migrate:

```bash
xencode vault init
xencode vault migrate --delete-after
```

---

##### Machine-to-Machine Migration

To move credentials to another machine, use export/import:

```bash
# On source machine
python -c "
from xencode.auth.json_file_vault import JsonFileCredentialVault
from pathlib import Path
vault = JsonFileCredentialVault()
vault.export_vault(Path('vault-export.json'))
"

# Copy vault-export.json to the target machine

# On target machine
python -c "
from xencode.auth.json_file_vault import JsonFileCredentialVault
from pathlib import Path
vault = JsonFileCredentialVault()
count = vault.import_vault(Path('vault-export.json'))
print(f'Imported {count} credentials')
"
```

> **Note:** Encrypted credentials from another machine cannot be decrypted without the original machine's key. Use a shared master key (`--master-key`) when initializing vaults on machines that need to share credentials.

---

##### What the Vault Protects Against

| Threat | Protection |
|---|---|
| Config file leaked to git | Plaintext keys never reach tracked config after `--delete-after` |
| Backup file exposed | Vault JSON is encrypted with Fernet |
| File read by unauthorized process | Vault file permissions are `0600` (owner only) |
| Credential rollover | Single vault path to audit and update |

#### Real-Time Health Monitoring (WebSocket)

Xencode exposes a WebSocket endpoint for streaming live vault health updates.

**Endpoint:** `ws://localhost:8000/api/v1/vault/health/ws`

##### Query Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `token` | string | *required* | JWT access token for authentication |
| `interval` | float | `5.0` | Polling interval in seconds (1–300) |
| `vault_path` | string | *optional* | Custom path to the vault file |

##### Authentication

All WebSocket connections require a valid JWT token passed as a query parameter:

```
ws://localhost:8000/api/v1/vault/health/ws?token=<JWT>
```

If the token is missing or invalid, the server closes the connection with code **4001**. The secret key is resolved using the same chain as REST endpoints (vault → `XENCODE_JWT_SECRET_KEY` env var → dev default).

##### Server → Client Messages

On connect the server immediately sends an initial `health_update` snapshot, then repeats every *interval* seconds.

```json
{
  "type": "health_update",
  "available": true,
  "vault_exists": true,
  "vault_path": "/home/user/.xencode/vault.json",
  "is_readable": true,
  "is_valid_json": true,
  "credential_count": 3,
  "encryption_available": true,
  "timestamp": "2026-06-01T12:00:00.000000"
}
```

##### Client → Server Messages

| Message | Response |
|---|---|
| `{"type": "ping"}` | `{"type": "pong", "timestamp": "..."}` |
| `{"type": "check_now"}` | Immediate `health_update` payload |
| `{"type": "set_interval", "interval": 10}` | `{"type": "interval_updated", "interval": 10, "timestamp": "..."}` |

##### CLI Monitor Command

The quickest way to watch vault health from a terminal:

```bash
# Connect with defaults (localhost:8000, 5 s interval)
xencode vault monitor

# Custom server and interval
xencode vault monitor --server-url http://my-host:8000 --interval 10

# Watch a non-default vault file
xencode vault monitor --vault-path /custom/path/vault.json

# Provide a token manually
xencode vault monitor --token eyJhbGci...
```

The command renders a live Rich table that refreshes on every update.

> **Requirements:** `websockets` and `PyJWT` must be installed (`pip install websockets PyJWT`).

##### Programmatic WebSocket Client

```python
import asyncio
import json
from datetime import datetime, timedelta
from websockets.asyncio.client import connect

import jwt
from xencode.api.auth import resolve_jwt_secret

# Generate a token
secret = resolve_jwt_secret()
now = datetime.utcnow()
token = jwt.encode(
    {
        "user_id": "monitor-client",
        "role": "admin",
        "type": "access",
        "iat": now,
        "exp": now + timedelta(hours=1),
    },
    secret,
    algorithm="HS256",
)

async def monitor():
    uri = f"ws://localhost:8000/api/v1/vault/health/ws?token={token}&interval=5"
    async with connect(uri) as ws:
        # Receive initial snapshot
        data = json.loads(await ws.recv())
        print(f"Vault exists: {data['vault_exists']}, credentials: {data['credential_count']}")

        # Request an on-demand check
        await ws.send(json.dumps({"type": "check_now"}))
        data = json.loads(await ws.recv())
        print(f"Updated credential count: {data['credential_count']}")

        # Change interval to 10 s
        await ws.send(json.dumps({"type": "set_interval", "interval": 10}))
        ack = json.loads(await ws.recv())  # interval_updated
        print(f"Server acknowledged: interval={ack['interval']}")

        # Listen for a few updates
        for _ in range(3):
            data = json.loads(await ws.recv())
            if data["type"] == "health_update":
                print(f"[{data['timestamp']}] credentials={data['credential_count']}")

asyncio.run(monitor())
```

##### REST Endpoints Summary

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/v1/vault/health` | Bearer JWT | One-shot vault health check |
| `POST` | `/api/v1/vault/init` | Bearer JWT | Initialize a new vault |
| `POST` | `/api/v1/vault/migrate` | Bearer JWT | Migrate plaintext keys into vault |
| `WS` | `/api/v1/vault/health/ws` | `?token=` | Real-time health monitoring stream |

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

### Vector Store
The canonical vector store implementation lives in `xencode.rag.vector_store` and includes:
- Synchronous `VectorStore`
- Asynchronous `AsyncVectorStore`
- Batch-optimized `OptimizedVectorStore`

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

### Natural Language Generation
You can also generate workflows automatically from natural language descriptions:

```python
# Generate from description
success = await builder.generate_from_description(
    "Create a workflow that takes user input, analyzes it with an AI model, and displays the result."
)
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