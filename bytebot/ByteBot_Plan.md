# ByteBot Integration Plan for Xencode

## Overview

This document outlines the comprehensive plan for integrating ByteBot as an agentic terminal IDE within the Xencode ecosystem. The goal is to create a terminal-based AI assistant that leverages both local and online models to provide intelligent command execution, project analysis, and development assistance.

## Current Architecture Analysis

### Xencode Core Components
- **xencode_core.py**: Main entry point with chat functionality, model management, and streaming capabilities
- **Shell Genie**: Natural language to shell command translation system
- **Agentic System**: Multi-agent coordination with specialized agents (code, research, execution, planning)
- **Advanced Tools**: Git operations, web search, code analysis, file operations
- **Visual Workflow Builder**: Drag-and-drop interface for creating AI workflows

### ByteBot External Project
The existing bytebot folder contains a separate project from bytebot-ai/bytebot that provides:
- Virtual desktop environment with Ubuntu Linux
- AI agent that can interact with GUI applications
- Task interface for natural language processing
- File upload and processing capabilities
- Live desktop view and takeover mode

## Integration Strategy

### 1. Single-Brain, Multi-Tool Architecture

#### Core Components
- **ByteBotEngine**: Single-brain system with execution modes
  - Planner (THINKS ONLY) - produces plan graphs
  - Executor (ACTS ONLY) - executes approved steps
  - SafetyGate (VETO POWER) - blocks dangerous operations
- **TerminalCognitionLayer**: The actual implementation layer that handles terminal interactions
- **Context Engine**: Event-driven context management that actively influences planning
- **Risk Scorer**: Evaluates command risk based on scope, privilege, and irreversibility

#### Integration Points
- Extend the existing Shell Genie functionality
- Integrate with the agentic system's execution tools
- Leverage the existing model management system
- Use the advanced caching system for command history

### 2. Execution Modes (Not Multiple Agents)

#### Replace agent roles with execution modes:

* **assist** → suggest only
* **execute** → auto-run safe steps
* **autonomous** → run everything except vetoed ops

This creates **single-brain, multi-tool** behavior without agent chaos.

#### Execution Flow
- Planner produces a **plan graph**, not commands
- Executor executes **only approved steps**
- SafetyGate can block or require confirmation

### 3. Two-Model Doctrine

#### Local Model
- ONE local model for 80% of tasks
  Example: `qwen3:4b` (fast, reliable, works offline)

#### Remote Model
- ONE "thinking" model for:
  - Planning
  - Large refactors
  - Multi-step reasoning
  - Complex analysis

Specialized models are an optimization, not a dependency.

### 4. Risk-Based Safety Model

#### Command Risk Scoring
Each command gets a **risk score** based on:
- Scope (single file vs system-wide)
- Privilege escalation
- Irreversibility
- Repo state (dirty git tree = higher risk)

#### Example
```json
{
  "command": "rm -rf build/",
  "risk": 0.32,
  "reason": "non-root, scoped, reversible via git"
}
```

#### Execution Policy
- Auto-execute low risk (< 0.3)
- Confirm medium risk (0.3-0.7)
- Hard block high risk (> 0.7)

### 5. Event-Driven Context Engine

#### Active Context Management
The Context Engine actively influences planning, not just prompt enrichment:

- On `git checkout`: re-evaluate project intent
- On dependency change: invalidate caches
- On test failure: raise context priority
- On file modification: update relevant context

## Implementation Roadmap

### Phase 1: Safe NL → Shell Execution (MVP - Required)
- Natural language to shell command translation
- Basic risk scoring and safety validation
- Simple context awareness (current directory, git status)
- Execution modes (assist/execute)
- Single local model integration (qwen3:4b)

### Phase 2: Deterministic Planning + Replay (Required)
- Plan graph generation and visualization
- Execution replay capability (`bytebot replay <task-id>`)
- Dry-run functionality (`bytebot plan --dry-run`)
- Plan serialization and storage
- Risk-based execution policies

### Phase 3: Context Awareness (Optional Expansion)
- Event-driven context engine
- Project state monitoring
- Git integration
- Dependency tracking

### Phase 4: Autonomy Gradients (Optional Expansion)
- Confirmation flows for medium-risk operations
- Supervision modes (assist/execute/autonomous)
- Advanced risk scoring

### Phase 5: UX + Delight (Optional Expansion)
- Terminal UI enhancements
- Progress indicators
- Error recovery
- Session management

## Execution Constraints

### Locked Scope
Only **Phase 1** and **Phase 2** are required for initial release.
Everything beyond these phases is optional expansion.

### Success Criteria for Initial Release
Before considering any additional phases, the system must reliably:
- Translate natural language to shell commands with >90% accuracy
- Execute commands with risk-based safety validation
- Generate and store plan graphs
- Successfully replay previous executions
- Support dry-run functionality

### Red Lines (Do Not Cross Until Success Criteria Met)
- Do not add more agents
- Do not add more models
- Do not add more autonomy
- Do not add more complex features

Focus on boring, stable, repeatable execution first.

## Technical Specifications

### ByteBotEngine Class
```python
class ByteBotEngine:
    def __init__(self, model_manager, tool_registry):
        self.model_manager = model_manager
        self.tool_registry = tool_registry
        self.context_engine = ContextEngine()
        self.planner = Planner()
        self.executor = Executor()
        self.safety_gate = SafetyGate()
        self.risk_scorer = RiskScorer()

    def process_intent(self, intent: str, mode: str = "execute") -> Dict[str, Any]:
        # Plan and execute user intent with specified mode
        pass

    def execute_plan_graph(self, plan_graph: Dict[str, Any]) -> Dict[str, Any]:
        # Execute plan graph with safety validation
        pass

    def replay_execution(self, execution_id: str) -> Dict[str, Any]:
        # Re-execute a previous task for debugging
        pass
```

### Execution Replay & Debugging
Every ByteBot action must be:
- Serializable
- Re-runnable
- Explainable

Store plans + commands as DAGs, allow `bytebot replay <task-id>`, and allow dry-run visualization.

### Risk Scoring System
- Context-aware risk evaluation
- Configurable risk thresholds
- Execution policy enforcement
- Audit trail for all decisions

## Integration Points

### With Existing Systems
1. **Model Management**: Use existing Ollama and API key integration
2. **Caching**: Leverage advanced caching system for command results
3. **Analytics**: Integrate with monitoring and analytics dashboard
4. **Security**: Use existing security governance system
5. **Plugins**: Enable plugin architecture for extended capabilities

### New Components
1. **Terminal Agent Coordinator**: Manages multiple terminal agents
2. **Command Safety Layer**: Validates and sanitizes commands
3. **Context Enrichment System**: Enhances commands with project context
4. **Execution Monitor**: Tracks and reports command execution status

## Expected Outcomes

### Immediate Benefits
- Natural language terminal commands with deterministic execution
- Context-aware development assistance with active context management
- Risk-based safety model with configurable autonomy levels
- Integration with existing tooling via single-brain, multi-tool architecture
- Support for both local and cloud models with two-model doctrine

### Long-term Vision
- Fully autonomous terminal-based development environment with replayable executions
- Intelligent project analysis with event-driven context awareness
- Deterministic planning with visualizable plan graphs
- Seamless integration with IDE workflows through TerminalCognitionLayer
- Advanced automation with configurable risk thresholds

## Risks and Mitigation

### Security Risks
- Risk: Unsafe command execution
- Mitigation: Risk-based scoring system with configurable thresholds and veto power

### Performance Risks
- Risk: Slow response times for complex tasks
- Mitigation: Single-brain architecture to avoid coordination overhead

### Usability Risks
- Risk: Overwhelming complexity for users
- Mitigation: Clear execution modes (assist/execute/autonomous) with progressive disclosure

### System Complexity Risks
- Risk: Agent coordination overhead and debugging difficulty
- Mitigation: Single-brain, multi-tool architecture instead of multi-agent system

## Success Metrics

### Quantitative
- Command execution success rate >95%
- Average response time <2 seconds for simple commands
- Risk assessment accuracy >98% (correctly identifying dangerous operations)
- Plan graph generation success rate >90%
- User satisfaction score >4.0/5.0

### Qualitative
- Positive user feedback on natural language interface
- Successful execution of complex multi-step operations with replay capability
- Seamless integration with existing workflows
- Effective safety measures preventing harmful operations while maintaining usability
- Deterministic behavior that users can predict and trust

## Conclusion

The ByteBot integration plan leverages Xencode's existing architecture while adding powerful agentic capabilities for terminal-based development. By adopting a single-brain, multi-tool architecture with risk-based safety and deterministic execution, we can create a sophisticated terminal-based AI assistant that enhances developer productivity while maintaining security and trustworthiness.

The focus on a locked scope of Phases 1 and 2 ensures we build a stable, debuggable foundation before expanding capabilities. The emphasis on replayability, risk-based safety, and two-model doctrine creates a system that is both maintainable and trustworthy.

Proceed with the locked scope: implement Safe NL → Shell Execution and Deterministic Planning + Replay first. Only after these core capabilities are stable and reliable should additional features be considered. This approach prioritizes structural soundness over feature breadth, ensuring ByteBot becomes engineering-grade infrastructure rather than a fragile demo.