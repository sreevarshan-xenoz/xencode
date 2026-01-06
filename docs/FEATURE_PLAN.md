# Xencode Feature Plan

Based on the robust foundation (FastAPI, Ollama integration, TUI, Agentic framework), this document outlines a curated, prioritized list of high-impact features. These are organized by development priority, moving from core "must-haves" to advanced ecosystem expansions.

---

## Phase 1: The Foundation (Core Intelligence & Access)
*Goal: Establish Xencode as the omniscient assistant that lives where the developer works.*

### 1. Deep Context: Local RAG (Retrieval-Augmented Generation)
*The Brain.*
- **The Feature**: Automatically index the user's project files into a local vector database. When the user asks "How does the auth middleware work?", the system retrieves the relevant code chunks for context.
- **Why Devs Love It**: It feels like the AI has actually read the code. Eliminates "copy-paste fatigue."
- **Implementation**:
  - Add a `vector_store` module (ChromaDB/Qdrant).
  - CLI command: `xencode index .` (with file watchers).
  - Update `agentic/coordinator.py` to prompt with retrieved context.

### 2. Natural Language Shell (Shell Genie)
*The Hands.*
- **The Feature**: Translate natural language instructions (e.g., "Find all large log files and gzip them") into safe, executable shell commands.
- **Why Devs Love It**: Solves the "how do I do `tar` again?" problem without leaving the terminal context.
- **Implementation**:
  - LLM translation layer with a "Do vs. Explain" safety prompt.
  - Interactive confirmation loop in TUI ("Run this? [y/N]").

### 3. Git Workflow Automation
*The Daily Driver.*
- **The Feature**: 
  - `xencode commit`: Stages files, analyzes diffs, and generates conventional commit messages.
  - `xencode review`: Analyzes local changes or PRs and suggests improvements/flags bugs.
- **Why Devs Love It**: Removes friction from the most repetitive part of the job.
- **Implementation**:
  - `GitPython` for local op operations.
  - `PyGithub` for PR review integration.

---

## Phase 2: Accelerator Tools (High-Frequency Utility)
*Goal: Automate the tedious parts of coding directly.*

### 4. Test & Documentation Generator
- **The Feature**: 
  - `xencode test generate <path>`: Reads logic and generates defined pytest cases/mocks.
  - `xencode docstring <path>`: Enforces style guides (Google/NumPy) on existing code.
- **Why Devs Love It**: Everyone hates writing boilerplate tests and docs, but everyone loves having them.
- **Implementation**:
  - Use `ast` analysis to locate function boundaries specifically.

### 5. DevOps & Infrastructure Generator
- **The Feature**: Auto-generate `Dockerfile`, `docker-compose.yml`, GitHub Actions workflows, and K8s manifests based on project analysis (`requirements.txt`, `package.json`, etc.).
- **Why Devs Love It**: Lowers the barrier to entry for deployment; removes "dependency hell."
- **Implementation**:
  - Template-based generation with LLM filling in version/env specifics.

### 6. Code Smell & Refactoring Agent
- **The Feature**: Active improvement suggestions. `xencode refactor <file> --goal="performance"` or `xencode fix <security-report>`.
- **Why Devs Love It**: It's like having a senior engineer pair program with you to clean up technical debt.
- **Implementation**:
  - Combine `security_analyzer` output with a patch-generation agent.

---

## Phase 3: Deep Integration (Workflow Embedding)
*Goal: Remove context switching entirely.*

### 7. The "IDE Bridge" (VS Code Extension)
- **The Feature**: A lightweight VS Code/JetBrains extension that talks to the local `xencode` server. Highlight code -> "Ask Xencode".
- **Why Devs Love It**: They never have to leave their primary editor environment.
- **Implementation**:
  - Add `/api/v1/chat` endpoint to FastAPI.
  - Simple TypeScript extension for the client side.

### 8. "Shadow Mode" Autocomplete
- **The Feature**: A privacy-first, local alternative to GitHub Copilot. Runs a small, fast model (like Codestral or DeepSeek-Coder) in the background to provide single-line or block completions.
- **Why Devs Love It**: Low latency, offline capable, and zero data leakage.
- **Implementation**:
  - Requires efficient local inference (e.g., `llama.cpp` server integration).
  - Editor plugin integration via LSP.

### 9. Smart Database Assistant
- **The Feature**: NL-to-SQL generation. "Show me the top 5 users by spend in the last month."
- **Why Devs Love It**: Makes ad-hoc data analysis trivial; prevents dangerous queries (e.g., missing `WHERE`).
- **Implementation**:
  - Introspect DB schema (via SQLAlchemy) to provide table context to the LLM.

---

## Phase 4: Advanced Visualization & Experimentation
*Goal: Provide higher-level understanding and transparency.*

### 10. Interactive Architecture Visualizer
- **The Feature**: Generate dynamic dependency graphs and flow charts (Mermaid/React Flow) from the codebase.
- **Why Devs Love It**: Incredible for onboarding and understanding legacy codebases.
- **Implementation**:
  - Static analysis -> Graph data structure -> Frontend renderer.

### 11. Multi-Model "Arena" Mode
- **The Feature**: Run a prompt against multiple local models simultaneously (e.g., Llama 3 vs. Mistral) to see which gives better code.
- **Why Devs Love It**: Allows rapid testing of new open-source models for specific tasks.
- **Implementation**:
  - Parallel API calls to Ollama; split-pane view in TUI.

### 12. System Health & Performance Profiling
- **The Feature**: Token velocity dashboard (Tokens/sec, TTFT) and cost estimation.
- **Why Devs Love It**: Essential for tuning local deployment interactions and hardware usage.

---

## Phase 5: Ecosystem & Future Tech
*Goal: Expand the platform's capabilities limitlessly.*

### 13. API Kitchen & Mock Server
- **The Feature**: Define an API spec (or just describe it), and Xencode spins up a live mock server with realistic data.
- **Why Devs Love It**: FE devs can work before BE is ready. Great for testing edge cases.
- **Implementation**:
  - Dynamic FastAPI route generation based on Pydantic models.

### 14. Voice Command Interface
- **The Feature**: "Xencode, run the test suite and tell me if it passes." Hands-free interaction.
- **Why Devs Love It**: Reduces RSI; great for "thinking out loud" workflows.
- **Implementation**:
  - Whisper (local) for STT -> Command Router -> TTS response.

### 15. Session Persistence & "Golden Prompts"
- **The Feature**: Save and organize successful prompt patterns; export chat history as Markdown/Notebooks.
- **Why Devs Love It**: Builds a reusable library of engineering knowledge.

### 16. Extensible Plugin System
- **The Feature**: `pip install xencode-plugin-jira`.
- **Why Devs Love It**: Connects Xencode to their unique proprietary tools/platforms.
- **Implementation**:
  - `pluggy` or standard Python entry-points.
