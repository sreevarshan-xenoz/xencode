# Xencode Feature Plan

Based on the robust foundation (FastAPI, Ollama integration, TUI, Agentic framework), this document outlines a curated list of high-impact features that developers typically love. These range from quality-of-life improvements to advanced AI capabilities that leverage the existing architecture.

## 1. Deep Context: Local RAG (Retrieval-Augmented Generation)
Currently, developers often have to copy-paste code or upload files to get context. Implementing a local Vector Store allows the AI to "know" the entire codebase instantly.

- The Feature: Automatically index the user's project files into a local vector database (like ChromaDB or Qdrant). When the user asks "How does the auth middleware work?", the system retrieves the relevant code chunks and sends them to the LLM as context.
- Why Devs Love It: It feels like the AI has actually read the code. It eliminates the "copy-paste fatigue."
- Implementation:
  - Add a `vector_store` module.
  - Create a CLI command: `xencode index .` (watches for changes).
  - Update `agentic/coordinator.py` to perform a similarity search before sending the prompt to Ollama.

## 2. Git Workflow Automation
Developers live in Git. Removing friction from Git tasks is a massive productivity booster.

- Automated Commit Messages: Run `xencode commit`. It stages files, runs `git diff`, sends the diff to the LLM, and generates a conventional commit message (e.g., `feat: add user login`).
- PR Reviewer Bot: Since you already have a FastAPI backend, create a GitHub App or a GitLab webhook integration. When a Pull Request is opened, Xencode analyzes the diff and leaves comments suggesting improvements or flagging potential bugs.
- Implementation:
  - Use `GitPython` library for local commands.
  - Use `PyGithub` for the API integration.
  - Add these as "Tools" in `agentic/advanced_tools.py`.

## 3. Test & Documentation Generation
Everyone hates writing unit tests and docstrings, but everyone loves having them.

- Unit Test Generator: `xencode test generate <path_to_file>`. The AI reads the function logic and generates pytest test cases, including edge cases.
- Docstring Standardizer: `xencode docstring <path>`. Enforces Google or NumPy style docstrings across the project.
- Implementation:
  - Leverage your `analyzers` module to parse AST (Abstract Syntax Trees) to understand function signatures without needing the LLM to parse raw text.
  - Add specific prompts in your TUI to prompt for "Generate Tests for current view."

## 4. The "IDE Bridge" (VS Code Extension)
You have a great TUI, but developers live in VS Code or JetBrains. A lightweight extension that talks to your local FastAPI server would be a game-changer.

- The Feature: A VS Code extension that sends the selected text to your running `xencode` server (localhost:8000) and displays the response inline, or opens a side panel for chat.
- Why Devs Love It: They never have to leave their editor.
- Implementation:
  - Your API is already FastAPI; just add an endpoint `/api/v1/chat` that accepts `{ context: str, query: str }`.
  - Build a simple JS extension that calls this endpoint.

## 5. Code Smell & Refactoring Agent
Move beyond simple "search" to active "improvement."

- Refactoring Suggestions: specific command `xencode refactor <file>`. Ask the AI to "Optimize for performance" or "Convert to Async/Await."
- Security Scanner (Enhanced): You have a `security_analyzer`. Enhance it to auto-fix vulnerabilities. Instead of just reporting "SQL Injection risk," generate the patched code using parameterized queries.
- Implementation:
  - Create a "Fix Mode" in the CLI/TUI that applies a patch directly to the file (with user confirmation).

## 6. Session Persistence & "Golden Prompts"
Developers often repeat the same complex instructions (e.g., "Act as a senior Python architect and review this code for PEP8 compliance...").

- Prompt Library: Allow users to save named prompts.
- Chat History Export: Export TUI sessions as Markdown or Jupyter Notebooks for documentation purposes.
- Implementation:
  - Store these in your SQLAlchemy DB (`UserSavedPrompts` table).
  - Add UI controls in `tui/app.py` to recall these prompts.

## 7. Performance Profiling (Cost & Speed)
Since you use local LLMs, developers care about resource usage.

- Token Velocity Dashboard: Extend your `analytics` module to show "Tokens/Second" and "Time to First Token" (TTFT) in the TUI sidebar.
- Why Devs Love It: Helps them tune their Ollama models (quantization, context window) for the best speed.

## 8. Extensible Plugin System
You have a plugin structure, but make it Python-package installable.

- The Feature: Allow users to `pip install xencode-plugin-jira` or `xencode-plugin-aws`.
- Why Devs Love It: It allows the community to build connectors for the tools they use (Jira, Linear, AWS, Terraform).
- Implementation:
  - Define a strict abstract base class in `plugins/base.py`.
  - Use entry points in `pyproject.toml` so your app automatically discovers installed plugins.

