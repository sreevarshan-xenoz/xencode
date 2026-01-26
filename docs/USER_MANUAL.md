# Xencode User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Command Reference](#command-reference)
7. [Troubleshooting](#troubleshooting)
8. [Examples](#examples)

## Introduction

Xencode is an AI-powered development assistant platform that integrates with local language models through Ollama. It provides intelligent code analysis, document processing, and workspace collaboration features with a focus on privacy and offline operation.

### Key Features
- **LangChain Agentic System**: Production-ready agent with 9 specialized tools
- **Multi-Agent Collaboration**: 4 specialized agents with intelligent task delegation
- **Ensemble Learning**: Multiple models working together for better results
- **Conversation Memory**: Persistent conversation history
- **Smart Model Selection**: Automatic model switching based on task type

## Installation

### Prerequisites
- Python 3.8 or higher
- Ollama installed and running
- 4GB+ RAM recommended

### Quick Install

**Linux / macOS**
```bash
git clone https://github.com/sreevarshan-xenoz/xencode.git
cd xencode
chmod +x install.sh && ./install.sh
```

**Windows**
```powershell
git clone https://github.com/sreevarshan-xenoz/xencode.git
cd xencode
.\install.ps1
```

### Manual Setup

1. Install Ollama from https://ollama.ai
2. Start Ollama service: `ollama serve`
3. Install a model: `ollama pull qwen3:4b`
4. Clone Xencode: `git clone https://github.com/sreevarshan-xenoz/xencode.git`
5. Navigate to directory: `cd xencode`
6. Install dependencies: `pip install -e .`

## Getting Started

### Starting Xencode

**Interactive Mode:**
```bash
./xencode.sh
```

**Inline Query:**
```bash
./xencode.sh "Explain how to reverse a linked list in Python"
```

### First-Time Setup

On first run, Xencode will guide you through:
1. Verifying Ollama installation
2. Checking available models
3. Installing a recommended model if none found
4. Configuring default settings

## Basic Usage

### Chat Mode
Enter chat mode by running `./xencode.sh` without arguments. You can then have a conversation with the AI:

```
$ ./xencode.sh
[Immersive banner appears]

You › Hello, can you help me with Python?

Xencode › [Thinking...]

Xencode › Of course! I'd be happy to help you with Python. What specifically would you like assistance with?
```

### Inline Queries
For quick questions, use inline mode:

```bash
$ ./xencode.sh "How do I sort a list in Python?"
You can sort a list in Python using the `sort()` method or the `sorted()` function...

$ ./xencode.sh "Debug this code: my_list = [3, 1, 4]; print(my_list.sort())"
The issue is that the `sort()` method sorts the list in-place and returns `None`...
```

### Model Selection
Specify a model with the `-m` flag:

```bash
./xencode.sh -m llama3.1:8b "Write a Python function to calculate factorial"
```

## Advanced Features

### Conversation Memory
Xencode remembers context from previous exchanges in the same session:

```
You › I'm working on a web scraping project in Python
Xencode › Great! What specific aspect of web scraping are you working on?

You › I need to extract product prices from an e-commerce site
Xencode › For extracting product prices, you can use libraries like BeautifulSoup...
```

### File Operations
Xencode can read, write, and modify files:

**Reading a file:**
```
You › /file read my_script.py
```

**Creating a file:**
```
You › /file create new_feature.py
[AI generates code and creates the file]
```

### Multi-Agent Collaboration
Xencode uses specialized agents for different tasks:
- **CodeAgent**: Programming tasks
- **ResearchAgent**: Web search and analysis  
- **ExecutionAgent**: File/command operations
- **GeneralAgent**: General queries

## Command Reference

### Chat Commands
- `/help` - Show help information
- `/clear` - Clear current conversation
- `/memory` - Show memory usage
- `/sessions` - List all sessions
- `/switch <id>` - Switch to a different session
- `/cache` - Show cache information
- `/status` - Show system status
- `/export` - Export conversation to file
- `/project` - Show project context
- `/theme <name>` - Change UI theme
- `/model <name>` - Switch to a different model
- `/models` - List available models with health status

### File Operations
- `/file create <path> <content>` - Create a file
- `/file read <path>` - Read a file
- `/file write <path> <content>` - Write to a file
- `/file delete <path>` - Delete a file

### Model Management
- `--list-models` - List installed models
- `--update <model>` - Update a model (use with -m)
- `-m <model>` - Specify a model for the query

## Examples

### Example 1: Code Review
```
$ ./xencode.sh
You › I wrote this Python function to calculate Fibonacci numbers. Can you review it?

def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

Xencode › [Thinking...]

Xencode › Your function correctly implements the Fibonacci sequence using recursion. However, there are some performance considerations...

You › Can you suggest a more efficient version?

Xencode › Certainly! Here's an iterative version that's much more efficient...
```

### Example 2: File Operations
```
You › /file create fibonacci.py
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

You › Thanks! Now can you add a test function to this file?

Xencode › [Thinking...]

Xencode › I'll add a test function to your fibonacci.py file...
[file is updated with test function]

You › /file read fibonacci.py
[Shows the updated file with both functions]
```

### Example 3: Complex Problem Solving
```
$ ./xencode.sh
You › I need to build a simple web scraper that extracts product names and prices from an e-commerce site. Can you help me design it?

Xencode › [Thinking...]

Xencode › I'll help you design a web scraper. Here's an approach using Python with requests and BeautifulSoup...

You › That looks good. Can you also add error handling and rate limiting?

Xencode › Absolutely! Here's the enhanced version with error handling and rate limiting...
```

### Example 4: Model Comparison
```
$ ./xencode.sh --list-models
📦 Installed Models
┌─────────────────────┬──────────┬───────────────┬────────────┐
│ Model               │ Status   │ Response Time │ Last Check │
├─────────────────────┼──────────┼───────────────┼────────────┤
│ llama3.1:8b         │ ✅ Healthy│ 1.234s       │ 14:30:22   │
│ mistral:7b          │ ✅ Healthy│ 0.987s       │ 14:30:21   │
│ qwen3:4b            │ ✅ Healthy│ 1.456s       │ 14:30:20   │
└─────────────────────┴──────────┴───────────────┴────────────┘

$ ./xencode.sh -m mistral:7b "Compare Python and JavaScript for backend development"
[Response from mistral:7b model]

$ /model qwen3:4b
✅ Model switched to qwen3:4b

$ "Same question but from your perspective?"
[Response from qwen3:4b model with different insights]
```

### Example 5: Session Management
```
$ ./xencode.sh
You › Let's start a new session about database design

Xencode › Sure! What specific aspect of database design are you interested in?

[After several exchanges...]

You › /sessions
💬 Conversation Sessions
┌────────────────────────────┬─────────┬────────┬────────────┬──────────────┐
│ Session ID                 │ Messages│ Model  │ Created    │ Last Updated │
├────────────────────────────┼─────────┼────────┼────────────┼──────────────┤
│ 🎯 session_1634567890     │ 5       │ qwen3:4b│ 2023-10-19 │ 14:35:42     │
│ session_1634567880        │ 12      │ llama3.1:8b│ 2023-10-19 │ 14:30:15     │
└────────────────────────────┴─────────┴────────┴────────────┴──────────────┘

You › /switch session_1634567880
✅ Switched to session: session_1634567880

You › What were we discussing in this session?
Xencode › In the previous session, we were discussing...
```

## Troubleshooting

### Common Issues

#### Ollama Not Running
**Problem:** "Cannot connect to Ollama service"
**Solution:** 
1. Start Ollama: `ollama serve`
2. Verify it's running: `curl http://localhost:11434/api/tags`

#### No Models Available
**Problem:** "No models found"
**Solution:**
1. Check available models: `ollama list`
2. Install a model: `ollama pull qwen3:4b`

#### Slow Responses
**Problem:** Long response times
**Solution:**
1. Check model health: `./xencode.sh --list-models`
2. Try a different model: `./xencode.sh -m mistral:7b "your query"`
3. Check system resources: `htop` or Task Manager

#### File Operation Errors
**Problem:** "Permission denied" or "File not found"
**Solution:**
1. Check file permissions
2. Verify file paths are correct
3. Ensure you have read/write permissions for the directory

### Performance Tips

1. **Use Caching**: Xencode caches responses to speed up repeated queries
2. **Choose Efficient Models**: Smaller models often respond faster
3. **Manage Memory**: Clear old sessions if experiencing slowdowns
4. **Optimize Prompts**: Clear, specific prompts yield faster responses

### Getting Help

- Use `/help` for command reference
- Check system status with `/status`
- Report issues on GitHub: https://github.com/sreevarshan-xenoz/xencode/issues
- Join discussions: https://github.com/sreevarshan-xenoz/xencode/discussions

## Best Practices

### Effective Prompting
- Be specific about what you need
- Provide context when relevant
- Break complex tasks into smaller queries
- Ask for explanations of code you don't understand

### Security
- Don't share sensitive information in prompts
- Verify code suggestions before running
- Keep your system updated
- Review file operations before confirming

### Productivity
- Use conversation memory to maintain context
- Leverage file operations for code generation
- Switch models based on task requirements
- Export important conversations for reference

---

For more information, visit the official documentation at https://github.com/sreevarshan-xenoz/xencode