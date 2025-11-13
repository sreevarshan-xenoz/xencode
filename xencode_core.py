#!/usr/bin/env python3

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

# Try to import prompt_toolkit for enhanced input handling
try:
    from prompt_toolkit import prompt

    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

# Import Enhanced CLI System with graceful fallback
try:
    from enhanced_cli_system import EnhancedXencodeCLI

    ENHANCED_CLI_AVAILABLE = True
except ImportError:
    ENHANCED_CLI_AVAILABLE = False
    EnhancedXencodeCLI = None

# Import enhanced systems for chat commands
try:
    from xencode.multi_model_system import MultiModelManager
except ImportError:
    MultiModelManager = None

try:
    from xencode.smart_context_system import SmartContextManager
except ImportError:
    SmartContextManager = None

try:
    from xencode.code_analysis_system import CodeAnalyzer
except ImportError:
    CodeAnalyzer = None

# Suppress Rich color encoding warnings and other terminal warnings
os.environ.setdefault('FORCE_COLOR', '1')
os.environ.setdefault('TERM', 'xterm-256color')
os.environ.setdefault('COLORTERM', 'truecolor')

console = Console(
    force_terminal=True, legacy_windows=False, color_system="256", stderr=False
)
DEFAULT_MODEL = "qwen3:4b"

# Enhanced Claude-style streaming timing configuration
THINKING_STREAM_DELAY = 0.045  # 40-60ms per token
ANSWER_STREAM_DELAY = 0.030  # 20-40ms per token
THINKING_TO_ANSWER_PAUSE = 0.5  # 500ms pause between sections
THINKING_LINE_PAUSE = 0.125  # 100-150ms between thinking lines

# Performance and caching configuration
CACHE_ENABLED = True
CACHE_DIR = Path.home() / ".xencode" / "cache"
MAX_CACHE_SIZE = 100  # Maximum cached responses
RESPONSE_TIMEOUT = 30  # API response timeout in seconds

# Conversation memory configuration
MEMORY_ENABLED = True
MAX_MEMORY_ITEMS = 50
MEMORY_FILE = Path.home() / ".xencode" / "conversation_memory.json"


class ConversationMemory:
    """Advanced conversation memory with context management"""

    def __init__(self, max_items: int = MAX_MEMORY_ITEMS) -> None:
        self.max_items: int = max_items
        self.conversations: Dict[str, Any] = {}
        self.current_session: Optional[str] = None
        self.load_memory()

    def load_memory(self) -> None:
        """Load conversation memory from disk"""
        try:
            if MEMORY_FILE.exists():
                with open(MEMORY_FILE, 'r') as f:
                    data = json.load(f)
                    self.conversations = data.get('conversations', {})
                    self.current_session = data.get('current_session')
        except Exception:
            self.conversations = {}
            self.current_session = None

    def save_memory(self) -> None:
        """Save conversation memory to disk"""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(MEMORY_FILE, 'w') as f:
                json.dump(
                    {
                        'conversations': self.conversations,
                        'current_session': self.current_session,
                        'last_updated': datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new conversation session"""
        if session_id is None:
            session_id = f"session_{int(time.time())}"

        self.current_session = session_id
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'messages': [],
                'model': DEFAULT_MODEL,
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
            }
        return session_id

    def add_message(self, role: str, content: str, model: Optional[str] = None) -> None:
        """Add a message to current session"""
        if self.current_session is None:
            self.start_session()

        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'model': model or DEFAULT_MODEL,
        }

        self.conversations[self.current_session]['messages'].append(message)
        self.conversations[self.current_session][
            'last_updated'
        ] = datetime.now().isoformat()

        # Trim old messages if exceeding limit
        if len(self.conversations[self.current_session]['messages']) > self.max_items:
            self.conversations[self.current_session]['messages'] = self.conversations[
                self.current_session
            ]['messages'][-self.max_items :]

        self.save_memory()

    def get_context(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation context for model input"""
        if (
            self.current_session is None
            or self.current_session not in self.conversations
        ):
            return []

        messages = self.conversations[self.current_session]['messages']
        return messages[-max_messages:] if len(messages) > max_messages else messages

    def list_sessions(self) -> List[str]:
        """List all conversation sessions"""
        return list(self.conversations.keys())

    def switch_session(self, session_id: str) -> bool:
        """Switch to a different conversation session"""
        if session_id in self.conversations:
            self.current_session = session_id
            return True
        return False


class ResponseCache:
    """Intelligent response caching for performance optimization"""

    def __init__(
        self, cache_dir: Path = CACHE_DIR, max_size: int = MAX_CACHE_SIZE
    ) -> None:
        self.cache_dir: Path = Path(cache_dir)
        self.max_size: int = max_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model"""
        content = f"{prompt}:{model}".encode('utf-8')
        return hashlib.md5(content).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[str]:
        """Get cached response if available"""
        if not CACHE_ENABLED:
            return None

        try:
            cache_key = self._get_cache_key(prompt, model)
            cache_file = self.cache_dir / f"{cache_key}.json"

            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Check if cache is still valid (24 hours)
                    if time.time() - data['timestamp'] < 86400:
                        return data['response']  # type: ignore
        except Exception:
            pass

        return None

    def set(self, prompt: str, model: str, response: str) -> None:
        """Cache a response"""
        if not CACHE_ENABLED:
            return

        try:
            cache_key = self._get_cache_key(prompt, model)
            cache_file = self.cache_dir / f"{cache_key}.json"

            data = {
                'prompt': prompt,
                'model': model,
                'response': response,
                'timestamp': time.time(),
            }

            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)

            # Clean up old cache files if exceeding limit
            self._cleanup_cache()
        except Exception:
            pass

    def _cleanup_cache(self) -> None:
        """Remove old cache files to maintain size limit"""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            if len(cache_files) > self.max_size:
                # Sort by modification time and remove oldest
                cache_files.sort(key=lambda x: x.stat().st_mtime)
                for old_file in cache_files[: -self.max_size]:
                    old_file.unlink()
        except Exception:
            pass


class ModelManager:
    """Advanced model management with health monitoring"""

    def __init__(self) -> None:
        self.available_models: List[str] = []
        self.current_model: str = DEFAULT_MODEL
        self.model_health: Dict[str, Any] = {}
        self.refresh_models()

    def refresh_models(self) -> None:
        """Refresh list of available models"""
        try:
            output = subprocess.check_output(["ollama", "list"], text=True, timeout=5)
            lines = output.strip().split('\n')
            self.available_models = [
                line.split()[0] for line in lines[1:] if line.strip()
            ]
        except Exception:
            self.available_models = []

    def check_model_health(self, model: str) -> bool:
        """Check if a model is healthy and responsive"""
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": "test", "stream": False},
                timeout=5,
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                self.model_health[model] = {
                    'status': 'healthy',
                    'response_time': response_time,
                    'last_check': time.time(),
                }
                return True
            else:
                self.model_health[model] = {
                    'status': 'error',
                    'error_code': response.status_code,
                    'last_check': time.time(),
                }
                return False
        except Exception as e:
            self.model_health[model] = {
                'status': 'unavailable',
                'error': str(e),
                'last_check': time.time(),
            }
            return False

    def get_best_model(self) -> str:
        """Get the best available model based on health and performance"""
        if not self.available_models:
            return DEFAULT_MODEL

        # Check health of all models
        healthy_models = []
        for model in self.available_models:
            if self.check_model_health(model):
                healthy_models.append(model)

        if not healthy_models:
            return DEFAULT_MODEL

        # Return the fastest healthy model
        fastest_model = min(
            healthy_models,
            key=lambda m: self.model_health.get(m, {}).get(
                'response_time', float('inf')
            ),
        )
        return fastest_model

    def switch_model(self, model: str) -> Tuple[bool, str]:
        """Switch to a different model"""
        if model in self.available_models:
            if self.check_model_health(model):
                self.current_model = model
                return True, "Model switched successfully"
            else:
                return False, f"Model {model} is not responding"
        else:
            return False, f"Model {model} not found"


# Initialize global instances
memory = ConversationMemory()
cache = ResponseCache()
model_manager = ModelManager()


def get_available_models() -> List[str]:
    """Get available models with enhanced error handling and caching"""
    try:
        # Use the model manager for better performance
        model_manager.refresh_models()
        return model_manager.available_models
    except Exception:
        return []


def list_models() -> None:
    """Enhanced model listing with health status and performance metrics"""
    try:
        # Refresh models and health status
        model_manager.refresh_models()

        if not model_manager.available_models:
            console.print(
                Panel(
                    "❌ No models found\n\nPlease install models with:\n• ollama pull qwen3:4b\n• ollama pull llama2\n• ollama pull mistral",
                    title="No Models Available",
                    style="red",
                    border_style="red",
                )
            )
            return

        # Create enhanced model table
        table = Table(
            title="📦 Installed Models", show_header=True, header_style="bold cyan"
        )
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Response Time", style="yellow")
        table.add_column("Last Check", style="dim")

        for model in model_manager.available_models:
            # Actively check model health
            try:
                is_healthy = model_manager.check_model_health(model)
                health = model_manager.model_health.get(model, {})
                status = health.get('status', 'unknown')

                # Color code status
                if status == 'healthy':
                    status_style = "✅ Healthy"
                    response_time = f"{health.get('response_time', 0):.3f}s"
                elif status == 'error':
                    status_style = "❌ Error"
                    response_time = "N/A"
                elif status == 'unavailable':
                    status_style = "⚠️ Unavailable"
                    response_time = "N/A"
                else:
                    status_style = "❓ Unknown"
                    response_time = "N/A"
            except Exception:
                status_style = "❓ Check Failed"
                response_time = "N/A"

            last_check = health.get('last_check', 0)
            if last_check:
                last_check_str = datetime.fromtimestamp(last_check).strftime("%H:%M:%S")
            else:
                last_check_str = "Never"

            table.add_row(model, status_style, response_time, last_check_str)

        console.print(table)

        # Show current model
        if model_manager.current_model:
            console.print(
                f"\n🎯 Current Model: [bold cyan]{model_manager.current_model}[/bold cyan]"
            )

        # Show recommendations
        if len(model_manager.available_models) == 1:
            console.print(
                "\n💡 Tip: Install more models for variety:\n• ollama pull llama2:7b\n• ollama pull mistral:7b"
            )

    except FileNotFoundError:
        # Ollama not installed error
        error_panel = Panel(
            "❌ Ollama not found\n\nPlease install Ollama:\n• Visit: https://ollama.ai\n• Or use your package manager",
            title="Missing Dependency",
            style="red",
            border_style="red",
        )
        console.print(error_panel)
    except Exception as e:
        # Generic error panel
        error_panel = Panel(
            f"❌ Error listing models: {str(e)}\n\nPlease check your Ollama installation.",
            title="Model List Error",
            style="red",
            border_style="red",
        )
        console.print(error_panel)


def update_model(model: str) -> None:
    """Enhanced model update with progress tracking and validation"""
    console.print(f"[yellow]🔄 Pulling latest model: {model}[/yellow]")

    try:
        # Check if model exists first
        available_models = get_available_models()
        if model not in available_models:
            console.print(
                f"[yellow]📥 Model {model} not found locally, pulling from Ollama library...[/yellow]"
            )

        # Show progress spinner
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Pulling {model}...", total=None)

            # Run the pull command
            result = subprocess.run(
                ["ollama", "pull", model], check=True, capture_output=True, text=True
            )

            progress.update(task, description=f"Validating {model}...")

            # Validate the model after pull
            if model_manager.check_model_health(model):
                progress.update(task, description=f"✅ {model} ready!")
                time.sleep(0.5)  # Show success briefly

                # Success panel with enhanced info
                success_panel = Panel(
                    f"✅ Successfully pulled and validated {model}\n\n"
                    f"📊 Model Status: [green]Healthy[/green]\n"
                    f"⚡ Response Time: [yellow]{model_manager.model_health[model]['response_time']:.3f}s[/yellow]\n"
                    f"🎯 Ready to use!",
                    title="Model Updated Successfully",
                    style="green",
                    border_style="green",
                )
                console.print(success_panel)

                # Refresh model list
                model_manager.refresh_models()
            else:
                raise Exception(f"Model {model} failed health check after pull")

    except FileNotFoundError:
        # Ollama not installed error
        error_panel = Panel(
            "❌ Ollama not found\n\nPlease install Ollama:\n• Visit: https://ollama.ai\n• Or use your package manager",
            title="Missing Dependency",
            style="red",
            border_style="red",
        )
        console.print(error_panel)
    except subprocess.CalledProcessError as e:
        # Model pull failed - could be missing model
        stderr_text = e.stderr.decode() if e.stderr else "Unknown error"
        if "not found" in stderr_text.lower():
            warning_panel = Panel(
                f"⚠️ Model '{model}' not found in Ollama library\n\n"
                f"💡 Try these popular models:\n"
                f"• ollama pull qwen3:4b (fast, efficient)\n"
                f"• ollama pull llama2:7b (balanced)\n"
                f"• ollama pull mistral:7b (code-focused)\n"
                f"• ollama pull codellama:7b (programming)",
                title="Model Not Found",
                style="yellow",
                border_style="yellow",
            )
            console.print(warning_panel)
        else:
            error_panel = Panel(
                f"❌ Failed to pull {model}\n\nError: {stderr_text}\n\n"
                f"🔧 Troubleshooting:\n"
                f"• Check internet connection\n"
                f"• Verify Ollama service is running\n"
                f"• Try: systemctl restart ollama",
                title="Update Failed",
                style="red",
                border_style="red",
            )
            console.print(error_panel)
    except Exception as e:
        error_panel = Panel(
            f"❌ Unexpected error updating {model}\n\nError: {str(e)}\n\n"
            f"🔧 Please check your setup and try again.",
            title="Update Error",
            style="red",
            border_style="red",
        )
        console.print(error_panel)


def run_query(model: str, prompt: str) -> str:
    """Enhanced non-streaming query with caching and conversation memory"""
    # Check cache first
    cached_response = cache.get(prompt, model)
    if cached_response:
        console.print("[dim]💾 Using cached response[/dim]")
        return cached_response

    # Add user message to memory
    memory.add_message("user", prompt, model)

    url = "http://localhost:11434/api/generate"

    # Build context-aware prompt
    context = memory.get_context(max_messages=5)
    if context:
        context_prompt = "\n\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in context]
        )
        enhanced_prompt = f"{context_prompt}\n\nuser: {prompt}"
    else:
        enhanced_prompt = prompt

    payload = {"model": model, "prompt": enhanced_prompt, "stream": False}

    try:
        # Show progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🤖 Processing...", total=None)

            r = requests.post(url, json=payload, timeout=RESPONSE_TIMEOUT)
            r.raise_for_status()

            response = r.json()["response"]

            # Cache the response
            cache.set(prompt, model, response)

            # Add AI response to memory
            memory.add_message("assistant", response, model)

            return response

    except requests.exceptions.ConnectionError:
        # Claude-style connection error panel
        error_panel = Panel(
            "❌ Cannot connect to Ollama service\n\nPlease check:\n• Is Ollama running? Try: systemctl start ollama\n• Is the service accessible at localhost:11434?",
            title="Connection Error",
            style="red",
            border_style="red",
        )
        console.print(error_panel)
        sys.exit(1)
    except requests.exceptions.Timeout:
        error_panel = Panel(
            f"⏰ Request timed out after {RESPONSE_TIMEOUT}s\n\n"
            f"🔧 Try:\n• Using a smaller model\n• Checking system resources\n• Restarting Ollama service",
            title="Request Timeout",
            style="red",
            border_style="red",
        )
        console.print(error_panel)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        # Generic API error panel
        error_panel = Panel(
            f"❌ API Error: {str(e)}\n\nPlease check your Ollama installation and try again.",
            title="API Error",
            style="red",
            border_style="red",
        )
        console.print(error_panel)
        sys.exit(1)


def run_streaming_query(model, prompt):
    """Enhanced real-time streaming query with conversation memory and context awareness"""
    # Add user message to memory
    memory.add_message("user", prompt, model)

    url = "http://localhost:11434/api/generate"

    # Build context-aware prompt
    context = memory.get_context(max_messages=5)
    if context:
        context_prompt = "\n\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in context]
        )
        enhanced_prompt = f"{context_prompt}\n\nuser: {prompt}"
    else:
        enhanced_prompt = prompt

    payload = {"model": model, "prompt": enhanced_prompt, "stream": True}

    try:
        response = requests.post(
            url, json=payload, stream=True, timeout=RESPONSE_TIMEOUT
        )
        response.raise_for_status()

        # Collect the full response first, then stream it with proper timing
        full_response = ""

        # Thinking indicator is shown by chat mode, not here

        # Collect all chunks with progress indication
        chunk_count = 0
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    chunk_count += 1

                    # Check if streaming is complete
                    if chunk.get('done', False):
                        break

                    if 'response' in chunk and chunk['response']:
                        token = chunk['response']
                        full_response += token

                except json.JSONDecodeError:
                    continue

        # Now stream the complete response with proper Claude-style timing
        thinking, answer = extract_thinking_and_answer(full_response)

        if thinking:
            # Stream thinking section
            stream_thinking_section(thinking)
            time.sleep(THINKING_TO_ANSWER_PAUSE)

        # Stream answer section
        if answer.strip():
            stream_answer_section(answer)

        # Add AI response to memory
        memory.add_message("assistant", full_response, model)

        return full_response

    except requests.exceptions.ConnectionError:
        # Claude-style connection error panel
        error_panel = Panel(
            "❌ Cannot connect to Ollama service\n\nPlease check:\n• Is Ollama running? Try: systemctl start ollama\n• Is the service accessible at localhost:11434?",
            title="Connection Error",
            style="red",
            border_style="red",
        )
        console.print(error_panel)
        sys.exit(1)
    except requests.exceptions.Timeout:
        error_panel = Panel(
            f"⏰ Request timed out after {RESPONSE_TIMEOUT}s\n\n"
            f"🔧 Try:\n• Using a smaller model\n• Checking system resources\n• Restarting Ollama service",
            title="Request Timeout",
            style="red",
            border_style="red",
        )
        console.print(error_panel)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        # Generic API error panel
        error_panel = Panel(
            f"❌ API Error: {str(e)}\n\nPlease check your Ollama installation and try again.",
            title="API Error",
            style="red",
            border_style="red",
        )
        console.print(error_panel)
        sys.exit(1)


def extract_thinking_and_answer(text: str) -> Tuple[str, str]:
    """Extract thinking section and answer from Qwen response"""
    thinking = ""
    answer = text

    # Look for thinking section in Qwen format

    if "<think>" in text and "</think>" in text:
        try:
            thinking = text.split("<think>")[1].split("</think>")[0].strip()
            answer = text.split("</think>")[1].strip()
        except IndexError:
            # If parsing fails, treat entire text as answer
            pass
    elif "🧠 Thinking:" in text:
        # Handle alternative thinking format
        try:
            thinking = text.split("🧠 Thinking:")[1].split("\n\n")[0].strip()
            answer = text.split("🧠 Thinking:")[1].split("\n\n", 1)[1].strip()
        except IndexError:
            pass

    return thinking, answer


def stream_thinking_section(thinking_text):
    """Stream thinking section with dim yellow italic styling and breathing pauses"""
    if not thinking_text:
        return

    console.print("🧠 Thinking...", style="dim italic yellow")

    lines = thinking_text.split('\n')
    for line in lines:
        if line.strip():  # Only process non-empty lines
            # Stream each character with timing using Rich console for consistent output
            for char in line:
                # Use Rich console for immediate display
                console.print(char, style="dim italic yellow", end="", highlight=False)
                time.sleep(THINKING_STREAM_DELAY)
            console.print()  # New line after each line
            time.sleep(THINKING_LINE_PAUSE)  # Breathing pause between lines


def stream_answer_section(answer_text):
    """Stream answer section with bold green styling and markdown support"""
    if not answer_text.strip():
        return

    console.print("\n📄 Answer", style="bold green")

    # Handle markdown in answer with streaming
    if answer_text.startswith("```") and "```" in answer_text[3:]:
        # For code blocks, stream the entire formatted block at once for readability
        parts = answer_text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Text
                if part.strip():
                    # Stream text character by character using Rich console
                    for char in part.strip():
                        console.print(char, end="", highlight=False)
                        time.sleep(ANSWER_STREAM_DELAY)
                    console.print()
            else:  # Code
                if part.strip():
                    lang = part.split('\n')[0] if '\n' in part else ""
                    code_content = part[part.find('\n') + 1 :] if '\n' in part else part
                    # Display code block immediately for readability
                    console.print(
                        Syntax(code_content, lang or "plaintext", theme="monokai")
                    )
    else:
        # Stream regular text character by character using Rich console
        for char in answer_text.strip():
            console.print(char, end="", highlight=False)
            time.sleep(ANSWER_STREAM_DELAY)
        console.print()


def stream_claude_response(thinking_text, answer_text):
    """Stream complete response with exact Claude timing and formatting"""
    # Stream thinking section with breathing pauses
    if thinking_text:
        stream_thinking_section(thinking_text)
        # 0.5s pause between thinking and answer sections
        time.sleep(THINKING_TO_ANSWER_PAUSE)

    # Stream answer section
    stream_answer_section(answer_text)


def format_output(text, streaming=False):
    """Format output in Claude Code style with optional streaming"""
    thinking, answer = extract_thinking_and_answer(text)

    if streaming:
        # Use Claude-style streaming
        stream_claude_response(thinking, answer)
    else:
        # Use existing non-streaming format for backward compatibility
        # 🧠 Thinking Section
        if thinking:
            console.print("[bold yellow]🧠 Thinking...[/bold yellow]")
            console.print(f"[dim]{thinking}[/dim]\n")

        # 📄 Answer Section
        if answer.strip():
            console.print("[bold green]📄 Answer[/bold green]")

            # Handle markdown in answer
            if answer.startswith("```") and "```" in answer[3:]:
                # Extract and format code blocks
                parts = answer.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 0:  # Text
                        if part.strip():
                            console.print(Markdown(part.strip()))
                    else:  # Code
                        if part.strip():
                            lang = part.split('\n')[0] if '\n' in part else ""
                            code_content = (
                                part[part.find('\n') + 1 :] if '\n' in part else part
                            )
                            console.print(
                                Syntax(
                                    code_content, lang or "plaintext", theme="monokai"
                                )
                            )
            else:
                console.print(Markdown(answer.strip()))

            console.print()


def display_chat_banner(model, online_status, is_update=False):
    """Display Claude-style centered banner with exact formatting"""
    if is_update:
        # Clear previous lines and redisplay banner for connectivity updates
        console.print("\033[2J\033[H", end="")  # Clear screen and move cursor to top

    # Claude-style centered banner with exact format
    banner_lines = [
        "╔══════════════════════════════════════════╗",
        f"║ Xencode AI (Claude-Code Style | {model})    ║",
        "╚══════════════════════════════════════════╝",
    ]

    for line in banner_lines:
        console.print(line, style="cyan", justify="center")

    console.print(
        "Offline-First | Hyprland Ready | Arch Optimized", style="dim", justify="center"
    )
    console.print()

    # Dynamic status line with appropriate emojis
    if online_status == "true":
        status_line = "🌐 Online Mode - using local+internet models"
    else:
        status_line = "📡 Offline Mode - local models only"

    console.print(status_line, style="bold", justify="center")
    console.print()


def display_prompt():
    """Display the chat prompt in bold white"""
    console.print("[bold white][You] >[/bold white] ", end="")


def get_multiline_input():
    """Get user input with multiline support using prompt_toolkit if available"""
    if PROMPT_TOOLKIT_AVAILABLE:
        try:
            # Use prompt_toolkit with default key bindings for reliability
            # Enter submits, Ctrl+N adds new line
            user_input = prompt(
                "",  # Empty prompt since we display our own
                multiline=True,
                wrap_lines=True,
                mouse_support=False,  # Disable mouse support to avoid conflicts
            )

            # Clean and return input
            if user_input:
                return user_input.strip()
            return ""

        except Exception as e:
            # Fall back to basic input if prompt_toolkit fails
            console.print(
                f"[dim]Warning: prompt_toolkit failed, using basic input: {e}[/dim]"
            )
            return input().strip()
    else:
        # Graceful fallback to basic input() if prompt_toolkit is not available
        return input().strip()


def update_online_status():
    """Check internet connectivity with lightweight ping"""
    try:
        # Use a lightweight ping check without blocking user interaction
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "1", "8.8.8.8"], capture_output=True, timeout=2
        )
        return "true" if result.returncode == 0 else "false"
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return "false"


def handle_chat_exit():
    """Display goodbye message using existing Rich formatting"""
    console.print()
    console.print(
        Panel(
            "[bold green]👋 Thanks for using Xencode! Goodbye![/bold green]",
            style="cyan",
            title="🤖 Xencode AI",
        )
    )
    console.print()


def handle_chat_command(command, current_model, current_online):
    """Handle enhanced chat commands with rich formatting"""
    cmd_parts = command.split()
    cmd = cmd_parts[0].lower()

    # Try enhanced chat commands first if available
    if ENHANCED_CLI_AVAILABLE:
        try:
            from enhanced_chat_commands import EnhancedChatCommands
            from enhanced_cli_system import FeatureDetector

            # Initialize enhanced chat commands if not already done
            if not hasattr(handle_chat_command, '_enhanced_commands'):
                detector = FeatureDetector()
                features = detector.detect_features()

                # Get enhanced systems if available
                enhanced_systems = {}
                if features.multi_model and MultiModelManager:
                    enhanced_systems['multi_model'] = MultiModelManager()
                if features.smart_context and SmartContextManager:
                    enhanced_systems['smart_context'] = SmartContextManager()
                if features.code_analysis and CodeAnalyzer:
                    enhanced_systems['code_analyzer'] = CodeAnalyzer()

                handle_chat_command._enhanced_commands = EnhancedChatCommands(
                    features, enhanced_systems
                )

            # Handle enhanced commands
            enhanced_commands = handle_chat_command._enhanced_commands

            # Check if this is an enhanced command
            enhanced_command_names = ["analyze", "model", "models", "context", "smart"]

            if cmd[1:] in enhanced_command_names:  # Remove leading /
                args = " ".join(cmd_parts[1:]) if len(cmd_parts) > 1 else ""
                response, new_model = enhanced_commands.handle_chat_command(
                    cmd[1:], args, current_model
                )

                console.print(Panel(response, style="cyan"))

                if new_model:
                    return (
                        "MODEL_SWITCH",
                        new_model,
                        f"Enhanced model switch to {new_model}",
                    )

                return True

            # Handle smart mode model suggestion for regular queries
            if not command.startswith("/") and enhanced_commands.smart_mode_enabled:
                suggested_model, reasoning = enhanced_commands.suggest_model_for_query(
                    command, current_model
                )

                if suggested_model != current_model:
                    console.print(f"[dim cyan]🤖 {reasoning}[/dim cyan]")
                    return "MODEL_SWITCH", suggested_model, reasoning

        except Exception as e:
            # Enhanced commands failed - fall back to legacy
            console.print(
                f"[dim yellow]⚠️ Enhanced commands unavailable: {e}[/dim yellow]"
            )

    # Legacy chat commands
    if cmd == "/help":
        show_help_panel()
        return True
    elif cmd == "/clear":
        memory.start_session()  # Start fresh session
        console.print(
            Panel("🧹 Conversation cleared. New session started.", style="green")
        )
        return True
    elif cmd == "/memory":
        show_memory_info()
        return True
    elif cmd == "/sessions":
        show_sessions_list()
        return True
    elif cmd == "/switch":
        if len(cmd_parts) > 1:
            session_id = cmd_parts[1]
            if memory.switch_session(session_id):
                console.print(
                    Panel(f"✅ Switched to session: {session_id}", style="green")
                )
            else:
                console.print(Panel(f"❌ Session not found: {session_id}", style="red"))
        else:
            console.print(Panel("❌ Usage: /switch <session_id>", style="red"))
        return True
    elif cmd == "/cache":
        show_cache_info()
        return True
    elif cmd == "/status":
        show_system_status(current_model, current_online)
        return True
    elif cmd == "/export":
        export_conversation()
        return True
    elif cmd == "/model":
        if len(cmd_parts) > 1:
            new_model = cmd_parts[1]
            success, message = model_manager.switch_model(new_model)
            if success:
                # Update the current model in the calling function
                # We'll need to return the new model
                return "MODEL_SWITCH", new_model, message
            else:
                console.print(Panel(f"❌ Model switch failed: {message}", style="red"))
                return True
        else:
            console.print(Panel("❌ Usage: /model <model_name>", style="red"))
            return True
    elif cmd == "/theme":
        if len(cmd_parts) > 1:
            change_theme(cmd_parts[1])
        else:
            show_available_themes()
        return True

    return False


def show_help_panel():
    """Display comprehensive help panel with all commands"""
    help_text = """
🎯 **Chat Commands:**
• /help - Show this help
• /clear - Clear current conversation
• /memory - Show memory usage
• /sessions - List all sessions
• /switch <id> - Switch to session
• /cache - Show cache info
• /status - System status
• /export - Export conversation
• /theme <name> - Change theme

🔧 **Model Commands:**
• /model <name> - Switch model
• /list-models - List available models
• /update - Update current model

💬 **Regular Input:**
• Type your message and press Enter
• Use Shift+Enter for multiline
• Type 'exit' or 'quit' to end
"""

    # Add enhanced commands if available
    if ENHANCED_CLI_AVAILABLE:
        try:
            from enhanced_cli_system import FeatureDetector

            detector = FeatureDetector()
            features = detector.detect_features()

            if features.enhanced_features_available:
                enhanced_text = "\n\n🚀 **Enhanced Commands:**\n"

                if features.code_analysis:
                    enhanced_text += "• /analyze [path] - Analyze code quality\n"

                if features.multi_model:
                    enhanced_text += "• /models - List models with capabilities\n"

                if features.smart_context:
                    enhanced_text += "• /context - Show project context\n"
                    enhanced_text += "• /context clear - Clear context cache\n"
                    enhanced_text += "• /context refresh - Refresh context\n"

                if features.multi_model and features.smart_context:
                    enhanced_text += "• /smart on|off - Toggle smart model selection\n"

                enhanced_text += f"\n🎚️ Feature Level: {features.feature_level.upper()}"

                help_text += enhanced_text
        except Exception:
            # Enhanced commands not available
            pass

    help_panel = Panel(help_text, title="📚 Xencode Help", style="cyan")
    console.print(help_panel)


def show_memory_info():
    """Display memory usage and statistics"""
    context = memory.get_context()
    sessions = memory.list_sessions()

    memory_text = f"""
🧠 **Memory Information:**
• Current Session: {memory.current_session}
• Messages in Context: {len(context)}
• Total Sessions: {len(sessions)}
• Memory Limit: {MAX_MEMORY_ITEMS} messages

📊 **Current Context:**
"""

    if context:
        for i, msg in enumerate(context[-5:], 1):  # Show last 5 messages
            role = msg['role'].capitalize()
            content_preview = (
                msg['content'][:50] + "..."
                if len(msg['content']) > 50
                else msg['content']
            )
            memory_text += f"• {i}. {role}: {content_preview}\n"
    else:
        memory_text += "• No messages in context\n"

    memory_panel = Panel(memory_text, title="🧠 Memory Status", style="blue")
    console.print(memory_panel)


def show_sessions_list():
    """Display list of all conversation sessions"""
    sessions = memory.list_sessions()

    if not sessions:
        console.print(Panel("❌ No sessions found", style="red"))
        return

    table = Table(
        title="💬 Conversation Sessions", show_header=True, header_style="bold cyan"
    )
    table.add_column("Session ID", style="cyan")
    table.add_column("Messages", style="green")
    table.add_column("Model", style="yellow")
    table.add_column("Created", style="dim")
    table.add_column("Last Updated", style="dim")

    for session_id in sessions:
        session_data = memory.conversations.get(session_id, {})
        messages_count = len(session_data.get('messages', []))
        model = session_data.get('model', 'Unknown')
        created = session_data.get('created', 'Unknown')
        last_updated = session_data.get('last_updated', 'Unknown')

        # Format timestamps
        try:
            created_dt = datetime.fromisoformat(created)
            created_str = created_dt.strftime("%Y-%m-%d %H:%M")
        except:
            created_str = created

        try:
            updated_dt = datetime.fromisoformat(last_updated)
            updated_str = updated_dt.strftime("%Y-%m-%d %H:%M")
        except:
            updated_str = last_updated

        # Highlight current session
        if session_id == memory.current_session:
            session_id = f"🎯 {session_id}"

        table.add_row(session_id, str(messages_count), model, created_str, updated_str)

    console.print(table)


def show_cache_info():
    """Display cache information and statistics"""
    try:
        cache_files = list(CACHE_DIR.glob("*.json"))
        cache_size = len(cache_files)

        cache_text = f"""
💾 **Cache Information:**
• Cache Directory: {CACHE_DIR}
• Cached Responses: {cache_size}
• Max Cache Size: {MAX_CACHE_SIZE}
• Cache Status: {'Enabled' if CACHE_ENABLED else 'Disabled'}
"""

        if cache_size > 0:
            cache_text += "\n📊 **Recent Cache Entries:**\n"
            # Show recent cache entries
            recent_files = sorted(
                cache_files, key=lambda x: x.stat().st_mtime, reverse=True
            )[:5]
            for i, cache_file in enumerate(recent_files, 1):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        prompt_preview = (
                            data.get('prompt', '')[:40] + "..."
                            if len(data.get('prompt', '')) > 40
                            else data.get('prompt', '')
                        )
                        cache_text += f"• {i}. {prompt_preview}\n"
                except:
                    cache_text += f"• {i}. [Error reading cache]\n"

        cache_panel = Panel(cache_text, title="💾 Cache Status", style="magenta")
        console.print(cache_panel)

    except Exception as e:
        console.print(Panel(f"❌ Error reading cache: {str(e)}", style="red"))


def show_system_status(current_model, current_online):
    """Display comprehensive system status"""
    # Check Ollama service
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_status = "✅ Running" if response.status_code == 200 else "❌ Error"
    except:
        ollama_status = "❌ Not accessible"

    # Check model health actively
    try:
        if model_manager.check_model_health(current_model):
            model_health = model_manager.model_health.get(current_model, {})
            response_time = f"{model_health.get('response_time', 0):.3f}s"
            model_status_display = f"✅ Healthy ({response_time})"
        else:
            model_health = model_manager.model_health.get(current_model, {})
            model_status_display = (
                f"❌ {model_health.get('status', 'Unavailable').capitalize()}"
            )
    except Exception:
        model_status_display = "❌ Check Failed"

    status_text = f"""
🖥️ **System Status:**
• Ollama Service: {ollama_status}
• Current Model: {current_model}
• Model Status: {model_status_display}
• Internet: {'🌐 Online' if current_online == 'true' else '🔌 Offline'}
• Memory Usage: {len(memory.get_context())} messages
• Cache Status: {'✅ Enabled' if CACHE_ENABLED else '❌ Disabled'}

📊 **Performance:**
• Available Models: {len(model_manager.available_models)}
• Cache Size: {len(list(CACHE_DIR.glob('*.json'))) if CACHE_DIR.exists() else 0}
• Session Count: {len(memory.conversations)}
"""

    status_panel = Panel(status_text, title="📊 System Status", style="green")
    console.print(status_panel)


def export_conversation():
    """Export current conversation to file"""
    try:
        context = memory.get_context()
        if not context:
            console.print(Panel("❌ No conversation to export", style="red"))
            return

        # Create export directory
        export_dir = Path.home() / ".xencode" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.md"
        export_path = export_dir / filename

        # Export as markdown
        with open(export_path, 'w') as f:
            f.write("# Xencode Conversation Export\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Session:** {memory.current_session}\n\n")

            for msg in context:
                role = msg['role'].capitalize()
                content = msg['content']
                timestamp = msg['timestamp']

                f.write(f"## {role}\n\n")
                f.write(f"*{timestamp}*\n\n")
                f.write(f"{content}\n\n")
                f.write("---\n\n")

        console.print(
            Panel(f"✅ Conversation exported to:\n{export_path}", style="green")
        )

    except Exception as e:
        console.print(Panel(f"❌ Export failed: {str(e)}", style="red"))


def change_theme(theme_name):
    """Change the visual theme (placeholder for future implementation)"""
    console.print(
        Panel(
            f"🎨 Theme '{theme_name}' not implemented yet.\n\nAvailable themes:\n• default\n• dark\n• light\n• colorful",
            style="yellow",
        )
    )


def show_available_themes():
    """Show available themes"""
    themes_text = """
🎨 **Available Themes:**
• default - Standard Xencode theme
• dark - Dark mode (coming soon)
• light - Light mode (coming soon)
• colorful - Enhanced colors (coming soon)

💡 Use: /theme <name> to change
"""

    themes_panel = Panel(themes_text, title="🎨 Themes", style="magenta")
    console.print(themes_panel)


def is_exit_command(user_input):
    """Check if user input is an exit command (exit, quit, q)"""
    exit_commands = ['exit', 'quit', 'q']
    return user_input.lower().strip() in exit_commands


def create_file(path, content):
    try:
        p = os.path.abspath(path)
        with open(p, 'w') as f:
            f.write(content)
        console.print(Panel(f"✅ {p}", title="Created", style="green"))
    except:
        console.print(Panel("❌ Failed", style="red"))


def read_file(path):
    try:
        console.print(
            Panel(open(os.path.abspath(path)).read(), title=path, style="cyan")
        )
    except:
        console.print(Panel("❌ Missing", style="red"))


def write_file(path, content):
    create_file(path, content)


def delete_file(path):
    try:
        os.remove(os.path.abspath(path))
        console.print(Panel(f"✅ {path}", title="Deleted", style="green"))
    except:
        console.print(Panel("❌ Failed", style="red"))


def chat_mode(model, online):
    """Enhanced interactive chat loop with advanced features and conversation management"""
    # Start a new conversation session
    session_id = memory.start_session()

    # Display initial banner
    display_chat_banner(model, online)

    # Track current online status for dynamic updates
    current_online = online
    current_model = model

    # Show session info
    console.print(f"[dim]💬 Session: {session_id}[/dim]")
    console.print(f"[dim]🧠 Memory: {len(memory.get_context())} messages[/dim]")

    while True:
        try:
            # Display prompt
            display_prompt()

            # Get user input with multiline support
            user_input = get_multiline_input()

            # Handle empty input gracefully without API calls
            if not user_input:
                console.print(
                    "[dim]Please enter a message or type 'exit' to quit.[/dim]"
                )
                continue

            # Enhanced command system
            if user_input.startswith("/"):
                command_result = handle_chat_command(
                    user_input, current_model, current_online
                )
                if command_result:
                    # Handle special return values
                    if (
                        isinstance(command_result, tuple)
                        and command_result[0] == "MODEL_SWITCH"
                    ):
                        _, new_model, message = command_result
                        current_model = new_model
                        console.print(
                            Panel(
                                f"✅ Model switched to [bold]{current_model}[/bold]\n{message}",
                                style="green",
                            )
                        )
                    continue
                else:
                    # If command not handled, treat as regular input
                    pass

            # Check for exit commands
            if is_exit_command(user_input):
                handle_chat_exit()
                break

            # Show thinking indicator with maximum 200ms latency
            console.print("[bold yellow]🧠 [Thinking...][/bold yellow]")

            # Check connectivity before API call
            new_online = update_online_status()
            if new_online != current_online:
                current_online = new_online
                display_chat_banner(current_model, current_online, is_update=True)
                # Don't re-display prompt or input to avoid confusion
                console.print("[dim]🌐 Connection status updated[/dim]")
                console.print("[bold yellow]🧠 [Thinking...][/bold yellow]")

            # Process the query using real-time streaming
            try:
                response = run_streaming_query(current_model, user_input)
                # No need for format_output since streaming is handled in run_streaming_query
            except Exception as e:
                # Claude-style error panel for chat mode
                error_panel = Panel(
                    f"❌ Error processing your request\n\n{str(e)}\n\nPlease try again or check your setup.",
                    title="Processing Error",
                    style="red",
                    border_style="red",
                )
                console.print(error_panel)

            console.print()  # Add spacing between interactions

        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C and Ctrl+D signal handling
            handle_chat_exit()
            break
        except Exception as e:
            # Generic error panel for unexpected errors in chat mode
            error_panel = Panel(
                f"❌ Unexpected error: {str(e)}\n\nThe chat session will continue. Please try again.",
                title="Unexpected Error",
                style="red",
                border_style="red",
            )
            console.print(error_panel)
            continue


def main():
    args = sys.argv[1:]
    online = "false"
    chat_mode_enabled = False

    # Initialize Enhanced CLI System if available
    enhanced_cli = None
    if ENHANCED_CLI_AVAILABLE and EnhancedXencodeCLI:
        try:
            enhanced_cli = EnhancedXencodeCLI()
        except Exception as e:
            # Enhanced CLI failed to initialize - continue with legacy mode
            console.print(
                f"[dim yellow]⚠️ Enhanced features unavailable: {e}[/dim yellow]"
            )

    # Try enhanced CLI processing first
    if enhanced_cli:
        try:
            # Create enhanced parser and parse arguments
            parser = enhanced_cli.create_parser()

            # Handle special case where no args means chat mode (legacy behavior)
            if not args:
                chat_mode_enabled = True
            else:
                # Parse enhanced arguments
                try:
                    parsed_args = parser.parse_args(args)

                    # Process enhanced commands
                    enhanced_result = enhanced_cli.process_enhanced_args(parsed_args)

                    if enhanced_result is not None:
                        # Enhanced command was processed
                        console.print(enhanced_result)
                        return

                    # Check if this should be handled as legacy command
                    if parsed_args.query:
                        # Legacy inline query mode
                        prompt = parsed_args.query
                        model = DEFAULT_MODEL

                        try:
                            response = run_query(model, prompt)
                            # For inline mode, only show the answer (no thinking section)
                            thinking, answer = extract_thinking_and_answer(response)
                            if answer.strip():
                                console.print(Markdown(answer.strip()))
                            else:
                                # Fallback to full response if no thinking tags
                                console.print(Markdown(response.strip()))
                        except Exception as e:
                            # Generic error panel for inline mode
                            error_panel = Panel(
                                f"❌ Unexpected error: {str(e)}\n\nPlease check your setup and try again.",
                                title="Error",
                                style="red",
                                border_style="red",
                            )
                            console.print(error_panel)
                        return

                except SystemExit:
                    # argparse called sys.exit (help or error)
                    return
                except Exception:
                    # Enhanced parsing failed - fall back to legacy parsing
                    pass
        except Exception:
            # Enhanced CLI processing failed - fall back to legacy
            pass

    # Legacy argument parsing (backward compatibility)
    # Parse online flag
    if "--online=true" in args:
        online = "true"
        args = [arg for arg in args if arg != "--online=true"]
    elif "--online=false" in args:
        online = "false"
        args = [arg for arg in args if arg != "--online=false"]

    # Parse chat mode flag
    if "--chat-mode" in args:
        chat_mode_enabled = True
        args = [arg for arg in args if arg != "--chat-mode"]

    # Auto-enable chat mode if no arguments (legacy behavior)
    if not args:
        chat_mode_enabled = True

    # Validate chat mode vs inline mode conflicts
    if (
        chat_mode_enabled
        and args
        and not any(flag in args for flag in ["--list-models", "--update", "-m"])
    ):
        error_panel = Panel(
            "❌ Invalid usage\n\nChat mode cannot be used with inline prompts.\n\nUse:\n• Chat mode: ./xencode.sh\n• Inline mode: ./xencode.sh \"your prompt\"",
            title="Usage Error",
            style="red",
            border_style="red",
        )
        console.print(error_panel)
        return

    # Handle --list-models
    if "--list-models" in args:
        list_models()
        return

    # Handle --update
    if "--update" in args:
        args = [arg for arg in args if arg != "--update"]
        if online == "true":
            model = DEFAULT_MODEL
            if "-m" in args:
                idx = args.index("-m")
                if idx + 1 < len(args):
                    model = args[idx + 1]
                    args.pop(idx)
                    args.pop(idx)
            update_model(model)
        else:
            warning_panel = Panel(
                "⚠️ No internet connection\n\nCannot update models while offline.\nPlease check your connection and try again.",
                title="Offline Mode",
                style="yellow",
                border_style="yellow",
            )
            console.print(warning_panel)
        return

    # Handle --status
    if "--status" in args:
        show_system_status(DEFAULT_MODEL, online)
        return

    # Handle --memory
    if "--memory" in args:
        show_memory_info()
        return

    # Handle --sessions
    if "--sessions" in args:
        show_sessions_list()
        return

    # Handle --cache
    if "--cache" in args:
        show_cache_info()
        return

    # Handle --export
    if "--export" in args:
        export_conversation()
        return

    # File ops
    if len(args) > 0 and args[0] == 'file':
        if len(args) < 2:
            exit(Panel("❌ No op", style="red"))
        op = args[1]
        if op == 'create' and len(args) >= 3:
            create_file(args[2], ' '.join(args[3:]))
        elif op == 'read' and len(args) >= 2:
            read_file(args[2])
        elif op == 'write' and len(args) >= 3:
            write_file(args[2], ' '.join(args[3:]))
        elif op == 'delete' and len(args) >= 2:
            delete_file(args[2])
        else:
            exit(Panel(f"❌ Invalid: {op}", style="red"))
        return

    # File operations
    if args and args[0] == 'file':
        op, args = args[1], args[2:]
        if op == 'create' and len(args) > 1:
            create_file(args[0], ' '.join(args[1:]))
        elif op == 'read' and args:
            read_file(args[0])
        elif op == 'write' and len(args) > 1:
            write_file(args[0], ' '.join(args[1:]))
        elif op == 'delete' and args:
            delete_file(args[0])
        return

    # Handle model specification
    model = DEFAULT_MODEL
    if "-m" in args:
        idx = args.index("-m")
        if idx + 1 < len(args):
            model = args[idx + 1]
            args.pop(idx)
            args.pop(idx)

    # Handle chat mode or inline prompt
    if chat_mode_enabled:
        chat_mode(model, online)
        return
    else:
        # Handle inline prompt
        if not args:
            warning_panel = Panel(
                "⚠️ No prompt provided\n\nUsage:\n• Inline mode: ./xencode.sh \"your prompt\"\n• Chat mode: ./xencode.sh",
                title="Missing Prompt",
                style="yellow",
                border_style="yellow",
            )
            console.print(warning_panel)
            return

        prompt = " ".join(args)

        try:
            response = run_query(model, prompt)
            # For inline mode, only show the answer (no thinking section)
            thinking, answer = extract_thinking_and_answer(response)
            if answer.strip():
                console.print(Markdown(answer.strip()))
            else:
                # Fallback to full response if no thinking tags
                console.print(Markdown(response.strip()))
        except Exception as e:
            # Generic error panel for inline mode
            error_panel = Panel(
                f"❌ Unexpected error: {str(e)}\n\nPlease check your setup and try again.",
                title="Error",
                style="red",
                border_style="red",
            )
            console.print(error_panel)


if __name__ == "__main__":
    main()
