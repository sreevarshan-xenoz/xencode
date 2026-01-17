"""
Model management module for Xencode
"""
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

# Smart default model selection - will be updated based on available models
DEFAULT_MODEL = None  # Will be set dynamically

class ModelManager:
    """Advanced model management with health monitoring"""

    def __init__(self) -> None:
        self.available_models: List[str] = []
        self.current_model: str = DEFAULT_MODEL
        self.model_health: Dict[str, Any] = {}
        # Import configuration to get API keys
        try:
            from xencode.smart_config_manager import get_config
            self.config = get_config()
        except ImportError:
            self.config = None
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

        # Add cloud models if API keys are configured
        if self.config:
            if self.config.api_keys.openai_api_key:
                # Add OpenAI models
                openai_models = [
                    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4",
                    "gpt-3.5-turbo", "gpt-4-32k", "gpt-3.5-turbo-16k"
                ]
                for model in openai_models:
                    if model not in self.available_models:
                        self.available_models.append(f"openai:{model}")

            if self.config.api_keys.google_gemini_api_key:
                # Add Google Gemini models
                gemini_models = [
                    "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"
                ]
                for model in gemini_models:
                    if model not in self.available_models:
                        self.available_models.append(f"google_gemini:{model}")

            if self.config.api_keys.openrouter_api_key:
                # Add OpenRouter models
                openrouter_models = [
                    "openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet",
                    "google/gemini-pro", "meta-llama/llama-3.1-8b-instruct"
                ]
                for model in openrouter_models:
                    if model not in self.available_models:
                        self.available_models.append(f"openrouter:{model}")

    def check_model_health(self, model: str) -> bool:
        """Check if a model is healthy and responsive"""
        # Check if this is a cloud model
        if model.startswith("openai:") or model.startswith("google_gemini:") or model.startswith("openrouter:"):
            return self.check_cloud_model_health(model)

        # Otherwise, it's an Ollama model
        try:
            start_time = time.time()
            # Use a minimal prompt to check model availability
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model.replace('ollama:', ''), "prompt": "hi", "stream": False},
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

    def check_cloud_model_health(self, model: str) -> bool:
        """Check health of cloud models"""
        try:
            start_time = time.time()

            if model.startswith("openai:"):
                import openai
                client = openai.OpenAI(api_key=self.config.api_keys.openai_api_key if self.config else "")

                # Test the model with a simple request
                model_name = model.replace("openai:", "")
                try:
                    # Just check if the model is accessible
                    response = client.models.retrieve(model_name)
                    response_time = time.time() - start_time

                    self.model_health[model] = {
                        'status': 'healthy',
                        'response_time': response_time,
                        'last_check': time.time(),
                    }
                    return True
                except Exception:
                    self.model_health[model] = {
                        'status': 'unavailable',
                        'error': 'Model not accessible with provided API key',
                        'last_check': time.time(),
                    }
                    return False

            elif model.startswith("google_gemini:"):
                import google.generativeai as genai
                genai.configure(api_key=self.config.api_keys.google_gemini_api_key if self.config else "")

                model_name = model.replace("google_gemini:", "")
                try:
                    # Test if we can access the model
                    test_model = genai.GenerativeModel(model_name)
                    response_time = time.time() - start_time

                    self.model_health[model] = {
                        'status': 'healthy',
                        'response_time': response_time,
                        'last_check': time.time(),
                    }
                    return True
                except Exception:
                    self.model_health[model] = {
                        'status': 'unavailable',
                        'error': 'Model not accessible with provided API key',
                        'last_check': time.time(),
                    }
                    return False

            elif model.startswith("openrouter:"):
                import openai
                client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.config.api_keys.openrouter_api_key if self.config else ""
                )

                model_name = model.replace("openrouter:", "")
                try:
                    # Test if we can access the model
                    response = client.models.retrieve(model_name)
                    response_time = time.time() - start_time

                    self.model_health[model] = {
                        'status': 'healthy',
                        'response_time': response_time,
                        'last_check': time.time(),
                    }
                    return True
                except Exception:
                    self.model_health[model] = {
                        'status': 'unavailable',
                        'error': 'Model not accessible with provided API key',
                        'last_check': time.time(),
                    }
                    return False

        except ImportError as e:
            self.model_health[model] = {
                'status': 'unavailable',
                'error': f'Missing required library: {str(e)}',
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


def get_available_models() -> List[str]:
    """Get available models with enhanced error handling and caching"""
    try:
        # Use the model manager for better performance
        model_manager = ModelManager()
        model_manager.refresh_models()
        return model_manager.available_models
    except Exception:
        return []


def get_smart_default_model() -> Optional[str]:
    """Intelligently select the best available model"""
    # Import here to avoid circular dependencies
    try:
        from xencode.tui.utils.model_checker import ModelChecker
        MODEL_CHECKER_AVAILABLE = True
    except ImportError:
        MODEL_CHECKER_AVAILABLE = False
        ModelChecker = None

    available = []

    if MODEL_CHECKER_AVAILABLE:
        available = ModelChecker.get_available_models()
    else:
        # Minimal fallback if checker not available
        try:
            from xencode.tui.utils.model_checker import ModelChecker as MC
            available = MC.get_available_models()
        except ImportError:
            pass

    if not available:
        # Try one last direct check if list is empty or checker failed
        try:
            import subprocess
            output = subprocess.check_output(["ollama", "list"], text=True)
            if "NAME" in output:
                lines = output.strip().split('\n')
                available = [line.split()[0] for line in lines[1:] if line.strip()]
        except Exception:
            pass

    if not available:
        # No models found at all
        return None

    # Filter out embedding models
    chat_models = [m for m in available if "embed" not in m]
    if not chat_models:
        # Only embedding models? Use first avail as fallback, though unlikely to work for chat
        return available[0]

    # Preferred models in order of preference
    preferred_models = [
        "qwen2.5:7b",
        "qwen2.5:3b",
        "qwen3:4b",
        "llama3.1:8b",
        "llama3.2:3b",
        "mistral:7b",
        "phi3:mini",
        "gemma2:2b",
    ]

    # Check for preferred models
    for preferred in preferred_models:
        for available_model in chat_models:
            if preferred in available_model.lower():
                return available_model

    # If no preferred model found, return the first available chat model
    return chat_models[0]


def list_models() -> None:
    """Enhanced model listing with health status and performance metrics"""
    model_manager = ModelManager()
    
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
            model_manager = ModelManager()
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