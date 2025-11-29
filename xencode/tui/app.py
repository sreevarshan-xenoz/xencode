#!/usr/bin/env python3
"""
Main Xencode TUI Application

VS Code-like terminal interface for Xencode.
"""

import asyncio
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer
from textual.binding import Binding

from xencode.tui.widgets.file_explorer import FileExplorer, FileSelected
from xencode.tui.widgets.editor import CodeEditor
from xencode.tui.widgets.chat import ChatPanel, ChatSubmitted
from xencode.tui.widgets.model_selector import ModelSelector, ModelSelected

# Import core functionality
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xencode_core import run_streaming_query, ModelManager, ConversationMemory


class XencodeApp(App):
    """Xencode TUI Application"""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #main-container {
        height: 100%;
    }
    
    #left-panel {
        width: 25%;
        max-width: 50;
    }
    
    #left-panel.hidden {
        display: none;
    }
    
    #center-panel {
        width: 1fr;
    }
    
    #right-panel {
        width: 35%;
        min-width: 40;
    }
    
    #model-selector-panel {
        height: 50%;
    }
    
    #model-selector-panel.hidden {
        display: none;
    }
    
    #chat-panel-container {
        height: 1fr;
    }
    
    #chat-panel-container.shrink {
        height: 50%;
    }
    
    FileExplorer {
        height: 100%;
        border: solid $primary;
    }
    """
    
    TITLE = "Xencode - AI-Powered Code Assistant"
    SUB_TITLE = "VS Code in Terminal"
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+e", "toggle_explorer", "Toggle Explorer"),
        Binding("ctrl+m", "toggle_models", "Models"),
        Binding("ctrl+l", "clear_chat", "Clear Chat"),
        Binding("f1", "help", "Help"),
    ]
    
    def __init__(self, root_path: Optional[Path] = None, *args, **kwargs):
        """Initialize Xencode app
        
        Args:
            root_path: Root directory for file explorer
        """
        super().__init__(*args, **kwargs)
        self.root_path = root_path or Path.cwd()
        
        # Core components
        self.model_manager = ModelManager()
        self.memory = ConversationMemory()
        self.current_model = "qwen2.5:7b"  # Default model
        
        # Widgets (will be set in compose)
        self.file_explorer: Optional[FileExplorer] = None
        self.code_editor: Optional[CodeEditor] = None
        self.chat_panel: Optional[ChatPanel] = None
        self.model_selector: Optional[ModelSelector] = None
        
        # Ensemble state
        self.use_ensemble = False
        self.ensemble_models = ["qwen2.5:7b"]
        self.ensemble_method = "vote"
    
    def compose(self) -> ComposeResult:
        """Compose the app layout"""
        yield Header()
        
        with Container(id="main-container"):
            with Horizontal():
                # Left panel: File Explorer
                with Vertical(id="left-panel"):
                    self.file_explorer = FileExplorer(self.root_path)
                    yield self.file_explorer
                
                # Center panel: Code Editor
                with Vertical(id="center-panel"):
                    self.code_editor = CodeEditor()
                    self.code_editor.border_title = "Code Viewer"
                    yield self.code_editor
                
                # Right panel: Model Selector (hidden by default) + Chat
                with Vertical(id="right-panel"):
                    # Model selector (initially hidden)
                    with Vertical(id="model-selector-panel", classes="hidden"):
                        self.model_selector = ModelSelector()
                        yield self.model_selector
                    
                    # Chat panel
                    with Vertical(id="chat-panel-container"):
                        self.chat_panel = ChatPanel()
                        yield self.chat_panel
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when app is mounted"""
        # Welcome message
        if self.chat_panel:
            self.chat_panel.add_system_message(
                f"Welcome to Xencode TUI! 🚀\n\n"
                f"Working directory: {self.root_path}\n"
                f"Mode: Single Model ({self.current_model})\n\n"
                f"💡 Press Ctrl+M to configure models/ensemble\n"
                f"Select a file from the explorer or start chatting!"
            )
    
    def on_file_selected(self, event: FileSelected) -> None:
        """Handle file selection from explorer
        
        Args:
            event: File selection event
        """
        if self.code_editor:
            self.code_editor.load_file(event.path)
        
        # Optionally notify chat
        if self.chat_panel:
            self.chat_panel.add_system_message(
                f"Opened: {event.path.name}"
            )
    
    def on_model_selected(self, event: ModelSelected) -> None:
        """Handle model selection changes
        
        Args:
            event: Model selection event
        """
        self.ensemble_models = event.models
        self.use_ensemble = event.is_ensemble
        self.ensemble_method = event.method
        
        # Update current model for single mode
        if not self.use_ensemble and self.ensemble_models:
            self.current_model = self.ensemble_models[0]
    
    async def on_chat_submitted(self, event: ChatSubmitted) -> None:
        """Handle chat submission
        
        Args:
            event: Chat submission event
        """
        user_query = event.content
        
        if not self.chat_panel:
            return
        
        # Add thinking indicator
        thinking_msg = self.chat_panel.add_assistant_message("Thinking...")
        
        try:
            # Build context from current file if open
            context_parts = []
            
            if self.code_editor and self.code_editor.current_file:
                file_path = self.code_editor.current_file
                try:
                    file_content = file_path.read_text(encoding="utf-8")
                    context_parts.append(
                        f"Current file: {file_path.name}\n"
                        f"```{self._get_language(file_path.suffix)}\n"
                        f"{file_content[:2000]}...\n"  # Limit context
                        f"```"
                    )
                except:
                    pass
            
            # Build enhanced prompt
            if context_parts:
                enhanced_prompt = (
                    f"Context:\n{''.join(context_parts)}\n\n"
                    f"User question: {user_query}"
                )
            else:
                enhanced_prompt = user_query
            
            # Stream response (ensemble or single model)
            full_response = ""
            if self.use_ensemble:
                # Use ensemble
                from xencode.ai_ensembles import EnsembleReasoner, QueryRequest, EnsembleMethod
                
                method_map = {
                    "vote": EnsembleMethod.VOTE,
                    "weighted": EnsembleMethod.WEIGHTED,
                    "consensus": EnsembleMethod.CONSENSUS,
                    "hybrid": EnsembleMethod.HYBRID,
                }
                
                reasoner = EnsembleReasoner()
                query = QueryRequest(
                    prompt=enhanced_prompt,
                    models=self.ensemble_models,
                    method=method_map.get(self.ensemble_method, EnsembleMethod.VOTE)
                )
                
                response = await reasoner.reason(query)
                full_response = response.fused_response
                self.chat_panel.update_streaming_message(full_response)
                
                # Show ensemble stats
                stats_msg = f"\n\n_Ensemble: {len(response.model_responses)} models, {response.total_time_ms:.0f}ms, consensus: {response.consensus_score:.2f}_"
                self.chat_panel.update_streaming_message(full_response + stats_msg)
            else:
                # Single model streaming
                async for chunk in self._stream_ai_response(enhanced_prompt):
                    full_response += chunk
                    self.chat_panel.update_streaming_message(full_response)
        
        except Exception as e:
            self.chat_panel.update_streaming_message(
                f"Error: {str(e)}"
            )
    
    async def _stream_ai_response(self, prompt: str):
        """Stream AI response chunks
        
        Args:
            prompt: The prompt to send
            
        Yields:
            Response chunks
        """
        # This is a simplified version - integrate with xencode_core streaming
        import requests
        
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.current_model,
            "prompt": prompt,
            "stream": True
        }
        
        try:
            response = requests.post(url, json=payload, stream=True, timeout=60)
            
            for line in response.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
        
        except Exception as e:
            yield f"\n\nError: {str(e)}"
    
    def _get_language(self, suffix: str) -> str:
        """Get language identifier for file
        
        Args:
            suffix: File extension
            
        Returns:
            Language name
        """
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".html": "html",
            ".css": "css",
        }
        return language_map.get(suffix.lower(), "text")
    
    def action_toggle_explorer(self) -> None:
        """Toggle file explorer visibility"""
        left_panel = self.query_one("#left-panel")
        if left_panel.has_class("hidden"):
            left_panel.remove_class("hidden")
        else:
            left_panel.add_class("hidden")
    
    def action_toggle_models(self) -> None:
        """Toggle model selector visibility"""
        model_panel = self.query_one("#model-selector-panel")
        chat_container = self.query_one("#chat-panel-container")
        
        if model_panel.has_class("hidden"):
            model_panel.remove_class("hidden")
            chat_container.add_class("shrink")
        else:
            model_panel.add_class("hidden")
            chat_container.remove_class("shrink")
    
    def action_clear_chat(self) -> None:
        """Clear chat history"""
        if self.chat_panel and self.chat_panel.history:
            self.chat_panel.history.clear_history()
            self.chat_panel.add_system_message("Chat cleared.")
    
    def action_help(self) -> None:
        """Show help"""
        if self.chat_panel:
            help_text = """
            # Xencode TUI Keybindings
            
            - **Ctrl+E**: Toggle file explorer
            - **Ctrl+M**: Toggle model selector
            - **Ctrl+L**: Clear chat history
            - **Ctrl+S**: Save current file (in editor)
            - **Ctrl+C**: Quit application
            - **F1**: Show this help
            - **Tab**: Switch focus between panels
            - **Ctrl+Enter**: Send chat message
            
            ## Ensemble Mode
            Select 2-4 models in Model Selector to enable ensemble.
            Choose method: Vote, Weighted, Consensus, or Hybrid.
            """
            self.chat_panel.add_system_message(help_text)


def run_tui(root_path: Optional[Path] = None):
    """Run the Xencode TUI
    
    Args:
        root_path: Root directory for file explorer
    """
    app = XencodeApp(root_path=root_path)
    app.run()


if __name__ == "__main__":
    run_tui()
