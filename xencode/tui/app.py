#!/usr/bin/env python3
import asyncio
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Label, Button
from textual.binding import Binding
from textual.screen import ModalScreen
import websockets

from xencode.tui.widgets.file_explorer import FileExplorer, FileSelected
from xencode.tui.widgets.editor import CodeEditor
from xencode.tui.widgets.chat import ChatPanel, ChatSubmitted
from xencode.tui.widgets.model_selector import ModelSelector, ModelSelected
from xencode.tui.widgets.collaboration import CollaborationPanel
from xencode.tui.widgets.commit_dialog import CommitDialog
from xencode.tui.widgets.terminal import TerminalPanel
from xencode.tui.widgets.agent_panel import AgentTaskSubmitted
from xencode.tui.widgets.bytebot_panel import ByteBotPanel, ByteBotTaskSubmitted
from xencode.tui.widgets.settings_panel import SettingsPanel
from xencode.tui.widgets.options_panel import OptionsPanel

from xencode.tui.utils.model_checker import ModelChecker

# Import core functionality
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xencode_core import run_streaming_query, ModelManager, ConversationMemory
from xencode.auth.qwen_auth import qwen_auth_manager, QwenAuthError


class OnboardingModal(ModalScreen):
    """First-run onboarding modal with account actions and key options."""

    DEFAULT_CSS = """
    OnboardingModal {
        align: center middle;
    }

    #onboarding-dialog {
        width: 80%;
        max-width: 100;
        height: auto;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }

    #onboarding-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #onboarding-actions {
        margin-top: 1;
        align: right middle;
        height: auto;
    }

    OnboardingModal Button {
        margin-left: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(id="onboarding-dialog"):
            yield Label("👋 Welcome to Xencode TUI", id="onboarding-title")
            yield Label(
                "Use the TUI as your main interface.\n"
                "\n"
                "Key options available in TUI:\n"
                "• Chat + code context\n"
                "• Model/ensemble controls\n"
                "• Collaboration + ByteBot\n"
                "• Terminal + Git actions\n"
                "• Settings panel (Ctrl+,)\n"
                "\n"
                "For Qwen cloud models, login/signup once to continue."
            )
            with Horizontal(id="onboarding-actions"):
                yield Button("Login", id="btn-login", variant="success")
                yield Button("Sign Up", id="btn-signup")
                yield Button("Continue", id="btn-continue", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-login":
            self.dismiss("login")
        elif event.button.id == "btn-signup":
            self.dismiss("signup")
        else:
            self.dismiss("continue")


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
    
    #collab-panel-container {
        height: 50%;
    }
    
    #collab-panel-container.hidden {
        display: none;
    }

    #bytebot-panel-container {
        height: 50%;
    }

    #bytebot-panel-container.hidden {
        display: none;
    }

    #settings-panel-container {
        height: 50%;
    }

    #settings-panel-container.hidden {
        display: none;
    }

    #options-panel-container {
        height: 50%;
    }

    #options-panel-container.hidden {
        display: none;
    }

    Screen.theme-midnight {
        background: #0f111a;
        color: #e6e6e6;
    }

    Screen.theme-ocean {
        background: #0b1b2b;
        color: #eaf4ff;
    }

    Screen.theme-forest {
        background: #0f1d14;
        color: #e9f5ea;
    }

    Screen.theme-sunset {
        background: #26160f;
        color: #fff1e6;
    }

    Screen.theme-violet {
        background: #191426;
        color: #f1e9ff;
    }

    Screen.theme-slate {
        background: #1b1f24;
        color: #e7edf3;
    }

    Screen.theme-terminal {
        background: #001100;
        color: #80ff80;
    }

    Screen.theme-desert {
        background: #2a2016;
        color: #fff4df;
    }

    Screen.theme-arctic {
        background: #eaf4ff;
        color: #102a43;
    }

    Screen.theme-rose {
        background: #26151c;
        color: #ffe8f0;
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
        Binding("ctrl+k", "toggle_collab", "Collab"),
        Binding("ctrl+b", "toggle_bytebot", "ByteBot"),
        Binding("ctrl+g", "refresh_git", "Git Refresh"),
        Binding("ctrl+shift+c", "commit_dialog", "Commit"),
        Binding("ctrl+t", "toggle_terminal", "Terminal"),
        Binding("ctrl+comma", "toggle_settings", "Settings"),
        Binding("ctrl+o", "toggle_options", "Options"),
        Binding("ctrl+l", "clear_chat", "Clear Chat"),
        Binding("ctrl+shift+l", "logout_qwen", "Logout Qwen"),
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
        
        # Detect available models
        available_models = ModelChecker.get_available_models()
        
        # Smart default selection
        if available_models:
            # Filter out embedding models for chat
            chat_models = [m for m in available_models if "embed" not in m]
            if chat_models:
                # Prefer qwen cloud models, then local qwen, then llama
                cloud_qwen_models = [m for m in chat_models if any(cloud_model in m.lower() for cloud_model in ["qwen-max", "qwen-plus", "qwen-chat", "chat.qwen.ai"])]
                local_qwen_models = [m for m in chat_models if "qwen" in m.lower() and not any(cloud_model in m.lower() for cloud_model in ["qwen-max", "qwen-plus", "qwen-chat", "chat.qwen.ai"])]
                llama_models = [m for m in chat_models if "llama" in m.lower()]

                # Prioritize cloud Qwen models first, then local Qwen, then Llama
                preferred = cloud_qwen_models or local_qwen_models or llama_models
                self.current_model = preferred[0] if preferred else chat_models[0]
            else:
                self.current_model = available_models[0]
        else:
            self.current_model = "qwen2.5:7b"  # Default fallback
        
        # Widgets (will be set in compose)
        self.file_explorer: Optional[FileExplorer] = None
        self.code_editor: Optional[CodeEditor] = None
        self.chat_panel: Optional[ChatPanel] = None
        self.model_selector: Optional[ModelSelector] = None
        self.collab_panel: Optional[CollaborationPanel] = None
        self.bytebot_panel: Optional[ByteBotPanel] = None
        self.settings_panel: Optional[SettingsPanel] = None
        self.options_panel: Optional[OptionsPanel] = None
        
        # Collaboration state
        self.server_process: Optional[subprocess.Popen] = None
        self.ws_connection: Optional[websockets.WebSocketClientProtocol] = None
        self.session_id: Optional[str] = None
        self.username: Optional[str] = None
        
        # Ensemble state
        self.use_ensemble = False
        self.ensemble_models = ["qwen2.5:7b"]
        self.ensemble_method = "vote"

        # Persisted TUI settings/state
        self.settings_path = Path.home() / ".xencode_tui_settings.json"
        self.ui_settings = self._load_ui_settings()
    
    def compose(self) -> ComposeResult:
        """Compose the app layout"""
        yield Header()
        
        with Container(id="main-container"):
            with Horizontal():
                # Left panel: File Explorer
                with Vertical(id="left-panel"):
                    self.file_explorer = FileExplorer(self.root_path)
                    yield self.file_explorer
                
                # Center panel: Code Editor + Terminal
                with Vertical(id="center-panel"):
                    self.code_editor = CodeEditor()
                    self.code_editor.border_title = "Code Viewer"
                    yield self.code_editor
                    yield TerminalPanel()
                
                # Right panel: Model Selector + Collab + Chat
                with Vertical(id="right-panel"):
                    # Model selector (initially hidden)
                    with Vertical(id="model-selector-panel", classes="hidden"):
                        self.model_selector = ModelSelector()
                        yield self.model_selector
                    
                    # Collaboration panel (initially hidden)
                    with Vertical(id="collab-panel-container", classes="hidden"):
                        self.collab_panel = CollaborationPanel()
                        yield self.collab_panel

                    # ByteBot panel (initially hidden)
                    with Vertical(id="bytebot-panel-container", classes="hidden"):
                        self.bytebot_panel = ByteBotPanel()
                        yield self.bytebot_panel

                    # Settings panel (initially hidden)
                    with Vertical(id="settings-panel-container", classes="hidden"):
                        self.settings_panel = SettingsPanel()
                        yield self.settings_panel

                    # Options panel (initially hidden)
                    with Vertical(id="options-panel-container", classes="hidden"):
                        self.options_panel = OptionsPanel()
                        yield self.options_panel

                    # Chat panel
                    with Vertical(id="chat-panel-container"):
                        self.chat_panel = ChatPanel()
                        yield self.chat_panel
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when app is mounted"""
        self._apply_ui_settings()

        if self.settings_panel:
            self.settings_panel.set_settings(self.ui_settings)

        # Welcome message
        if self.chat_panel:
            self.chat_panel.add_system_message(
                f"Welcome to Xencode TUI! 🚀\n\n"
                f"Working directory: {self.root_path}\n"
                f"Mode: Single Model ({self.current_model})\n\n"
                f"💡 Press Ctrl+M to configure models/ensemble\n"
                f"⚙️ Settings: Press Ctrl+,\n"
                f"🧠 ByteBot: Press Ctrl+B to open widget\n"
                f"Select a file from the explorer or start chatting!"
            )

        if self._is_first_run():
            self.push_screen(OnboardingModal(), self._handle_onboarding_result)
    
    def on_file_selected(self, event: FileSelected) -> None:
        """Handle file selection from explorer
        
        Args:
            event: File selection event
        """
        if self.code_editor:
            self.code_editor.load_file(event.path)
        
        # Broadcast file open if connected
        if self.ws_connection and self.session_id:
            try:
                # Calculate relative path
                try:
                    rel_path = event.path.relative_to(self.root_path)
                    asyncio.create_task(self.ws_connection.send(json.dumps({
                        "type": "file_open",
                        "content": str(rel_path)
                    })))
                except ValueError:
                    # Path not relative to root (e.g. external file)
                    pass
            except Exception as e:
                self.notify(f"Failed to broadcast file open: {e}", severity="error")
        
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
    
    async def on_collaboration_panel_host_session(self, event: CollaborationPanel.HostSession) -> None:
        """Handle host session request"""
        try:
            # Start server
            self.server_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "xencode.server.app:app", "--host", "0.0.0.0", "--port", "8000"],
                cwd=str(Path.cwd())
            )
            self.notify("Starting collaboration server...", severity="information")
            await asyncio.sleep(2)  # Wait for server to start
            
            # Create session via API (using requests for simplicity, could use aiohttp)
            import requests
            response = requests.post("http://localhost:8000/sessions/create", params={"username": "Host"})
            if response.status_code == 200:
                data = response.json()
                self.session_id = str(data["session_id"])
                invite_code = data["invite_code"]
                self.username = "Host"
                
                # Connect WebSocket
                await self._connect_websocket("localhost:8000", self.session_id, "Host")
                
                # Update UI
                if self.collab_panel:
                    self.collab_panel.set_connected("host", invite_code)
                    self.collab_panel.add_user("Host (You)")
            else:
                self.notify(f"Failed to create session: {response.text}", severity="error")
                
        except Exception as e:
            self.notify(f"Error hosting session: {e}", severity="error")

    async def on_collaboration_panel_join_session(self, event: CollaborationPanel.JoinSession) -> None:
        """Handle join session request"""
        try:
            import requests
            # Get session ID from invite code
            response = requests.get(f"http://localhost:8000/sessions/{event.invite_code}")
            if response.status_code == 200:
                data = response.json()
                self.session_id = str(data["session_id"])
                self.username = event.username
                
                # Connect WebSocket
                await self._connect_websocket("localhost:8000", self.session_id, event.username)
                
                # Update UI
                if self.collab_panel:
                    self.collab_panel.set_connected("guest", event.invite_code)
                    self.collab_panel.add_user(f"{event.username} (You)")
            else:
                self.notify("Invalid invite code or session not found", severity="error")
                
        except Exception as e:
            self.notify(f"Error joining session: {e}", severity="error")

    async def _connect_websocket(self, host: str, session_id: str, username: str):
        """Connect to collaboration WebSocket"""
        uri = f"ws://{host}/ws/{session_id}/{username}"
        try:
            self.ws_connection = await websockets.connect(uri)
            self.notify(f"Connected to session as {username}", severity="information")
            
            # Start listening loop
            asyncio.create_task(self._listen_websocket())
            
        except Exception as e:
            self.notify(f"WebSocket connection failed: {e}", severity="error")

    async def _listen_websocket(self):
        """Listen for WebSocket messages"""
        if not self.ws_connection:
            return
            
        try:
            async for message in self.ws_connection:
                data = json.loads(message)
                await self._handle_collab_message(data)
        except websockets.exceptions.ConnectionClosed:
            self.notify("Disconnected from session", severity="warning")
            self.ws_connection = None
            
    async def _handle_collab_message(self, data: dict):
        """Handle incoming collaboration message"""
        msg_type = data.get("type")
        content = data.get("content")
        sender = data.get("sender", "System")
        
        if msg_type == "chat":
            if self.chat_panel:
                self.chat_panel.add_message("user", content, sender) # Re-using user role for now
        elif msg_type == "system":
            if self.collab_panel:
                # Update user list based on system messages (simplified)
                if "joined" in content:
                    user = content.split(" ")[0]
                    self.collab_panel.add_user(user)
            self.notify(content, severity="information")
            
        elif msg_type == "file_open":
            # Handle remote file open (Follow Mode)
            try:
                rel_path = content
                full_path = self.root_path / rel_path
                if full_path.exists() and full_path.is_file():
                    if self.code_editor:
                        self.code_editor.load_file(full_path)
                    self.notify(f"{sender} opened {rel_path}", severity="information")
                else:
                    self.notify(f"{sender} opened {rel_path} (not found)", severity="warning")
            except Exception as e:
                self.notify(f"Error syncing file: {e}", severity="error")

    async def on_chat_submitted(self, event: ChatSubmitted) -> None:
        """Handle chat submission
        
        Args:
            event: Chat submission event
        """
        user_query = event.content
        
        if not self.chat_panel:
            return

        # ByteBot slash command handling (do not broadcast)
        bytebot_payload = self._parse_bytebot_command(user_query)
        if bytebot_payload:
            intent_text = bytebot_payload["intent"]
            mode = bytebot_payload["mode"]

            if not intent_text:
                self.chat_panel.add_system_message(
                    "ByteBot usage: /bytebot --mode assist|execute|autonomous <intent>"
                )
                return

            thinking_msg = self.chat_panel.add_assistant_message("ByteBot is working...")
            try:
                result_text = await self._run_bytebot_intent(intent_text, mode)
                self.chat_panel.update_streaming_message(result_text)
            except Exception as e:
                self.chat_panel.update_streaming_message(f"ByteBot error: {e}")
            return
            
        # Broadcast if connected
        if self.ws_connection and self.session_id:
            try:
                await self.ws_connection.send(json.dumps({
                    "type": "chat",
                    "content": user_query
                }))
            except Exception as e:
                self.notify(f"Failed to broadcast message: {e}", severity="error")
        
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

    async def on_bytebot_task_submitted(self, event: ByteBotTaskSubmitted) -> None:
        """Handle ByteBot task submission from widget"""
        if not self.bytebot_panel:
            return

        try:
            result_text = await self._run_bytebot_intent(event.intent, event.mode)
            self.bytebot_panel.log_result(result_text)
        except Exception as e:
            self.bytebot_panel.log_result(f"ByteBot error: {e}")
        finally:
            self.bytebot_panel.set_idle()

    def _parse_bytebot_command(self, user_query: str) -> Optional[dict]:
        """Parse /bytebot or /bb command from chat input"""
        try:
            tokens = shlex.split(user_query)
        except ValueError:
            return None

        if not tokens:
            return None

        command = tokens[0].lower()
        if command not in ("/bytebot", "/bb"):
            return None

        mode = "assist"
        intent_parts = []
        i = 1
        while i < len(tokens):
            token = tokens[i]
            if token in ("--mode", "-m") and i + 1 < len(tokens):
                mode = tokens[i + 1].lower()
                i += 2
                continue
            if token.startswith("mode="):
                mode = token.split("=", 1)[1].lower()
                i += 1
                continue
            intent_parts.append(token)
            i += 1

        return {
            "mode": mode if mode in ("assist", "execute", "autonomous") else "assist",
            "intent": " ".join(intent_parts).strip()
        }

    async def _run_bytebot_intent(self, intent: str, mode: str) -> str:
        """Run ByteBot in a worker thread and format the result"""
        from xencode.bytebot import ByteBotEngine

        engine = ByteBotEngine()
        result = await asyncio.to_thread(engine.process_intent, intent, mode)

        status = result.get("status", "unknown")
        summary = result.get("summary") or result.get("message") or ""
        output_lines = [
            f"**ByteBot**",
            f"- Mode: {mode}",
            f"- Status: {status}",
        ]
        if summary:
            output_lines.append(f"- Summary: {summary}")

        steps = result.get("suggested_steps") or result.get("execution_results") or []
        if steps:
            output_lines.append("\n**Steps**")
            for idx, step in enumerate(steps, 1):
                command = step.get("command", "")
                step_status = step.get("status", "")
                risk = step.get("risk_score")
                risk_text = f"{risk:.2f}" if isinstance(risk, (int, float)) else ""
                output_lines.append(
                    f"{idx}. [{step_status}] {command} (risk: {risk_text})"
                )

        return "\n".join(output_lines)
    
    async def on_agent_task_submitted(self, event: "AgentTaskSubmitted") -> None:
        """Handle agent task submission

        Args:
            event: Agent task submission event
        """
        user_query = event.task

        if not self.chat_panel:
            return

        # Add thinking indicator
        thinking_msg = self.chat_panel.add_assistant_message("Processing with agents...")

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

            # Handle different collaboration types
            if event.collaboration_type == "sequential" and event.agent_sequence:
                # Sequential collaboration
                from xencode.agentic.coordinator import AgentCoordinator, AgentType

                coordinator = AgentCoordinator()
                result = coordinator.sequential_collaboration(enhanced_prompt, event.agent_sequence)

                # Format the result
                response_parts = [
                    f"SEQUENTIAL COLLABORATION RESULT:",
                    f"Original Task: {result['original_task']}",
                    f"Agent Sequence: {' -> '.join(result['agent_sequence'])}",
                    f"Total Steps: {result['total_steps']}",
                    "",
                    "INTERMEDIATE RESULTS:",
                ]

                for step_result in result['intermediate_results']:
                    response_parts.append(
                        f"Step {step_result['step']} ({step_result['agent']}): "
                        f"{step_result['output']}"
                    )

                response_parts.append("")
                response_parts.append(f"FINAL RESULT: {result['final_result']}")

                full_response = "\n".join(response_parts)
                self.chat_panel.update_streaming_message(full_response)

            elif event.collaboration_type == "adaptive":
                # Adaptive collaboration
                from xencode.agentic.coordinator import AgentCoordinator

                coordinator = AgentCoordinator()
                result = coordinator.adaptive_collaboration(enhanced_prompt)

                # Format the result based on the approach taken
                if result.get('approach') == 'standard_delegation':
                    full_response = f"Standard delegation result: {result['result']}"
                else:
                    # For sequential or parallel results
                    full_response = f"Adaptive collaboration result: {result.get('final_result', result.get('synthesized_result', 'No result'))}"

                self.chat_panel.update_streaming_message(full_response)

            elif event.use_multi_agent:
                # Multi-agent collaboration
                from xencode.agentic.coordinator import AgentCoordinator, AgentType

                coordinator = AgentCoordinator()
                result = coordinator.multi_agent_task([{
                    "task": enhanced_prompt,
                    "agent_type": None  # Will be classified automatically
                }])

                # Format multi-agent results
                response_parts = ["MULTI-AGENT RESULTS:"]
                for res in result:
                    response_parts.append(
                        f"Agent: {res['selected_agent']}, "
                        f"Result: {res['result'][:200]}..."
                    )

                full_response = "\n".join(response_parts)
                self.chat_panel.update_streaming_message(full_response)

            else:
                # Single agent (default behavior)
                async for chunk in self._stream_ai_response(enhanced_prompt):
                    full_response += chunk
                    self.chat_panel.update_streaming_message(full_response)

        except Exception as e:
            self.chat_panel.update_streaming_message(
                f"Error in agent collaboration: {str(e)}"
            )

    async def _stream_ai_response(self, prompt: str):
        """Stream AI response chunks

        Args:
            prompt: The prompt to send

        Yields:
            Response chunks
        """
        import aiohttp
        import json

        # Check if we're using a Qwen model that requires authentication
        if "qwen" in self.current_model.lower() and any(qwen_model in self.current_model.lower() for qwen_model in ["qwen-max", "qwen-plus", "qwen-max-coder", "qwen-chat", "chat.qwen.ai"]):
            # Use Qwen AI API with authentication via provider
            try:
                from xencode.model_providers import QwenProvider

                # Create Qwen provider instance
                provider = QwenProvider()

                # Format messages for chat API
                messages = [{"role": "user", "content": prompt}]

                # Call Qwen completion API via provider
                full_response = ""
                async for chunk in provider.chat(messages, self.current_model, max_tokens=2048, temperature=0.7):
                    full_response += chunk
                    yield chunk

            except Exception as e:
                yield f"\n\nError calling Qwen API: {str(e)}"
        else:
            # Use local Ollama API (existing behavior)
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "stream": True
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as response:
                        if response.status != 200:
                            yield f"\n\nError: API returned status {response.status}"
                            return

                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line)
                                    if "response" in data:
                                        yield data["response"]
                                    if "error" in data:
                                        yield f"\n\nError: {data['error']}"
                                except json.JSONDecodeError:
                                    pass

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
        collab_panel = self.query_one("#collab-panel-container")
        
        # Hide collab if open
        if not collab_panel.has_class("hidden"):
            collab_panel.add_class("hidden")
        
        if model_panel.has_class("hidden"):
            model_panel.remove_class("hidden")
            chat_container.add_class("shrink")
        else:
            model_panel.add_class("hidden")
            chat_container.remove_class("shrink")

    def action_toggle_collab(self) -> None:
        """Toggle collaboration panel visibility"""
        collab_panel = self.query_one("#collab-panel-container")
        chat_container = self.query_one("#chat-panel-container")
        model_panel = self.query_one("#model-selector-panel")
        bytebot_panel = self.query_one("#bytebot-panel-container")
        
        # Hide models if open
        if not model_panel.has_class("hidden"):
            model_panel.add_class("hidden")
        if not bytebot_panel.has_class("hidden"):
            bytebot_panel.add_class("hidden")
            
        if collab_panel.has_class("hidden"):
            collab_panel.remove_class("hidden")
            chat_container.add_class("shrink")
        else:
            collab_panel.add_class("hidden")
            chat_container.remove_class("shrink")

    def action_toggle_bytebot(self) -> None:
        """Toggle ByteBot panel visibility"""
        bytebot_panel = self.query_one("#bytebot-panel-container")
        chat_container = self.query_one("#chat-panel-container")
        model_panel = self.query_one("#model-selector-panel")
        collab_panel = self.query_one("#collab-panel-container")
        settings_panel = self.query_one("#settings-panel-container")

        # Hide other panels if open
        if not model_panel.has_class("hidden"):
            model_panel.add_class("hidden")
        if not collab_panel.has_class("hidden"):
            collab_panel.add_class("hidden")
        if not settings_panel.has_class("hidden"):
            settings_panel.add_class("hidden")

        if bytebot_panel.has_class("hidden"):
            bytebot_panel.remove_class("hidden")
            chat_container.add_class("shrink")
        else:
            bytebot_panel.add_class("hidden")
            chat_container.remove_class("shrink")

    def action_toggle_settings(self) -> None:
        """Toggle settings panel visibility"""
        settings_panel = self.query_one("#settings-panel-container")
        chat_container = self.query_one("#chat-panel-container")
        model_panel = self.query_one("#model-selector-panel")
        collab_panel = self.query_one("#collab-panel-container")
        bytebot_panel = self.query_one("#bytebot-panel-container")
        options_panel = self.query_one("#options-panel-container")

        if not model_panel.has_class("hidden"):
            model_panel.add_class("hidden")
        if not collab_panel.has_class("hidden"):
            collab_panel.add_class("hidden")
        if not bytebot_panel.has_class("hidden"):
            bytebot_panel.add_class("hidden")
        if not options_panel.has_class("hidden"):
            options_panel.add_class("hidden")

        if settings_panel.has_class("hidden"):
            settings_panel.remove_class("hidden")
            chat_container.add_class("shrink")
        else:
            settings_panel.add_class("hidden")
            chat_container.remove_class("shrink")

    def action_toggle_options(self) -> None:
        """Toggle options panel visibility."""
        options_panel = self.query_one("#options-panel-container")
        chat_container = self.query_one("#chat-panel-container")
        model_panel = self.query_one("#model-selector-panel")
        collab_panel = self.query_one("#collab-panel-container")
        bytebot_panel = self.query_one("#bytebot-panel-container")
        settings_panel = self.query_one("#settings-panel-container")

        if not model_panel.has_class("hidden"):
            model_panel.add_class("hidden")
        if not collab_panel.has_class("hidden"):
            collab_panel.add_class("hidden")
        if not bytebot_panel.has_class("hidden"):
            bytebot_panel.add_class("hidden")
        if not settings_panel.has_class("hidden"):
            settings_panel.add_class("hidden")

        if options_panel.has_class("hidden"):
            options_panel.remove_class("hidden")
            chat_container.add_class("shrink")
        else:
            options_panel.add_class("hidden")
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
            - **Ctrl+B**: Toggle ByteBot panel
            - **Ctrl+,**: Toggle settings panel
            - **Ctrl+O**: Toggle options panel
            - **Ctrl+L**: Clear chat history
            - **Ctrl+S**: Save current file (in editor)
            - **Ctrl+C**: Quit application
            - **Ctrl+Shift+L**: Logout of Qwen AI
            - **F1**: Show this help
            - **Tab**: Switch focus between panels
            - **Ctrl+Enter**: Send chat message

            ## Ensemble Mode
            Select 2-4 models in Model Selector to enable ensemble.
            Choose method: Vote, Weighted, Consensus, or Hybrid.

            ## Git Integration
            - **Ctrl+G**: Refresh Git status
            - Status indicators: M (Modified), A (Added), D (Deleted), ? (Untracked)

            ## ByteBot
            - **Ctrl+B**: Open ByteBot panel and run intents
            """
            self.chat_panel.add_system_message(help_text)

    def action_refresh_git(self) -> None:
        """Refresh Git status in file explorer"""
        if self.file_explorer:
            asyncio.create_task(self.file_explorer.refresh_git_status())
            self.notify("Git status refreshed", severity="information")

    async def action_commit_dialog(self) -> None:
        """Show commit dialog"""
        if not self.file_explorer or not self.file_explorer.git_manager:
            self.notify("Git integration not available", severity="error")
            return
            
        diff = await self.file_explorer.git_manager.get_diff(staged=True)
        if not diff:
            self.notify("No staged changes to commit", severity="warning")
            return
            
        def commit_callback(message: Optional[str]) -> None:
            if message:
                asyncio.create_task(self._do_commit(message))
                
        await self.push_screen(CommitDialog(diff, self._generate_commit_message), commit_callback)
        
    async def _do_commit(self, message: str) -> None:
        success = await self.file_explorer.git_manager.commit(message)
        if success:
            self.notify("Changes committed successfully", severity="information")
        else:
            self.notify("Failed to commit changes", severity="error")
            
    async def _generate_commit_message(self, diff: str) -> str:
        """Generate commit message from diff"""
        prompt = f"Generate a conventional commit message for the following git diff. Return ONLY the message.\n\n{diff[:2000]}"
        # Use existing streaming method but collect result
        response = ""
        async for chunk in self._stream_ai_response(prompt):
            if "Error:" not in chunk:
                response += chunk
        return response.strip()

    def action_toggle_terminal(self) -> None:
        """Toggle terminal visibility"""
        terminal = self.query_one(TerminalPanel)
        if terminal.has_class("visible"):
            terminal.remove_class("visible")
        else:
            terminal.add_class("visible")
            terminal.query_one("Input").focus()

    def action_logout_qwen(self) -> None:
        """Log out of Qwen authentication and clear cached credentials"""
        try:
            success = qwen_auth_manager.clear_credentials()
            if success:
                self.notify("Successfully logged out of Qwen AI", severity="information")
            else:
                self.notify("Failed to clear Qwen credentials", severity="warning")
        except Exception as e:
            self.notify(f"Error clearing Qwen credentials: {e}", severity="error")

    async def on_settings_panel_login_requested(self, event: SettingsPanel.LoginRequested) -> None:
        """Handle login action from settings panel."""
        await self._authenticate_qwen("login")

    async def on_settings_panel_signup_requested(self, event: SettingsPanel.SignupRequested) -> None:
        """Handle signup action from settings panel."""
        await self._authenticate_qwen("signup")

    def on_settings_panel_save_requested(self, event: SettingsPanel.SaveRequested) -> None:
        """Persist settings from settings panel."""
        self.ui_settings.update(event.settings)
        self._save_ui_settings()
        self._apply_ui_settings()
        self.notify("Settings saved", severity="information")

    def on_settings_panel_theme_changed(self, event: SettingsPanel.ThemeChanged) -> None:
        """Apply selected theme immediately for live preview."""
        self._apply_theme(event.theme)

    async def on_options_panel_command_requested(self, event: OptionsPanel.CommandRequested) -> None:
        """Run option command through embedded terminal."""
        command = event.command.strip()
        if not command:
            return

        if command == "tui":
            self.notify("You are already in TUI", severity="information")
            return

        if command == "query":
            if self.chat_panel:
                self.chat_panel.add_system_message("Use the chat panel directly for `query` in TUI.")
            return

        self.action_toggle_terminal()
        terminal = self.query_one(TerminalPanel)
        terminal.output.write(f"[bold green]➜ xencode {command}[/bold green]")
        await terminal._run_command(f"xencode {command}")

    def _handle_onboarding_result(self, result: Optional[str]) -> None:
        """Handle onboarding modal result."""
        self.ui_settings["onboarding_completed"] = True
        self._save_ui_settings()

        if result == "login":
            asyncio.create_task(self._authenticate_qwen("login"))
        elif result == "signup":
            asyncio.create_task(self._authenticate_qwen("signup"))

    async def _authenticate_qwen(self, intent: str) -> None:
        """Authenticate Qwen via device flow for login/signup paths."""
        action_name = "Sign up" if intent == "signup" else "Login"
        self.notify(f"{action_name} to Qwen started. Follow terminal/browser instructions.", severity="information")

        try:
            await qwen_auth_manager.get_or_authenticate()
            self.notify("Qwen authentication successful", severity="information")
            if self.chat_panel:
                self.chat_panel.add_system_message("✅ Qwen authentication completed.")
        except QwenAuthError as e:
            self.notify(f"Qwen authentication failed: {e}", severity="error")
        except Exception as e:
            self.notify(f"Unexpected auth error: {e}", severity="error")

    def _is_first_run(self) -> bool:
        """Determine whether onboarding should be shown."""
        return not bool(self.ui_settings.get("onboarding_completed", False))

    def _load_ui_settings(self) -> dict:
        """Load persisted UI settings from disk."""
        defaults = {
            "onboarding_completed": False,
            "show_explorer": True,
            "show_model_selector": False,
            "use_ensemble_default": False,
            "prompt_qwen_auth_on_first_run": True,
            "theme": "midnight",
        }

        try:
            if self.settings_path.exists():
                with open(self.settings_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                defaults.update(payload)
        except Exception:
            pass

        return defaults

    def _save_ui_settings(self) -> None:
        """Save persisted UI settings to disk."""
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.settings_path, "w", encoding="utf-8") as handle:
                json.dump(self.ui_settings, handle, indent=2)
        except Exception as e:
            self.notify(f"Failed to save settings: {e}", severity="error")

    def _apply_ui_settings(self) -> None:
        """Apply startup settings to the current UI."""
        try:
            left_panel = self.query_one("#left-panel")
            model_panel = self.query_one("#model-selector-panel")
            chat_panel_container = self.query_one("#chat-panel-container")

            if self.ui_settings.get("show_explorer", True):
                left_panel.remove_class("hidden")
            else:
                left_panel.add_class("hidden")

            if self.ui_settings.get("show_model_selector", False):
                model_panel.remove_class("hidden")
                chat_panel_container.add_class("shrink")
            else:
                model_panel.add_class("hidden")

            self.use_ensemble = bool(self.ui_settings.get("use_ensemble_default", False))
            self._apply_theme(self.ui_settings.get("theme", "midnight"))
        except Exception:
            pass

    def _apply_theme(self, theme_name: str) -> None:
        """Apply one of the predefined TUI themes."""
        themes = [
            "midnight", "ocean", "forest", "sunset", "violet",
            "slate", "terminal", "desert", "arctic", "rose"
        ]

        try:
            for theme in themes:
                self.screen.remove_class(f"theme-{theme}")
        except Exception:
            pass

        if theme_name not in themes:
            theme_name = "midnight"

        self.screen.add_class(f"theme-{theme_name}")

    def on_unmount(self) -> None:
        """Called when app is unmounted"""
        # Stop server if running
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            
        # Close WebSocket
        if self.ws_connection:
            asyncio.create_task(self.ws_connection.close())


def run_tui(root_path: Optional[Path] = None):
    """Run the Xencode TUI
    
    Args:
        root_path: Root directory for file explorer
    """
    app = XencodeApp(root_path=root_path)
    app.run()


if __name__ == "__main__":
    run_tui()
