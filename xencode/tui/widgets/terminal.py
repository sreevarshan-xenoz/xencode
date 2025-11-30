#!/usr/bin/env python3
"""
Embedded Terminal Widget
"""

import asyncio
import shlex
import subprocess
from collections import deque
from typing import Deque

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, Container
from textual.widgets import Input, RichLog, Static, Label
from textual.binding import Binding
from textual.message import Message

class TerminalPanel(Container):
    """Embedded terminal for running commands"""
    
    DEFAULT_CSS = """
    TerminalPanel {
        height: 30%;
        dock: bottom;
        border-top: solid $accent;
        background: $surface;
        display: none;
    }
    
    TerminalPanel.visible {
        display: block;
    }
    
    #term-output {
        height: 1fr;
        background: $surface-darken-1;
        border: none;
        padding: 0 1;
    }
    
    #term-input-container {
        height: 3;
        background: $surface;
        padding: 0 1;
    }
    
    #term-input {
        border: none;
        background: $surface;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.history: Deque[str] = deque(maxlen=50)
        self.history_index = -1
        self.current_process = None
        
    def compose(self) -> ComposeResult:
        yield Label("💻 Terminal", classes="section-title")
        yield RichLog(id="term-output", markup=True, wrap=True)
        
        with Container(id="term-input-container"):
            yield Input(placeholder="Enter command...", id="term-input")
            
    def on_mount(self) -> None:
        self.output = self.query_one("#term-output", RichLog)
        self.input = self.query_one("#term-input", Input)
        self.output.write("[dim]Xencode Terminal Ready. Type a command and press Enter.[/dim]")
        
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        command = event.value.strip()
        if not command:
            return
            
        self.input.value = ""
        self.history.append(command)
        self.history_index = -1
        
        self.output.write(f"[bold green]➜ {command}[/bold green]")
        
        if command == "clear":
            self.output.clear()
            return
            
        await self._run_command(command)
        
    async def _run_command(self, command: str) -> None:
        """Run command asynchronously"""
        try:
            # Use shell=True equivalent for better command support (e.g. dir, pipes)
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=None, # Use current working directory
            )
            
            self.current_process = process
            
            # Read output
            async def read_stream(stream, color):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode().rstrip()
                    self.output.write(f"[{color}]{text}[/{color}]")
            
            await asyncio.gather(
                read_stream(process.stdout, "white"),
                read_stream(process.stderr, "red")
            )
            
            await process.wait()
            
            if process.returncode != 0:
                self.output.write(f"[red]Exited with code {process.returncode}[/red]")
                
        except FileNotFoundError:
            self.output.write(f"[red]Command not found: {command}[/red]")
        except Exception as e:
            self.output.write(f"[red]Error: {str(e)}[/red]")
        finally:
            self.current_process = None
            
    def key_up(self) -> None:
        """History navigation up"""
        if not self.history:
            return
            
        if self.history_index == -1:
            self.history_index = len(self.history) - 1
        elif self.history_index > 0:
            self.history_index -= 1
            
        self.input.value = self.history[self.history_index]
        
    def key_down(self) -> None:
        """History navigation down"""
        if self.history_index == -1:
            return
            
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.input.value = self.history[self.history_index]
        else:
            self.history_index = -1
            self.input.value = ""
