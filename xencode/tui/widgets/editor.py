#!/usr/bin/env python3
"""
Code Editor/Viewer Widget for Xencode TUI

Displays and edits code files with syntax highlighting.
"""

from pathlib import Path
from typing import Optional

from rich.syntax import Syntax
from textual.widgets import TextArea
from textual.containers import Container, VerticalScroll
from textual.binding import Binding


class CodeEditor(Container):
    """Code editor widget with syntax highlighting and editing"""
    
    DEFAULT_CSS = """
    CodeEditor {
        background: $surface;
        border: solid $primary;
        height: 100%;
    }
    
    CodeEditor TextArea {
        height: 100%;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+s", "save_file", "Save File", show=True),
    ]
    
    def __init__(self, *args, **kwargs):
        """Initialize code editor"""
        super().__init__(*args, **kwargs)
        self.current_file: Optional[Path] = None
        self.text_area: Optional[TextArea] = None
        self.is_modified = False
    
    def compose(self):
        """Compose the editor"""
        self.text_area = TextArea(
            text="No file selected",
            language=None,
            theme="monokai",
            show_line_numbers=True,
            read_only=False,
        )
        self.text_area.can_focus = True
        yield self.text_area
    
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text changes"""
        if self.current_file:
            self.is_modified = True
            self._update_title()
    
    def load_file(self, file_path: Path) -> None:
        """Load a file into the editor
        
        Args:
            file_path: Path to the file to load
        """
        self.current_file = file_path
        self.is_modified = False
        
        try:
            # Check file size (10MB limit)
            MAX_FILE_SIZE = 10 * 1024 * 1024
            file_size = file_path.stat().st_size
            
            if file_size > MAX_FILE_SIZE:
                if self.text_area:
                    self.text_area.text = (
                        f"File too large ({file_size / 1024 / 1024:.1f}MB)\n"
                        f"Maximum: {MAX_FILE_SIZE / 1024 / 1024}MB\n\n"
                        f"Use a regular text editor for large files."
                    )
                    self.text_area.read_only = True
                self.border_title = f"📦 {file_path.name} (Too Large)"
                return
            
            # Read file content
            content = file_path.read_text(encoding="utf-8")
            
            # Limit lines for performance (max 1000 lines)
            lines = content.split('\n')
            if len(lines) > 1000:
                content = '\n'.join(lines[:1000])
                truncated = True
            else:
                truncated = False
            
            # Determine language from file extension
            language = self._get_language(file_path.suffix)
            
            # Update text area
            if self.text_area:
                self.text_area.text = content
                self.text_area.language = language
                self.text_area.read_only = False
            
            # Update border title
            self._update_title(truncated)
        
        except UnicodeDecodeError:
            # Binary file
            if self.text_area:
                self.text_area.text = f"Binary file: {file_path.name}\n\nCannot edit binary files."
                self.text_area.read_only = True
            self.border_title = f"📦 {file_path.name} (Binary)"
        
        except Exception as e:
            # Error reading file
            if self.text_area:
                self.text_area.text = f"Error loading file: {e}"
                self.text_area.read_only = True
            self.border_title = f"❌ {file_path.name}"
    
    def _update_title(self, truncated: bool = False) -> None:
        """Update the border title"""
        if not self.current_file:
            self.border_title = "Code Editor"
            return
        
        title = f"📝 {self.current_file.name}"
        if self.is_modified:
            title += " ●"  # Modified indicator
        if truncated:
            title += " (first 1000 lines)"
        self.border_title = title
    
    def action_save_file(self) -> None:
        """Save the current file"""
        if not self.current_file or not self.text_area:
            return
        
        try:
            content = self.text_area.text
            self.current_file.write_text(content, encoding="utf-8")
            self.is_modified = False
            self._update_title()
            
            # Show saved message
            self.app.notify(f"Saved: {self.current_file.name}", severity="information")
        
        except Exception as e:
            self.app.notify(f"Error saving: {e}", severity="error")
    
    def _get_language(self, suffix: str) -> str:
        """Get language name for syntax highlighting
        
        Args:
            suffix: File extension (e.g., '.py')
            
        Returns:
            Language name
        """
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "jsx",
            ".tsx": "tsx",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".md": "markdown",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "bash",
            ".bat": "batch",
            ".ps1": "powershell",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".sql": "sql",
            ".xml": "xml",
        }
        return language_map.get(suffix.lower(), "text")
    
    def clear(self) -> None:
        """Clear the editor"""
        self.current_file = None
        self.is_modified = False
        if self.text_area:
            self.text_area.text = "No file selected"
            self.text_area.language = None
        self.border_title = "Code Editor"
