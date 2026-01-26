"""
Enhanced File Explorer for Xencode TUI

Advanced file explorer with Git integration, file previews, and enhanced navigation.
"""

from typing import Dict, List, Optional, Any
import os
import subprocess
from pathlib import Path
from datetime import datetime
import mimetypes
from rich.text import Text
from textual.widgets import Static, Tree, DirectoryTree, Button, Input, Label, DataTable
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.message import Message
from textual.reactive import reactive
from textual.events import Click
from git import Repo, InvalidGitRepositoryError
import asyncio


class GitStatus:
    """Class to represent Git status information"""
    def __init__(self):
        self.is_tracked = False
        self.is_modified = False
        self.is_added = False
        self.is_deleted = False
        self.is_unmerged = False
        self.is_ignored = False
        self.status_char = " "  # Default status character


class EnhancedFileExplorer(Container):
    """Enhanced file explorer with Git integration and file previews"""

    DEFAULT_CSS = """
    EnhancedFileExplorer {
        height: 100%;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }

    .explorer-header {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    .file-tree {
        height: 1fr;
        border: solid $secondary;
        padding: 1;
    }

    .preview-pane {
        height: 1fr;
        border: solid $secondary;
        padding: 1;
        background: $boost;
    }

    .git-status {
        text-style: bold;
    }

    .status-modified {
        color: $warning;
    }

    .status-added {
        color: $success;
    }

    .status-deleted {
        color: $error;
    }

    .status-untracked {
        color: $text-muted;
    }

    .file-info {
        margin-top: 1;
        border: solid $secondary;
        padding: 1;
    }

    .search-controls {
        margin-bottom: 1;
    }

    .git-diff-view {
        height: 20;
        border: solid $secondary;
        padding: 1;
        background: $panel;
    }

    .breadcrumb {
        margin-bottom: 1;
    }

    .breadcrumb-separator {
        color: $text-muted;
        margin: 0 0.5;
    }
    """

    class FileSelected(Message):
        """Message sent when a file is selected"""
        def __init__(self, path: Path, file_info: Dict[str, Any] = None):
            self.path = path
            self.file_info = file_info or {}
            super().__init__()

    class DirectoryChanged(Message):
        """Message sent when directory changes"""
        def __init__(self, path: Path):
            self.path = path
            super().__init__()

    class GitStatusChanged(Message):
        """Message sent when Git status changes"""
        def __init__(self, path: Path, git_status: GitStatus):
            self.path = path
            self.git_status = git_status
            super().__init__()

    # Reactive properties
    current_path = reactive(Path.cwd())
    git_repo = reactive(None)
    git_status_cache = reactive(lambda: {})
    search_query = reactive("")

    def __init__(self, root_path: Optional[Path] = None, *args, **kwargs):
        """Initialize the enhanced file explorer"""
        super().__init__(*args, **kwargs)
        self.border_title = "📁 Enhanced File Explorer"
        self.root_path = root_path or Path.cwd()
        self.current_path = self.root_path
        self.git_repo = self._find_git_repo(self.root_path)
        self.git_status_cache = {}
        self.file_tree = None
        self.preview_pane = None
        self.file_info_pane = None
        self.breadcrumb = None

    def compose(self):
        """Compose the enhanced file explorer"""
        yield Label("📁 Enhanced File Explorer", classes="explorer-header")

        # Breadcrumb navigation
        self.breadcrumb = Static(id="breadcrumb", classes="breadcrumb")
        yield self.breadcrumb

        # Search controls
        with Horizontal(classes="search-controls"):
            self.search_input = Input(placeholder="Search files...", id="search-input")
            yield self.search_input
            yield Button("🔍 Search", id="search-btn", variant="primary")
            yield Button("🔄 Refresh", id="refresh-btn", variant="default")
            yield Button("Git Status", id="git-status-btn", variant="success")

        # Main content area
        with Horizontal():
            # File tree
            self.file_tree = DirectoryTree(self.root_path, id="file-tree", classes="file-tree")
            yield self.file_tree

            # Preview and info pane
            with Vertical():
                # Preview pane
                self.preview_pane = Static(id="preview-pane", classes="preview-pane")
                self.preview_pane.border_title = "File Preview"
                yield self.preview_pane

                # File info pane
                self.file_info_pane = Static(id="file-info", classes="file-info")
                self.file_info_pane.border_title = "File Information"
                yield self.file_info_pane

        # Git status details (when available)
        self.git_status_details = Static(id="git-status-details", classes="git-diff-view")
        self.git_status_details.border_title = "Git Status"
        self.git_status_details.visible = False
        yield self.git_status_details

    def on_mount(self) -> None:
        """Called when widget is mounted"""
        self.update_breadcrumb()
        self.refresh_git_status()

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection in the tree"""
        file_path = event.path
        file_info = self._get_file_info(file_path)
        
        # Update preview
        self.update_preview(file_path)
        
        # Update file info
        self.update_file_info(file_path, file_info)
        
        # Update Git status if applicable
        self.update_git_status_for_file(file_path)
        
        # Post message
        self.post_message(self.FileSelected(file_path, file_info))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search input submission"""
        if event.input.id == "search-input":
            self.perform_search(event.input.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "search-btn":
            self.perform_search(self.search_input.value)
        elif event.button.id == "refresh-btn":
            self.refresh_view()
        elif event.button.id == "git-status-btn":
            self.show_git_status()

    def update_breadcrumb(self):
        """Update the breadcrumb navigation"""
        if not self.current_path:
            return

        parts = []
        path = self.current_path.resolve()

        # Add home indicator
        if path.is_relative_to(Path.home()):
            parts.append(("~", str(Path.home())))
        else:
            # Add drive letter on Windows
            if os.name == 'nt':
                drive = path.drive
                if drive:
                    parts.append((drive, drive))

        # Add path segments
        relative_to_root = path.relative_to(self.root_path) if path.is_relative_to(self.root_path) else path.relative_to(path.anchor)
        path_parts = list(relative_to_root.parts) if relative_to_root.parts != ('.',) else []

        for i, part in enumerate(path_parts):
            if part:
                # Build path up to this point
                current_segment_path = self.root_path.joinpath(*path_parts[:i+1])
                parts.append((part, str(current_segment_path)))

        # Create breadcrumb text
        breadcrumb_items = []
        for i, (name, path_str) in enumerate(parts):
            if i > 0:
                breadcrumb_items.append(Text(" / ", classes="breadcrumb-separator"))
            breadcrumb_items.append(
                Text(name, style="underline", classes="clickable")
            )

        self.breadcrumb.update(Text.assemble(*breadcrumb_items))

    def _find_git_repo(self, path: Path) -> Optional[Repo]:
        """Find the Git repository for a given path"""
        try:
            # Walk up the directory tree to find a .git folder
            current = path.resolve()
            while current != current.parent:
                if (current / ".git").exists():
                    return Repo(current)
                current = current.parent
            return None
        except InvalidGitRepositoryError:
            return None
        except Exception:
            return None

    def _get_git_status(self, file_path: Path) -> GitStatus:
        """Get Git status for a specific file"""
        if not self.git_repo:
            return GitStatus()

        try:
            # Get the relative path from the repo root
            rel_path = file_path.relative_to(self.git_repo.working_dir)
            rel_path_str = str(rel_path)

            # Check if file is in the index
            if rel_path_str in self.git_repo.untracked_files:
                status = GitStatus()
                status.is_tracked = False
                status.is_modified = False
                status.is_added = False
                status.is_deleted = False
                status.is_unmerged = False
                status.is_ignored = False
                status.status_char = "?"
                return status

            # Get the diff to determine status
            diffs = list(self.git_repo.index.diff(None))
            staged_diffs = list(self.git_repo.head.commit.diff())

            status = GitStatus()
            status.is_tracked = True

            # Check for modifications
            for diff in diffs:
                if diff.a_path == rel_path_str or diff.b_path == rel_path_str:
                    status.is_modified = True
                    status.status_char = "M"
                    break

            # Check for staged changes
            for diff in staged_diffs:
                if diff.a_path == rel_path_str or diff.b_path == rel_path_str:
                    status.is_added = True
                    if status.status_char == "M":
                        status.status_char = "AM"  # Both added and modified
                    else:
                        status.status_char = "A"
                    break

            return status

        except Exception:
            # Return default status if there's an error
            return GitStatus()

    def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get detailed information about a file"""
        try:
            stat = file_path.stat()
            mime_type, _ = mimetypes.guess_type(str(file_path))

            info = {
                "name": file_path.name,
                "path": str(file_path),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "created": datetime.fromtimestamp(stat.st_ctime),
                "is_file": file_path.is_file(),
                "is_dir": file_path.is_dir(),
                "mime_type": mime_type or "unknown",
                "extension": file_path.suffix.lower(),
                "permissions": oct(stat.st_mode)[-3:],
            }

            # Add Git status if available
            if self.git_repo:
                git_status = self._get_git_status(file_path)
                info["git_status"] = git_status

            return info
        except Exception as e:
            return {
                "name": file_path.name,
                "path": str(file_path),
                "error": str(e),
                "is_file": file_path.is_file(),
                "is_dir": file_path.is_dir(),
            }

    def update_preview(self, file_path: Path):
        """Update the file preview pane"""
        if not file_path.is_file():
            self.preview_pane.update("Directory - No preview available")
            return

        try:
            # Determine if file is text-based
            mime_type, _ = mimetypes.guess_type(str(file_path))
            is_text = mime_type and mime_type.startswith('text/')

            if is_text or file_path.suffix.lower() in ['.py', '.js', '.ts', '.html', '.css', '.json', '.md', '.txt', '.yaml', '.yml']:
                # Read text file
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read(1000)  # Limit preview to first 1000 chars

                # Truncate if too long
                if len(content) > 1000:
                    content = content[:1000] + "... (truncated)"

                # Add syntax highlighting for code files
                if file_path.suffix.lower() in ['.py', '.js', '.ts', '.html', '.css', '.json', '.md', '.yaml', '.yml']:
                    from rich.syntax import Syntax
                    syntax = Syntax(content, lexer=file_path.suffix.lower().lstrip('.'), theme="monokai", line_numbers=True)
                    self.preview_pane.update(syntax)
                else:
                    self.preview_pane.update(content)
            else:
                # For binary files, show file info instead
                self.preview_pane.update(f"Binary file: {file_path.name}\\nSize: {file_path.stat().st_size} bytes\\nType: {mime_type}")

        except Exception as e:
            self.preview_pane.update(f"Error reading file: {str(e)}")

    def update_file_info(self, file_path: Path, file_info: Dict[str, Any]):
        """Update the file information pane"""
        try:
            size = file_info.get('size', 0)
            size_str = self._format_file_size(size)

            modified = file_info.get('modified', 'Unknown')
            created = file_info.get('created', 'Unknown')

            info_lines = [
                f"Name: {file_info.get('name', 'Unknown')}",
                f"Path: {file_info.get('path', 'Unknown')}",
                f"Size: {size_str}",
                f"Modified: {modified}",
                f"Created: {created}",
                f"Type: {file_info.get('mime_type', 'Unknown')}",
                f"Permissions: {file_info.get('permissions', 'Unknown')}"
            ]

            # Add Git status if available
            if 'git_status' in file_info:
                git_status = file_info['git_status']
                status_text = f"Git: {git_status.status_char}"
                if git_status.is_modified:
                    status_text += " (Modified)"
                elif not git_status.is_tracked:
                    status_text += " (Untracked)"
                info_lines.append(status_text)

            self.file_info_pane.update("\\n".join(info_lines))

        except Exception as e:
            self.file_info_pane.update(f"Error showing file info: {str(e)}")

    def update_git_status_for_file(self, file_path: Path):
        """Update Git status display for a specific file"""
        if not self.git_repo:
            self.git_status_details.visible = False
            return

        try:
            # Get diff for the specific file
            rel_path = file_path.relative_to(self.git_repo.working_dir)
            diff = self.git_repo.git.diff("HEAD", str(rel_path), unified=3)

            if diff:
                self.git_status_details.update(diff)
                self.git_status_details.visible = True
                self.git_status_details.border_title = f"Git Diff: {file_path.name}"
            else:
                self.git_status_details.update("No changes in Git")
                self.git_status_details.visible = True
                self.git_status_details.border_title = f"Git Status: {file_path.name}"

        except Exception as e:
            self.git_status_details.update(f"Error getting Git status: {str(e)}")
            self.git_status_details.visible = True
            self.git_status_details.border_title = f"Git Error: {file_path.name}"

    def refresh_git_status(self):
        """Refresh Git status for all files"""
        if not self.git_repo:
            return

        try:
            # Get all changed files
            changed_files = self.git_repo.git.diff("--name-status", "HEAD").splitlines()
            staged_files = self.git_repo.git.diff("--name-status", "--cached").splitlines()
            untracked_files = self.git_repo.untracked_files

            # Update status cache
            for f in changed_files + staged_files:
                if f:
                    parts = f.split("\\t")
                    if len(parts) >= 2:
                        file_path = self.git_repo.working_dir / parts[1]
                        self.git_status_cache[str(file_path)] = self._get_git_status(file_path)

            for f in untracked_files:
                file_path = self.git_repo.working_dir / f
                self.git_status_cache[str(file_path)] = self._get_git_status(file_path)

        except Exception:
            pass  # Silently fail if Git repo is not available

    def show_git_status(self):
        """Show overall Git status"""
        if not self.git_repo:
            self.git_status_details.update("Not in a Git repository")
            self.git_status_details.visible = True
            self.git_status_details.border_title = "Git Status"
            return

        try:
            # Get overall status
            status_output = self.git_repo.git.status(short=True, branch=True)
            self.git_status_details.update(status_output)
            self.git_status_details.visible = True
            self.git_status_details.border_title = "Git Status"

        except Exception as e:
            self.git_status_details.update(f"Error getting Git status: {str(e)}")
            self.git_status_details.visible = True
            self.git_status_details.border_title = "Git Status Error"

    def perform_search(self, query: str):
        """Perform file search"""
        if not query:
            return

        try:
            results = []
            for root, dirs, files in os.walk(self.root_path):
                for name in dirs + files:
                    if query.lower() in name.lower():
                        path = Path(root) / name
                        results.append(str(path))

            # Limit results to prevent overwhelming
            results = results[:50]

            if results:
                result_text = "\\n".join(results)
                self.preview_pane.update(f"Search results for '{query}':\\n\\n{result_text}")
            else:
                self.preview_pane.update(f"No files found matching '{query}'")

        except Exception as e:
            self.preview_pane.update(f"Error during search: {str(e)}")

    def refresh_view(self):
        """Refresh the file explorer view"""
        # Update the tree by recreating it
        if self.file_tree:
            self.file_tree.path = self.current_path
            self.file_tree.reload()

        # Refresh Git status
        self.git_repo = self._find_git_repo(self.current_path)
        self.refresh_git_status()

        # Update breadcrumb
        self.update_breadcrumb()

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        if size_bytes == 0:
            return "0B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"

    def on_click(self, event: Click) -> None:
        """Handle click events in the breadcrumb"""
        # This would handle clicking on breadcrumb segments to navigate
        # Implementation would depend on how we associate click positions with segments
        pass