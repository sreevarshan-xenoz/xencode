#!/usr/bin/env python3
"""
File Explorer Widget for Xencode TUI

A tree-based file explorer similar to VS Code's sidebar.
"""

from pathlib import Path
from typing import Optional

from rich.text import Text
from textual.widgets import Tree
from textual.widgets.tree import TreeNode
from textual import events


class FileExplorer(Tree):
    """File explorer tree widget"""

    def __init__(
        self,
        root_path: Optional[Path] = None,
        *args,
        **kwargs
    ):
        """Initialize file explorer
        
        Args:
            root_path: Root directory to explore (defaults to current directory)
        """
        self.root_path = root_path or Path.cwd()
        
        # Create root node
        root_label = Text("📁 " + self.root_path.name, style="bold cyan")
        super().__init__(root_label, *args, **kwargs)
        
        self.root.data = self.root_path
        self._populate_node(self.root)
    
    def _populate_node(self, node: TreeNode) -> None:
        """Populate a tree node with its children
        
        Args:
            node: The tree node to populate
        """
        path: Path = node.data
        
        if not path.is_dir():
            return
        
        # Depth limit for performance (max 5 levels deep)
        current_depth = 0
        temp_node = node
        while temp_node.parent is not None:
            current_depth += 1
            temp_node = temp_node.parent
        
        if current_depth > 5:
            node.add_leaf("⚠️ Max depth reached", data=None)
            return
        
        try:
            # Get all items in directory (limit to first 100 for performance)
            items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            
            # Limit items shown
            MAX_ITEMS = 100
            if len(items) > MAX_ITEMS:
                items = items[:MAX_ITEMS]
                show_truncated = True
            else:
                show_truncated = False
            
            for item in items:
                # Skip hidden files and common ignore patterns
                if item.name.startswith(".") or item.name in ["__pycache__", "node_modules", ".git", "venv", ".venv"]:
                    continue
                
                # Create label with icon
                if item.is_dir():
                    icon = "📁"
                    style = "bold blue"
                else:
                    icon = self._get_file_icon(item.suffix)
                    style = "white"
                
                label = Text(f"{icon} {item.name}", style=style)
                
                # Add child node
                child = node.add(label, data=item)
                
                # If it's a directory, mark it as expandable
                if item.is_dir():
                    # Add a placeholder to make it expandable
                    child.allow_expand = True
            
            if show_truncated:
                node.add_leaf(f"⚠️ {len(list(path.iterdir())) - MAX_ITEMS} more items...", data=None)
        
        except PermissionError:
            # Can't read this directory
            node.add_leaf("🔒 Permission Denied", data=None)
    
    def _get_file_icon(self, suffix: str) -> str:
        """Get icon for file type
        
        Args:
            suffix: File extension (e.g., '.py')
            
        Returns:
            Icon character
        """
        icon_map = {
            ".py": "🐍",
            ".js": "📜",
            ".ts": "📘",
            ".html": "🌐",
            ".css": "🎨",
            ".json": "📋",
            ".md": "📝",
            ".txt": "📄",
            ".yaml": "⚙️",
            ".yml": "⚙️",
            ".toml": "⚙️",
            ".sh": "⚡",
            ".bat": "⚡",
            ".exe": "⚙️",
            ".jpg": "🖼️",
            ".png": "🖼️",
            ".gif": "🖼️",
            ".pdf": "📕",
        }
        return icon_map.get(suffix.lower(), "📄")
    
    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        """Called when a node is expanded
        
        Args:
            event: The expand event
        """
        node = event.node
        
        # Remove placeholder children and populate with real content
        if node.children:
            for child in list(node.children):
                child.remove()
        
        self._populate_node(node)
    
    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Called when a node is selected
        
        Args:
            event: The selection event
        """
        node = event.node
        path: Optional[Path] = node.data
        
        if path and path.is_file():
            # Post a custom message to the app
            self.post_message(FileSelected(path))


class FileSelected(events.Message):
    """Message sent when a file is selected in the explorer"""
    
    def __init__(self, path: Path) -> None:
        """Initialize message
        
        Args:
            path: The selected file path
        """
        super().__init__()
        self.path = path
