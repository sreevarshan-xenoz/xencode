import ast
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

class CodeGraphExtractor:
    """
    Extracts code relationships (AST-based) for Graph-RAG.
    Currently supports Python using the native 'ast' module.
    """
    
    def __init__(self):
        self.relationships: List[Tuple[str, str, str, Dict[str, Any]]] = []
        self.nodes: List[Tuple[str, str, Dict[str, Any]]] = []
        
    def extract_from_file(self, file_path: str) -> None:
        """Extract nodes and relationships from a single file."""
        path = Path(file_path)
        if path.suffix != '.py':
            return # Currently only Python supported
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            file_node_id = str(path)
            
            # Add file node
            self.nodes.append((file_node_id, "file", {"name": path.name}))
            
            self._process_node(tree, file_node_id, file_node_id)
            
        except Exception as e:
            # print(f"Error parsing {file_path}: {e}")
            pass

    def _process_node(self, node: ast.AST, parent_id: str, file_id: str) -> None:
        """Recursively process AST nodes."""
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                class_id = f"{file_id}::{child.name}"
                self.nodes.append((class_id, "class", {"name": child.name, "line": child.lineno}))
                self.relationships.append((parent_id, class_id, "contains", {}))
                
                # Bases (inheritance)
                for base in child.bases:
                    if isinstance(base, ast.Name):
                        self.relationships.append((class_id, base.id, "inherits", {"is_external": True}))
                
                self._process_node(child, class_id, file_id)
                
            elif isinstance(child, ast.FunctionDef) or isinstance(child, ast.AsyncFunctionDef):
                func_id = f"{parent_id}.{child.name}" if "::" in parent_id else f"{file_id}::{child.name}"
                self.nodes.append((func_id, "function", {"name": child.name, "line": child.lineno}))
                self.relationships.append((parent_id, func_id, "contains", {}))
                
                # Process calls inside function
                self._extract_calls(child, func_id)
                
                # Don't recurse into function body for nested functions for now to keep it simple
                # but we could if needed.
                
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    self.relationships.append((file_id, alias.name, "imports", {"is_external": True}))
                    
            elif isinstance(child, ast.ImportFrom):
                module = child.module or ""
                self.relationships.append((file_id, module, "imports", {"is_external": True}))

    def _extract_calls(self, node: ast.AST, caller_id: str) -> None:
        """Extract function calls within a function body."""
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Call):
                if isinstance(sub_node.func, ast.Name):
                    self.relationships.append((caller_id, sub_node.func.id, "calls", {"is_external": True}))
                elif isinstance(sub_node.func, ast.Attribute):
                    # For method calls like obj.method(), we track the method name
                    self.relationships.append((caller_id, sub_node.func.attr, "calls_method", {"is_external": True}))

    def get_data(self) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        """Return the extracted nodes and relationships."""
        return self.nodes, self.relationships
    
    def clear(self):
        """Clear the current extraction data."""
        self.nodes = []
        self.relationships = []
