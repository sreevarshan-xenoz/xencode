#!/usr/bin/env python3
"""
Smart Context System for Xencode
Intelligently manages conversation context and project awareness
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import subprocess
import fnmatch

@dataclass
class ContextItem:
    """A single context item (file, conversation, etc.)"""
    type: str  # "file", "conversation", "command", "error"
    content: str
    metadata: Dict[str, Any]
    timestamp: float
    relevance_score: float = 0.0

class SmartContextManager:
    """Intelligent context management system"""
    
    def __init__(self, max_context_size: int = 8192):
        self.max_context_size = max_context_size
        self.context_items: List[ContextItem] = []
        self.project_root = self.find_project_root()
        self.file_cache = {}
        self.conversation_memory = []
        
    def find_project_root(self) -> Optional[Path]:
        """Find the root of the current project"""
        current = Path.cwd()
        
        # Look for common project indicators
        indicators = [
            ".git", "package.json", "requirements.txt", "Cargo.toml",
            "go.mod", "pom.xml", "build.gradle", "Makefile", "CMakeLists.txt"
        ]
        
        while current != current.parent:
            for indicator in indicators:
                if (current / indicator).exists():
                    return current
            current = current.parent
        
        return Path.cwd()  # Fallback to current directory
    
    def scan_project_files(self, extensions: List[str] = None) -> List[Path]:
        """Scan project for relevant files"""
        if not self.project_root:
            return []
        
        if extensions is None:
            extensions = [
                ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
                ".rs", ".go", ".php", ".rb", ".swift", ".kt", ".scala", ".cs",
                ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"
            ]
        
        files = []
        ignore_patterns = [
            "node_modules/*", ".git/*", "__pycache__/*", "*.pyc", 
            ".venv/*", "venv/*", "env/*", "build/*", "dist/*",
            ".next/*", ".nuxt/*", "target/*", "bin/*", "obj/*"
        ]
        
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                # Check if file should be ignored
                relative_path = file_path.relative_to(self.project_root)
                if any(fnmatch.fnmatch(str(relative_path), pattern) for pattern in ignore_patterns):
                    continue
                
                # Check if file has relevant extension
                if file_path.suffix.lower() in extensions:
                    files.append(file_path)
        
        return files[:100]  # Limit to prevent overwhelming
    
    def analyze_file_relevance(self, file_path: Path, query: str) -> float:
        """Analyze how relevant a file is to the current query"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Basic keyword matching
            query_words = query.lower().split()
            content_lower = content.lower()
            
            # Score based on keyword matches
            keyword_score = sum(1 for word in query_words if word in content_lower)
            
            # Boost score for certain file types
            file_type_boost = 0
            if file_path.suffix in [".py", ".js", ".ts"]:
                file_type_boost = 0.2
            elif file_path.suffix in [".md", ".txt"]:
                file_type_boost = 0.1
            
            # Boost score for main files
            main_file_boost = 0
            if file_path.name in ["main.py", "index.js", "app.py", "server.py"]:
                main_file_boost = 0.3
            
            # Calculate final relevance score
            relevance = (keyword_score / len(query_words)) + file_type_boost + main_file_boost
            return min(relevance, 1.0)
            
        except Exception:
            return 0.0
    
    def get_file_summary(self, file_path: Path) -> str:
        """Get a concise summary of a file's content"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # For code files, extract key elements
            if file_path.suffix == ".py":
                return self.summarize_python_file(content)
            elif file_path.suffix in [".js", ".ts"]:
                return self.summarize_javascript_file(content)
            elif file_path.suffix == ".md":
                return self.summarize_markdown_file(content)
            else:
                # Generic summary - first few lines
                lines = content.split('\n')[:10]
                return '\n'.join(lines)
                
        except Exception:
            return f"[Could not read {file_path.name}]"
    
    def summarize_python_file(self, content: str) -> str:
        """Summarize a Python file"""
        lines = content.split('\n')
        summary_parts = []
        
        # Extract docstring if present
        if '"""' in content:
            start = content.find('"""')
            end = content.find('"""', start + 3)
            if end != -1:
                docstring = content[start+3:end].strip()
                summary_parts.append(f"Purpose: {docstring[:200]}...")
        
        # Extract function and class definitions
        functions = []
        classes = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('def '):
                func_name = stripped.split('(')[0].replace('def ', '')
                functions.append(func_name)
            elif stripped.startswith('class '):
                class_name = stripped.split('(')[0].replace('class ', '').replace(':', '')
                classes.append(class_name)
        
        if classes:
            summary_parts.append(f"Classes: {', '.join(classes[:5])}")
        if functions:
            summary_parts.append(f"Functions: {', '.join(functions[:10])}")
        
        return '\n'.join(summary_parts) if summary_parts else content[:300]
    
    def summarize_javascript_file(self, content: str) -> str:
        """Summarize a JavaScript/TypeScript file"""
        lines = content.split('\n')
        summary_parts = []
        
        # Extract exports and functions
        exports = []
        functions = []
        
        for line in lines:
            stripped = line.strip()
            if 'export' in stripped:
                exports.append(stripped[:50])
            elif stripped.startswith('function ') or 'function(' in stripped:
                func_name = stripped.split('(')[0].replace('function ', '')
                functions.append(func_name)
        
        if exports:
            summary_parts.append(f"Exports: {', '.join(exports[:3])}")
        if functions:
            summary_parts.append(f"Functions: {', '.join(functions[:5])}")
        
        return '\n'.join(summary_parts) if summary_parts else content[:300]
    
    def summarize_markdown_file(self, content: str) -> str:
        """Summarize a Markdown file"""
        lines = content.split('\n')
        headers = []
        
        for line in lines:
            if line.startswith('#'):
                headers.append(line.strip())
        
        if headers:
            return '\n'.join(headers[:10])
        else:
            return content[:300]
    
    def add_context_from_query(self, query: str):
        """Add relevant context based on the current query"""
        # Scan for relevant files
        project_files = self.scan_project_files()
        
        for file_path in project_files:
            relevance = self.analyze_file_relevance(file_path, query)
            
            if relevance > 0.3:  # Only include relevant files
                summary = self.get_file_summary(file_path)
                
                context_item = ContextItem(
                    type="file",
                    content=f"File: {file_path.name}\n{summary}",
                    metadata={
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime
                    },
                    timestamp=time.time(),
                    relevance_score=relevance
                )
                
                self.context_items.append(context_item)
        
        # Sort by relevance and keep only the most relevant
        self.context_items.sort(key=lambda x: x.relevance_score, reverse=True)
        self.context_items = self.context_items[:10]  # Keep top 10
    
    def add_conversation_context(self, role: str, content: str, model: str = None):
        """Add conversation message to context"""
        context_item = ContextItem(
            type="conversation",
            content=f"{role}: {content}",
            metadata={
                "role": role,
                "model": model,
                "length": len(content)
            },
            timestamp=time.time(),
            relevance_score=1.0  # Conversation is always relevant
        )
        
        self.context_items.append(context_item)
        
        # Keep only recent conversation items
        conversation_items = [item for item in self.context_items if item.type == "conversation"]
        if len(conversation_items) > 20:
            # Remove oldest conversation items
            oldest_conversation = min(conversation_items, key=lambda x: x.timestamp)
            self.context_items.remove(oldest_conversation)
    
    def get_context_for_query(self, query: str, max_tokens: int = None) -> str:
        """Get the most relevant context for a query"""
        if max_tokens is None:
            max_tokens = self.max_context_size
        
        # Add context based on current query
        self.add_context_from_query(query)
        
        # Build context string
        context_parts = []
        current_tokens = 0
        
        # Add project information
        if self.project_root:
            project_info = f"Project: {self.project_root.name}\nLocation: {self.project_root}"
            context_parts.append(project_info)
            current_tokens += len(project_info.split())
        
        # Add most relevant context items
        for item in sorted(self.context_items, key=lambda x: x.relevance_score, reverse=True):
            item_tokens = len(item.content.split())
            
            if current_tokens + item_tokens > max_tokens:
                break
            
            context_parts.append(f"[{item.type.upper()}] {item.content}")
            current_tokens += item_tokens
        
        return "\n\n".join(context_parts)
    
    def clear_context(self):
        """Clear all context items"""
        self.context_items = []
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current context"""
        context_by_type = {}
        for item in self.context_items:
            if item.type not in context_by_type:
                context_by_type[item.type] = []
            context_by_type[item.type].append(item)
        
        summary = {
            "total_items": len(self.context_items),
            "by_type": {k: len(v) for k, v in context_by_type.items()},
            "project_root": str(self.project_root) if self.project_root else None,
            "estimated_tokens": sum(len(item.content.split()) for item in self.context_items)
        }
        
        return summary

# Example usage
def demo_smart_context():
    """Demonstrate the smart context system"""
    manager = SmartContextManager()
    
    print("🧠 Smart Context System Demo")
    print("=" * 40)
    
    # Show project info
    print(f"\n📁 Project Root: {manager.project_root}")
    
    # Scan files
    files = manager.scan_project_files()
    print(f"📄 Found {len(files)} relevant files")
    
    # Test context for different queries
    test_queries = [
        "How do I fix a Python import error?",
        "Show me the main application code",
        "What's in the README file?"
    ]
    
    for query in test_queries:
        print(f"\n🎯 Query: '{query}'")
        context = manager.get_context_for_query(query)
        print(f"Context length: {len(context)} characters")
        print("Context preview:")
        print(context[:300] + "..." if len(context) > 300 else context)
        
        # Show context summary
        summary = manager.get_context_summary()
        print(f"Summary: {summary}")
        
        manager.clear_context()

if __name__ == "__main__":
    demo_smart_context()