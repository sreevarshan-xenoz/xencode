"""Enhanced tools for the agentic system with additional capabilities."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Type, List, Dict, Any
from datetime import datetime

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from git import Repo, InvalidGitRepositoryError
from duckduckgo_search import DDGS

# Import existing code analyzer
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from multimodal.code_analyzer import CodeAnalyzer
except ImportError:
    # If multimodal module is not available, create a dummy class
    class CodeAnalyzer:
        def analyze_python_file(self, path):
            return {"error": "CodeAnalyzer not available"}

        def analyze_directory(self, path):
            return {"error": "CodeAnalyzer not available"}


# ============================================================================
# Enhanced Git Operations Tools
# ============================================================================

class GitBranchSchema(BaseModel):
    repo_path: str = Field(description="Path to the git repository", default=".")
    branch_name: Optional[str] = Field(description="Name of the branch to create/switch to", default=None)


class GitBranchTool(BaseTool):
    name: str = "git_branch"
    description: str = "Create, list, or switch git branches"
    args_schema: Type[BaseModel] = GitBranchSchema

    def _run(self, repo_path: str = ".", branch_name: Optional[str] = None) -> str:
        try:
            repo = Repo(repo_path)
            
            if branch_name:
                # Create or switch to branch
                if branch_name in [branch.name for branch in repo.branches]:
                    repo.heads[branch_name].checkout()
                    return f"Switched to existing branch: {branch_name}"
                else:
                    repo.create_head(branch_name)
                    repo.heads[branch_name].checkout()
                    return f"Created and switched to new branch: {branch_name}"
            else:
                # List branches
                current_branch = repo.active_branch.name
                branches = [branch.name for branch in repo.branches]
                branch_list = []
                for branch in branches:
                    marker = "*" if branch == current_branch else " "
                    branch_list.append(f"{marker} {branch}")
                
                return f"Branches:\n" + "\n".join(branch_list)
        except InvalidGitRepositoryError:
            return f"Error: {repo_path} is not a git repository"
        except Exception as e:
            return f"Error managing git branches: {str(e)}"


class GitPushSchema(BaseModel):
    repo_path: str = Field(description="Path to the git repository", default=".")
    remote: str = Field(description="Remote name (e.g., 'origin')", default="origin")
    branch: Optional[str] = Field(description="Branch name to push", default=None)


class GitPushTool(BaseTool):
    name: str = "git_push"
    description: str = "Push commits to a remote repository"
    args_schema: Type[BaseModel] = GitPushSchema

    def _run(self, repo_path: str = ".", remote: str = "origin", branch: Optional[str] = None) -> str:
        try:
            repo = Repo(repo_path)
            
            if not branch:
                branch = repo.active_branch.name
            
            result = repo.remotes[remote].push(refspec=f'{branch}:{branch}')
            
            if result:
                return f"Push result: {result[0].summary if result else 'Success'}"
            else:
                return f"Successfully pushed {branch} to {remote}"
        except Exception as e:
            return f"Error pushing to remote: {str(e)}"


class GitPullSchema(BaseModel):
    repo_path: str = Field(description="Path to the git repository", default=".")
    remote: str = Field(description="Remote name (e.g., 'origin')", default="origin")
    branch: Optional[str] = Field(description="Branch name to pull", default=None)


class GitPullTool(BaseTool):
    name: str = "git_pull"
    description: str = "Pull latest changes from a remote repository"
    args_schema: Type[BaseModel] = GitPullSchema

    def _run(self, repo_path: str = ".", remote: str = "origin", branch: Optional[str] = None) -> str:
        try:
            repo = Repo(repo_path)
            
            if not branch:
                branch = repo.active_branch.name
            
            result = repo.remotes[remote].pull(refspec=f'{branch}:{branch}')
            
            return f"Successfully pulled {branch} from {remote}. Updates: {len(result)}"
        except Exception as e:
            return f"Error pulling from remote: {str(e)}"


# ============================================================================
# Enhanced File Operations Tools
# ============================================================================

class FindFileSchema(BaseModel):
    directory: str = Field(description="Directory to search in", default=".")
    pattern: str = Field(description="Pattern to search for (supports wildcards)")


class FindFileTool(BaseTool):
    name: str = "find_file"
    description: str = "Find files in a directory matching a pattern"
    args_schema: Type[BaseModel] = FindFileSchema

    def _run(self, directory: str = ".", pattern: str = "*") -> str:
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return f"Error: Directory {directory} does not exist"
            
            # Use glob to find matching files
            matches = list(dir_path.glob(pattern))
            
            if not matches:
                return f"No files found matching pattern '{pattern}' in {directory}"
            
            result = [str(match) for match in matches[:50]]  # Limit to 50 results
            
            return f"Found {len(result)} files:\n" + "\n".join(result)
        except Exception as e:
            return f"Error finding files: {str(e)}"


class FileStatSchema(BaseModel):
    file_path: str = Field(description="Path to the file to get stats for")


class FileStatTool(BaseTool):
    name: str = "file_stat"
    description: str = "Get detailed information about a file (size, permissions, etc.)"
    args_schema: Type[BaseModel] = FileStatSchema

    def _run(self, file_path: str) -> str:
        try:
            path = Path(file_path)
            if not path.exists():
                return f"Error: File {file_path} does not exist"
            
            stat = path.stat()
            
            info = [
                f"File: {file_path}",
                f"Size: {stat.st_size} bytes",
                f"Created: {datetime.fromtimestamp(stat.st_ctime)}",
                f"Modified: {datetime.fromtimestamp(stat.st_mtime)}",
                f"Permissions: {oct(stat.st_mode)[-3:]}"
            ]
            
            return "\n".join(info)
        except Exception as e:
            return f"Error getting file stats: {str(e)}"


# ============================================================================
# Enhanced Code Analysis Tools
# ============================================================================

class DependencyAnalysisSchema(BaseModel):
    path: str = Field(description="Path to project directory to analyze")


class DependencyAnalysisTool(BaseTool):
    name: str = "dependency_analysis"
    description: str = "Analyze project dependencies and create dependency graph"
    args_schema: Type[BaseModel] = DependencyAnalysisSchema

    def _run(self, path: str) -> str:
        try:
            project_path = Path(path)
            
            # Look for common dependency files
            deps_info = []
            
            # Python requirements
            req_files = list(project_path.glob("*requirements*.txt")) + list(project_path.glob("Pipfile*")) + list(project_path.glob("pyproject.toml"))
            if req_files:
                deps_info.append("Python dependencies found:")
                for req_file in req_files:
                    deps_info.append(f"- {req_file.name}")
            
            # Package managers for other languages
            js_files = list(project_path.glob("package.json"))
            if js_files:
                deps_info.append("JavaScript dependencies found:")
                for js_file in js_files:
                    deps_info.append(f"- {js_file.name}")
            
            # Look for import statements in Python files
            python_files = list(project_path.glob("**/*.py"))
            imports = set()
            for py_file in python_files[:10]:  # Limit to first 10 files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Simple import detection
                        import_lines = [line.strip() for line in content.split('\n') if line.strip().startswith('import ') or line.strip().startswith('from ')]
                        for line in import_lines:
                            parts = line.split()
                            if len(parts) > 1:
                                module = parts[1]
                                if '.' in module:
                                    module = module.split('.')[0]
                                if module not in ['os', 'sys', 'json', 'datetime', 'typing', 'pathlib']:  # Skip stdlib
                                    imports.add(module)
                except:
                    continue
            
            if imports:
                deps_info.append(f"Potential third-party imports found: {', '.join(list(imports)[:10])}")
            
            if not deps_info:
                return f"No dependency information found in {path}"
            
            return "\n".join(deps_info)
        except Exception as e:
            return f"Error analyzing dependencies: {str(e)}"


# ============================================================================
# System and Environment Tools
# ============================================================================

class SystemInfoSchema(BaseModel):
    info_type: str = Field(description="Type of system info to retrieve: 'os', 'cpu', 'memory', 'disk', 'network'", default="os")


class SystemInfoTool(BaseTool):
    name: str = "system_info"
    description: str = "Get system information (OS, CPU, memory, disk, network)"
    args_schema: Type[BaseModel] = SystemInfoSchema

    def _run(self, info_type: str = "os") -> str:
        try:
            import platform
            import psutil
            
            if info_type == "os":
                info = [
                    f"System: {platform.system()}",
                    f"Release: {platform.release()}",
                    f"Version: {platform.version()}",
                    f"Machine: {platform.machine()}",
                    f"Processor: {platform.processor()}"
                ]
            elif info_type == "cpu":
                info = [
                    f"Physical cores: {psutil.cpu_count(logical=False)}",
                    f"Total cores: {psutil.cpu_count(logical=True)}",
                    f"Max Frequency: {psutil.cpu_freq().max:.2f}Mhz",
                    f"Current Frequency: {psutil.cpu_freq().current:.2f}Mhz",
                    f"CPU Usage: {psutil.cpu_percent(interval=1)}%"
                ]
            elif info_type == "memory":
                svmem = psutil.virtual_memory()
                info = [
                    f"Total: {svmem.total / (1024**3):.2f} GB",
                    f"Available: {svmem.available / (1024**3):.2f} GB",
                    f"Used: {svmem.used / (1024**3):.2f} GB",
                    f"Percentage: {svmem.percent}%"
                ]
            elif info_type == "disk":
                partition = psutil.disk_usage('/')
                info = [
                    f"Total: {partition.total / (1024**3):.2f} GB",
                    f"Used: {partition.used / (1024**3):.2f} GB",
                    f"Free: {partition.free / (1024**3):.2f} GB",
                    f"Percentage: {partition.percent}%"
                ]
            elif info_type == "network":
                addrs = psutil.net_if_addrs()
                info = ["Network interfaces:"]
                for interface, addresses in addrs.items():
                    info.append(f"- {interface}")
                    for addr in addresses:
                        if addr.family.name == 'AF_INET':
                            info.append(f"  IPv4: {addr.address}")
                        elif addr.family.name == 'AF_PACKET':
                            info.append(f"  MAC: {addr.address}")
            else:
                return f"Unknown info type: {info_type}. Use: os, cpu, memory, disk, or network"
            
            return "\n".join(info)
        except Exception as e:
            return f"Error getting system info: {str(e)}"


class ProcessInfoSchema(BaseModel):
    filter_term: Optional[str] = Field(description="Filter processes by name or keyword", default=None)


class ProcessInfoTool(BaseTool):
    name: str = "process_info"
    description: str = "Get information about running processes"
    args_schema: Type[BaseModel] = ProcessInfoSchema

    def _run(self, filter_term: Optional[str] = None) -> str:
        try:
            import psutil
            
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'username', 'memory_percent', 'cpu_percent']):
                try:
                    pinfo = proc.info
                    if filter_term is None or filter_term.lower() in pinfo['name'].lower():
                        processes.append(f"PID: {pinfo['pid']}, Name: {pinfo['name']}, CPU: {pinfo['cpu_percent']}%, Mem: {pinfo['memory_percent']:.1f}%")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            if not processes:
                return f"No processes found matching filter: {filter_term}" if filter_term else "No processes found"
            
            return f"Found {len(processes)} processes:\n" + "\n".join(processes[:20])  # Limit to 20
        except Exception as e:
            return f"Error getting process info: {str(e)}"


# ============================================================================
# Enhanced Web Search with Filtering
# ============================================================================

class WebSearchDetailedSchema(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(description="Maximum number of results to return", default=5)
    region: str = Field(description="Region for search results", default="wt-wt")
    safesearch: str = Field(description="Safe search level: 'moderate', 'strict', 'off'", default="moderate")


class WebSearchDetailedTool(BaseTool):
    name: str = "web_search_detailed"
    description: str = "Search the web using DuckDuckGo with advanced options"
    args_schema: Type[BaseModel] = WebSearchDetailedSchema

    def _run(self, query: str, max_results: int = 5, region: str = "wt-wt", safesearch: str = "moderate") -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query, 
                    max_results=max_results,
                    region=region,
                    safesearch=safesearch
                ))

            if not results:
                return "No results found"

            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. {result.get('title', 'No title')}\n"
                    f"   URL: {result.get('href', 'No URL')}\n"
                    f"   {result.get('body', 'No description')}\n"
                )

            return "\n".join(formatted_results)
        except Exception as e:
            return f"Error performing web search: {str(e)}"


# ============================================================================
# Enhanced Tool Registry
# ============================================================================

class EnhancedToolRegistry:
    """Enhanced registry for managing available tools with categorization and discovery."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register all default tools."""
        # Import base tools
        from .tools import ReadFileTool, WriteFileTool, ExecuteCommandTool
        from .advanced_tools import (
            GitStatusTool, GitDiffTool, GitLogTool, GitCommitTool,
            WebSearchTool, CodeAnalysisTool
        )

        default_tools = [
            # Base tools
            ReadFileTool(),
            WriteFileTool(),
            ExecuteCommandTool(),
            # Enhanced file tools
            FindFileTool(),
            FileStatTool(),
            # Git tools
            GitStatusTool(),
            GitDiffTool(),
            GitLogTool(),
            GitCommitTool(),
            GitBranchTool(),
            GitPushTool(),
            GitPullTool(),
            # Search tools
            WebSearchTool(),
            WebSearchDetailedTool(),
            # Analysis tools
            CodeAnalysisTool(),
            DependencyAnalysisTool(),
            # System tools
            SystemInfoTool(),
            ProcessInfoTool(),
        ]

        for tool in default_tools:
            self.register_tool(tool)

    def register_tool(self, tool: BaseTool, category: str = "general"):
        """Register a new tool with category."""
        self._tools[tool.name] = tool
        
        if category not in self._categories:
            self._categories[category] = []
        if tool.name not in self._categories[category]:
            self._categories[category].append(tool.name)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get tools by category."""
        if category in self._categories:
            return [self._tools[name] for name in self._categories[category] if name in self._tools]
        else:
            # Fallback to keyword matching
            category_lower = category.lower()
            return [tool for tool in self._tools.values() 
                   if category_lower in tool.name.lower() or category_lower in tool.description.lower()]

    def get_categories(self) -> List[str]:
        """Get all available categories."""
        return list(self._categories.keys())

    def search_tools(self, keyword: str) -> List[BaseTool]:
        """Search for tools by keyword in name or description."""
        keyword_lower = keyword.lower()
        return [
            tool for tool in self._tools.values()
            if keyword_lower in tool.name.lower() or keyword_lower in tool.description.lower()
        ]

    def get_tool_suggestions(self, task_description: str) -> List[str]:
        """Suggest relevant tools based on task description."""
        task_lower = task_description.lower()
        suggestions = []
        
        # Map common terms to relevant tools
        term_to_tool = {
            'git': ['git_status', 'git_diff', 'git_log', 'git_commit', 'git_branch', 'git_push', 'git_pull'],
            'file': ['read_file', 'write_file', 'find_file', 'file_stat'],
            'code': ['code_analysis', 'dependency_analysis'],
            'search': ['web_search', 'web_search_detailed'],
            'system': ['system_info', 'process_info'],
            'branch': ['git_branch'],
            'push': ['git_push'],
            'pull': ['git_pull'],
            'dependency': ['dependency_analysis'],
            'requirement': ['dependency_analysis']
        }
        
        for term, tools in term_to_tool.items():
            if term in task_lower:
                suggestions.extend(tools)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)
        
        return unique_suggestions