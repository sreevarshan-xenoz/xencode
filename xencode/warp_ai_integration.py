#!/usr/bin/env python3
"""
Warp Terminal AI Integration

Advanced AI integration for context-aware command suggestions,
smart completion, and project-specific intelligence.
"""

import asyncio
import json
import os
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import time
import logging

from rich.console import Console

# Import Xencode's existing AI systems
try:
    from .intelligent_model_selector import IntelligentModelSelector, HardwareDetector
    from .advanced_cache_system import get_cache_manager
    from .smart_config_manager import ConfigurationManager
    from .enhancement_integration import get_enhancement_integration
    XENCODE_AI_AVAILABLE = True
except ImportError:
    XENCODE_AI_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ProjectContext:
    """Context information about the current project"""
    project_type: str  # "python", "javascript", "rust", "go", "docker", "git", etc.
    git_status: Dict[str, Any]
    package_files: List[str]  # package.json, requirements.txt, Cargo.toml, etc.
    recent_files: List[str]
    working_directory: Path
    environment_info: Dict[str, Any]


@dataclass
class CommandSuggestionContext:
    """Context for generating command suggestions"""
    recent_commands: List[str]
    current_directory: Path
    project_context: ProjectContext
    user_preferences: Dict[str, Any]
    time_of_day: str
    error_history: List[str]


class ProjectAnalyzer:
    """Analyzes the current project to understand context"""
    
    def __init__(self):
        self.console = Console()
        
    async def analyze_project(self, directory: Path) -> ProjectContext:
        """Analyze the current project directory"""
        project_type = await self._detect_project_type(directory)
        git_status = await self._get_git_status(directory)
        package_files = await self._find_package_files(directory)
        recent_files = await self._get_recent_files(directory)
        environment_info = await self._get_environment_info()
        
        return ProjectContext(
            project_type=project_type,
            git_status=git_status,
            package_files=package_files,
            recent_files=recent_files,
            working_directory=directory,
            environment_info=environment_info
        )
    
    async def _detect_project_type(self, directory: Path) -> str:
        """Detect the type of project based on files present"""
        project_indicators = {
            "python": ["requirements.txt", "setup.py", "pyproject.toml", "Pipfile", "poetry.lock"],
            "javascript": ["package.json", "yarn.lock", "package-lock.json"],
            "typescript": ["tsconfig.json", "package.json"],
            "rust": ["Cargo.toml", "Cargo.lock"],
            "go": ["go.mod", "go.sum"],
            "java": ["pom.xml", "build.gradle", "gradle.properties"],
            "docker": ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"],
            "kubernetes": ["deployment.yaml", "service.yaml", "ingress.yaml"],
            "terraform": ["main.tf", "variables.tf", "terraform.tfvars"],
            "react": ["package.json", "src/App.js", "src/App.tsx"],
            "vue": ["package.json", "vue.config.js", "src/App.vue"],
            "angular": ["package.json", "angular.json", "src/app"],
        }
        
        detected_types = []
        
        for project_type, indicators in project_indicators.items():
            for indicator in indicators:
                if (directory / indicator).exists():
                    detected_types.append(project_type)
                    break
        
        # Return the most specific type
        if "react" in detected_types:
            return "react"
        elif "vue" in detected_types:
            return "vue"
        elif "angular" in detected_types:
            return "angular"
        elif "typescript" in detected_types:
            return "typescript"
        elif "javascript" in detected_types:
            return "javascript"
        elif "python" in detected_types:
            return "python"
        elif detected_types:
            return detected_types[0]
        else:
            return "general"
    
    async def _get_git_status(self, directory: Path) -> Dict[str, Any]:
        """Get git status information"""
        try:
            # Check if it's a git repository
            git_dir = directory / ".git"
            if not git_dir.exists():
                return {"is_git_repo": False}
            
            # Get git status
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return {"is_git_repo": True, "error": result.stderr}
            
            # Parse git status
            lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
            modified = []
            untracked = []
            staged = []
            
            for line in lines:
                if line.startswith('M '):
                    modified.append(line[3:])
                elif line.startswith('??'):
                    untracked.append(line[3:])
                elif line.startswith('A '):
                    staged.append(line[3:])
            
            # Get current branch
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"
            
            return {
                "is_git_repo": True,
                "current_branch": current_branch,
                "modified": modified,
                "untracked": untracked,
                "staged": staged,
                "has_changes": bool(modified or untracked or staged)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get git status: {e}")
            return {"is_git_repo": False, "error": str(e)}
    
    async def _find_package_files(self, directory: Path) -> List[str]:
        """Find package/dependency files in the project"""
        package_files = []
        
        common_package_files = [
            "package.json", "requirements.txt", "Pipfile", "poetry.lock",
            "Cargo.toml", "go.mod", "pom.xml", "build.gradle",
            "composer.json", "Gemfile", "setup.py", "pyproject.toml"
        ]
        
        for file_name in common_package_files:
            file_path = directory / file_name
            if file_path.exists():
                package_files.append(file_name)
        
        return package_files
    
    async def _get_recent_files(self, directory: Path, limit: int = 10) -> List[str]:
        """Get recently modified files"""
        try:
            # Use find to get recently modified files
            result = subprocess.run(
                ["find", str(directory), "-type", "f", "-not", "-path", "*/.*", 
                 "-printf", "%T@ %p\\n"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return []
            
            # Parse and sort by modification time
            files = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        timestamp, filepath = parts
                        relative_path = Path(filepath).relative_to(directory)
                        files.append((float(timestamp), str(relative_path)))
            
            # Sort by timestamp (newest first) and return file paths
            files.sort(reverse=True)
            return [filepath for _, filepath in files[:limit]]
            
        except Exception as e:
            logger.warning(f"Failed to get recent files: {e}")
            return []
    
    async def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information"""
        env_info = {}
        
        # Check for common tools
        tools_to_check = [
            "python", "node", "npm", "yarn", "docker", "git", 
            "cargo", "go", "java", "mvn", "gradle"
        ]
        
        for tool in tools_to_check:
            try:
                result = subprocess.run(
                    [tool, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    env_info[tool] = result.stdout.strip().split('\n')[0]
            except Exception:
                pass
        
        # Get shell and terminal info
        env_info.update({
            "shell": os.environ.get("SHELL", "unknown"),
            "term": os.environ.get("TERM", "unknown"),
            "pwd": os.getcwd(),
        })
        
        return env_info


class AdvancedAISuggester:
    """Advanced AI suggester with context awareness"""
    
    def __init__(self):
        self.console = Console()
        self.project_analyzer = ProjectAnalyzer()
        self.cache_manager = None
        self.model_selector = None
        
        # Initialize Xencode AI systems if available
        if XENCODE_AI_AVAILABLE:
            try:
                self.cache_manager = get_cache_manager()
                self.model_selector = IntelligentModelSelector()
            except Exception as e:
                logger.warning(f"Failed to initialize Xencode AI systems: {e}")
        
        # Command templates for different project types
        self.command_templates = {
            "python": [
                "python -m pip install {package}",
                "python -m pytest",
                "python -m black .",
                "python -m flake8",
                "python -m mypy .",
                "python setup.py install",
                "python -m venv venv",
                "source venv/bin/activate",
                "pip freeze > requirements.txt"
            ],
            "javascript": [
                "npm install {package}",
                "npm run build",
                "npm run test",
                "npm run dev",
                "npm run start",
                "yarn install",
                "yarn build",
                "yarn test",
                "npx {command}"
            ],
            "docker": [
                "docker build -t {image} .",
                "docker run -it {image}",
                "docker ps",
                "docker images",
                "docker logs {container}",
                "docker exec -it {container} bash",
                "docker-compose up",
                "docker-compose down",
                "docker system prune"
            ],
            "git": [
                "git add .",
                "git commit -m '{message}'",
                "git push",
                "git pull",
                "git status",
                "git log --oneline",
                "git branch",
                "git checkout {branch}",
                "git merge {branch}"
            ]
        }
    
    async def get_context_aware_suggestions(
        self, 
        recent_commands: List[str], 
        current_directory: Optional[Path] = None
    ) -> List[str]:
        """Get AI suggestions based on project context"""
        
        if current_directory is None:
            current_directory = Path.cwd()
        
        # Analyze project context
        project_context = await self.project_analyzer.analyze_project(current_directory)
        
        # Create suggestion context
        import logging
        logger = logging.getLogger(__name__)
        
        user_prefs = {}
        try:
            pass
        except Exception:
            logger.warning("User preferences loading not implemented")
        
        error_hist = []
        try:
            pass
        except Exception:
            logger.warning("Error history tracking not implemented")
        
        suggestion_context = CommandSuggestionContext(
            recent_commands=recent_commands,
            current_directory=current_directory,
            project_context=project_context,
            user_preferences=user_prefs,
            time_of_day=time.strftime("%H:%M"),
            error_history=error_hist
        )
        
        # Generate suggestions
        suggestions = []
        
        # 1. Project-specific suggestions
        project_suggestions = await self._get_project_specific_suggestions(suggestion_context)
        suggestions.extend(project_suggestions)
        
        # 2. Git-based suggestions
        git_suggestions = await self._get_git_based_suggestions(suggestion_context)
        suggestions.extend(git_suggestions)
        
        # 3. Context-aware suggestions
        context_suggestions = await self._get_context_suggestions(suggestion_context)
        suggestions.extend(context_suggestions)
        
        # 4. AI model suggestions (if available)
        if XENCODE_AI_AVAILABLE and self.model_selector:
            ai_suggestions = await self._get_ai_model_suggestions(suggestion_context)
            suggestions.extend(ai_suggestions)
        
        # Remove duplicates and limit results
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:8]  # Return top 8 suggestions
    
    async def _get_project_specific_suggestions(self, context: CommandSuggestionContext) -> List[str]:
        """Get suggestions based on project type"""
        project_type = context.project_context.project_type
        suggestions = []
        
        # Get templates for this project type
        templates = self.command_templates.get(project_type, [])
        
        # Add project-specific commands
        if project_type == "python":
            if "requirements.txt" in context.project_context.package_files:
                suggestions.append("pip install -r requirements.txt")
            if "setup.py" in context.project_context.package_files:
                suggestions.append("python setup.py develop")
            if "pyproject.toml" in context.project_context.package_files:
                suggestions.append("pip install -e .")
        
        elif project_type == "javascript":
            if "package.json" in context.project_context.package_files:
                suggestions.extend(["npm install", "npm run build", "npm test"])
            if "yarn.lock" in context.project_context.package_files:
                suggestions.extend(["yarn install", "yarn build"])
        
        elif project_type == "docker":
            if "Dockerfile" in context.project_context.package_files:
                suggestions.append("docker build -t app .")
            if "docker-compose.yml" in context.project_context.package_files:
                suggestions.extend(["docker-compose up", "docker-compose down"])
        
        # Add some general templates
        suggestions.extend(templates[:3])
        
        return suggestions
    
    async def _get_git_based_suggestions(self, context: CommandSuggestionContext) -> List[str]:
        """Get suggestions based on git status"""
        git_status = context.project_context.git_status
        suggestions = []
        
        if not git_status.get("is_git_repo"):
            return suggestions
        
        # Suggest based on git state
        if git_status.get("has_changes"):
            if git_status.get("modified"):
                suggestions.append("git add .")
                suggestions.append("git diff")
            if git_status.get("staged"):
                suggestions.append("git commit -m 'Update files'")
            if git_status.get("untracked"):
                suggestions.append("git add .")
        else:
            # Clean working directory
            suggestions.extend(["git pull", "git log --oneline -5", "git branch"])
        
        return suggestions
    
    async def _get_context_suggestions(self, context: CommandSuggestionContext) -> List[str]:
        """Get suggestions based on recent commands and context"""
        suggestions = []
        recent_commands = context.recent_commands
        
        # Analyze recent command patterns
        if any("git" in cmd for cmd in recent_commands):
            suggestions.extend(["git status", "git log --oneline -5"])
        
        if any("docker" in cmd for cmd in recent_commands):
            suggestions.extend(["docker ps", "docker images"])
        
        if any("npm" in cmd for cmd in recent_commands):
            suggestions.extend(["npm run build", "npm test"])
        
        if any("python" in cmd for cmd in recent_commands):
            suggestions.extend(["python -m pytest", "python -m pip list"])
        
        # Time-based suggestions
        current_hour = int(time.strftime("%H"))
        if 9 <= current_hour <= 17:  # Work hours
            suggestions.extend(["git status", "npm run dev", "python -m pytest"])
        
        return suggestions
    
    async def _get_ai_model_suggestions(self, context: CommandSuggestionContext) -> List[str]:
        """Get suggestions from Xencode's AI models"""
        if not self.model_selector:
            return []
        
        try:
            # Create a prompt for the AI model
            prompt = self._create_ai_prompt(context)
            
            # Get the optimal model for command suggestion
            # Note: This is a simplified integration - in practice, you'd use the actual model
            # For now, we'll return some intelligent suggestions based on context
            
            suggestions = []
            
            # Simulate AI model response based on context
            if context.project_context.project_type == "python":
                suggestions.extend([
                    "python -m black . --check",
                    "python -m isort . --check-only",
                    "python -m bandit -r ."
                ])
            
            elif context.project_context.project_type == "javascript":
                suggestions.extend([
                    "npm audit",
                    "npm run lint",
                    "npx prettier --check ."
                ])
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"AI model suggestions failed: {e}")
            return []
    
    def _create_ai_prompt(self, context: CommandSuggestionContext) -> str:
        """Create a prompt for the AI model"""
        prompt = f"""
Based on the following context, suggest the most relevant terminal commands:

Project Type: {context.project_context.project_type}
Current Directory: {context.current_directory}
Recent Commands: {', '.join(context.recent_commands[-5:])}
Git Status: {context.project_context.git_status}
Package Files: {', '.join(context.project_context.package_files)}

Please suggest 3-5 relevant terminal commands that would be useful in this context.
Focus on commands that are commonly used in {context.project_context.project_type} projects.
"""
        return prompt


class WarpAIIntegration:
    """Main integration class for Warp Terminal AI features"""
    
    def __init__(self):
        self.ai_suggester = AdvancedAISuggester()
        self.console = Console()
        
        # Integration with enhancement systems
        if XENCODE_AI_AVAILABLE:
            try:
                self.enhancement_integration = get_enhancement_integration()
            except Exception as e:
                logger.warning(f"Enhancement integration failed: {e}")
                self.enhancement_integration = None
    
    async def get_smart_suggestions(
        self, 
        recent_commands: List[str], 
        current_directory: Optional[Path] = None
    ) -> List[str]:
        """Get smart AI suggestions for the terminal"""
        
        try:
            # Get context-aware suggestions
            suggestions = await self.ai_suggester.get_context_aware_suggestions(
                recent_commands, current_directory
            )
            
            # Track suggestion usage if enhancement integration is available
            if self.enhancement_integration:
                await self.enhancement_integration.track_feature_usage(
                    "ai_suggestions",
                    {"suggestion_count": len(suggestions), "context": "warp_terminal"}
                )
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Smart suggestions failed: {e}")
            # Fallback to simple suggestions
            return self._get_fallback_suggestions(recent_commands)
    
    def _get_fallback_suggestions(self, recent_commands: List[str]) -> List[str]:
        """Fallback suggestions when AI fails"""
        fallback = []
        
        # Simple pattern-based suggestions
        if any("git" in cmd for cmd in recent_commands):
            fallback.extend(["git status", "git add .", "git commit"])
        
        if any("docker" in cmd for cmd in recent_commands):
            fallback.extend(["docker ps", "docker images"])
        
        if any("npm" in cmd for cmd in recent_commands):
            fallback.extend(["npm install", "npm run build"])
        
        # Default suggestions
        fallback.extend(["ls -la", "pwd", "git status"])
        
        return fallback[:5]


# Global AI integration instance
_warp_ai_integration: Optional[WarpAIIntegration] = None


def get_warp_ai_integration() -> WarpAIIntegration:
    """Get the global Warp AI integration instance"""
    global _warp_ai_integration
    if _warp_ai_integration is None:
        _warp_ai_integration = WarpAIIntegration()
    return _warp_ai_integration


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_ai_integration():
        """Test the AI integration"""
        console = Console()
        console.print("[bold blue]Testing Warp AI Integration[/bold blue]")
        
        ai_integration = get_warp_ai_integration()
        
        # Test with sample commands
        recent_commands = ["git status", "ls -la", "npm install"]
        
        console.print("\n[yellow]Getting smart suggestions...[/yellow]")
        suggestions = await ai_integration.get_smart_suggestions(recent_commands)
        
        console.print(f"\n[green]AI Suggestions:[/green]")
        for i, suggestion in enumerate(suggestions, 1):
            console.print(f"  {i}. {suggestion}")
    
    asyncio.run(test_ai_integration())