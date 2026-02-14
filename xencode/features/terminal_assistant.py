"""
Terminal Assistant Feature

Provides context-aware command suggestions, intelligent error handling,
and learning capabilities for terminal users.
"""

import os
import re
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter

from .base import FeatureBase, FeatureConfig, FeatureError


@dataclass
class TerminalAssistantConfig:
    """Configuration for Terminal Assistant"""
    history_size: int = 1000
    context_aware: bool = True
    learning_enabled: bool = True
    suggestion_limit: int = 5
    error_fix_enabled: bool = True
    shell_type: str = "bash"  # bash, zsh, fish, powershell
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TerminalAssistantConfig':
        """Create config from dictionary"""
        return cls(
            history_size=data.get('history_size', 1000),
            context_aware=data.get('context_aware', True),
            learning_enabled=data.get('learning_enabled', True),
            suggestion_limit=data.get('suggestion_limit', 5),
            error_fix_enabled=data.get('error_fix_enabled', True),
            shell_type=data.get('shell_type', 'bash')
        )


class TerminalAssistantFeature(FeatureBase):
    """Terminal Assistant feature implementation"""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.ta_config = TerminalAssistantConfig.from_dict(config.config)
        self.command_predictor = None
        self.context_analyzer = None
        self.learning_engine = None
        self.error_handler = None
    
    @property
    def name(self) -> str:
        """Feature name"""
        return "terminal_assistant"
    
    @property
    def description(self) -> str:
        """Feature description"""
        return "Context-aware command suggestions and intelligent error handling"
    
    async def _initialize(self) -> None:
        """Initialize Terminal Assistant components"""
        # Initialize command predictor
        self.command_predictor = CommandPredictor(
            history_size=self.ta_config.history_size,
            suggestion_limit=self.ta_config.suggestion_limit
        )
        
        # Initialize context analyzer
        self.context_analyzer = ContextAnalyzer(
            enabled=self.ta_config.context_aware
        )
        
        # Initialize learning engine
        self.learning_engine = LearningEngine(
            enabled=self.ta_config.learning_enabled
        )
        
        # Initialize error handler
        self.error_handler = ErrorHandler(
            enabled=self.ta_config.error_fix_enabled
        )
        
        # Load command history
        await self.command_predictor.load_history()
        
        # Load user preferences
        await self.learning_engine.load_preferences()
    
    async def _shutdown(self) -> None:
        """Shutdown Terminal Assistant"""
        # Save command history
        if self.command_predictor:
            await self.command_predictor.save_history()
        
        # Save user preferences
        if self.learning_engine:
            await self.learning_engine.save_preferences()
    
    async def suggest_commands(self, context: str = None, partial: str = None) -> List[Dict[str, Any]]:
        """
        Suggest commands based on context and partial input
        
        Args:
            context: Current context (directory, project type, etc.)
            partial: Partial command input
            
        Returns:
            List of command suggestions with explanations
        """
        suggestions = []
        
        # Get context information
        context_info = await self.context_analyzer.analyze(context)
        
        # Get predictions from command predictor
        predictions = await self.command_predictor.predict(
            partial=partial,
            context=context_info
        )
        
        # Enhance predictions with learning engine
        enhanced = await self.learning_engine.enhance_suggestions(
            predictions,
            context_info
        )
        
        return enhanced
    
    async def explain_command(self, command: str) -> Dict[str, Any]:
        """
        Explain what a command does
        
        Args:
            command: Command to explain
            
        Returns:
            Explanation with details
        """
        explanation = {
            'command': command,
            'description': '',
            'arguments': [],
            'examples': [],
            'warnings': []
        }
        
        # Parse command
        parsed = self._parse_command(command)
        
        # Get explanation from knowledge base
        explanation['description'] = self._get_command_description(parsed['base'])
        explanation['arguments'] = self._explain_arguments(parsed)
        explanation['examples'] = self._get_command_examples(parsed['base'])
        explanation['warnings'] = self._get_command_warnings(parsed)
        
        return explanation
    
    async def fix_error(self, command: str, error: str) -> List[Dict[str, Any]]:
        """
        Suggest fixes for command errors
        
        Args:
            command: Command that failed
            error: Error message
            
        Returns:
            List of fix suggestions
        """
        return await self.error_handler.suggest_fixes(command, error)
    
    async def search_history(self, pattern: str) -> List[Dict[str, Any]]:
        """
        Search command history
        
        Args:
            pattern: Search pattern
            
        Returns:
            Matching commands from history
        """
        return await self.command_predictor.search_history(pattern)
    
    async def record_command(self, command: str, success: bool = True, 
                           context: Dict[str, Any] = None) -> None:
        """
        Record a command execution
        
        Args:
            command: Executed command
            success: Whether command succeeded
            context: Execution context
        """
        # Record in history
        await self.command_predictor.record(command, success)
        
        # Update learning engine
        await self.learning_engine.learn(command, success, context)
    
    def _parse_command(self, command: str) -> Dict[str, Any]:
        """Parse command into components"""
        parts = command.strip().split()
        if not parts:
            return {'base': '', 'args': [], 'flags': []}
        
        base = parts[0]
        args = []
        flags = []
        
        for part in parts[1:]:
            if part.startswith('-'):
                flags.append(part)
            else:
                args.append(part)
        
        return {
            'base': base,
            'args': args,
            'flags': flags,
            'full': command
        }
    
    def _get_command_description(self, command: str) -> str:
        """Get description for a command"""
        # Basic command descriptions
        descriptions = {
            'ls': 'List directory contents',
            'cd': 'Change directory',
            'pwd': 'Print working directory',
            'mkdir': 'Create directory',
            'rm': 'Remove files or directories',
            'cp': 'Copy files or directories',
            'mv': 'Move or rename files',
            'cat': 'Display file contents',
            'grep': 'Search text patterns',
            'find': 'Search for files',
            'git': 'Version control system',
            'python': 'Python interpreter',
            'pip': 'Python package installer',
            'npm': 'Node package manager',
            'docker': 'Container platform',
            'kubectl': 'Kubernetes CLI',
        }
        return descriptions.get(command, f'Execute {command} command')
    
    def _explain_arguments(self, parsed: Dict[str, Any]) -> List[Dict[str, str]]:
        """Explain command arguments"""
        explanations = []
        for arg in parsed['args']:
            explanations.append({
                'value': arg,
                'description': f'Argument: {arg}'
            })
        for flag in parsed['flags']:
            explanations.append({
                'value': flag,
                'description': self._explain_flag(parsed['base'], flag)
            })
        return explanations
    
    def _explain_flag(self, command: str, flag: str) -> str:
        """Explain a command flag"""
        # Common flag explanations
        flag_map = {
            '-l': 'Long format listing',
            '-a': 'Show all files including hidden',
            '-r': 'Recursive operation',
            '-f': 'Force operation',
            '-v': 'Verbose output',
            '-h': 'Human-readable format',
        }
        return flag_map.get(flag, f'Flag: {flag}')
    
    def _get_command_examples(self, command: str) -> List[str]:
        """Get example usage for command"""
        examples = {
            'ls': ['ls -la', 'ls -lh /path', 'ls *.py'],
            'git': ['git status', 'git add .', 'git commit -m "message"'],
            'docker': ['docker ps', 'docker build -t name .', 'docker run image'],
        }
        return examples.get(command, [])
    
    def _get_command_warnings(self, parsed: Dict[str, Any]) -> List[str]:
        """Get warnings for potentially dangerous commands"""
        warnings = []
        command = parsed['base']
        
        if command == 'rm' and '-rf' in ' '.join(parsed['flags']):
            warnings.append('⚠️  This will permanently delete files without confirmation')
        
        if command in ['sudo', 'su']:
            warnings.append('⚠️  This command requires elevated privileges')
        
        if command == 'chmod' and '777' in parsed['args']:
            warnings.append('⚠️  Setting 777 permissions is a security risk')
        
        return warnings
    
    def get_cli_commands(self) -> List[Any]:
        """Get CLI commands for Terminal Assistant"""
        # Will be implemented with actual CLI framework
        return []
    
    def get_tui_components(self) -> List[Any]:
        """Get TUI components for Terminal Assistant"""
        # Will be implemented with actual TUI framework
        return []
    
    def get_api_endpoints(self) -> List[Any]:
        """Get API endpoints for Terminal Assistant"""
        return [
            {
                'path': '/api/terminal/suggest',
                'method': 'POST',
                'handler': self.suggest_commands
            },
            {
                'path': '/api/terminal/explain',
                'method': 'POST',
                'handler': self.explain_command
            },
            {
                'path': '/api/terminal/fix',
                'method': 'POST',
                'handler': self.fix_error
            },
            {
                'path': '/api/terminal/history',
                'method': 'GET',
                'handler': self.search_history
            }
        ]


class CommandPredictor:
    """Predicts commands based on history and context"""
    
    def __init__(self, history_size: int = 1000, suggestion_limit: int = 5):
        self.history_size = history_size
        self.suggestion_limit = suggestion_limit
        self.history: List[Dict[str, Any]] = []
        self.command_frequency = Counter()
        self.command_sequences = defaultdict(Counter)
        
        # Advanced analysis features
        self.command_patterns: Dict[str, List[str]] = defaultdict(list)
        self.temporal_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.context_patterns: Dict[str, Counter] = defaultdict(Counter)
        self.success_rates: Dict[str, Dict[str, int]] = defaultdict(lambda: {'success': 0, 'failure': 0})
    
    async def load_history(self) -> None:
        """Load command history from file"""
        history_file = Path.home() / '.xencode' / 'terminal_history.json'
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.history = data.get('history', [])
                    self.command_frequency = Counter(data.get('frequency', {}))
                    self.command_sequences = defaultdict(Counter, {
                        k: Counter(v) for k, v in data.get('sequences', {}).items()
                    })
                    self.command_patterns = defaultdict(list, data.get('patterns', {}))
                    self.temporal_patterns = defaultdict(
                        lambda: defaultdict(int),
                        {k: defaultdict(int, v) for k, v in data.get('temporal_patterns', {}).items()}
                    )
                    self.context_patterns = defaultdict(Counter, {
                        k: Counter(v) for k, v in data.get('context_patterns', {}).items()
                    })
                    self.success_rates = defaultdict(
                        lambda: {'success': 0, 'failure': 0},
                        data.get('success_rates', {})
                    )
            except Exception:
                pass
    
    async def save_history(self) -> None:
        """Save command history to file"""
        history_file = Path.home() / '.xencode' / 'terminal_history.json'
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(history_file, 'w') as f:
                json.dump({
                    'history': self.history[-self.history_size:],
                    'frequency': dict(self.command_frequency),
                    'sequences': {k: dict(v) for k, v in self.command_sequences.items()},
                    'patterns': dict(self.command_patterns),
                    'temporal_patterns': {k: dict(v) for k, v in self.temporal_patterns.items()},
                    'context_patterns': {k: dict(v) for k, v in self.context_patterns.items()},
                    'success_rates': dict(self.success_rates)
                }, f, indent=2)
        except Exception:
            pass
    
    async def predict(self, partial: str = None, 
                     context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Predict commands based on partial input and context"""
        suggestions = []
        
        if partial:
            # Find commands matching partial input
            for cmd_data in reversed(self.history):
                cmd = cmd_data.get('command', '')
                if cmd.startswith(partial) and cmd not in [s['command'] for s in suggestions]:
                    score = self._calculate_command_score(cmd, context)
                    suggestions.append({
                        'command': cmd,
                        'score': score,
                        'source': 'history'
                    })
        else:
            # Suggest based on context and frequency
            for cmd, freq in self.command_frequency.most_common(self.suggestion_limit * 2):
                score = self._calculate_command_score(cmd, context)
                suggestions.append({
                    'command': cmd,
                    'score': score,
                    'source': 'frequency'
                })
        
        # Add sequence-based suggestions
        sequence_suggestions = self._get_sequence_suggestions()
        suggestions.extend(sequence_suggestions)
        
        # Add temporal pattern suggestions
        temporal_suggestions = self._get_temporal_suggestions()
        suggestions.extend(temporal_suggestions)
        
        # Add context-based suggestions
        if context:
            context_suggestions = self._get_context_suggestions(context)
            suggestions.extend(context_suggestions)
        
        # Add pattern-based suggestions
        pattern_suggestions = self._get_pattern_suggestions(partial)
        suggestions.extend(pattern_suggestions)
        
        # Sort by score and limit
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        return suggestions[:self.suggestion_limit]
    
    def _calculate_command_score(self, command: str, context: Dict[str, Any] = None) -> float:
        """Calculate comprehensive score for a command"""
        score = 0.0
        
        # Base frequency score
        score += self.command_frequency.get(command, 0) * 1.0
        
        # Success rate bonus
        stats = self.success_rates.get(command, {'success': 0, 'failure': 0})
        total = stats['success'] + stats['failure']
        if total > 0:
            success_rate = stats['success'] / total
            score += success_rate * 5.0
        
        # Context relevance
        if context:
            project_type = context.get('project_type')
            if project_type and command in self.context_patterns.get(project_type, {}):
                score += self.context_patterns[project_type][command] * 2.0
        
        # Temporal relevance (recent usage)
        current_hour = datetime.now().hour
        hour_key = f"hour_{current_hour}"
        if command in self.temporal_patterns.get(hour_key, {}):
            score += self.temporal_patterns[hour_key][command] * 1.5
        
        return score
    
    def _get_context_suggestions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get suggestions based on context"""
        suggestions = []
        project_type = context.get('project_type')
        
        if project_type == 'python':
            suggestions.extend([
                {'command': 'python -m pytest', 'score': 10, 'source': 'context'},
                {'command': 'pip install -r requirements.txt', 'score': 9, 'source': 'context'},
            ])
        elif project_type == 'node':
            suggestions.extend([
                {'command': 'npm install', 'score': 10, 'source': 'context'},
                {'command': 'npm test', 'score': 9, 'source': 'context'},
            ])
        elif project_type == 'git':
            suggestions.extend([
                {'command': 'git status', 'score': 10, 'source': 'context'},
                {'command': 'git pull', 'score': 9, 'source': 'context'},
            ])
        
        return suggestions
    
    def _get_sequence_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions based on command sequences"""
        suggestions = []
        
        if len(self.history) >= 1:
            last_cmd = self.history[-1].get('command')
            if last_cmd in self.command_sequences:
                for next_cmd, count in self.command_sequences[last_cmd].most_common(3):
                    suggestions.append({
                        'command': next_cmd,
                        'score': count * 3.0,
                        'source': 'sequence',
                        'reason': f'Often follows "{last_cmd}"'
                    })
        
        return suggestions
    
    def _get_temporal_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions based on temporal patterns"""
        suggestions = []
        current_hour = datetime.now().hour
        hour_key = f"hour_{current_hour}"
        
        if hour_key in self.temporal_patterns:
            for cmd, count in Counter(self.temporal_patterns[hour_key]).most_common(3):
                suggestions.append({
                    'command': cmd,
                    'score': count * 2.0,
                    'source': 'temporal',
                    'reason': f'Commonly used at this time'
                })
        
        return suggestions
    
    def _get_pattern_suggestions(self, partial: str = None) -> List[Dict[str, Any]]:
        """Get suggestions based on detected patterns"""
        suggestions = []
        
        if not partial:
            return suggestions
        
        # Extract pattern (e.g., "git" from "git commit")
        base_cmd = partial.split()[0] if partial else ''
        
        if base_cmd in self.command_patterns:
            for pattern in self.command_patterns[base_cmd][:3]:
                if pattern.startswith(partial):
                    suggestions.append({
                        'command': pattern,
                        'score': 8.0,
                        'source': 'pattern',
                        'reason': f'Common {base_cmd} pattern'
                    })
        
        return suggestions
    
    def _detect_command_pattern(self, command: str) -> Optional[str]:
        """Detect and categorize command patterns"""
        parts = command.split()
        if len(parts) < 2:
            return None
        
        base = parts[0]
        
        # Git patterns
        if base == 'git':
            if len(parts) >= 2:
                return f"git_{parts[1]}"
        
        # Docker patterns
        elif base == 'docker':
            if len(parts) >= 2:
                return f"docker_{parts[1]}"
        
        # Python patterns
        elif base == 'python':
            if '-m' in parts:
                idx = parts.index('-m')
                if idx + 1 < len(parts):
                    return f"python_module_{parts[idx + 1]}"
        
        # NPM patterns
        elif base == 'npm':
            if len(parts) >= 2:
                return f"npm_{parts[1]}"
        
        return None
    
    def _analyze_temporal_pattern(self, command: str, timestamp: str) -> None:
        """Analyze and record temporal patterns"""
        try:
            dt = datetime.fromisoformat(timestamp)
            hour_key = f"hour_{dt.hour}"
            self.temporal_patterns[hour_key][command] += 1
            
            # Day of week pattern
            day_key = f"day_{dt.weekday()}"
            self.temporal_patterns[day_key][command] += 1
        except Exception:
            pass
    
    def _analyze_context_pattern(self, command: str, context: Dict[str, Any]) -> None:
        """Analyze and record context-based patterns"""
        if not context:
            return
        
        project_type = context.get('project_type')
        if project_type:
            self.context_patterns[project_type][command] += 1
        
        # Directory-based patterns
        directory = context.get('directory')
        if directory:
            dir_name = Path(directory).name
            self.context_patterns[f"dir_{dir_name}"][command] += 1
    
    async def record(self, command: str, success: bool = True, context: Dict[str, Any] = None) -> None:
        """Record a command execution"""
        timestamp = datetime.now().isoformat()
        
        self.history.append({
            'command': command,
            'timestamp': timestamp,
            'success': success,
            'context': context
        })
        
        # Update frequency
        if success:
            self.command_frequency[command] += 1
        
        # Update success rates
        if success:
            self.success_rates[command]['success'] += 1
        else:
            self.success_rates[command]['failure'] += 1
        
        # Update sequences
        if len(self.history) >= 2:
            prev_cmd = self.history[-2].get('command')
            self.command_sequences[prev_cmd][command] += 1
        
        # Detect and record patterns
        pattern = self._detect_command_pattern(command)
        if pattern:
            base = command.split()[0]
            if command not in self.command_patterns[base]:
                self.command_patterns[base].append(command)
        
        # Analyze temporal patterns
        self._analyze_temporal_pattern(command, timestamp)
        
        # Analyze context patterns
        if context:
            self._analyze_context_pattern(command, context)
        
        # Trim history
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]
    
    async def search_history(self, pattern: str) -> List[Dict[str, Any]]:
        """Search command history"""
        results = []
        regex = re.compile(pattern, re.IGNORECASE)
        
        for cmd_data in reversed(self.history):
            cmd = cmd_data.get('command', '')
            if regex.search(cmd):
                results.append(cmd_data)
        
        return results


class ContextAnalyzer:
    """Analyzes current context for command suggestions"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    async def analyze(self, context: str = None) -> Dict[str, Any]:
        """Analyze current context"""
        if not self.enabled:
            return {}
        
        cwd = Path.cwd()
        
        context_info = {
            'directory': str(cwd),
            'project_type': self._detect_project_type(cwd),
            'git_repo': self._is_git_repo(cwd),
            'files': self._get_relevant_files(cwd),
            'os': os.name
        }
        
        return context_info
    
    def _detect_project_type(self, path: Path) -> Optional[str]:
        """Detect project type from directory contents"""
        if (path / 'package.json').exists():
            return 'node'
        elif (path / 'requirements.txt').exists() or (path / 'setup.py').exists():
            return 'python'
        elif (path / 'Cargo.toml').exists():
            return 'rust'
        elif (path / 'go.mod').exists():
            return 'go'
        elif (path / 'pom.xml').exists():
            return 'java'
        elif (path / '.git').exists():
            return 'git'
        return None
    
    def _is_git_repo(self, path: Path) -> bool:
        """Check if directory is a git repository"""
        return (path / '.git').exists()
    
    def _get_relevant_files(self, path: Path) -> List[str]:
        """Get relevant files in directory"""
        try:
            files = [f.name for f in path.iterdir() if f.is_file()]
            return files[:20]  # Limit to 20 files
        except Exception:
            return []


class LearningEngine:
    """Learns user preferences and patterns"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.preferences: Dict[str, Any] = {}
        self.patterns: Dict[str, int] = Counter()
    
    async def load_preferences(self) -> None:
        """Load user preferences"""
        prefs_file = Path.home() / '.xencode' / 'terminal_preferences.json'
        if prefs_file.exists():
            try:
                with open(prefs_file, 'r') as f:
                    data = json.load(f)
                    self.preferences = data.get('preferences', {})
                    self.patterns = Counter(data.get('patterns', {}))
            except Exception:
                pass
    
    async def save_preferences(self) -> None:
        """Save user preferences"""
        prefs_file = Path.home() / '.xencode' / 'terminal_preferences.json'
        prefs_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(prefs_file, 'w') as f:
                json.dump({
                    'preferences': self.preferences,
                    'patterns': dict(self.patterns)
                }, f, indent=2)
        except Exception:
            pass
    
    async def enhance_suggestions(self, suggestions: List[Dict[str, Any]], 
                                 context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance suggestions based on learned preferences"""
        if not self.enabled:
            return suggestions
        
        # Boost scores based on user patterns
        for suggestion in suggestions:
            cmd = suggestion['command']
            if cmd in self.patterns:
                suggestion['score'] += self.patterns[cmd] * 0.5
        
        # Add explanations
        for suggestion in suggestions:
            suggestion['explanation'] = self._generate_explanation(
                suggestion['command'],
                context
            )
        
        return suggestions
    
    def _generate_explanation(self, command: str, context: Dict[str, Any]) -> str:
        """Generate explanation for command suggestion"""
        base = command.split()[0] if command else ''
        
        explanations = {
            'git': 'Version control operation',
            'npm': 'Node package management',
            'pip': 'Python package management',
            'docker': 'Container operation',
            'python': 'Run Python script',
        }
        
        return explanations.get(base, f'Execute {command}')
    
    async def learn(self, command: str, success: bool, 
                   context: Dict[str, Any] = None) -> None:
        """Learn from command execution"""
        if not self.enabled:
            return
        
        if success:
            self.patterns[command] += 1


class ErrorHandler:
    """Handles command errors and suggests fixes"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    async def suggest_fixes(self, command: str, error: str) -> List[Dict[str, Any]]:
        """Suggest fixes for command errors"""
        if not self.enabled:
            return []
        
        fixes = []
        
        # Common error patterns
        if 'command not found' in error.lower():
            fixes.extend(self._fix_command_not_found(command))
        elif 'permission denied' in error.lower():
            fixes.extend(self._fix_permission_denied(command))
        elif 'no such file or directory' in error.lower():
            fixes.extend(self._fix_file_not_found(command))
        elif 'syntax error' in error.lower():
            fixes.extend(self._fix_syntax_error(command))
        
        return fixes
    
    def _fix_command_not_found(self, command: str) -> List[Dict[str, Any]]:
        """Fix 'command not found' errors"""
        base = command.split()[0] if command else ''
        
        suggestions = []
        
        # Common typos
        typo_map = {
            'pyhton': 'python',
            'gti': 'git',
            'cd..': 'cd ..',
            'sl': 'ls',
        }
        
        if base in typo_map:
            fixed = command.replace(base, typo_map[base], 1)
            suggestions.append({
                'fix': fixed,
                'explanation': f'Did you mean "{typo_map[base]}"?',
                'confidence': 0.9
            })
        
        # Installation suggestions
        install_map = {
            'docker': 'Install Docker: https://docs.docker.com/get-docker/',
            'kubectl': 'Install kubectl: https://kubernetes.io/docs/tasks/tools/',
            'git': 'Install Git: https://git-scm.com/downloads',
        }
        
        if base in install_map:
            suggestions.append({
                'fix': None,
                'explanation': install_map[base],
                'confidence': 0.8
            })
        
        return suggestions
    
    def _fix_permission_denied(self, command: str) -> List[Dict[str, Any]]:
        """Fix permission denied errors"""
        return [{
            'fix': f'sudo {command}',
            'explanation': 'Try running with sudo (elevated privileges)',
            'confidence': 0.7
        }]
    
    def _fix_file_not_found(self, command: str) -> List[Dict[str, Any]]:
        """Fix file not found errors"""
        return [{
            'fix': None,
            'explanation': 'Check if the file path is correct and the file exists',
            'confidence': 0.6
        }]
    
    def _fix_syntax_error(self, command: str) -> List[Dict[str, Any]]:
        """Fix syntax errors"""
        return [{
            'fix': None,
            'explanation': 'Check command syntax and arguments',
            'confidence': 0.5
        }]
