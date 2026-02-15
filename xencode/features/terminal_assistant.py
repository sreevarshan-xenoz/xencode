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
from .error_handler_enhanced import EnhancedErrorHandler, ErrorFix


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
        self.error_handler = EnhancedErrorHandler(
            enabled=self.ta_config.error_fix_enabled,
            command_history=[]
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
    
    async def fix_error(self, command: str, error: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Suggest fixes for command errors
        
        Args:
            command: Command that failed
            error: Error message
            context: Execution context
            
        Returns:
            List of fix suggestions with confidence scores
        """
        # Get context if not provided
        if context is None and self.context_analyzer:
            context = await self.context_analyzer.analyze()
        
        # Update error handler with current command history
        if self.command_predictor:
            self.error_handler.command_history = self.command_predictor.history
        
        # Get fix suggestions
        fixes = await self.error_handler.suggest_fixes(command, error, context)
        
        # Convert ErrorFix objects to dictionaries
        return [
            {
                'fix': fix.fix_command,
                'explanation': fix.explanation,
                'confidence': fix.confidence,
                'category': fix.category,
                'requires_sudo': fix.requires_sudo,
                'requires_install': fix.requires_install,
                'install_command': fix.install_command,
                'documentation_url': fix.documentation_url,
                'alternative_commands': fix.alternative_commands
            }
            for fix in fixes
        ]
    
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
        await self.command_predictor.record(command, success, context)
        
        # Update learning engine
        await self.learning_engine.learn(command, success, context)
    
    async def record_successful_fix(self, original_command: str, error: str, 
                                   fix_command: str) -> None:
        """
        Record a successful error fix for learning
        
        Args:
            original_command: The command that failed
            error: The error message
            fix_command: The fix that worked
        """
        await self.error_handler.record_successful_fix(original_command, error, fix_command)
    
    async def get_statistics(self, command: str = None) -> Dict[str, Any]:
        """
        Get command statistics
        
        Args:
            command: Specific command to get stats for (None for overall stats)
            
        Returns:
            Statistics dictionary
        """
        return await self.command_predictor.get_command_statistics(command)
    
    async def analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyze all detected patterns in command history
        
        Returns:
            Pattern analysis results
        """
        return await self.command_predictor.analyze_patterns()
    
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
            },
            {
                'path': '/api/terminal/statistics',
                'method': 'GET',
                'handler': self.get_statistics
            },
            {
                'path': '/api/terminal/patterns',
                'method': 'GET',
                'handler': self.analyze_patterns
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
    
    async def get_command_statistics(self, command: str = None) -> Dict[str, Any]:
        """Get statistics for a specific command or all commands"""
        if command:
            stats = {
                'command': command,
                'frequency': self.command_frequency.get(command, 0),
                'success_rate': self._calculate_success_rate(command),
                'last_used': self._get_last_used(command),
                'common_sequences': self._get_common_sequences(command),
                'temporal_usage': self._get_temporal_usage(command)
            }
        else:
            stats = {
                'total_commands': len(self.history),
                'unique_commands': len(self.command_frequency),
                'most_frequent': self.command_frequency.most_common(10),
                'success_rate': self._calculate_overall_success_rate(),
                'patterns_detected': sum(len(v) for v in self.command_patterns.values())
            }
        
        return stats
    
    def _calculate_success_rate(self, command: str) -> float:
        """Calculate success rate for a command"""
        stats = self.success_rates.get(command, {'success': 0, 'failure': 0})
        total = stats['success'] + stats['failure']
        if total == 0:
            return 0.0
        return stats['success'] / total
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate"""
        total_success = sum(stats['success'] for stats in self.success_rates.values())
        total_failure = sum(stats['failure'] for stats in self.success_rates.values())
        total = total_success + total_failure
        if total == 0:
            return 0.0
        return total_success / total
    
    def _get_last_used(self, command: str) -> Optional[str]:
        """Get last usage timestamp for a command"""
        for cmd_data in reversed(self.history):
            if cmd_data.get('command') == command:
                return cmd_data.get('timestamp')
        return None
    
    def _get_common_sequences(self, command: str) -> List[Tuple[str, int]]:
        """Get common command sequences following this command"""
        if command in self.command_sequences:
            return self.command_sequences[command].most_common(5)
        return []
    
    def _get_temporal_usage(self, command: str) -> Dict[str, int]:
        """Get temporal usage pattern for a command"""
        usage = {}
        for time_key, commands in self.temporal_patterns.items():
            if command in commands:
                usage[time_key] = commands[command]
        return usage
    
    async def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze all detected patterns in command history"""
        analysis = {
            'command_patterns': {},
            'sequence_patterns': [],
            'temporal_patterns': {},
            'context_patterns': {}
        }
        
        # Analyze command patterns
        for base, patterns in self.command_patterns.items():
            analysis['command_patterns'][base] = {
                'count': len(patterns),
                'examples': patterns[:5]
            }
        
        # Analyze sequence patterns
        for cmd, sequences in self.command_sequences.items():
            if sequences:
                top_sequence = sequences.most_common(1)[0]
                analysis['sequence_patterns'].append({
                    'from': cmd,
                    'to': top_sequence[0],
                    'frequency': top_sequence[1]
                })
        
        # Analyze temporal patterns
        for time_key, commands in self.temporal_patterns.items():
            if commands:
                analysis['temporal_patterns'][time_key] = Counter(commands).most_common(5)
        
        # Analyze context patterns
        for context_key, commands in self.context_patterns.items():
            if commands:
                analysis['context_patterns'][context_key] = commands.most_common(5)
        
        return analysis


class ContextAnalyzer:
    """Analyzes current context for command suggestions"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._git_cache: Dict[str, Any] = {}
        self._env_cache: Dict[str, str] = {}

    async def analyze(self, context: str = None) -> Dict[str, Any]:
        """
        Analyze current context comprehensively

        Returns:
            Dictionary with context information including:
            - directory: Current working directory
            - project_type: Detected project type
            - git_info: Git repository information
            - environment: Relevant environment variables
            - filesystem: File system context
            - processes: Running process context
            - network: Network context
        """
        if not self.enabled:
            return {}

        cwd = Path.cwd()

        context_info = {
            'directory': str(cwd),
            'project_type': self._detect_project_type(cwd),
            'git_info': await self._analyze_git_repo(cwd),
            'environment': self._analyze_environment(),
            'filesystem': self._analyze_filesystem(cwd),
            'processes': self._analyze_processes(),
            'network': self._analyze_network(),
            'os': os.name,
            'files': self._get_relevant_files(cwd)
        }

        return context_info

    def _detect_project_type(self, path: Path) -> Optional[str]:
        """
        Detect project type from directory contents with advanced detection

        Supports: Python, Node.js, Rust, Go, Java, Ruby, PHP, C/C++, .NET, Docker, Kubernetes
        """
        # Python projects
        if (path / 'pyproject.toml').exists():
            return 'python-poetry'
        elif (path / 'setup.py').exists():
            return 'python-setuptools'
        elif (path / 'requirements.txt').exists():
            return 'python'
        elif (path / 'Pipfile').exists():
            return 'python-pipenv'
        elif (path / 'conda.yaml').exists() or (path / 'environment.yml').exists():
            return 'python-conda'

        # Node.js projects
        elif (path / 'package.json').exists():
            # Check for specific frameworks
            try:
                with open(path / 'package.json', 'r') as f:
                    pkg = json.load(f)
                    deps = {**pkg.get('dependencies', {}), **pkg.get('devDependencies', {})}

                    if 'react' in deps:
                        return 'node-react'
                    elif 'vue' in deps:
                        return 'node-vue'
                    elif 'angular' in deps or '@angular/core' in deps:
                        return 'node-angular'
                    elif 'next' in deps:
                        return 'node-nextjs'
                    elif 'express' in deps:
                        return 'node-express'
                    else:
                        return 'node'
            except Exception:
                return 'node'

        # Rust projects
        elif (path / 'Cargo.toml').exists():
            return 'rust'

        # Go projects
        elif (path / 'go.mod').exists():
            return 'go'

        # Java projects
        elif (path / 'pom.xml').exists():
            return 'java-maven'
        elif (path / 'build.gradle').exists() or (path / 'build.gradle.kts').exists():
            return 'java-gradle'

        # Ruby projects
        elif (path / 'Gemfile').exists():
            return 'ruby'

        # PHP projects
        elif (path / 'composer.json').exists():
            return 'php'

        # .NET projects
        elif list(path.glob('*.csproj')) or list(path.glob('*.fsproj')):
            return 'dotnet'

        # C/C++ projects
        elif (path / 'CMakeLists.txt').exists():
            return 'cpp-cmake'
        elif (path / 'Makefile').exists():
            return 'c-make'

        # Docker projects
        elif (path / 'Dockerfile').exists() or (path / 'docker-compose.yml').exists():
            return 'docker'

        # Kubernetes projects
        elif list(path.glob('*.yaml')) and any('kind:' in f.read_text() for f in path.glob('*.yaml') if f.is_file()):
            return 'kubernetes'

        # Git repository
        elif (path / '.git').exists():
            return 'git'

        return None

    async def _analyze_git_repo(self, path: Path) -> Dict[str, Any]:
        """
        Analyze Git repository information

        Returns:
            - is_repo: Whether directory is a git repo
            - branch: Current branch name
            - status: Git status (clean, modified, etc.)
            - remotes: List of remote repositories
            - ahead_behind: Commits ahead/behind remote
            - stashed: Number of stashed changes
        """
        git_info = {
            'is_repo': False,
            'branch': None,
            'status': None,
            'remotes': [],
            'ahead_behind': {'ahead': 0, 'behind': 0},
            'stashed': 0,
            'has_changes': False,
            'untracked_files': 0
        }

        if not (path / '.git').exists():
            return git_info

        git_info['is_repo'] = True

        try:
            # Get current branch
            branch_file = path / '.git' / 'HEAD'
            if branch_file.exists():
                head_content = branch_file.read_text().strip()
                if head_content.startswith('ref: refs/heads/'):
                    git_info['branch'] = head_content.replace('ref: refs/heads/', '')
                else:
                    git_info['branch'] = head_content[:7]  # Detached HEAD

            # Get remotes
            config_file = path / '.git' / 'config'
            if config_file.exists():
                config_content = config_file.read_text()
                remotes = re.findall(r'\[remote "(.+?)"\]', config_content)
                git_info['remotes'] = remotes

            # Check for changes (simplified - would use git commands in production)
            # This is a basic check; real implementation would use subprocess
            git_info['status'] = 'unknown'

        except Exception:
            pass

        return git_info

    def _analyze_environment(self) -> Dict[str, Any]:
        """
        Analyze relevant environment variables

        Returns:
            Dictionary with categorized environment variables:
            - paths: PATH and related variables
            - development: Development-related variables
            - cloud: Cloud provider variables
            - shell: Shell configuration
            - custom: Project-specific variables
        """
        env_info = {
            'paths': {},
            'development': {},
            'cloud': {},
            'shell': {},
            'custom': {}
        }

        # Path variables
        path_vars = ['PATH', 'PYTHONPATH', 'NODE_PATH', 'GOPATH', 'CARGO_HOME']
        for var in path_vars:
            if var in os.environ:
                env_info['paths'][var] = os.environ[var]

        # Development variables
        dev_vars = [
            'VIRTUAL_ENV', 'CONDA_DEFAULT_ENV', 'NVM_DIR', 'PYENV_ROOT',
            'RBENV_ROOT', 'JAVA_HOME', 'ANDROID_HOME', 'EDITOR', 'VISUAL'
        ]
        for var in dev_vars:
            if var in os.environ:
                env_info['development'][var] = os.environ[var]

        # Cloud provider variables
        cloud_vars = [
            'AWS_PROFILE', 'AWS_REGION', 'AWS_DEFAULT_REGION',
            'GOOGLE_CLOUD_PROJECT', 'GCP_PROJECT',
            'AZURE_SUBSCRIPTION_ID', 'AZURE_RESOURCE_GROUP',
            'KUBECONFIG', 'KUBERNETES_SERVICE_HOST'
        ]
        for var in cloud_vars:
            if var in os.environ:
                env_info['cloud'][var] = os.environ[var]

        # Shell variables
        shell_vars = ['SHELL', 'TERM', 'LANG', 'LC_ALL', 'HOME', 'USER']
        for var in shell_vars:
            if var in os.environ:
                env_info['shell'][var] = os.environ[var]

        # Custom variables (project-specific)
        # Look for variables with common prefixes
        custom_prefixes = ['PROJECT_', 'APP_', 'API_', 'DB_', 'REDIS_', 'MONGO_']
        for key, value in os.environ.items():
            if any(key.startswith(prefix) for prefix in custom_prefixes):
                env_info['custom'][key] = value

        return env_info

    def _analyze_filesystem(self, path: Path) -> Dict[str, Any]:
        """
        Analyze file system context

        Returns:
            - disk_usage: Disk space information
            - permissions: Directory permissions
            - file_counts: Count of different file types
            - recent_files: Recently modified files
        """
        fs_info = {
            'disk_usage': {},
            'permissions': {},
            'file_counts': {},
            'recent_files': []
        }

        try:
            # Disk usage
            import shutil
            usage = shutil.disk_usage(path)
            fs_info['disk_usage'] = {
                'total': usage.total,
                'used': usage.used,
                'free': usage.free,
                'percent_used': (usage.used / usage.total) * 100 if usage.total > 0 else 0
            }
        except Exception:
            pass

        try:
            # Directory permissions
            stat_info = path.stat()
            fs_info['permissions'] = {
                'readable': os.access(path, os.R_OK),
                'writable': os.access(path, os.W_OK),
                'executable': os.access(path, os.X_OK),
                'mode': oct(stat_info.st_mode)
            }
        except Exception:
            pass

        try:
            # File counts by extension
            file_counts = Counter()
            recent_files = []

            for item in path.iterdir():
                if item.is_file():
                    ext = item.suffix.lower() or 'no_extension'
                    file_counts[ext] += 1

                    # Track recent files (modified in last 24 hours)
                    try:
                        mtime = item.stat().st_mtime
                        age_hours = (datetime.now().timestamp() - mtime) / 3600
                        if age_hours < 24:
                            recent_files.append({
                                'name': item.name,
                                'age_hours': round(age_hours, 1)
                            })
                    except Exception:
                        pass

            fs_info['file_counts'] = dict(file_counts.most_common(10))
            fs_info['recent_files'] = sorted(recent_files, key=lambda x: x['age_hours'])[:10]

        except Exception:
            pass

        return fs_info

    def _analyze_processes(self) -> Dict[str, Any]:
        """
        Analyze running process context

        Returns:
            - development_servers: Running dev servers
            - databases: Running database processes
            - containers: Running containers
            - relevant_processes: Other relevant processes
        """
        process_info = {
            'development_servers': [],
            'databases': [],
            'containers': [],
            'relevant_processes': []
        }

        try:
            import psutil

            # Common development server process names
            dev_servers = ['node', 'npm', 'yarn', 'webpack', 'vite', 'python', 'flask', 'django', 'rails']
            databases = ['postgres', 'mysql', 'mongodb', 'redis', 'elasticsearch']
            containers = ['docker', 'containerd', 'podman']

            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    name = proc.info['name'].lower()

                    if any(srv in name for srv in dev_servers):
                        process_info['development_servers'].append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': ' '.join(proc.info['cmdline'][:3]) if proc.info['cmdline'] else ''
                        })
                    elif any(db in name for db in databases):
                        process_info['databases'].append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name']
                        })
                    elif any(cnt in name for cnt in containers):
                        process_info['containers'].append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Limit results
            process_info['development_servers'] = process_info['development_servers'][:5]
            process_info['databases'] = process_info['databases'][:5]
            process_info['containers'] = process_info['containers'][:5]

        except ImportError:
            # psutil not available
            pass
        except Exception:
            pass

        return process_info

    def _analyze_network(self) -> Dict[str, Any]:
        """
        Analyze network context where relevant

        Returns:
            - localhost_ports: Ports listening on localhost
            - vpn_active: Whether VPN is active
            - network_interfaces: Available network interfaces
        """
        network_info = {
            'localhost_ports': [],
            'vpn_active': False,
            'network_interfaces': []
        }

        try:
            import psutil

            # Check for common development ports
            common_ports = [3000, 3001, 4200, 5000, 5173, 8000, 8080, 8888, 9000]
            connections = psutil.net_connections(kind='inet')

            for conn in connections:
                if conn.status == 'LISTEN' and conn.laddr.port in common_ports:
                    network_info['localhost_ports'].append({
                        'port': conn.laddr.port,
                        'address': conn.laddr.ip
                    })

            # Check network interfaces
            interfaces = psutil.net_if_addrs()
            for iface_name in interfaces:
                # Check for VPN interfaces
                if any(vpn in iface_name.lower() for vpn in ['vpn', 'tun', 'tap', 'ppp']):
                    network_info['vpn_active'] = True

                network_info['network_interfaces'].append(iface_name)

        except ImportError:
            # psutil not available
            pass
        except Exception:
            pass

        return network_info

    def _get_relevant_files(self, path: Path) -> List[str]:
        """Get relevant files in directory"""
        try:
            files = [f.name for f in path.iterdir() if f.is_file()]
            return files[:20]  # Limit to 20 files
        except Exception:
            return []




class LearningEngine:
    """Learns user preferences and patterns with advanced learning capabilities"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.preferences: Dict[str, Any] = {}
        self.patterns: Dict[str, int] = Counter()
        self.command_contexts: Dict[str, List[Dict[str, Any]]] = {}
        self.user_skill_level: Dict[str, float] = {}  # Track skill per command category
        self.learning_progress: Dict[str, Dict[str, Any]] = {}
        self.preference_weights: Dict[str, float] = {
            'frequency': 0.3,
            'recency': 0.2,
            'success_rate': 0.3,
            'context_match': 0.2
        }

    async def load_preferences(self) -> None:
        """Load user preferences"""
        prefs_file = Path.home() / '.xencode' / 'terminal_preferences.json'
        if prefs_file.exists():
            try:
                with open(prefs_file, 'r') as f:
                    data = json.load(f)
                    self.preferences = data.get('preferences', {})
                    self.patterns = Counter(data.get('patterns', {}))
                    self.command_contexts = data.get('command_contexts', {})
                    self.user_skill_level = data.get('user_skill_level', {})
                    self.learning_progress = data.get('learning_progress', {})
                    self.preference_weights = data.get('preference_weights', self.preference_weights)
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
                    'patterns': dict(self.patterns),
                    'command_contexts': self.command_contexts,
                    'user_skill_level': self.user_skill_level,
                    'learning_progress': self.learning_progress,
                    'preference_weights': self.preference_weights
                }, f, indent=2)
        except Exception:
            pass

    async def enhance_suggestions(self, suggestions: List[Dict[str, Any]],
                                 context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance suggestions based on learned preferences with personalized scoring"""
        if not self.enabled:
            return suggestions

        enhanced = []
        for suggestion in suggestions:
            cmd = suggestion['command']

            # Calculate personalized score
            personalized_score = self._calculate_personalized_score(cmd, context)
            suggestion['score'] = suggestion.get('score', 0.0) + personalized_score

            # Add detailed explanation
            suggestion['explanation'] = self._generate_explanation(cmd, context)

            # Add difficulty level
            suggestion['difficulty'] = self._estimate_difficulty(cmd)

            # Add learning hints if user is learning
            if self._is_learning_command(cmd):
                suggestion['learning_hint'] = self._generate_learning_hint(cmd)

            enhanced.append(suggestion)

        # Sort by personalized score
        enhanced.sort(key=lambda x: x['score'], reverse=True)

        return enhanced

    def _calculate_personalized_score(self, command: str, context: Dict[str, Any]) -> float:
        """Calculate personalized score based on user preferences and patterns"""
        score = 0.0

        # Frequency score
        frequency_score = self.patterns.get(command, 0) * self.preference_weights['frequency']
        score += frequency_score

        # Recency score
        recency_score = self._calculate_recency_score(command) * self.preference_weights['recency']
        score += recency_score

        # Success rate score
        success_score = self._calculate_success_score(command) * self.preference_weights['success_rate']
        score += success_score

        # Context match score
        context_score = self._calculate_context_match(command, context) * self.preference_weights['context_match']
        score += context_score

        return score

    def _calculate_recency_score(self, command: str) -> float:
        """Calculate recency score based on last usage"""
        if command not in self.command_contexts:
            return 0.0

        contexts = self.command_contexts[command]
        if not contexts:
            return 0.0

        # Get most recent usage
        last_used = contexts[-1].get('timestamp', '')
        if not last_used:
            return 0.0

        try:
            from datetime import datetime
            last_time = datetime.fromisoformat(last_used)
            now = datetime.now()
            hours_ago = (now - last_time).total_seconds() / 3600

            # Exponential decay: more recent = higher score
            return max(0.0, 1.0 - (hours_ago / 168))  # 168 hours = 1 week
        except Exception:
            return 0.0

    def _calculate_success_score(self, command: str) -> float:
        """Calculate success rate score"""
        if command not in self.command_contexts:
            return 0.5  # Neutral score for unknown commands

        contexts = self.command_contexts[command]
        if not contexts:
            return 0.5

        successes = sum(1 for ctx in contexts if ctx.get('success', False))
        total = len(contexts)

        return successes / total if total > 0 else 0.5

    def _calculate_context_match(self, command: str, context: Dict[str, Any]) -> float:
        """Calculate how well command matches current context"""
        if command not in self.command_contexts or not context:
            return 0.0

        contexts = self.command_contexts[command]
        if not contexts:
            return 0.0

        # Compare current context with historical contexts
        match_score = 0.0
        current_dir = context.get('current_directory', '')
        current_project = context.get('project_type', '')

        for ctx in contexts[-10:]:  # Check last 10 usages
            score = 0.0

            # Directory match
            if ctx.get('current_directory', '') == current_dir:
                score += 0.5

            # Project type match
            if ctx.get('project_type', '') == current_project:
                score += 0.5

            match_score += score

        return match_score / min(10, len(contexts)) if contexts else 0.0

    def _estimate_difficulty(self, command: str) -> str:
        """Estimate command difficulty level"""
        base_cmd = command.split()[0] if command else ''

        # Advanced commands (pipes, redirects, complex flags) - check first
        if '|' in command or '>' in command or '<' in command:
            return 'advanced'

        # Count flags
        flag_count = command.count('-')
        if flag_count > 3:
            return 'advanced'

        # Simple commands
        simple_commands = {'ls', 'cd', 'pwd', 'echo', 'cat', 'mkdir', 'touch', 'rm', 'cp', 'mv'}
        if base_cmd in simple_commands:
            return 'beginner'

        # Intermediate commands
        intermediate_commands = {'grep', 'find', 'sed', 'awk', 'git', 'npm', 'pip', 'curl', 'wget'}
        if base_cmd in intermediate_commands:
            return 'intermediate'

        # Check flag count for intermediate
        if flag_count > 1:
            return 'intermediate'

        return 'beginner'

    def _is_learning_command(self, command: str) -> bool:
        """Check if user is learning this command"""
        base_cmd = command.split()[0] if command else ''

        # Check if command has low usage count (learning phase)
        if self.patterns.get(command, 0) < 5:
            return True

        # Check if command category has low skill level
        if base_cmd in self.user_skill_level:
            return self.user_skill_level[base_cmd] < 0.5

        return False

    def _generate_learning_hint(self, command: str) -> str:
        """Generate learning hint for command"""
        base_cmd = command.split()[0] if command else ''

        hints = {
            'git': 'Tip: Use "git status" frequently to check your repository state',
            'npm': 'Tip: Use "npm install --save" to add dependencies to package.json',
            'pip': 'Tip: Use "pip install -r requirements.txt" to install all dependencies',
            'docker': 'Tip: Use "docker ps" to see running containers',
            'grep': 'Tip: Use -r flag for recursive search in directories',
            'find': 'Tip: Use -name flag to search by filename pattern',
        }

        return hints.get(base_cmd, f'Practice {base_cmd} to improve your skills')

    def _generate_explanation(self, command: str, context: Dict[str, Any]) -> str:
        """Generate detailed explanation for command suggestion"""
        base = command.split()[0] if command else ''

        # Context-aware explanations
        project_type = context.get('project_type', '') if context else ''

        explanations = {
            'git': f'Version control operation{" for " + project_type if project_type else ""}',
            'npm': 'Node package management - install, update, or run scripts',
            'pip': 'Python package management - install or manage dependencies',
            'docker': 'Container operation - build, run, or manage containers',
            'python': 'Execute Python script or start Python interpreter',
            'pytest': 'Run Python tests with pytest framework',
            'cargo': 'Rust package manager and build tool',
            'go': 'Go language compiler and tool',
            'make': 'Build automation using Makefile',
            'cmake': 'Cross-platform build system generator',
        }

        explanation = explanations.get(base, f'Execute {command}')

        # Add usage frequency context
        usage_count = self.patterns.get(command, 0)
        if usage_count > 10:
            explanation += f' (frequently used: {usage_count} times)'
        elif usage_count > 0:
            explanation += f' (used {usage_count} times)'

        return explanation

    async def learn(self, command: str, success: bool,
                   context: Dict[str, Any] = None) -> None:
        """Learn from command execution with advanced pattern recognition"""
        if not self.enabled:
            return

        # Update command patterns
        if success:
            self.patterns[command] += 1

        # Store command context
        if command not in self.command_contexts:
            self.command_contexts[command] = []

        from datetime import datetime
        context_entry = {
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'current_directory': context.get('current_directory', '') if context else '',
            'project_type': context.get('project_type', '') if context else '',
        }

        self.command_contexts[command].append(context_entry)

        # Keep only last 50 contexts per command
        if len(self.command_contexts[command]) > 50:
            self.command_contexts[command] = self.command_contexts[command][-50:]

        # Update skill level
        await self._update_skill_level(command, success)

        # Update learning progress
        await self._update_learning_progress(command, success)

        # Adapt preference weights based on user behavior
        await self._adapt_preference_weights()

    async def _update_skill_level(self, command: str, success: bool) -> None:
        """Update user skill level for command category"""
        base_cmd = command.split()[0] if command else ''

        if base_cmd not in self.user_skill_level:
            self.user_skill_level[base_cmd] = 0.0

        # Increase skill on success, decrease slightly on failure
        if success:
            self.user_skill_level[base_cmd] = min(1.0, self.user_skill_level[base_cmd] + 0.05)
        else:
            self.user_skill_level[base_cmd] = max(0.0, self.user_skill_level[base_cmd] - 0.02)

    async def _update_learning_progress(self, command: str, success: bool) -> None:
        """Track learning progress for commands"""
        base_cmd = command.split()[0] if command else ''

        if base_cmd not in self.learning_progress:
            self.learning_progress[base_cmd] = {
                'total_uses': 0,
                'successful_uses': 0,
                'first_used': datetime.now().isoformat(),
                'mastery_level': 0.0
            }

        progress = self.learning_progress[base_cmd]
        progress['total_uses'] += 1
        if success:
            progress['successful_uses'] += 1

        # Calculate mastery level (0.0 to 1.0)
        success_rate = progress['successful_uses'] / progress['total_uses']
        usage_factor = min(1.0, progress['total_uses'] / 20)  # Mastery after ~20 uses
        progress['mastery_level'] = success_rate * usage_factor

    async def _adapt_preference_weights(self) -> None:
        """Adapt preference weights based on user behavior patterns"""
        # Analyze which factors lead to successful command selections
        # This is a simplified adaptation - could be more sophisticated

        total_commands = sum(self.patterns.values())
        if total_commands < 50:
            return  # Not enough data to adapt

        # Slightly adjust weights based on usage patterns
        # Users who use many different commands might prefer context over frequency
        unique_commands = len(self.patterns)
        diversity_ratio = unique_commands / total_commands

        if diversity_ratio > 0.5:  # High diversity
            self.preference_weights['context_match'] = 0.3
            self.preference_weights['frequency'] = 0.2
        else:  # Low diversity (repetitive user)
            self.preference_weights['frequency'] = 0.4
            self.preference_weights['context_match'] = 0.1

    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics and progress"""
        return {
            'total_commands_learned': len(self.patterns),
            'total_executions': sum(self.patterns.values()),
            'skill_levels': self.user_skill_level,
            'learning_progress': self.learning_progress,
            'preference_weights': self.preference_weights,
            'mastered_commands': [
                cmd for cmd, progress in self.learning_progress.items()
                if progress.get('mastery_level', 0.0) > 0.8
            ]
        }

    async def adjust_difficulty(self, current_level: str) -> str:
        """Adjust difficulty level based on user performance"""
        # Calculate average mastery across all commands
        if not self.learning_progress:
            return current_level

        avg_mastery = sum(
            p.get('mastery_level', 0.0)
            for p in self.learning_progress.values()
        ) / len(self.learning_progress)

        # Suggest difficulty adjustment
        if avg_mastery > 0.8 and current_level == 'beginner':
            return 'intermediate'
        elif avg_mastery > 0.8 and current_level == 'intermediate':
            return 'advanced'
        elif avg_mastery < 0.3 and current_level == 'advanced':
            return 'intermediate'
        elif avg_mastery < 0.3 and current_level == 'intermediate':
            return 'beginner'

        return current_level

