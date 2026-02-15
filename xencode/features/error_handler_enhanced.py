"""
Enhanced Error Handler for Terminal Assistant

Provides comprehensive intelligent error handling including:
- Advanced error pattern recognition
- Context-aware fix suggestions
- Multiple fix alternatives with confidence scores
- Learning from successful fixes
- Integration with command history
"""

import re
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import json


@dataclass
class ErrorFix:
    """Represents a potential fix for an error"""
    fix_command: Optional[str]
    explanation: str
    confidence: float
    category: str
    requires_sudo: bool = False
    requires_install: bool = False
    install_command: Optional[str] = None
    documentation_url: Optional[str] = None
    alternative_commands: List[str] = None
    
    def __post_init__(self):
        if self.alternative_commands is None:
            self.alternative_commands = []


class ErrorPattern:
    """Defines an error pattern and its fixes"""
    
    def __init__(self, pattern: str, category: str, 
                 fix_generator: callable, priority: int = 5):
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.category = category
        self.fix_generator = fix_generator
        self.priority = priority
    
    def matches(self, error: str) -> bool:
        """Check if error matches this pattern"""
        return bool(self.pattern.search(error))



class EnhancedErrorHandler:
    """
    Enhanced error handler with intelligent error recognition and fix suggestions
    
    Features:
    - Advanced error pattern recognition
    - Context-aware fix suggestions
    - Multiple fix alternatives with confidence scores
    - Learning from successful fixes
    - Integration with command history
    """
    
    def __init__(self, enabled: bool = True, command_history: List[Dict[str, Any]] = None):
        self.enabled = enabled
        self.command_history = command_history or []
        self.error_patterns: List[ErrorPattern] = []
        self.successful_fixes: Dict[str, Counter] = defaultdict(Counter)
        self.error_frequency: Counter = Counter()
        self.context_fixes: Dict[str, List[str]] = defaultdict(list)
        
        # Initialize error patterns
        self._initialize_patterns()
        
        # Load learning data
        self._load_learning_data()
    
    def _initialize_patterns(self) -> None:
        """Initialize all error patterns"""
        # Command not found patterns
        self.error_patterns.append(ErrorPattern(
            r'command not found|not recognized as an internal or external command',
            'command_not_found',
            self._fix_command_not_found,
            priority=10
        ))
        
        # Permission denied patterns
        self.error_patterns.append(ErrorPattern(
            r'permission denied|access denied|operation not permitted',
            'permission_denied',
            self._fix_permission_denied,
            priority=9
        ))
        
        # File not found patterns
        self.error_patterns.append(ErrorPattern(
            r'no such file or directory|cannot find|file not found',
            'file_not_found',
            self._fix_file_not_found,
            priority=8
        ))
        
        # Syntax error patterns
        self.error_patterns.append(ErrorPattern(
            r'syntax error|invalid syntax|unexpected token|unexpected EOF|unmatched|unknown option',
            'syntax_error',
            self._fix_syntax_error,
            priority=7
        ))
        
        # Port already in use
        self.error_patterns.append(ErrorPattern(
            r'port.*already in use|address already in use|bind.*failed|EADDRINUSE|Errno 98',
            'port_in_use',
            self._fix_port_in_use,
            priority=8
        ))
        
        # Module/package not found
        self.error_patterns.append(ErrorPattern(
            r'module.*not found|no module named|cannot import|package.*not found|cannot find module',
            'module_not_found',
            self._fix_module_not_found,
            priority=9
        ))
        
        # Git errors
        self.error_patterns.append(ErrorPattern(
            r'fatal:.*not a git repository|not a git repository',
            'not_git_repo',
            self._fix_not_git_repo,
            priority=8
        ))
        
        # Network errors
        self.error_patterns.append(ErrorPattern(
            r'connection refused|connection timed out|network unreachable|could not resolve host|name or service not known',
            'network_error',
            self._fix_network_error,
            priority=7
        ))
        
        # Disk space errors
        self.error_patterns.append(ErrorPattern(
            r'no space left on device|disk full|out of space',
            'disk_space',
            self._fix_disk_space,
            priority=9
        ))
        
        # Docker errors
        self.error_patterns.append(ErrorPattern(
            r'docker.*not running|cannot connect to.*docker daemon',
            'docker_not_running',
            self._fix_docker_not_running,
            priority=8
        ))
        
        # Environment variable errors
        self.error_patterns.append(ErrorPattern(
            r'environment variable.*not set|undefined variable',
            'env_var_missing',
            self._fix_env_var_missing,
            priority=7
        ))
    
    async def suggest_fixes(self, command: str, error: str, 
                          context: Dict[str, Any] = None) -> List[ErrorFix]:
        """
        Suggest fixes for command errors with context awareness
        
        Args:
            command: The command that failed
            error: The error message
            context: Additional context (directory, project type, etc.)
            
        Returns:
            List of ErrorFix objects sorted by confidence
        """
        if not self.enabled:
            return []
        
        fixes = []
        
        # Record error frequency
        self.error_frequency[error[:100]] += 1
        
        # Match error patterns
        matched_patterns = []
        for pattern in self.error_patterns:
            if pattern.matches(error):
                matched_patterns.append(pattern)
        
        # Sort by priority
        matched_patterns.sort(key=lambda p: p.priority, reverse=True)
        
        # Generate fixes from matched patterns
        for pattern in matched_patterns:
            pattern_fixes = pattern.fix_generator(command, error, context)
            fixes.extend(pattern_fixes)
        
        # Add context-aware fixes
        if context:
            context_fixes = self._get_context_aware_fixes(command, error, context)
            fixes.extend(context_fixes)
        
        # Add learning-based fixes
        learning_fixes = self._get_learning_based_fixes(command, error)
        fixes.extend(learning_fixes)
        
        # Adjust confidence based on historical success
        fixes = self._adjust_confidence_from_history(fixes, command, error)
        
        # Sort by confidence and remove duplicates
        fixes = self._deduplicate_fixes(fixes)
        fixes.sort(key=lambda f: f.confidence, reverse=True)
        
        return fixes[:5]  # Return top 5 fixes
    
    def _fix_command_not_found(self, command: str, error: str, 
                               context: Dict[str, Any] = None) -> List[ErrorFix]:
        """Fix 'command not found' errors"""
        base_cmd = command.split()[0] if command else ''
        fixes = []
        
        # Check for common typos using fuzzy matching
        common_commands = [
            'python', 'pip', 'git', 'npm', 'node', 'docker', 'kubectl',
            'cargo', 'rustc', 'go', 'java', 'javac', 'gcc', 'make',
            'cmake', 'ls', 'cd', 'pwd', 'cat', 'grep', 'find', 'sed',
            'awk', 'curl', 'wget', 'ssh', 'scp', 'rsync', 'tar', 'zip'
        ]
        
        # Find close matches
        close_matches = difflib.get_close_matches(base_cmd, common_commands, n=3, cutoff=0.6)
        for match in close_matches:
            fixed_cmd = command.replace(base_cmd, match, 1)
            fixes.append(ErrorFix(
                fix_command=fixed_cmd,
                explanation=f'Did you mean "{match}"? (typo correction)',
                confidence=0.85 if difflib.SequenceMatcher(None, base_cmd, match).ratio() > 0.8 else 0.7,
                category='typo_correction'
            ))
        
        # Check command history for similar commands
        if self.command_history:
            similar_cmds = self._find_similar_commands_in_history(command)
            for sim_cmd in similar_cmds[:2]:
                fixes.append(ErrorFix(
                    fix_command=sim_cmd,
                    explanation=f'Similar command from your history',
                    confidence=0.75,
                    category='history_suggestion'
                ))
        
        # Installation suggestions
        install_map = {
            'docker': ('sudo apt-get install docker.io', 'https://docs.docker.com/get-docker/'),
            'kubectl': ('curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"', 
                       'https://kubernetes.io/docs/tasks/tools/'),
            'git': ('sudo apt-get install git', 'https://git-scm.com/downloads'),
            'python3': ('sudo apt-get install python3', 'https://www.python.org/downloads/'),
            'pip': ('sudo apt-get install python3-pip', 'https://pip.pypa.io/en/stable/installation/'),
            'npm': ('sudo apt-get install npm', 'https://nodejs.org/'),
            'cargo': ('curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh', 'https://www.rust-lang.org/tools/install'),
            'go': ('sudo apt-get install golang', 'https://golang.org/doc/install'),
        }
        
        if base_cmd in install_map:
            install_cmd, doc_url = install_map[base_cmd]
            fixes.append(ErrorFix(
                fix_command=None,
                explanation=f'Install {base_cmd}',
                confidence=0.9,
                category='installation',
                requires_install=True,
                install_command=install_cmd,
                documentation_url=doc_url
            ))
        
        return fixes
    
    def _fix_permission_denied(self, command: str, error: str, 
                               context: Dict[str, Any] = None) -> List[ErrorFix]:
        """Fix permission denied errors"""
        fixes = []
        
        # Suggest sudo
        fixes.append(ErrorFix(
            fix_command=f'sudo {command}',
            explanation='Run with elevated privileges (sudo)',
            confidence=0.8,
            category='permission',
            requires_sudo=True
        ))
        
        # Check if it's a file permission issue (script execution)
        if '.sh' in command or '.py' in command or './' in command:
            # Extract file path from command
            parts = command.split()
            for part in parts:
                if '/' in part or part.endswith('.sh') or part.endswith('.py'):
                    fixes.append(ErrorFix(
                        fix_command=f'chmod +x {part}',
                        explanation=f'Make {part} executable',
                        confidence=0.75,
                        category='permission',
                        alternative_commands=[f'chmod 755 {part}', f'chmod u+x {part}']
                    ))
                    break
        
        # Docker-specific permission fix
        if 'docker' in command.lower():
            fixes.append(ErrorFix(
                fix_command='sudo usermod -aG docker $USER',
                explanation='Add your user to the docker group (requires logout/login)',
                confidence=0.75,
                category='permission',
                documentation_url='https://docs.docker.com/engine/install/linux-postinstall/'
            ))
        
        return fixes
    
    def _fix_file_not_found(self, command: str, error: str, 
                           context: Dict[str, Any] = None) -> List[ErrorFix]:
        """Fix file not found errors"""
        fixes = []
        
        # Extract potential file path from command
        parts = command.split()
        file_paths = [p for p in parts if '/' in p or '.' in p]
        
        for file_path in file_paths:
            # Check if file exists with different extension
            path = Path(file_path)
            if path.parent.exists():
                similar_files = list(path.parent.glob(f'{path.stem}*'))
                for similar in similar_files[:3]:
                    fixed_cmd = command.replace(file_path, str(similar))
                    fixes.append(ErrorFix(
                        fix_command=fixed_cmd,
                        explanation=f'Did you mean {similar.name}?',
                        confidence=0.75,
                        category='file_suggestion'
                    ))
            
            # Suggest creating the file/directory
            if path.suffix:  # It's a file
                fixes.append(ErrorFix(
                    fix_command=f'touch {file_path}',
                    explanation=f'Create empty file {file_path}',
                    confidence=0.6,
                    category='file_creation'
                ))
            else:  # It's a directory
                fixes.append(ErrorFix(
                    fix_command=f'mkdir -p {file_path}',
                    explanation=f'Create directory {file_path}',
                    confidence=0.6,
                    category='directory_creation'
                ))
        
        # Check current directory
        fixes.append(ErrorFix(
            fix_command='ls -la',
            explanation='List files in current directory to verify path',
            confidence=0.5,
            category='diagnostic'
        ))
        
        return fixes
    
    def _fix_syntax_error(self, command: str, error: str, 
                         context: Dict[str, Any] = None) -> List[ErrorFix]:
        """Fix syntax errors"""
        fixes = []
        
        # Check for common syntax issues
        if '"' in command or "'" in command:
            # Quote mismatch
            if command.count('"') % 2 != 0 or command.count("'") % 2 != 0:
                fixes.append(ErrorFix(
                    fix_command=None,
                    explanation='Check for unmatched quotes in your command',
                    confidence=0.8,
                    category='syntax'
                ))
        
        # Check for missing operators
        if '=' in command and ' = ' not in command:
            fixed = command.replace('=', ' = ')
            fixes.append(ErrorFix(
                fix_command=fixed,
                explanation='Add spaces around = operator',
                confidence=0.6,
                category='syntax'
            ))
        
        # Command-specific syntax help
        base_cmd = command.split()[0] if command else ''
        syntax_help = {
            'git': 'git <command> [options] - Try: git --help',
            'docker': 'docker <command> [options] - Try: docker --help',
            'npm': 'npm <command> [options] - Try: npm help',
            'pip': 'pip <command> [options] - Try: pip --help',
        }
        
        if base_cmd in syntax_help:
            fixes.append(ErrorFix(
                fix_command=f'{base_cmd} --help',
                explanation=f'View {base_cmd} command syntax: {syntax_help[base_cmd]}',
                confidence=0.7,
                category='help',
                documentation_url=f'https://docs.{base_cmd}.com' if base_cmd != 'pip' else 'https://pip.pypa.io'
            ))
        
        return fixes
    
    def _fix_port_in_use(self, command: str, error: str, 
                        context: Dict[str, Any] = None) -> List[ErrorFix]:
        """Fix port already in use errors"""
        fixes = []
        
        # Extract port number from error or command
        port_match = re.search(r':(\d+)|port\s+(\d+)|server\s+(\d+)', error + ' ' + command)
        if port_match:
            port = port_match.group(1) or port_match.group(2) or port_match.group(3)
            
            # Find and kill process using port
            fixes.append(ErrorFix(
                fix_command=f'lsof -ti:{port} | xargs kill -9',
                explanation=f'Kill process using port {port}',
                confidence=0.85,
                category='port_management',
                alternative_commands=[
                    f'fuser -k {port}/tcp',
                    f'netstat -tulpn | grep {port}'
                ]
            ))
            
            # Suggest using different port
            new_port = int(port) + 1
            if any(cmd in command for cmd in ['npm', 'node', 'python', 'flask', 'django']):
                fixes.append(ErrorFix(
                    fix_command=None,
                    explanation=f'Use a different port (e.g., {new_port})',
                    confidence=0.7,
                    category='port_management'
                ))
        else:
            # Generic port in use fix
            fixes.append(ErrorFix(
                fix_command=None,
                explanation='A port is already in use. Check running processes or use a different port.',
                confidence=0.6,
                category='port_management'
            ))
        
        return fixes
    
    def _fix_module_not_found(self, command: str, error: str, 
                             context: Dict[str, Any] = None) -> List[ErrorFix]:
        """Fix module/package not found errors"""
        fixes = []
        
        # Extract module name
        module_match = re.search(r"module['\"]?\s+['\"]?(\w+)|no module named\s+['\"]?(\w+)|cannot import\s+['\"]?(\w+)|cannot find module\s+['\"]?(\w+)", error, re.IGNORECASE)
        if module_match:
            module = module_match.group(1) or module_match.group(2) or module_match.group(3) or module_match.group(4)
            
            # Python package installation
            if 'python' in command.lower() or context and context.get('project_type', '').startswith('python'):
                fixes.append(ErrorFix(
                    fix_command=f'pip install {module}',
                    explanation=f'Install Python package {module}',
                    confidence=0.9,
                    category='package_install',
                    requires_install=True,
                    alternative_commands=[
                        f'pip3 install {module}',
                        f'python -m pip install {module}',
                        f'poetry add {module}',
                        f'pipenv install {module}'
                    ]
                ))
            
            # Node.js package installation
            elif 'node' in command.lower() or 'npm' in command.lower() or context and context.get('project_type', '').startswith('node'):
                fixes.append(ErrorFix(
                    fix_command=f'npm install {module}',
                    explanation=f'Install Node.js package {module}',
                    confidence=0.9,
                    category='package_install',
                    requires_install=True,
                    alternative_commands=[
                        f'npm install --save {module}',
                        f'yarn add {module}',
                        f'pnpm add {module}'
                    ]
                ))
        
        return fixes
    
    def _fix_not_git_repo(self, command: str, error: str, 
                         context: Dict[str, Any] = None) -> List[ErrorFix]:
        """Fix 'not a git repository' errors"""
        fixes = []
        
        fixes.append(ErrorFix(
            fix_command='git init',
            explanation='Initialize a new Git repository',
            confidence=0.85,
            category='git_init'
        ))
        
        fixes.append(ErrorFix(
            fix_command='git clone <repository-url>',
            explanation='Clone an existing repository',
            confidence=0.75,
            category='git_clone',
            documentation_url='https://git-scm.com/docs/git-clone'
        ))
        
        return fixes
    
    def _fix_network_error(self, command: str, error: str, 
                          context: Dict[str, Any] = None) -> List[ErrorFix]:
        """Fix network-related errors"""
        fixes = []
        
        # Check network connectivity
        fixes.append(ErrorFix(
            fix_command='ping -c 3 8.8.8.8',
            explanation='Check internet connectivity',
            confidence=0.7,
            category='diagnostic'
        ))
        
        # DNS resolution check
        if 'could not resolve' in error.lower() or 'name or service not known' in error.lower():
            fixes.append(ErrorFix(
                fix_command='nslookup google.com',
                explanation='Check DNS resolution',
                confidence=0.75,
                category='diagnostic'
            ))
        
        # Proxy/VPN suggestion
        fixes.append(ErrorFix(
            fix_command=None,
            explanation='Check if you need to configure proxy or VPN settings',
            confidence=0.6,
            category='network_config'
        ))
        
        return fixes
    
    def _fix_disk_space(self, command: str, error: str, 
                       context: Dict[str, Any] = None) -> List[ErrorFix]:
        """Fix disk space errors"""
        fixes = []
        
        fixes.append(ErrorFix(
            fix_command='df -h',
            explanation='Check disk space usage',
            confidence=0.9,
            category='diagnostic'
        ))
        
        fixes.append(ErrorFix(
            fix_command='du -sh * | sort -h',
            explanation='Find large directories in current location',
            confidence=0.8,
            category='diagnostic'
        ))
        
        # Docker-specific cleanup
        if context and 'docker' in context.get('project_type', '').lower():
            fixes.append(ErrorFix(
                fix_command='docker system prune -a',
                explanation='Clean up Docker images and containers',
                confidence=0.85,
                category='cleanup'
            ))
        
        return fixes
    
    def _fix_docker_not_running(self, command: str, error: str, 
                                context: Dict[str, Any] = None) -> List[ErrorFix]:
        """Fix Docker daemon not running errors"""
        fixes = []
        
        fixes.append(ErrorFix(
            fix_command='sudo systemctl start docker',
            explanation='Start Docker daemon (systemd)',
            confidence=0.9,
            category='service_start',
            requires_sudo=True,
            alternative_commands=[
                'sudo service docker start',
                'dockerd'
            ]
        ))
        
        fixes.append(ErrorFix(
            fix_command='sudo systemctl enable docker',
            explanation='Enable Docker to start on boot',
            confidence=0.7,
            category='service_config',
            requires_sudo=True
        ))
        
        return fixes
    
    def _fix_env_var_missing(self, command: str, error: str, 
                            context: Dict[str, Any] = None) -> List[ErrorFix]:
        """Fix missing environment variable errors"""
        fixes = []
        
        # Extract variable name
        var_match = re.search(r'variable\s+["\']?(\w+)|(\w+).*not set', error, re.IGNORECASE)
        if var_match:
            var_name = var_match.group(1) or var_match.group(2)
            
            fixes.append(ErrorFix(
                fix_command=f'export {var_name}=<value>',
                explanation=f'Set environment variable {var_name}',
                confidence=0.8,
                category='env_config'
            ))
            
            fixes.append(ErrorFix(
                fix_command=f'echo "export {var_name}=<value>" >> ~/.bashrc',
                explanation=f'Permanently set {var_name} in .bashrc',
                confidence=0.7,
                category='env_config',
                alternative_commands=[
                    f'echo "export {var_name}=<value>" >> ~/.zshrc',
                    f'echo "export {var_name}=<value>" >> ~/.profile'
                ]
            ))
        
        return fixes
    
    def _get_context_aware_fixes(self, command: str, error: str, 
                                 context: Dict[str, Any]) -> List[ErrorFix]:
        """Generate context-aware fixes based on project type and environment"""
        fixes = []
        project_type = context.get('project_type', '')
        
        # Python project context
        if project_type and 'python' in project_type.lower():
            if 'module' in error.lower() or 'import' in error.lower():
                fixes.append(ErrorFix(
                    fix_command='pip install -r requirements.txt',
                    explanation='Install all project dependencies',
                    confidence=0.75,
                    category='dependency_install'
                ))
        
        # Node.js project context
        elif project_type and 'node' in project_type.lower():
            if 'module' in error.lower() or 'cannot find' in error.lower():
                fixes.append(ErrorFix(
                    fix_command='npm install',
                    explanation='Install all project dependencies',
                    confidence=0.75,
                    category='dependency_install'
                ))
        
        # Git repository context
        if context.get('git_info', {}).get('is_repo'):
            if 'branch' in error.lower():
                fixes.append(ErrorFix(
                    fix_command='git branch -a',
                    explanation='List all branches',
                    confidence=0.7,
                    category='git_diagnostic'
                ))
        
        return fixes
    
    def _get_learning_based_fixes(self, command: str, error: str) -> List[ErrorFix]:
        """Generate fixes based on learned successful patterns"""
        fixes = []
        
        # Check if we've seen this error before
        error_key = error[:100]  # Use first 100 chars as key
        
        if error_key in self.successful_fixes:
            # Get most successful fixes for this error
            for fix_cmd, count in self.successful_fixes[error_key].most_common(3):
                fixes.append(ErrorFix(
                    fix_command=fix_cmd,
                    explanation=f'This fix worked {count} time(s) before',
                    confidence=min(0.95, 0.6 + (count * 0.1)),
                    category='learned_fix'
                ))
        
        return fixes
    
    def _find_similar_commands_in_history(self, command: str) -> List[str]:
        """Find similar commands in command history"""
        if not self.command_history:
            return []
        
        similar = []
        base_cmd = command.split()[0] if command else ''
        
        for hist_entry in reversed(self.command_history[-100:]):  # Check last 100 commands
            hist_cmd = hist_entry.get('command', '')
            hist_base = hist_cmd.split()[0] if hist_cmd else ''
            
            # Same base command
            if hist_base == base_cmd:
                similarity = difflib.SequenceMatcher(None, command, hist_cmd).ratio()
                if similarity > 0.5 and hist_cmd != command:
                    similar.append(hist_cmd)
            
            if len(similar) >= 5:
                break
        
        return similar
    
    def _adjust_confidence_from_history(self, fixes: List[ErrorFix], 
                                       command: str, error: str) -> List[ErrorFix]:
        """Adjust confidence scores based on historical success"""
        error_key = error[:100]
        
        for fix in fixes:
            if fix.fix_command:
                # Boost confidence if this fix was successful before
                success_count = self.successful_fixes[error_key].get(fix.fix_command, 0)
                if success_count > 0:
                    fix.confidence = min(0.99, fix.confidence + (success_count * 0.05))
        
        return fixes
    
    def _deduplicate_fixes(self, fixes: List[ErrorFix]) -> List[ErrorFix]:
        """Remove duplicate fixes"""
        seen = set()
        unique_fixes = []
        
        for fix in fixes:
            key = (fix.fix_command, fix.explanation)
            if key not in seen:
                seen.add(key)
                unique_fixes.append(fix)
        
        return unique_fixes
    
    async def record_successful_fix(self, command: str, error: str, 
                                   fix_command: str) -> None:
        """Record a successful fix for learning"""
        error_key = error[:100]
        self.successful_fixes[error_key][fix_command] += 1
        
        # Save learning data
        await self._save_learning_data()
    
    def _load_learning_data(self) -> None:
        """Load learning data from file"""
        learning_file = Path.home() / '.xencode' / 'error_handler_learning.json'
        if learning_file.exists():
            try:
                with open(learning_file, 'r') as f:
                    data = json.load(f)
                    self.successful_fixes = defaultdict(Counter, {
                        k: Counter(v) for k, v in data.get('successful_fixes', {}).items()
                    })
                    self.error_frequency = Counter(data.get('error_frequency', {}))
            except Exception:
                pass
    
    async def _save_learning_data(self) -> None:
        """Save learning data to file"""
        learning_file = Path.home() / '.xencode' / 'error_handler_learning.json'
        learning_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(learning_file, 'w') as f:
                json.dump({
                    'successful_fixes': {k: dict(v) for k, v in self.successful_fixes.items()},
                    'error_frequency': dict(self.error_frequency)
                }, f, indent=2)
        except Exception:
            pass
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about errors and fixes"""
        return {
            'total_errors_seen': sum(self.error_frequency.values()),
            'unique_errors': len(self.error_frequency),
            'most_common_errors': self.error_frequency.most_common(10),
            'learned_fixes': sum(len(fixes) for fixes in self.successful_fixes.values()),
            'patterns_registered': len(self.error_patterns)
        }
