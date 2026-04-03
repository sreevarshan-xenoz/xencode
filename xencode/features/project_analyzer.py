"""
Project Analyzer Feature

Analyzes project structure, generates documentation, tracks metrics,
and provides insights into project health and architecture.
"""

import os
import ast
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter

from .base import FeatureBase, FeatureConfig, FeatureStatus


@dataclass
class ProjectAnalyzerConfig:
    """Configuration for Project Analyzer"""
    enabled: bool = True
    max_file_size: int = 1024 * 1024  # 1MB
    exclude_patterns: List[str] = field(default_factory=lambda: [
        '__pycache__', '.git', '.venv', 'node_modules', 'dist', 'build'
    ])
    include_extensions: List[str] = field(default_factory=lambda: [
        '.py', '.js', '.ts', '.go', '.rs', '.java', '.cpp', '.c', '.h'
    ])
    generate_diagrams: bool = True
    track_metrics: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectAnalyzerConfig':
        """Create config from dictionary"""
        return cls(
            enabled=data.get('enabled', True),
            max_file_size=data.get('max_file_size', 1024 * 1024),
            exclude_patterns=data.get('exclude_patterns', [
                '__pycache__', '.git', '.venv', 'node_modules', 'dist', 'build'
            ]),
            include_extensions=data.get('include_extensions', [
                '.py', '.js', '.ts', '.go', '.rs', '.java', '.cpp', '.c', '.h'
            ]),
            generate_diagrams=data.get('generate_diagrams', True),
            track_metrics=data.get('track_metrics', True)
        )



class ProjectAnalyzerFeature(FeatureBase):
    """Project Analyzer feature implementation"""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.pa_config = ProjectAnalyzerConfig.from_dict(config.config)
        self.structure_scanner = None
        self.dependency_analyzer = None
        self.metrics_calculator = None
        self.architecture_visualizer = None
        self.tech_debt_detector = None
    
    @property
    def name(self) -> str:
        """Feature name"""
        return "project_analyzer"
    
    @property
    def description(self) -> str:
        """Feature description"""
        return "Analyzes project structure, dependencies, metrics, and technical debt"
    
    async def _initialize(self) -> None:
        """Initialize Project Analyzer components"""
        self.structure_scanner = StructureScanner(
            exclude_patterns=self.pa_config.exclude_patterns,
            include_extensions=self.pa_config.include_extensions,
            max_file_size=self.pa_config.max_file_size
        )
        
        self.dependency_analyzer = DependencyAnalyzer()
        self.metrics_calculator = MetricsCalculator()
        self.architecture_visualizer = ArchitectureVisualizer()
        self.tech_debt_detector = TechnicalDebtDetector()
    
    async def _shutdown(self) -> None:
        """Shutdown Project Analyzer"""
        self.structure_scanner = None
        self.dependency_analyzer = None
        self.metrics_calculator = None
        self.architecture_visualizer = None
        self.tech_debt_detector = None
    
    async def analyze_project(self, project_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive project analysis
        
        Args:
            project_path: Path to project directory
            
        Returns:
            Complete analysis results
        """
        path = Path(project_path)
        if not path.exists():
            raise FileNotFoundError(f"Project not found: {project_path}")
        
        # Scan project structure
        structure = await self.structure_scanner.scan(path)
        
        # Analyze dependencies
        dependencies = await self.dependency_analyzer.analyze(path, structure)
        
        # Calculate metrics
        metrics = await self.metrics_calculator.calculate(path, structure)
        
        # Generate architecture visualization
        architecture = await self.architecture_visualizer.generate(path, structure, dependencies)
        
        # Detect technical debt
        tech_debt = await self.tech_debt_detector.detect(path, structure, metrics)
        
        return {
            'project_path': str(path),
            'analyzed_at': datetime.now().isoformat(),
            'structure': structure,
            'dependencies': dependencies,
            'metrics': metrics,
            'architecture': architecture,
            'technical_debt': tech_debt,
            'summary': self._generate_summary(structure, dependencies, metrics, tech_debt)
        }

    
    def _generate_summary(self, structure: Dict[str, Any], dependencies: Dict[str, Any],
                         metrics: Dict[str, Any], tech_debt: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary"""
        return {
            'total_files': structure.get('total_files', 0),
            'total_lines': metrics.get('total_lines', 0),
            'languages': list(structure.get('languages', {}).keys()),
            'dependency_count': len(dependencies.get('dependencies', [])),
            'complexity_score': metrics.get('average_complexity', 0),
            'maintainability_index': metrics.get('maintainability_index', 0),
            'tech_debt_items': len(tech_debt.get('issues', [])),
            'health_score': self._calculate_health_score(metrics, tech_debt)
        }
    
    def _calculate_health_score(self, metrics: Dict[str, Any], tech_debt: Dict[str, Any]) -> float:
        """Calculate overall project health score (0-100)"""
        score = 100.0
        
        # Deduct for high complexity
        avg_complexity = metrics.get('average_complexity', 0)
        if avg_complexity > 10:
            score -= min(20, (avg_complexity - 10) * 2)
        
        # Deduct for low maintainability
        maintainability = metrics.get('maintainability_index', 100)
        if maintainability < 65:
            score -= (65 - maintainability) * 0.5
        
        # Deduct for technical debt
        debt_count = len(tech_debt.get('issues', []))
        score -= min(30, debt_count * 2)
        
        return max(0, min(100, score))
    
    def get_cli_commands(self) -> List[Any]:
        """Get CLI commands for Project Analyzer"""
        return []
    
    def get_tui_components(self) -> List[Any]:
        """Get TUI components for Project Analyzer"""
        return []
    
    def get_api_endpoints(self) -> List[Any]:
        """Get API endpoints for Project Analyzer"""
        return [
            {
                'path': '/api/analyzer/analyze',
                'method': 'POST',
                'handler': self.analyze_project
            }
        ]



class StructureScanner:
    """Scans and analyzes project structure"""
    
    def __init__(self, exclude_patterns: List[str], include_extensions: List[str], max_file_size: int):
        self.exclude_patterns = exclude_patterns
        self.include_extensions = include_extensions
        self.max_file_size = max_file_size
    
    async def scan(self, project_path: Path) -> Dict[str, Any]:
        """Scan project structure"""
        files = []
        directories = []
        languages = Counter()
        total_size = 0
        
        for root, dirs, filenames in os.walk(project_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not self._should_exclude(d)]
            
            root_path = Path(root)
            directories.append(str(root_path.relative_to(project_path)))
            
            for filename in filenames:
                file_path = root_path / filename
                
                # Skip excluded files
                if self._should_exclude(filename):
                    continue
                
                # Check file extension
                ext = file_path.suffix.lower()
                if ext not in self.include_extensions:
                    continue
                
                # Check file size
                try:
                    size = file_path.stat().st_size
                    if size > self.max_file_size:
                        continue
                    
                    total_size += size
                    
                    # Detect language
                    language = self._detect_language(ext)
                    languages[language] += 1
                    
                    files.append({
                        'path': str(file_path.relative_to(project_path)),
                        'size': size,
                        'language': language,
                        'extension': ext
                    })
                except Exception:
                    continue
        
        return {
            'total_files': len(files),
            'total_directories': len(directories),
            'total_size': total_size,
            'files': files,
            'directories': directories,
            'languages': dict(languages),
            'project_type': self._detect_project_type(project_path, files)
        }
    
    def _should_exclude(self, name: str) -> bool:
        """Check if file/directory should be excluded"""
        for pattern in self.exclude_patterns:
            if pattern in name or name.startswith('.'):
                return True
        return False
    
    def _detect_language(self, extension: str) -> str:
        """Detect programming language from extension"""
        lang_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.go': 'Go',
            '.rs': 'Rust',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C/C++',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin'
        }
        return lang_map.get(extension, 'Unknown')
    
    def _detect_project_type(self, project_path: Path, files: List[Dict[str, Any]]) -> str:
        """Detect project type based on files"""
        # Check for specific files
        if (project_path / 'package.json').exists():
            return 'Node.js'
        if (project_path / 'requirements.txt').exists() or (project_path / 'setup.py').exists():
            return 'Python'
        if (project_path / 'Cargo.toml').exists():
            return 'Rust'
        if (project_path / 'go.mod').exists():
            return 'Go'
        if (project_path / 'pom.xml').exists() or (project_path / 'build.gradle').exists():
            return 'Java'
        
        # Detect by most common language
        languages = Counter(f['language'] for f in files)
        if languages:
            return languages.most_common(1)[0][0]
        
        return 'Unknown'



class DependencyAnalyzer:
    """Analyzes project dependencies"""
    
    async def analyze(self, project_path: Path, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze project dependencies"""
        project_type = structure.get('project_type', 'Unknown')
        
        dependencies = []
        dependency_graph = defaultdict(list)
        
        # Analyze based on project type
        if project_type == 'Python':
            dependencies = await self._analyze_python_deps(project_path)
        elif project_type == 'Node.js':
            dependencies = await self._analyze_nodejs_deps(project_path)
        elif project_type == 'Rust':
            dependencies = await self._analyze_rust_deps(project_path)
        elif project_type == 'Go':
            dependencies = await self._analyze_go_deps(project_path)
        
        # Analyze internal dependencies (imports between modules)
        internal_deps = await self._analyze_internal_deps(project_path, structure)
        
        return {
            'dependencies': dependencies,
            'dependency_count': len(dependencies),
            'internal_dependencies': internal_deps,
            'dependency_graph': dict(dependency_graph),
            'circular_dependencies': self._detect_circular_deps(internal_deps)
        }
    
    async def _analyze_python_deps(self, project_path: Path) -> List[Dict[str, Any]]:
        """Analyze Python dependencies"""
        deps = []
        
        # Check requirements.txt
        req_file = project_path / 'requirements.txt'
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Parse package name and version
                            parts = line.split('==')
                            name = parts[0].strip()
                            version = parts[1].strip() if len(parts) > 1 else 'latest'
                            deps.append({
                                'name': name,
                                'version': version,
                                'type': 'external'
                            })
            except Exception:
                pass
        
        # Check setup.py
        setup_file = project_path / 'setup.py'
        if setup_file.exists():
            try:
                with open(setup_file, 'r') as f:
                    content = f.read()
                    # Simple regex to find install_requires
                    import re
                    match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
                    if match:
                        requires = match.group(1)
                        for line in requires.split(','):
                            line = line.strip().strip('"\'')
                            if line:
                                parts = line.split('>=')
                                if len(parts) == 1:
                                    parts = line.split('==')
                                name = parts[0].strip()
                                version = parts[1].strip() if len(parts) > 1 else 'latest'
                                if name not in [d['name'] for d in deps]:
                                    deps.append({
                                        'name': name,
                                        'version': version,
                                        'type': 'external'
                                    })
            except Exception:
                pass
        
        return deps
    
    async def _analyze_nodejs_deps(self, project_path: Path) -> List[Dict[str, Any]]:
        """Analyze Node.js dependencies"""
        deps = []
        
        package_json = project_path / 'package.json'
        if package_json.exists():
            try:
                with open(package_json, 'r') as f:
                    data = json.load(f)
                    
                    # Regular dependencies
                    for name, version in data.get('dependencies', {}).items():
                        deps.append({
                            'name': name,
                            'version': version,
                            'type': 'external'
                        })
                    
                    # Dev dependencies
                    for name, version in data.get('devDependencies', {}).items():
                        deps.append({
                            'name': name,
                            'version': version,
                            'type': 'dev'
                        })
            except Exception:
                pass
        
        return deps
    
    async def _analyze_rust_deps(self, project_path: Path) -> List[Dict[str, Any]]:
        """Analyze Rust dependencies"""
        deps = []
        
        cargo_toml = project_path / 'Cargo.toml'
        if cargo_toml.exists():
            try:
                with open(cargo_toml, 'r') as f:
                    content = f.read()
                    # Simple parsing of [dependencies] section
                    in_deps = False
                    for line in content.split('\n'):
                        line = line.strip()
                        if line == '[dependencies]':
                            in_deps = True
                            continue
                        if line.startswith('[') and in_deps:
                            break
                        if in_deps and '=' in line:
                            parts = line.split('=')
                            name = parts[0].strip()
                            version = parts[1].strip().strip('"\'')
                            deps.append({
                                'name': name,
                                'version': version,
                                'type': 'external'
                            })
            except Exception:
                pass
        
        return deps
    
    async def _analyze_go_deps(self, project_path: Path) -> List[Dict[str, Any]]:
        """Analyze Go dependencies"""
        deps = []
        
        go_mod = project_path / 'go.mod'
        if go_mod.exists():
            try:
                with open(go_mod, 'r') as f:
                    in_require = False
                    for line in f:
                        line = line.strip()
                        if line.startswith('require'):
                            in_require = True
                            continue
                        if in_require and line == ')':
                            break
                        if in_require and line:
                            parts = line.split()
                            if len(parts) >= 2:
                                name = parts[0]
                                version = parts[1]
                                deps.append({
                                    'name': name,
                                    'version': version,
                                    'type': 'external'
                                })
            except Exception:
                pass
        
        return deps
    
    async def _analyze_internal_deps(self, project_path: Path, structure: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze internal module dependencies"""
        internal_deps = defaultdict(list)
        
        for file_info in structure.get('files', []):
            if file_info['language'] == 'Python':
                file_path = project_path / file_info['path']
                imports = await self._extract_python_imports(file_path)
                internal_deps[file_info['path']] = imports
        
        return dict(internal_deps)
    
    async def _extract_python_imports(self, file_path: Path) -> List[str]:
        """Extract imports from Python file"""
        imports = []
        
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
        except Exception:
            pass
        
        return imports
    
    def _detect_circular_deps(self, internal_deps: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular dependencies"""
        circular = []
        
        def has_path(start: str, end: str, visited: Set[str]) -> bool:
            if start == end:
                return True
            if start in visited:
                return False
            visited.add(start)
            for dep in internal_deps.get(start, []):
                if has_path(dep, end, visited):
                    return True
            return False
        
        # Check each pair
        for module in internal_deps:
            for dep in internal_deps[module]:
                if dep in internal_deps and has_path(dep, module, set()):
                    cycle = [module, dep]
                    if cycle not in circular and list(reversed(cycle)) not in circular:
                        circular.append(cycle)
        
        return circular



class MetricsCalculator:
    """Calculates code metrics"""
    
    async def calculate(self, project_path: Path, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate project metrics"""
        total_lines = 0
        total_code_lines = 0
        total_comment_lines = 0
        total_blank_lines = 0
        complexity_scores = []
        
        for file_info in structure.get('files', []):
            file_path = project_path / file_info['path']
            
            if file_info['language'] == 'Python':
                metrics = await self._analyze_python_file(file_path)
            else:
                metrics = await self._analyze_generic_file(file_path)
            
            total_lines += metrics['total_lines']
            total_code_lines += metrics['code_lines']
            total_comment_lines += metrics['comment_lines']
            total_blank_lines += metrics['blank_lines']
            
            if metrics['complexity'] > 0:
                complexity_scores.append(metrics['complexity'])
        
        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
        
        # Calculate maintainability index
        maintainability = self._calculate_maintainability_index(
            total_lines, avg_complexity, total_comment_lines
        )
        
        return {
            'total_lines': total_lines,
            'code_lines': total_code_lines,
            'comment_lines': total_comment_lines,
            'blank_lines': total_blank_lines,
            'average_complexity': round(avg_complexity, 2),
            'max_complexity': max(complexity_scores) if complexity_scores else 0,
            'maintainability_index': round(maintainability, 2),
            'comment_ratio': round(total_comment_lines / total_lines * 100, 2) if total_lines > 0 else 0
        }
    
    async def _analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze Python file metrics"""
        total_lines = 0
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        complexity = 0
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                total_lines = len(lines)
                
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        blank_lines += 1
                    elif stripped.startswith('#'):
                        comment_lines += 1
                    else:
                        code_lines += 1
                
                # Calculate cyclomatic complexity
                try:
                    tree = ast.parse(content)
                    complexity = self._calculate_complexity(tree)
                except Exception:
                    complexity = 0
        except Exception:
            pass
        
        return {
            'total_lines': total_lines,
            'code_lines': code_lines,
            'comment_lines': comment_lines,
            'blank_lines': blank_lines,
            'complexity': complexity
        }
    
    async def _analyze_generic_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze generic file metrics"""
        total_lines = 0
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                total_lines = len(lines)
                
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        blank_lines += 1
                    elif stripped.startswith('//') or stripped.startswith('#'):
                        comment_lines += 1
                    else:
                        code_lines += 1
        except Exception:
            pass
        
        return {
            'total_lines': total_lines,
            'code_lines': code_lines,
            'comment_lines': comment_lines,
            'blank_lines': blank_lines,
            'complexity': 0
        }
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity for Python AST"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            # Count decision points
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _calculate_maintainability_index(self, lines: int, complexity: float, comments: int) -> float:
        """
        Calculate maintainability index (0-100)
        Based on Microsoft's maintainability index formula
        """
        import math
        
        if lines == 0:
            return 100.0
        
        # Simplified formula
        volume = lines * math.log2(lines + 1)
        mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(lines)
        
        # Normalize to 0-100
        mi = max(0, min(100, mi * 100 / 171))
        
        return mi



class ArchitectureVisualizer:
    """Generates architecture visualizations"""
    
    async def generate(self, project_path: Path, structure: Dict[str, Any], 
                      dependencies: Dict[str, Any]) -> Dict[str, Any]:
        """Generate architecture visualization"""
        # Generate component relationships
        components = self._identify_components(structure)
        
        # Generate module structure
        modules = self._analyze_module_structure(structure)
        
        # Generate dependency graph
        dep_graph = self._generate_dependency_graph(dependencies)
        
        # Generate Mermaid diagram
        mermaid_diagram = self._generate_mermaid_diagram(components, dep_graph)
        
        return {
            'components': components,
            'modules': modules,
            'dependency_graph': dep_graph,
            'mermaid_diagram': mermaid_diagram,
            'visualization_type': 'mermaid'
        }
    
    def _identify_components(self, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify major components in the project"""
        components = []
        
        # Group files by directory
        dir_groups = defaultdict(list)
        for file_info in structure.get('files', []):
            dir_path = str(Path(file_info['path']).parent)
            dir_groups[dir_path].append(file_info)
        
        # Create components from directories
        for dir_path, files in dir_groups.items():
            if dir_path == '.':
                component_name = 'root'
            else:
                component_name = dir_path.replace('/', '_').replace('\\', '_')
            
            components.append({
                'name': component_name,
                'path': dir_path,
                'file_count': len(files),
                'languages': list(set(f['language'] for f in files))
            })
        
        return components
    
    def _analyze_module_structure(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze module structure"""
        modules = defaultdict(list)
        
        for file_info in structure.get('files', []):
            path_parts = Path(file_info['path']).parts
            if len(path_parts) > 1:
                module = path_parts[0]
                modules[module].append(file_info['path'])
        
        return dict(modules)
    
    def _generate_dependency_graph(self, dependencies: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate simplified dependency graph"""
        graph = {}
        
        internal_deps = dependencies.get('internal_dependencies', {})
        for module, deps in internal_deps.items():
            # Simplify to module level
            module_name = str(Path(module).parts[0]) if Path(module).parts else module
            graph[module_name] = list(set(
                str(Path(d).parts[0]) if Path(d).parts else d 
                for d in deps
            ))
        
        return graph
    
    def _generate_mermaid_diagram(self, components: List[Dict[str, Any]], 
                                 dep_graph: Dict[str, List[str]]) -> str:
        """Generate Mermaid diagram"""
        lines = ['graph TD']
        
        # Add components
        for comp in components[:10]:  # Limit to 10 components
            name = comp['name']
            lines.append(f'    {name}["{name}<br/>{comp["file_count"]} files"]')
        
        # Add dependencies
        for source, targets in list(dep_graph.items())[:20]:  # Limit edges
            for target in targets[:3]:  # Limit targets per source
                if source != target:
                    lines.append(f'    {source} --> {target}')
        
        return '\n'.join(lines)



class TechnicalDebtDetector:
    """Detects technical debt and code smells"""
    
    async def detect(self, project_path: Path, structure: Dict[str, Any], 
                    metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect technical debt"""
        issues = []
        
        # Check for high complexity
        if metrics.get('average_complexity', 0) > 10:
            issues.append({
                'type': 'high_complexity',
                'severity': 'high',
                'message': f"Average complexity is {metrics['average_complexity']:.1f}, consider refactoring",
                'suggestion': 'Break down complex functions into smaller, more manageable pieces'
            })
        
        # Check for low maintainability
        if metrics.get('maintainability_index', 100) < 65:
            issues.append({
                'type': 'low_maintainability',
                'severity': 'high',
                'message': f"Maintainability index is {metrics['maintainability_index']:.1f}",
                'suggestion': 'Improve code structure, add comments, and reduce complexity'
            })
        
        # Check for low comment ratio
        if metrics.get('comment_ratio', 0) < 10:
            issues.append({
                'type': 'low_documentation',
                'severity': 'medium',
                'message': f"Comment ratio is only {metrics['comment_ratio']:.1f}%",
                'suggestion': 'Add more documentation and comments to improve code understanding'
            })
        
        # Detect code smells in files
        code_smells = await self._detect_code_smells(project_path, structure)
        issues.extend(code_smells)
        
        # Detect anti-patterns
        anti_patterns = await self._detect_anti_patterns(project_path, structure)
        issues.extend(anti_patterns)
        
        return {
            'issues': issues,
            'total_issues': len(issues),
            'by_severity': self._group_by_severity(issues),
            'by_type': self._group_by_type(issues)
        }
    
    async def _detect_code_smells(self, project_path: Path, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect code smells"""
        smells = []
        
        for file_info in structure.get('files', []):
            if file_info['language'] != 'Python':
                continue
            
            file_path = project_path / file_info['path']
            
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                    # Check for long functions
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                            if func_lines > 50:
                                smells.append({
                                    'type': 'long_function',
                                    'severity': 'medium',
                                    'file': file_info['path'],
                                    'function': node.name,
                                    'lines': func_lines,
                                    'message': f"Function '{node.name}' is {func_lines} lines long",
                                    'suggestion': 'Consider breaking this function into smaller functions'
                                })
                        
                        # Check for too many parameters
                        if isinstance(node, ast.FunctionDef):
                            param_count = len(node.args.args)
                            if param_count > 5:
                                smells.append({
                                    'type': 'too_many_parameters',
                                    'severity': 'low',
                                    'file': file_info['path'],
                                    'function': node.name,
                                    'parameters': param_count,
                                    'message': f"Function '{node.name}' has {param_count} parameters",
                                    'suggestion': 'Consider using a configuration object or reducing parameters'
                                })
            except Exception:
                pass
        
        return smells
    
    async def _detect_anti_patterns(self, project_path: Path, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anti-patterns"""
        patterns = []
        
        for file_info in structure.get('files', []):
            if file_info['language'] != 'Python':
                continue
            
            file_path = project_path / file_info['path']
            
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                    # Check for bare except
                    if 'except:' in content:
                        patterns.append({
                            'type': 'bare_except',
                            'severity': 'medium',
                            'file': file_info['path'],
                            'message': 'Bare except clause found',
                            'suggestion': 'Specify exception types to catch'
                        })
                    
                    # Check for global variables
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Global):
                            patterns.append({
                                'type': 'global_variable',
                                'severity': 'low',
                                'file': file_info['path'],
                                'message': 'Global variable usage detected',
                                'suggestion': 'Consider using class attributes or function parameters'
                            })
            except Exception:
                pass
        
        return patterns
    
    def _group_by_severity(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group issues by severity"""
        severity_counts = Counter(issue['severity'] for issue in issues)
        return dict(severity_counts)
    
    def _group_by_type(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group issues by type"""
        type_counts = Counter(issue['type'] for issue in issues)
        return dict(type_counts)
