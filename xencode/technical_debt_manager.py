#!/usr/bin/env python3
"""
Technical Debt Management System for Xencode

Implements automated technical debt tracking, prioritization, and management
to maintain code quality while scaling rapidly.
"""

import ast
import json
import sqlite3
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class DebtType(Enum):
    """Types of technical debt"""
    CODE_COMPLEXITY = "code_complexity"
    DUPLICATION = "duplication"
    OUTDATED_DEPENDENCIES = "outdated_dependencies"
    MISSING_TESTS = "missing_tests"
    PERFORMANCE_ISSUE = "performance_issue"
    SECURITY_VULNERABILITY = "security_vulnerability"
    DOCUMENTATION_GAP = "documentation_gap"
    ARCHITECTURE_VIOLATION = "architecture_violation"
    TODO_COMMENT = "todo_comment"
    DEPRECATED_API = "deprecated_api"


class DebtSeverity(Enum):
    """Severity levels for technical debt"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class TechnicalDebtItem:
    """Technical debt item data structure"""
    id: str
    debt_type: DebtType
    severity: DebtSeverity
    file_path: str
    line_number: Optional[int]
    description: str
    estimated_effort_hours: float
    business_impact: str
    created_date: datetime
    last_updated: datetime
    resolved: bool = False
    resolution_date: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class DebtMetrics:
    """Technical debt metrics"""
    total_items: int
    total_effort_hours: float
    items_by_severity: Dict[str, int]
    items_by_type: Dict[str, int]
    trend_7_days: float  # Change in debt over last 7 days
    debt_ratio: float  # Debt effort / total codebase size
    resolution_rate: float  # Items resolved per week


class TechnicalDebtDetector:
    """Detects various types of technical debt"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        
    async def detect_code_complexity(self) -> List[TechnicalDebtItem]:
        """Detect overly complex code using cyclomatic complexity"""
        debt_items = []
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                complexity_analyzer = ComplexityAnalyzer()
                complexity_analyzer.visit(tree)
                
                for func_name, complexity, line_no in complexity_analyzer.complexities:
                    if complexity > 10:  # McCabe complexity threshold
                        severity = self._get_complexity_severity(complexity)
                        debt_items.append(TechnicalDebtItem(
                            id=f"complexity_{py_file.name}_{line_no}",
                            debt_type=DebtType.CODE_COMPLEXITY,
                            severity=severity,
                            file_path=str(py_file.relative_to(self.project_root)),
                            line_number=line_no,
                            description=f"Function '{func_name}' has high cyclomatic complexity: {complexity}",
                            estimated_effort_hours=complexity * 0.5,  # Rough estimate
                            business_impact="Reduced maintainability and increased bug risk",
                            created_date=datetime.now(),
                            last_updated=datetime.now(),
                            metadata={"complexity_score": complexity, "function_name": func_name}
                        ))
            except Exception as e:
                logger.warning(f"Failed to analyze complexity for {py_file}: {e}")
        
        return debt_items
    
    async def detect_code_duplication(self) -> List[TechnicalDebtItem]:
        """Detect code duplication"""
        debt_items = []
        
        # Simple duplication detection based on similar function signatures
        functions = {}
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                duplication_analyzer = DuplicationAnalyzer()
                duplication_analyzer.visit(tree)
                
                for func_signature, line_no in duplication_analyzer.functions:
                    if func_signature in functions:
                        # Found potential duplication
                        original_file, original_line = functions[func_signature]
                        debt_items.append(TechnicalDebtItem(
                            id=f"duplication_{py_file.name}_{line_no}",
                            debt_type=DebtType.DUPLICATION,
                            severity=DebtSeverity.MEDIUM,
                            file_path=str(py_file.relative_to(self.project_root)),
                            line_number=line_no,
                            description=f"Potential code duplication with {original_file}:{original_line}",
                            estimated_effort_hours=2.0,
                            business_impact="Increased maintenance burden and inconsistency risk",
                            created_date=datetime.now(),
                            last_updated=datetime.now(),
                            metadata={"duplicate_of": f"{original_file}:{original_line}"}
                        ))
                    else:
                        functions[func_signature] = (str(py_file.relative_to(self.project_root)), line_no)
                        
            except Exception as e:
                logger.warning(f"Failed to analyze duplication for {py_file}: {e}")
        
        return debt_items
    
    async def detect_todo_comments(self) -> List[TechnicalDebtItem]:
        """Detect TODO, FIXME, and HACK comments"""
        debt_items = []
        todo_patterns = [
            (r'#\s*TODO[:\s]*(.*)', DebtSeverity.MEDIUM),
            (r'#\s*FIXME[:\s]*(.*)', DebtSeverity.HIGH),
            (r'#\s*HACK[:\s]*(.*)', DebtSeverity.HIGH),
            (r'#\s*XXX[:\s]*(.*)', DebtSeverity.MEDIUM),
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line_no, line in enumerate(lines, 1):
                    for pattern, severity in todo_patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            comment = match.group(1).strip() if match.group(1) else "No description"
                            debt_items.append(TechnicalDebtItem(
                                id=f"todo_{py_file.name}_{line_no}",
                                debt_type=DebtType.TODO_COMMENT,
                                severity=severity,
                                file_path=str(py_file.relative_to(self.project_root)),
                                line_number=line_no,
                                description=f"TODO comment: {comment}",
                                estimated_effort_hours=1.0,
                                business_impact="Incomplete functionality or known issues",
                                created_date=datetime.now(),
                                last_updated=datetime.now(),
                                metadata={"comment_text": line.strip()}
                            ))
                            
            except Exception as e:
                logger.warning(f"Failed to analyze TODOs for {py_file}: {e}")
        
        return debt_items
    
    async def detect_missing_tests(self) -> List[TechnicalDebtItem]:
        """Detect files without corresponding test files"""
        debt_items = []
        
        source_files = set()
        test_files = set()
        
        # Collect source files
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            if not py_file.name.startswith("test_"):
                source_files.add(py_file)
        
        # Collect test files
        for py_file in self.project_root.rglob("test_*.py"):
            test_files.add(py_file.name[5:])  # Remove "test_" prefix
        
        # Find source files without tests
        for source_file in source_files:
            expected_test = f"test_{source_file.name}"
            if expected_test not in [tf.name for tf in test_files]:
                debt_items.append(TechnicalDebtItem(
                    id=f"missing_test_{source_file.name}",
                    debt_type=DebtType.MISSING_TESTS,
                    severity=DebtSeverity.MEDIUM,
                    file_path=str(source_file.relative_to(self.project_root)),
                    line_number=None,
                    description=f"No test file found for {source_file.name}",
                    estimated_effort_hours=4.0,
                    business_impact="Reduced confidence in code changes and higher bug risk",
                    created_date=datetime.now(),
                    last_updated=datetime.now(),
                    metadata={"expected_test_file": expected_test}
                ))
        
        return debt_items
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during analysis"""
        skip_patterns = [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            ".pytest_cache",
            "htmlcov"
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _get_complexity_severity(self, complexity: int) -> DebtSeverity:
        """Get severity based on complexity score"""
        if complexity > 20:
            return DebtSeverity.CRITICAL
        elif complexity > 15:
            return DebtSeverity.HIGH
        elif complexity > 10:
            return DebtSeverity.MEDIUM
        else:
            return DebtSeverity.LOW


class ComplexityAnalyzer(ast.NodeVisitor):
    """AST visitor to calculate cyclomatic complexity"""
    
    def __init__(self):
        self.complexities = []
        self.current_complexity = 0
        self.current_function = None
        self.current_line = None
    
    def visit_FunctionDef(self, node):
        old_complexity = self.current_complexity
        old_function = self.current_function
        old_line = self.current_line
        
        self.current_complexity = 1  # Base complexity
        self.current_function = node.name
        self.current_line = node.lineno
        
        self.generic_visit(node)
        
        self.complexities.append((self.current_function, self.current_complexity, self.current_line))
        
        self.current_complexity = old_complexity
        self.current_function = old_function
        self.current_line = old_line
    
    def visit_If(self, node):
        self.current_complexity += 1
        self.generic_visit(node)
    
    def visit_While(self, node):
        self.current_complexity += 1
        self.generic_visit(node)
    
    def visit_For(self, node):
        self.current_complexity += 1
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node):
        self.current_complexity += 1
        self.generic_visit(node)


class DuplicationAnalyzer(ast.NodeVisitor):
    """AST visitor to detect potential code duplication"""
    
    def __init__(self):
        self.functions = []
    
    def visit_FunctionDef(self, node):
        # Create a simple signature based on function structure
        signature = self._create_signature(node)
        self.functions.append((signature, node.lineno))
        self.generic_visit(node)
    
    def _create_signature(self, node):
        """Create a simple signature for duplication detection"""
        # This is a simplified approach - in practice, you'd want more sophisticated analysis
        arg_count = len(node.args.args)
        body_length = len(node.body)
        return f"{arg_count}_{body_length}"


class TechnicalDebtManager:
    """Manages technical debt tracking and reporting"""
    
    def __init__(self, project_root: Path, db_path: Optional[Path] = None):
        self.project_root = project_root
        self.db_path = db_path or project_root / ".xencode" / "technical_debt.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.detector = TechnicalDebtDetector(project_root)
        self._init_database()
    
    def _init_database(self):
        """Initialize the technical debt database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS debt_items (
                    id TEXT PRIMARY KEY,
                    debt_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    line_number INTEGER,
                    description TEXT NOT NULL,
                    estimated_effort_hours REAL NOT NULL,
                    business_impact TEXT NOT NULL,
                    created_date TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_date TEXT,
                    resolution_notes TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS debt_scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_date TEXT NOT NULL,
                    total_items INTEGER NOT NULL,
                    total_effort_hours REAL NOT NULL,
                    scan_duration_seconds REAL NOT NULL
                )
            """)
            
            conn.commit()
    
    async def run_full_scan(self) -> DebtMetrics:
        """Run a comprehensive technical debt scan"""
        start_time = time.time()
        logger.info("Starting technical debt scan...")
        
        all_debt_items = []
        
        # Run all detectors
        detectors = [
            self.detector.detect_code_complexity(),
            self.detector.detect_code_duplication(),
            self.detector.detect_todo_comments(),
            self.detector.detect_missing_tests(),
        ]
        
        for detector_coro in detectors:
            try:
                items = await detector_coro
                all_debt_items.extend(items)
            except Exception as e:
                logger.error(f"Detector failed: {e}")
        
        # Store results
        await self._store_debt_items(all_debt_items)
        
        # Calculate metrics
        metrics = await self.get_debt_metrics()
        
        # Record scan
        scan_duration = time.time() - start_time
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO debt_scans (scan_date, total_items, total_effort_hours, scan_duration_seconds)
                VALUES (?, ?, ?, ?)
            """, (datetime.now().isoformat(), metrics.total_items, metrics.total_effort_hours, scan_duration))
            conn.commit()
        
        logger.info(f"Technical debt scan completed in {scan_duration:.2f}s. Found {metrics.total_items} items.")
        return metrics
    
    async def _store_debt_items(self, debt_items: List[TechnicalDebtItem]):
        """Store debt items in database"""
        with sqlite3.connect(self.db_path) as conn:
            # Clear existing unresolved items (they'll be re-detected if still present)
            conn.execute("DELETE FROM debt_items WHERE resolved = FALSE")
            
            for item in debt_items:
                conn.execute("""
                    INSERT OR REPLACE INTO debt_items 
                    (id, debt_type, severity, file_path, line_number, description, 
                     estimated_effort_hours, business_impact, created_date, last_updated, 
                     resolved, resolution_date, resolution_notes, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item.id,
                    item.debt_type.value,
                    item.severity.value,
                    item.file_path,
                    item.line_number,
                    item.description,
                    item.estimated_effort_hours,
                    item.business_impact,
                    item.created_date.isoformat(),
                    item.last_updated.isoformat(),
                    item.resolved,
                    item.resolution_date.isoformat() if item.resolution_date else None,
                    item.resolution_notes,
                    json.dumps(item.metadata) if item.metadata else None
                ))
            
            conn.commit()
    
    async def get_debt_metrics(self) -> DebtMetrics:
        """Calculate current technical debt metrics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total items and effort
            total_data = conn.execute("""
                SELECT COUNT(*), SUM(estimated_effort_hours) 
                FROM debt_items WHERE resolved = FALSE
            """).fetchone()
            
            total_items = total_data[0] or 0
            total_effort = total_data[1] or 0.0
            
            # Items by severity
            severity_data = conn.execute("""
                SELECT severity, COUNT(*) FROM debt_items 
                WHERE resolved = FALSE GROUP BY severity
            """).fetchall()
            items_by_severity = {row[0]: row[1] for row in severity_data}
            
            # Items by type
            type_data = conn.execute("""
                SELECT debt_type, COUNT(*) FROM debt_items 
                WHERE resolved = FALSE GROUP BY debt_type
            """).fetchall()
            items_by_type = {row[0]: row[1] for row in type_data}
            
            # 7-day trend
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            trend_data = conn.execute("""
                SELECT COUNT(*) FROM debt_items 
                WHERE created_date > ? AND resolved = FALSE
            """, (week_ago,)).fetchone()
            trend_7_days = trend_data[0] or 0
            
            # Resolution rate (items resolved per week)
            resolved_data = conn.execute("""
                SELECT COUNT(*) FROM debt_items 
                WHERE resolved = TRUE AND resolution_date > ?
            """, (week_ago,)).fetchone()
            resolution_rate = resolved_data[0] or 0
            
            # Debt ratio (simplified - debt effort per 1000 lines of code)
            total_lines = await self._count_total_lines()
            debt_ratio = (total_effort / max(total_lines / 1000, 1)) if total_lines > 0 else 0
            
            return DebtMetrics(
                total_items=total_items,
                total_effort_hours=total_effort,
                items_by_severity=items_by_severity,
                items_by_type=items_by_type,
                trend_7_days=trend_7_days,
                debt_ratio=debt_ratio,
                resolution_rate=resolution_rate
            )
    
    async def _count_total_lines(self) -> int:
        """Count total lines of code in the project"""
        total_lines = 0
        for py_file in self.project_root.rglob("*.py"):
            if self.detector._should_skip_file(py_file):
                continue
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            except Exception:
                continue
        return total_lines
    
    async def resolve_debt_item(self, item_id: str, resolution_notes: str):
        """Mark a debt item as resolved"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE debt_items 
                SET resolved = TRUE, resolution_date = ?, resolution_notes = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), resolution_notes, item_id))
            conn.commit()
    
    async def get_prioritized_debt_items(self, limit: int = 20) -> List[TechnicalDebtItem]:
        """Get prioritized list of debt items for resolution"""
        with sqlite3.connect(self.db_path) as conn:
            # Priority scoring: Critical=4, High=3, Medium=2, Low=1
            priority_scores = {
                DebtSeverity.CRITICAL.value: 4,
                DebtSeverity.HIGH.value: 3,
                DebtSeverity.MEDIUM.value: 2,
                DebtSeverity.LOW.value: 1,
                DebtSeverity.INFO.value: 0
            }
            
            rows = conn.execute("""
                SELECT * FROM debt_items 
                WHERE resolved = FALSE 
                ORDER BY 
                    CASE severity
                        WHEN 'critical' THEN 4
                        WHEN 'high' THEN 3
                        WHEN 'medium' THEN 2
                        WHEN 'low' THEN 1
                        ELSE 0
                    END DESC,
                    estimated_effort_hours ASC
                LIMIT ?
            """, (limit,)).fetchall()
            
            items = []
            for row in rows:
                items.append(TechnicalDebtItem(
                    id=row[0],
                    debt_type=DebtType(row[1]),
                    severity=DebtSeverity(row[2]),
                    file_path=row[3],
                    line_number=row[4],
                    description=row[5],
                    estimated_effort_hours=row[6],
                    business_impact=row[7],
                    created_date=datetime.fromisoformat(row[8]),
                    last_updated=datetime.fromisoformat(row[9]),
                    resolved=row[10],
                    resolution_date=datetime.fromisoformat(row[11]) if row[11] else None,
                    resolution_notes=row[12],
                    metadata=json.loads(row[13]) if row[13] else None
                ))
            
            return items


# Global debt manager instance
_debt_manager: Optional[TechnicalDebtManager] = None


def get_debt_manager(project_root: Optional[Path] = None) -> TechnicalDebtManager:
    """Get the global debt manager instance"""
    global _debt_manager
    if _debt_manager is None:
        root = project_root or Path.cwd()
        _debt_manager = TechnicalDebtManager(root)
    return _debt_manager