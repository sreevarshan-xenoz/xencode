#!/usr/bin/env python3
"""
Code Analyzers Package

Contains specialized analyzers for different aspects of code analysis:
- SecurityAnalyzer: Security vulnerability detection
- ErrorDetector: Error and bug detection
- PerformanceAnalyzer: Performance issue detection
- RefactoringEngine: Refactoring suggestions
"""

from typing import List, Optional

# Import analyzers with graceful fallback
try:
    from .security_analyzer import SecurityAnalyzer
    SECURITY_ANALYZER_AVAILABLE = True
except ImportError:
    SecurityAnalyzer = None
    SECURITY_ANALYZER_AVAILABLE = False

try:
    from .error_detector import ErrorDetector
    ERROR_DETECTOR_AVAILABLE = True
except ImportError:
    ErrorDetector = None
    ERROR_DETECTOR_AVAILABLE = False

try:
    from .refactoring_engine import RefactoringEngine
    REFACTORING_ENGINE_AVAILABLE = True
except ImportError:
    RefactoringEngine = None
    REFACTORING_ENGINE_AVAILABLE = False


def get_available_analyzers() -> List[str]:
    """Get list of available analyzer names"""
    available = []
    
    if SECURITY_ANALYZER_AVAILABLE:
        available.append("SecurityAnalyzer")
    
    if ERROR_DETECTOR_AVAILABLE:
        available.append("ErrorDetector")
    
    if REFACTORING_ENGINE_AVAILABLE:
        available.append("RefactoringEngine")
    
    return available


def get_analyzer_status() -> dict:
    """Get status of all analyzers"""
    return {
        "security_available": SECURITY_ANALYZER_AVAILABLE,
        "error_detector_available": ERROR_DETECTOR_AVAILABLE,
        "refactoring_available": REFACTORING_ENGINE_AVAILABLE,
        "total_available": len(get_available_analyzers())
    }


__all__ = [
    'SecurityAnalyzer',
    'ErrorDetector',
    'RefactoringEngine',
    'get_available_analyzers',
    'get_analyzer_status',
    'SECURITY_ANALYZER_AVAILABLE',
    'ERROR_DETECTOR_AVAILABLE',
    'REFACTORING_ENGINE_AVAILABLE'
]