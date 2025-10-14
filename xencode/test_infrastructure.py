#!/usr/bin/env python3
"""
Enhanced Test Infrastructure for Xencode

Provides comprehensive test framework with dependency resolution,
proper isolation, mocking, and coverage reporting.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import Mock, patch

import pytest
import coverage


@dataclass
class TestResults:
    """Test execution results"""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)
    coverage_percentage: float = 0.0
    execution_time: float = 0.0
    failed_tests: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'total_tests': self.total_tests,
            'passed': self.passed,
            'failed': self.failed,
            'skipped': self.skipped,
            'errors': self.errors,
            'coverage_percentage': self.coverage_percentage,
            'execution_time': self.execution_time,
            'failed_tests': self.failed_tests,
            'success_rate': self.success_rate
        }


@dataclass
class IntegrationIssue:
    """Integration issue detected during testing"""
    component: str
    issue_type: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    suggested_fix: Optional[str] = None


class DependencyResolver:
    """Resolves pytest and integration dependencies"""
    
    def __init__(self):
        self.resolved_conflicts: List[str] = []
        self.missing_dependencies: List[str] = []
    
    async def resolve_dependencies(self) -> bool:
        """Resolve all pytest and integration dependencies"""
        try:
            # Check for common pytest conflicts
            conflicts = await self._detect_conflicts()
            if conflicts:
                await self._resolve_conflicts(conflicts)
            
            # Check for missing dependencies
            missing = await self._detect_missing_dependencies()
            if missing:
                await self._install_missing_dependencies(missing)
            
            return len(self.resolved_conflicts) == 0 and len(self.missing_dependencies) == 0
            
        except Exception as e:
            print(f"Error resolving dependencies: {e}")
            return False
    
    async def _detect_conflicts(self) -> List[str]:
        """Detect version conflicts in pytest ecosystem"""
        conflicts = []
        
        try:
            # Run pip check to detect conflicts
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'check'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                # Parse conflicts from output
                for line in result.stdout.split('\n'):
                    if 'pytest' in line.lower() or 'test' in line.lower():
                        conflicts.append(line.strip())
                        
        except subprocess.TimeoutExpired:
            conflicts.append("Dependency check timed out")
        except Exception as e:
            conflicts.append(f"Error checking dependencies: {e}")
        
        return conflicts
    
    async def _resolve_conflicts(self, conflicts: List[str]) -> None:
        """Resolve detected conflicts"""
        for conflict in conflicts:
            try:
                # Basic conflict resolution strategies
                if 'pytest' in conflict:
                    # Upgrade pytest to latest compatible version
                    subprocess.run([
                        sys.executable, '-m', 'pip', 'install', 
                        '--upgrade', 'pytest>=7.0.0'
                    ], check=True)
                    self.resolved_conflicts.append(f"Upgraded pytest: {conflict}")
                    
            except subprocess.CalledProcessError as e:
                self.resolved_conflicts.append(f"Failed to resolve: {conflict} - {e}")
    
    async def _detect_missing_dependencies(self) -> List[str]:
        """Detect missing test dependencies"""
        required_packages = [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0', 
            'pytest-mock>=3.10.0',
            'pytest-asyncio>=0.21.0',
            'coverage>=7.0.0'
        ]
        
        missing = []
        for package in required_packages:
            try:
                # Try importing the package
                package_name = package.split('>=')[0].replace('-', '_')
                __import__(package_name)
            except ImportError:
                missing.append(package)
        
        return missing
    
    async def _install_missing_dependencies(self, missing: List[str]) -> None:
        """Install missing dependencies"""
        for package in missing:
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], check=True)
                print(f"Installed missing dependency: {package}")
            except subprocess.CalledProcessError as e:
                self.missing_dependencies.append(f"Failed to install {package}: {e}")


class CoverageTracker:
    """Tracks and reports test coverage metrics"""
    
    def __init__(self, source_dirs: Optional[List[str]] = None):
        self.source_dirs = source_dirs or ['xencode', 'xencode_core']
        self.coverage_data: Optional[coverage.Coverage] = None
        
    def start_coverage(self) -> None:
        """Start coverage tracking"""
        self.coverage_data = coverage.Coverage(
            source=self.source_dirs,
            omit=[
                '*/tests/*',
                '*/venv/*', 
                '*/__pycache__/*',
                '*/.*',
                'setup.py',
                'scripts/*'
            ]
        )
        self.coverage_data.start()
    
    def stop_coverage(self) -> float:
        """Stop coverage tracking and return percentage"""
        if not self.coverage_data:
            return 0.0
            
        self.coverage_data.stop()
        self.coverage_data.save()
        
        # Generate coverage report
        return self.coverage_data.report()
    
    def generate_html_report(self, output_dir: str = 'htmlcov') -> None:
        """Generate HTML coverage report"""
        if self.coverage_data:
            self.coverage_data.html_report(directory=output_dir)
    
    def generate_xml_report(self, output_file: str = 'coverage.xml') -> None:
        """Generate XML coverage report for CI"""
        if self.coverage_data:
            self.coverage_data.xml_report(outfile=output_file)


class EnhancedTestRunner:
    """Enhanced test runner with proper isolation and reporting"""
    
    def __init__(self, 
                 test_dirs: Optional[List[str]] = None,
                 coverage_tracker: Optional[CoverageTracker] = None):
        self.test_dirs = test_dirs or ['tests']
        self.coverage_tracker = coverage_tracker or CoverageTracker()
        self.mock_registry: Dict[str, Mock] = {}
        
    async def run_tests(self, 
                       test_pattern: Optional[str] = None,
                       parallel: bool = True,
                       verbose: bool = True) -> TestResults:
        """Run tests with enhanced features"""
        
        start_time = time.time()
        results = TestResults()
        
        try:
            # Start coverage tracking
            self.coverage_tracker.start_coverage()
            
            # Build pytest command
            cmd = [sys.executable, '-m', 'pytest']
            
            # Add test directories
            for test_dir in self.test_dirs:
                if Path(test_dir).exists():
                    cmd.append(test_dir)
            
            # Add options
            if verbose:
                cmd.append('-v')
            
            if parallel:
                cmd.extend(['-n', 'auto'])  # Requires pytest-xdist
            
            if test_pattern:
                cmd.extend(['-k', test_pattern])
            
            # Add coverage options
            cmd.extend([
                '--cov=xencode',
                '--cov=xencode_core', 
                '--cov-report=term-missing',
                '--cov-report=html:htmlcov',
                '--cov-report=xml',
                '--tb=short'
            ])
            
            # Run tests
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Parse results
            results = self._parse_pytest_output(result.stdout, result.stderr)
            results.execution_time = time.time() - start_time
            
            # Stop coverage and get percentage
            coverage_pct = self.coverage_tracker.stop_coverage()
            results.coverage_percentage = coverage_pct
            
            # Generate coverage reports
            self.coverage_tracker.generate_html_report()
            self.coverage_tracker.generate_xml_report()
            
        except subprocess.TimeoutExpired:
            results.errors.append("Test execution timed out")
        except Exception as e:
            results.errors.append(f"Test execution failed: {e}")
        
        return results
    
    def _parse_pytest_output(self, stdout: str, stderr: str) -> TestResults:
        """Parse pytest output to extract results"""
        results = TestResults()
        
        # Parse the summary line - look for patterns like "1 failed, 9 passed in 2.5s"
        for line in stdout.split('\n'):
            line = line.strip()
            if ('passed' in line or 'failed' in line or 'skipped' in line) and ' in ' in line:
                # Split by commas and spaces to parse counts
                parts = line.replace(',', '').split()
                
                i = 0
                while i < len(parts) - 1:
                    try:
                        count = int(parts[i])
                        status = parts[i + 1]
                        
                        if status == 'passed':
                            results.passed = count
                        elif status == 'failed':
                            results.failed = count
                        elif status == 'skipped':
                            results.skipped = count
                        elif status == 'error' or status == 'errors':
                            results.failed += count  # Treat errors as failures
                            
                    except (ValueError, IndexError):
                        pass
                    i += 1
                
                # If we found any counts, break
                if results.passed > 0 or results.failed > 0 or results.skipped > 0:
                    break
        
        results.total_tests = results.passed + results.failed + results.skipped
        
        # Extract failed test names
        in_failures = False
        for line in stdout.split('\n'):
            if 'FAILURES' in line:
                in_failures = True
            elif in_failures and line.startswith('FAILED'):
                test_name = line.split('::')[0].replace('FAILED ', '')
                results.failed_tests.append(test_name)
        
        # Add stderr as errors if present
        if stderr.strip():
            results.errors.append(stderr.strip())
        
        return results
    
    def setup_test_isolation(self) -> None:
        """Set up test isolation and mocking"""
        # Mock external services that might be imported
        mock_targets = [
            'requests.get',
            'requests.post', 
            'subprocess.run',
            'os.system'
        ]
        
        for target in mock_targets:
            try:
                mock_obj = Mock()
                self.mock_registry[target] = patch(target, mock_obj).start()
            except Exception:
                # Skip if target doesn't exist
                pass
    
    def teardown_test_isolation(self) -> None:
        """Clean up test isolation"""
        # Stop all patches
        patch.stopall()
        self.mock_registry.clear()


class TestInfrastructure:
    """Main test infrastructure coordinator"""
    
    def __init__(self):
        self.dependency_resolver = DependencyResolver()
        self.test_runner = EnhancedTestRunner()
        self.coverage_tracker = CoverageTracker()
        
    async def resolve_dependencies(self) -> bool:
        """Resolve all pytest and integration dependencies"""
        return await self.dependency_resolver.resolve_dependencies()
    
    async def run_comprehensive_tests(self, 
                                    test_pattern: Optional[str] = None) -> TestResults:
        """Run full test suite with coverage reporting"""
        
        # Ensure dependencies are resolved
        deps_resolved = await self.resolve_dependencies()
        if not deps_resolved:
            results = TestResults()
            results.errors.append("Failed to resolve dependencies")
            return results
        
        # Set up test isolation
        self.test_runner.setup_test_isolation()
        
        try:
            # Run tests
            results = await self.test_runner.run_tests(
                test_pattern=test_pattern,
                parallel=True,
                verbose=True
            )
            
            return results
            
        finally:
            # Clean up
            self.test_runner.teardown_test_isolation()
    
    async def validate_integration_points(self) -> List[IntegrationIssue]:
        """Validate all system integration points"""
        issues = []
        
        # Check core component integrations
        integration_checks = [
            ('enhanced_cli_system', 'xencode.enhanced_cli_system'),
            ('context_cache_manager', 'xencode.context_cache_manager'),
            ('model_stability_manager', 'xencode.model_stability_manager'),
            ('security_manager', 'xencode.security_manager'),
            ('resource_monitor', 'xencode.resource_monitor')
        ]
        
        for component_name, module_path in integration_checks:
            try:
                # Try importing the module
                __import__(module_path)
            except ImportError as e:
                issues.append(IntegrationIssue(
                    component=component_name,
                    issue_type='import_error',
                    description=f"Failed to import {module_path}: {e}",
                    severity='high',
                    suggested_fix=f"Check if {module_path} exists and has correct dependencies"
                ))
            except Exception as e:
                issues.append(IntegrationIssue(
                    component=component_name,
                    issue_type='initialization_error', 
                    description=f"Error initializing {module_path}: {e}",
                    severity='medium',
                    suggested_fix=f"Check {module_path} initialization code"
                ))
        
        return issues
    
    def generate_test_report(self, results: TestResults, output_file: str = 'test_report.json') -> None:
        """Generate comprehensive test report"""
        report = {
            'timestamp': time.time(),
            'results': results.to_dict(),
            'dependency_status': {
                'resolved_conflicts': self.dependency_resolver.resolved_conflicts,
                'missing_dependencies': self.dependency_resolver.missing_dependencies
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Test report generated: {output_file}")


# Global instance for easy access
test_infrastructure = TestInfrastructure()


async def main():
    """Main entry point for running enhanced tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Test Infrastructure')
    parser.add_argument('--pattern', '-k', help='Test pattern to match')
    parser.add_argument('--validate', action='store_true', help='Validate integration points')
    parser.add_argument('--report', default='test_report.json', help='Output report file')
    
    args = parser.parse_args()
    
    if args.validate:
        print("Validating integration points...")
        issues = await test_infrastructure.validate_integration_points()
        
        if issues:
            print(f"Found {len(issues)} integration issues:")
            for issue in issues:
                print(f"  - {issue.component}: {issue.description}")
        else:
            print("All integration points validated successfully!")
        return
    
    print("Running comprehensive tests...")
    results = await test_infrastructure.run_comprehensive_tests(args.pattern)
    
    print(f"\nTest Results:")
    print(f"  Total: {results.total_tests}")
    print(f"  Passed: {results.passed}")
    print(f"  Failed: {results.failed}")
    print(f"  Skipped: {results.skipped}")
    print(f"  Success Rate: {results.success_rate:.1f}%")
    print(f"  Coverage: {results.coverage_percentage:.1f}%")
    print(f"  Execution Time: {results.execution_time:.2f}s")
    
    if results.errors:
        print(f"\nErrors:")
        for error in results.errors:
            print(f"  - {error}")
    
    if results.failed_tests:
        print(f"\nFailed Tests:")
        for test in results.failed_tests:
            print(f"  - {test}")
    
    # Generate report
    test_infrastructure.generate_test_report(results, args.report)


if __name__ == '__main__':
    asyncio.run(main())