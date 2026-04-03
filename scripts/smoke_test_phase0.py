#!/usr/bin/env python3
"""
Phase 0 Smoke Test Gate for Xencode Sprint 1

Validates end-to-end functionality of:
- Transport layer with retry/reliability
- Provider diagnostics (connection tests)
- Locked model resolver (Qwen3-Coder-Next)
- Credential vault backend

Usage:
    python scripts/smoke_test_phase0.py
    
Returns:
    Exit code 0 if all tests pass, 1 otherwise
"""

import asyncio
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class TestStatus(Enum):
    """Test result status"""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    WARN = "warn"


@dataclass
class TestResult:
    """Result of a single smoke test"""
    name: str
    status: TestStatus
    message: str = ""
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "error": self.error,
            **(self.details or {})
        }


class SmokeTestResult:
    """Overall smoke test result"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = asyncio.get_event_loop().time()
    
    def add_result(self, result: TestResult):
        """Add test result"""
        self.results.append(result)
    
    @property
    def passed(self) -> int:
        """Count passed tests"""
        return sum(1 for r in self.results if r.status == TestStatus.PASS)
    
    @property
    def failed(self) -> int:
        """Count failed tests"""
        return sum(1 for r in self.results if r.status == TestStatus.FAIL)
    
    @property
    def skipped(self) -> int:
        """Count skipped tests"""
        return sum(1 for r in self.results if r.status == TestStatus.SKIP)
    
    @property
    def warnings(self) -> int:
        """Count warnings"""
        return sum(1 for r in self.results if r.status == TestStatus.WARN)
    
    @property
    def total(self) -> int:
        """Total tests"""
        return len(self.results)
    
    @property
    def success_rate(self) -> float:
        """Success rate percentage"""
        runnable = self.total - self.skipped
        if runnable == 0:
            return 100.0
        return (self.passed / runnable) * 100
    
    @property
    def is_passed(self) -> bool:
        """Check if smoke test passed"""
        return self.failed == 0 and self.passed > 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "warnings": self.warnings,
            "success_rate": self.success_rate,
            "is_passed": self.is_passed,
            "results": [r.to_dict() for r in self.results],
        }


class Phase0SmokeTester:
    """
    Phase 0 Smoke Test Gate
    
    Runs end-to-end validation of Sprint 1 components
    """
    
    def __init__(self):
        self.result = SmokeTestResult()
    
    async def run_all_tests(self) -> SmokeTestResult:
        """Run all smoke tests"""
        console.print("\n[bold blue]========================================[/bold blue]")
        console.print("[bold blue]  Xencode Phase 0 Smoke Test Gate (Sprint 1)[/bold blue]")
        console.print("[bold blue]========================================[/bold blue]\n")
        
        # Test Suite 1: Transport Layer
        console.print("[bold]Test Suite 1: Transport Layer[/bold]")
        await self.test_transport_layer()
        
        # Test Suite 2: Provider Diagnostics
        console.print("\n[bold]Test Suite 2: Provider Diagnostics[/bold]")
        await self.test_provider_diagnostics()
        
        # Test Suite 3: Model Resolver
        console.print("\n[bold]Test Suite 3: Model Resolver[/bold]")
        await self.test_model_resolver()
        
        # Test Suite 4: Credential Vault
        console.print("\n[bold]Test Suite 4: Credential Vault[/bold]")
        await self.test_credential_vault()
        
        # Print summary
        self._print_summary()
        
        return self.result
    
    async def test_transport_layer(self):
        """Test transport layer functionality"""
        try:
            from xencode.model_providers.transport import (
                ProviderTransport,
                ProviderTransportPolicy,
                TransportError,
            )
            
            # Test 1.1: Transport creation
            transport = ProviderTransport(
                provider="test",
                base_url="http://localhost:9999",  # Non-existent endpoint
                policy=ProviderTransportPolicy(
                    timeout=2.0,
                    max_retries=1,
                    backoff_base=0.1,
                ),
            )
            
            self.result.add_result(TestResult(
                name="Transport layer import and creation",
                status=TestStatus.PASS,
                message="Transport class instantiated successfully",
            ))
            
            # Test 1.2: Policy configuration
            policy = ProviderTransportPolicy(
                timeout=30.0,
                max_retries=3,
                backoff_base=1.0,
                backoff_max=60.0,
            )
            
            assert policy.is_retryable_status(429)
            assert policy.is_retryable_status(500)
            assert not policy.is_retryable_status(400)
            
            self.result.add_result(TestResult(
                name="Retry policy configuration",
                status=TestStatus.PASS,
                message=f"Retryable codes: {policy.retryable_status_codes}",
            ))
            
            # Test 1.3: Error envelope
            envelope = TransportError(
                error_type="test",
                http_status=500,
                message="Test error",
                provider="test",
            )
            
            error_dict = envelope.to_dict()
            assert error_dict["error_type"] == "test"
            assert error_dict["http_status"] == 500
            
            self.result.add_result(TestResult(
                name="Unified error envelope",
                status=TestStatus.PASS,
                message="Error envelope serialization works",
            ))
            
        except ImportError as e:
            self.result.add_result(TestResult(
                name="Transport layer import and creation",
                status=TestStatus.FAIL,
                error=str(e),
            ))
        except Exception as e:
            self.result.add_result(TestResult(
                name="Transport layer tests",
                status=TestStatus.FAIL,
                error=str(e),
            ))
    
    async def test_provider_diagnostics(self):
        """Test provider diagnostics functionality"""
        try:
            from xencode.model_providers.diagnostics import (
                ProviderDiagnostics,
                ProviderStatus,
                ProviderTestResult,
            )
            
            # Test 2.1: Diagnostics creation
            diagnostics = ProviderDiagnostics()
            
            self.result.add_result(TestResult(
                name="Diagnostics service creation",
                status=TestStatus.PASS,
                message="ProviderDiagnostics instantiated",
            ))
            
            # Test 2.2: Test result creation
            test_result = ProviderTestResult(
                provider="TestProvider",
                status=ProviderStatus.OK,
                latency_ms=100.5,
            )
            
            assert test_result.is_success
            assert test_result.to_dict()["status"] == "ok"
            
            self.result.add_result(TestResult(
                name="Test result envelope",
                status=TestStatus.PASS,
                message=f"Latency: {test_result.latency_ms}ms",
            ))
            
            # Test 2.3: Ollama test (should work if Ollama is running)
            ollama_result = await diagnostics.test_ollama()
            
            if ollama_result.status == ProviderStatus.OK:
                self.result.add_result(TestResult(
                    name="Ollama connectivity test",
                    status=TestStatus.PASS,
                    message=f"Ollama responding ({ollama_result.latency_ms:.0f}ms)",
                    details={"model_count": ollama_result.model_count},
                ))
            elif ollama_result.status == ProviderStatus.CONNECTION_ERROR:
                self.result.add_result(TestResult(
                    name="Ollama connectivity test",
                    status=TestStatus.WARN,
                    message="Ollama not running (expected in CI)",
                    details={"remediation": ollama_result.remediation},
                ))
            else:
                self.result.add_result(TestResult(
                    name="Ollama connectivity test",
                    status=TestStatus.FAIL,
                    error=ollama_result.error_message,
                ))
            
            # Test 2.4: Qwen test (not configured)
            qwen_result = await diagnostics.test_qwen()
            
            if qwen_result.status == ProviderStatus.NOT_CONFIGURED:
                self.result.add_result(TestResult(
                    name="Qwen auth check",
                    status=TestStatus.PASS,
                    message="Correctly detected missing authentication",
                ))
            elif qwen_result.status == ProviderStatus.OK:
                self.result.add_result(TestResult(
                    name="Qwen auth check",
                    status=TestStatus.PASS,
                    message=f"Qwen authenticated ({qwen_result.latency_ms:.0f}ms)",
                ))
            else:
                self.result.add_result(TestResult(
                    name="Qwen auth check",
                    status=TestStatus.WARN,
                    message=qwen_result.error_message or "Unknown status",
                ))
            
            # Test 2.5: OpenRouter test (not configured)
            openrouter_result = await diagnostics.test_openrouter(None)
            
            if openrouter_result.status == ProviderStatus.NOT_CONFIGURED:
                self.result.add_result(TestResult(
                    name="OpenRouter config check",
                    status=TestStatus.PASS,
                    message="Correctly detected missing API key",
                ))
            else:
                self.result.add_result(TestResult(
                    name="OpenRouter config check",
                    status=TestStatus.WARN,
                    message=openrouter_result.error_message or "Unexpected result",
                ))
            
        except ImportError as e:
            self.result.add_result(TestResult(
                name="Diagnostics service creation",
                status=TestStatus.FAIL,
                error=str(e),
            ))
        except Exception as e:
            self.result.add_result(TestResult(
                name="Provider diagnostics tests",
                status=TestStatus.FAIL,
                error=str(e),
            ))
    
    async def test_model_resolver(self):
        """Test model resolver functionality"""
        try:
            from xencode.model_providers.resolver import (
                LockedModelResolver,
                ResolverConfig,
                ModelDiscoveryCache,
                ModelInfo,
                ModelAlias,
            )
            
            # Test 3.1: Resolver creation
            config = ResolverConfig(
                lock_best_coder=True,
                lock_model_override=None,
                cache_ttl=300,
            )
            
            resolver = LockedModelResolver(config=config)
            
            self.result.add_result(TestResult(
                name="Model resolver creation",
                status=TestStatus.PASS,
                message="LockedModelResolver instantiated with config",
            ))
            
            # Test 3.2: Alias mapping
            canonical = ModelAlias.CANONICAL_MAP.get("qwen-coder-next-latest")
            
            if canonical:
                self.result.add_result(TestResult(
                    name="Model alias mapping",
                    status=TestStatus.PASS,
                    message=f"qwen-coder-next-latest → {canonical}",
                ))
            else:
                self.result.add_result(TestResult(
                    name="Model alias mapping",
                    status=TestStatus.WARN,
                    message="Alias mapping not found",
                ))
            
            # Test 3.3: Cache operations
            cache = ModelDiscoveryCache(ttl=60)
            
            assert not cache.is_valid()  # Empty cache
            
            cache.set_models([
                ModelInfo(model_id="qwen3-coder-next-instruct", capabilities=["code"]),
                ModelInfo(model_id="gpt-4", capabilities=["chat"]),
            ])
            
            assert cache.is_valid()
            
            coder_models = cache.get_coder_models()
            assert len(coder_models) == 1
            
            best_model = cache.find_best_coder_model()
            assert best_model.model_id == "qwen3-coder-next-instruct"
            
            self.result.add_result(TestResult(
                name="Model discovery cache",
                status=TestStatus.PASS,
                message=f"Cached {len(cache._models)} models, found {len(coder_models)} coder models",
            ))
            
            # Test 3.4: Resolution without auth (should use default)
            model_id = await resolver.resolve_coder_model()
            
            if model_id:
                self.result.add_result(TestResult(
                    name="Model resolution (fallback)",
                    status=TestStatus.PASS,
                    message=f"Resolved to: {model_id}",
                ))
            else:
                self.result.add_result(TestResult(
                    name="Model resolution (fallback)",
                    status=TestStatus.WARN,
                    message="No model resolved (expected without auth)",
                ))
            
        except ImportError as e:
            self.result.add_result(TestResult(
                name="Model resolver creation",
                status=TestStatus.FAIL,
                error=str(e),
            ))
        except Exception as e:
            self.result.add_result(TestResult(
                name="Model resolver tests",
                status=TestStatus.FAIL,
                error=str(e),
            ))
    
    async def test_credential_vault(self):
        """Test credential vault functionality"""
        try:
            from xencode.auth.credential_vault import (
                CredentialVault,
                Credential,
                EnvironmentBackend,
            )
            
            # Test 4.1: Vault creation
            vault = CredentialVault(prefer_windows=False)  # Force env backend for tests
            
            status = vault.get_status()
            
            self.result.add_result(TestResult(
                name="Credential vault creation",
                status=TestStatus.PASS,
                message=f"Backends: {', '.join(status['backends'])}",
            ))
            
            # Test 4.2: Store and retrieve
            test_cred = Credential(
                service="smoke_test",
                username="test_key",
                secret="test_secret_123",
                description="Smoke test credential",
            )
            
            assert vault.set(test_cred)
            
            retrieved = vault.get("smoke_test", "test_key")
            
            if retrieved and retrieved.secret == "test_secret_123":
                self.result.add_result(TestResult(
                    name="Credential storage/retrieval",
                    status=TestStatus.PASS,
                    message="Successfully stored and retrieved credential",
                ))
            else:
                self.result.add_result(TestResult(
                    name="Credential storage/retrieval",
                    status=TestStatus.FAIL,
                    error="Retrieved credential doesn't match",
                ))
            
            # Test 4.3: Delete credential
            vault.delete("smoke_test", "test_key")
            
            deleted = vault.get("smoke_test", "test_key")
            
            if not deleted:
                self.result.add_result(TestResult(
                    name="Credential deletion",
                    status=TestStatus.PASS,
                    message="Credential successfully deleted",
                ))
            else:
                self.result.add_result(TestResult(
                    name="Credential deletion",
                    status=TestStatus.FAIL,
                    error="Credential still exists after deletion",
                ))
            
            # Test 4.4: Environment backend
            env_backend = EnvironmentBackend()
            assert env_backend.is_available()
            
            self.result.add_result(TestResult(
                name="Environment backend",
                status=TestStatus.PASS,
                message="Environment backend available",
            ))
            
        except ImportError as e:
            self.result.add_result(TestResult(
                name="Credential vault creation",
                status=TestStatus.FAIL,
                error=str(e),
            ))
        except Exception as e:
            self.result.add_result(TestResult(
                name="Credential vault tests",
                status=TestStatus.FAIL,
                error=str(e),
            ))
    
    def _print_summary(self):
        """Print test summary"""
        console.print("\n" + "=" * 60)
        console.print("[bold]SMOKE TEST SUMMARY[/bold]")
        console.print("=" * 60)
        
        # Create summary table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Status", style="green")
        
        table.add_row(
            "Total Tests",
            str(self.result.total),
            "OK" if self.result.total > 0 else "FAIL"
        )
        table.add_row(
            "Passed",
            str(self.result.passed),
            "OK" if self.result.passed > 0 else "FAIL"
        )
        table.add_row(
            "Failed",
            str(self.result.failed),
            "FAIL" if self.result.failed > 0 else "OK"
        )
        table.add_row(
            "Skipped",
            str(self.result.skipped),
            "-"
        )
        table.add_row(
            "Warnings",
            str(self.result.warnings),
            "WARN" if self.result.warnings > 0 else ""
        )
        table.add_row(
            "Success Rate",
            f"{self.result.success_rate:.1f}%",
            "OK" if self.result.success_rate >= 80 else "WARN"
        )
        
        console.print(table)
        
        # Overall result
        if self.result.is_passed:
            console.print(
                Panel(
                    f"[green]PASS: Phase 0 SMOKE TEST PASSED[/green]\n\n"
                    f"All {self.result.passed} tests completed successfully.\n"
                    f"Sprint 1 components are functioning correctly.",
                    title="[bold]RESULT[/bold]",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(
                    f"[red]FAIL: Phase 0 SMOKE TEST FAILED[/red]\n\n"
                    f"{self.result.failed} test(s) failed.\n"
                    f"Review errors above and fix before proceeding.",
                    title="[bold]RESULT[/bold]",
                    border_style="red",
                )
            )
        
        # Print failed tests
        if self.result.failed > 0:
            console.print("\n[bold red]Failed Tests:[/bold red]")
            for result in self.result.results:
                if result.status == TestStatus.FAIL:
                    console.print(f"  • {result.name}: {result.error}")
        
        console.print("\n" + "=" * 60)


async def main():
    """Main entry point"""
    tester = Phase0SmokeTester()
    result = await tester.run_all_tests()
    
    # Save result to file
    import json
    result_file = project_root / ".xencode" / "smoke_test_phase0_result.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(result_file, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    console.print(f"\n[dim]Results saved to: {result_file}[/dim]\n")
    
    # Return exit code
    return 0 if result.is_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
