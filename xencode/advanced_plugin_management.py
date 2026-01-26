#!/usr/bin/env python3
"""
Advanced Plugin Management System for Xencode

Extends the basic plugin system with advanced features like plugin bundling,
dependency resolution, security scanning, and performance monitoring.
"""

import asyncio
import hashlib
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import aiofiles
from pydantic import BaseModel

from xencode.plugin_system import PluginManager, PluginMetadata, LoadedPlugin
from xencode.security_manager import SecurityManager
from xencode.analyzers.security_analyzer import SecurityAnalyzer


class PluginBundle(BaseModel):
    """Represents a bundle of related plugins"""
    id: str
    name: str
    version: str
    description: str
    plugins: List[str]  # List of plugin IDs in the bundle
    dependencies: List[str] = []
    author: str = ""
    license: str = "MIT"
    created_at: datetime = field(default_factory=datetime.now)
    checksum: str = ""


class PluginSecurityScanResult(BaseModel):
    """Result of security scan for a plugin"""
    plugin_id: str
    scan_date: datetime
    vulnerabilities: List[Dict[str, Any]]
    security_score: float  # 0.0 to 1.0
    issues_found: int
    status: str  # safe, warning, critical
    recommendations: List[str]


class PluginDependencyGraph(BaseModel):
    """Represents the dependency graph of plugins"""
    plugin_id: str
    dependencies: List[str]
    dependents: List[str]
    conflicts: List[str]


class AdvancedPluginManager:
    """Advanced plugin management with additional features"""
    
    def __init__(self):
        from pathlib import Path
        self.plugin_manager = PluginManager(Path("./plugins"))  # Use a default plugins directory
        self.security_manager = SecurityManager()
        self.security_analyzer = SecurityAnalyzer()
        self.bundles: Dict[str, PluginBundle] = {}
        self.dependency_graph: Dict[str, PluginDependencyGraph] = {}
        
    async def create_bundle(self, name: str, description: str,
                          plugin_ids: List[str],
                          author: str = "",
                          license: str = "MIT") -> PluginBundle:
        """Create a bundle of related plugins"""
        bundle_id = f"bundle_{hashlib.md5(f'{name}_{datetime.now()}'.encode()).hexdigest()[:8]}"

        # Verify all plugins exist by checking the loaded plugins
        loaded_plugins = self.plugin_manager.list_plugins()
        for plugin_id in plugin_ids:
            if plugin_id not in loaded_plugins:
                raise ValueError(f"Plugin {plugin_id} does not exist")

        # Get dependencies for all plugins in the bundle
        all_dependencies = set()
        for plugin_id in plugin_ids:
            if plugin_id in loaded_plugins:
                plugin = loaded_plugins[plugin_id]
                if plugin.metadata.dependencies:
                    all_dependencies.update(plugin.metadata.dependencies)

        bundle = PluginBundle(
            id=bundle_id,
            name=name,
            version="1.0.0",  # Bundle version
            description=description,
            plugins=plugin_ids,
            dependencies=list(all_dependencies),
            author=author,
            license=license
        )

        # Calculate checksum
        bundle_content = json.dumps(bundle.dict(), default=str).encode()
        bundle.checksum = hashlib.sha256(bundle_content).hexdigest()

        self.bundles[bundle_id] = bundle
        return bundle
    
    async def install_bundle(self, bundle_id: str, 
                           verify_signature: bool = True,
                           auto_enable: bool = True) -> bool:
        """Install all plugins in a bundle"""
        if bundle_id not in self.bundles:
            raise ValueError(f"Bundle {bundle_id} does not exist")
        
        bundle = self.bundles[bundle_id]
        
        # Install each plugin in the bundle
        for plugin_id in bundle.plugins:
            try:
                # For now, we'll simulate installing from marketplace
                # In a real implementation, this would install the specific plugin
                success = await self.plugin_manager.install_from_marketplace(
                    plugin_id,
                    verify_signature=verify_signature,
                    auto_enable=auto_enable
                )
                if not success:
                    raise Exception(f"Failed to install plugin {plugin_id}")
            except Exception as e:
                # Rollback: uninstall any plugins that were already installed
                for installed_plugin in bundle.plugins[:bundle.plugins.index(plugin_id)]:
                    try:
                        await self.plugin_manager.uninstall_plugin(installed_plugin)
                    except:
                        pass  # Ignore errors during rollback
                raise e
        
        return True
    
    async def scan_plugin_security(self, plugin_id: str) -> PluginSecurityScanResult:
        """Perform security scan on a plugin"""
        # Check if plugin exists by checking the loaded plugins
        loaded_plugins = self.plugin_manager.list_plugins()
        if plugin_id not in loaded_plugins:
            raise ValueError(f"Plugin {plugin_id} does not exist")

        scan_date = datetime.now()
        vulnerabilities = []
        recommendations = []

        # Analyze plugin code for security issues
        try:
            # Get plugin file path (this is a mock implementation)
            plugin_path = Path(f"/tmp/plugins/{plugin_id}")  # Mock path

            # In a real implementation, we would scan the actual plugin files
            # For now, we'll simulate a basic scan
            simulated_issues = [
                {"type": "code_injection", "severity": "medium", "location": "main.py:45"},
                {"type": "unsafe_import", "severity": "low", "location": "utils.py:12"}
            ]

            vulnerabilities = simulated_issues
            issues_found = len(vulnerabilities)

            # Calculate security score (simplified)
            if issues_found == 0:
                security_score = 1.0
                status = "safe"
            elif issues_found <= 2:
                security_score = 0.7
                status = "warning"
                recommendations = ["Review code for potential security issues"]
            else:
                security_score = 0.3
                status = "critical"
                recommendations = [
                    "Immediate security review required",
                    "Consider disabling plugin until issues are fixed"
                ]

        except Exception as e:
            vulnerabilities = [{"type": "scan_error", "severity": "critical", "message": str(e)}]
            security_score = 0.0
            status = "critical"
            issues_found = 1
            recommendations = ["Security scan failed, manual review required"]

        return PluginSecurityScanResult(
            plugin_id=plugin_id,
            scan_date=scan_date,
            vulnerabilities=vulnerabilities,
            security_score=security_score,
            issues_found=issues_found,
            status=status,
            recommendations=recommendations
        )
    
    async def resolve_dependencies(self, plugin_id: str) -> List[PluginDependencyGraph]:
        """Resolve dependencies for a plugin and create dependency graph"""
        # Check if plugin exists by checking the loaded plugins
        loaded_plugins = self.plugin_manager.list_plugins()
        if plugin_id not in loaded_plugins:
            raise ValueError(f"Plugin {plugin_id} does not exist")

        plugin = loaded_plugins[plugin_id]

        # Build dependency graph (simplified implementation)
        graph = PluginDependencyGraph(
            plugin_id=plugin_id,
            dependencies=plugin.metadata.dependencies if plugin.metadata.dependencies else [],
            dependents=[],  # Would need to scan all plugins to find dependents
            conflicts=[]  # Would need to check for conflicts
        )

        # Store in cache
        self.dependency_graph[plugin_id] = graph

        # Return the graph and any related graphs
        result = [graph]

        # Add dependency graphs for each dependency
        for dep_id in graph.dependencies:
            if dep_id in self.dependency_graph:
                result.append(self.dependency_graph[dep_id])
            else:
                # Create a basic graph for the dependency
                if dep_id in loaded_plugins:
                    dep_plugin = loaded_plugins[dep_id]
                    dep_graph = PluginDependencyGraph(
                        plugin_id=dep_id,
                        dependencies=dep_plugin.metadata.dependencies if dep_plugin.metadata.dependencies else [],
                        dependents=[plugin_id],  # This plugin depends on it
                        conflicts=[]
                    )
                    self.dependency_graph[dep_id] = dep_graph
                    result.append(dep_graph)

        return result
    
    async def get_performance_metrics(self, plugin_id: str, 
                                    time_range: str = "24h") -> Dict[str, Any]:
        """Get performance metrics for a plugin"""
        # This would typically connect to a metrics collection system
        # For now, we'll return mock data
        
        # Get plugin stats from the base manager
        stats = await self.plugin_manager.get_plugin_stats(plugin_id)
        
        # Calculate additional metrics
        total_executions = stats.get('total_executions', 0)
        successful_executions = stats.get('successful_executions', 0)
        failed_executions = stats.get('failed_executions', 0)
        
        success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 100
        error_rate = (failed_executions / total_executions * 100) if total_executions > 0 else 0
        
        return {
            "plugin_id": plugin_id,
            "time_range": time_range,
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate_percent": round(success_rate, 2),
            "error_rate_percent": round(error_rate, 2),
            "average_response_time_ms": stats.get('avg_execution_time_ms', 0.0),
            "peak_memory_usage_mb": stats.get('peak_memory_mb', 0.0),
            "average_cpu_usage_percent": stats.get('avg_cpu_percent', 0.0),
            "last_activity": stats.get('last_executed', None),
            "uptime_percentage": stats.get('uptime_percent', 100.0)
        }
    
    async def backup_plugin(self, plugin_id: str, backup_path: str = None) -> str:
        """Create a backup of a plugin"""
        if not backup_path:
            backup_path = f"/tmp/plugin_backups/{plugin_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        plugin = await self.plugin_manager.get_plugin(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {plugin_id} does not exist")
        
        # Create backup directory if it doesn't exist
        Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
        
        # In a real implementation, this would zip the actual plugin files
        # For now, we'll create a mock backup with plugin metadata
        backup_data = {
            "plugin_metadata": plugin.metadata.dict(),
            "permissions": plugin.permissions,
            "config": await self.plugin_manager.get_plugin_config(plugin_id),
            "backup_date": datetime.now().isoformat(),
            "plugin_id": plugin_id
        }
        
        with zipfile.ZipFile(backup_path, 'w') as zip_file:
            zip_file.writestr(f"{plugin_id}_metadata.json", json.dumps(backup_data, default=str))
        
        return backup_path
    
    async def restore_plugin(self, backup_path: str, plugin_id: str = None) -> bool:
        """Restore a plugin from backup"""
        if not Path(backup_path).exists():
            raise FileNotFoundError(f"Backup file {backup_path} does not exist")
        
        # Extract backup data
        with zipfile.ZipFile(backup_path, 'r') as zip_file:
            metadata_content = zip_file.read(f"{plugin_id}_metadata.json").decode()
            backup_data = json.loads(metadata_content)
        
        # Restore plugin (in a real implementation, this would restore files and config)
        restored_plugin_id = backup_data["plugin_id"]
        
        # In a real implementation, we would:
        # 1. Extract plugin files to the appropriate location
        # 2. Restore configuration
        # 3. Update plugin registry
        # 4. Verify integrity
        
        # For now, we'll just return success
        return True
    
    async def get_compatibility_report(self, plugin_id: str, 
                                     target_environments: List[str]) -> Dict[str, Any]:
        """Get compatibility report for plugin across different environments"""
        plugin = await self.plugin_manager.get_plugin(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {plugin_id} does not exist")
        
        report = {
            "plugin_id": plugin_id,
            "target_environments": target_environments,
            "compatibility_results": {}
        }
        
        for env in target_environments:
            # Simulate compatibility check
            # In a real implementation, this would check system requirements, 
            # dependencies, and environment compatibility
            compatibility_result = {
                "environment": env,
                "compatible": True,  # Simplified
                "issues": [],
                "recommendations": [],
                "required_changes": []
            }
            
            # Add some mock issues based on environment
            if env == "resource_constrained":
                compatibility_result["issues"].append({
                    "type": "resource_requirement",
                    "severity": "medium",
                    "description": "Plugin may exceed memory limits in constrained environments"
                })
                compatibility_result["recommendations"].append(
                    "Consider optimizing plugin for low-resource environments"
                )
            
            report["compatibility_results"][env] = compatibility_result
        
        return report


# Example usage and demo function
async def demo_advanced_plugin_features():
    """Demonstrate advanced plugin management features"""
    print("Demonstrating Advanced Plugin Management Features")
    
    # Create advanced plugin manager
    advanced_manager = AdvancedPluginManager()
    
    # Create a mock plugin bundle
    print("\nCreating plugin bundle...")
    try:
        bundle = await advanced_manager.create_bundle(
            name="Development Toolkit",
            description="Essential plugins for development",
            plugin_ids=["code-analyzer", "formatter", "tester"],
            author="Xencode Team"
        )
        print(f"   SUCCESS: Created bundle: {bundle.name} (ID: {bundle.id})")
        print(f"   SUCCESS: Contains {len(bundle.plugins)} plugins")
    except Exception as e:
        print(f"   ERROR: Failed to create bundle: {e}")

    # Perform security scan (mock)
    print("\nPerforming security scan...")
    try:
        scan_result = await advanced_manager.scan_plugin_security("code-analyzer")
        print(f"   SUCCESS: Security scan completed for code-analyzer")
        print(f"   SUCCESS: Security score: {scan_result.security_score}")
        print(f"   SUCCESS: Status: {scan_result.status}")
        print(f"   SUCCESS: Issues found: {scan_result.issues_found}")
    except Exception as e:
        print(f"   ERROR: Security scan failed: {e}")

    # Resolve dependencies (mock)
    print("\nResolving dependencies...")
    try:
        dependency_graphs = await advanced_manager.resolve_dependencies("code-analyzer")
        print(f"   SUCCESS: Resolved dependencies for code-analyzer")
        print(f"   SUCCESS: Found {len(dependency_graphs)} related dependency graphs")
    except Exception as e:
        print(f"   ERROR: Dependency resolution failed: {e}")

    # Get performance metrics (mock)
    print("\nGetting performance metrics...")
    try:
        metrics = await advanced_manager.get_performance_metrics("code-analyzer")
        print(f"   SUCCESS: Retrieved performance metrics for code-analyzer")
        print(f"   SUCCESS: Success rate: {metrics['success_rate_percent']}%")
        print(f"   SUCCESS: Error rate: {metrics['error_rate_percent']}%")
        print(f"   SUCCESS: Avg response time: {metrics['average_response_time_ms']}ms")
    except Exception as e:
        print(f"   ERROR: Performance metrics retrieval failed: {e}")

    # Create backup (mock)
    print("\nCreating plugin backup...")
    try:
        backup_path = await advanced_manager.backup_plugin("code-analyzer")
        print(f"   SUCCESS: Created backup at: {backup_path}")
    except Exception as e:
        print(f"   ERROR: Backup creation failed: {e}")

    # Get compatibility report (mock)
    print("\nChecking compatibility...")
    try:
        compat_report = await advanced_manager.get_compatibility_report(
            "code-analyzer",
            ["standard", "resource_constrained", "secure_mode"]
        )
        print(f"   SUCCESS: Generated compatibility report for code-analyzer")
        print(f"   SUCCESS: Checked {len(compat_report['target_environments'])} environments")
    except Exception as e:
        print(f"   ERROR: Compatibility check failed: {e}")

    print("\nAdvanced Plugin Management Demo Completed")


if __name__ == "__main__":
    asyncio.run(demo_advanced_plugin_features())