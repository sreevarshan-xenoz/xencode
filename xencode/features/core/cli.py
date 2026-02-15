"""
Feature CLI Integration

CLI command group for feature management.
"""

import click
from typing import List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..base import FeatureBase, FeatureConfig, FeatureStatus


console = Console()


class FeatureCommandGroup:
    """CLI command group for feature management"""
    
    def __init__(self, feature_manager):
        self.feature_manager = feature_manager
    
    def create_cli_group(self) -> click.Group:
        """Create the main feature CLI group"""
        
        @click.group()
        def feature_group():
            """Manage Xencode features"""
            pass
        
        # Add subcommands
        feature_group.add_command(self._list_command())
        feature_group.add_command(self._enable_command())
        feature_group.add_command(self._disable_command())
        feature_group.add_command(self._status_command())
        feature_group.add_command(self._info_command())
        
        return feature_group
    
    def _list_command(self) -> click.Command:
        """Create the list command"""
        
        @click.command(name='list')
        def list_features():
            """List all available features"""
            features = self.feature_manager.get_available_features()
            
            table = Table(title="Available Features")
            table.add_column("Name", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Version", style="yellow")
            
            for feature_name in sorted(features):
                feature = self.feature_manager.get_feature(feature_name)
                if feature:
                    status = feature.get_status().value
                    version = feature.version
                else:
                    status = "not_loaded"
                    version = "unknown"
                
                table.add_row(feature_name, status, version)
            
            console.print(table)
        
        return list_features
    
    def _enable_command(self) -> click.Command:
        """Create the enable command"""
        
        @click.command(name='enable')
        @click.argument('feature_name')
        async def enable_feature(feature_name):
            """Enable a feature"""
            success = await self.feature_manager.initialize_feature(feature_name)
            
            if success:
                console.print(f"[green]✅ Feature '{feature_name}' enabled successfully![/green]")
            else:
                console.print(f"[red]❌ Failed to enable feature '{feature_name}'[/red]")
        
        return enable_feature
    
    def _disable_command(self) -> click.Command:
        """Create the disable command"""
        
        @click.command(name='disable')
        @click.argument('feature_name')
        async def disable_feature(feature_name):
            """Disable a feature"""
            success = await self.feature_manager.shutdown_feature(feature_name)
            
            if success:
                console.print(f"[green]✅ Feature '{feature_name}' disabled successfully![/green]")
            else:
                console.print(f"[red]❌ Failed to disable feature '{feature_name}'[/red]")
        
        return disable_feature
    
    def _status_command(self) -> click.Command:
        """Create the status command"""
        
        @click.command(name='status')
        @click.argument('feature_name', required=False)
        async def feature_status(feature_name):
            """Show feature status"""
            if feature_name:
                feature = self.feature_manager.get_feature(feature_name)
                if feature:
                    status = feature.get_status()
                    console.print(Panel(
                        f"[bold]{feature_name}[/bold]\n"
                        f"Status: {status.value}\n"
                        f"Version: {feature.version}\n"
                        f"Enabled: {feature.is_enabled}\n"
                        f"Initialized: {feature.is_initialized}",
                        title="Feature Status"
                    ))
                else:
                    console.print(f"[red]Feature '{feature_name}' not found[/red]")
            else:
                # Show all features status
                features = self.feature_manager.get_all_features()
                
                table = Table(title="Feature Status")
                table.add_column("Name", style="cyan")
                table.add_column("Status", style="green")
                table.add_column("Enabled", style="yellow")
                
                for name, feature in features.items():
                    table.add_row(
                        name,
                        feature.get_status().value,
                        "Yes" if feature.is_enabled else "No"
                    )
                
                console.print(table)
        
        return feature_status
    
    def _info_command(self) -> click.Command:
        """Create the info command"""
        
        @click.command(name='info')
        @click.argument('feature_name')
        def feature_info(feature_name):
            """Show feature information"""
            feature_class = self.feature_manager.get_feature_class(feature_name)
            
            if not feature_class:
                console.print(f"[red]Feature '{feature_name}' not found[/red]")
                return
            
            # Get docstring
            doc = feature_class.__doc__ or "No description available"
            
            console.print(Panel(
                f"[bold]{feature_name}[/bold]\n\n"
                f"{doc}\n\n"
                f"Version: {feature_class.__dict__.get('version', 'unknown')}",
                title="Feature Information"
            ))
        
        return feature_info
