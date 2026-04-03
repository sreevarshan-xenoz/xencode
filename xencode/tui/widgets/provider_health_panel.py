#!/usr/bin/env python3
"""
Provider Health Dashboard TUI Widget

Real-time provider health monitoring widget for the Xencode TUI.
"""

from datetime import datetime
from typing import Dict, List, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Label, Button, ProgressBar
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.reactive import reactive

from rich.table import Table
from rich.panel import Panel
from rich.text import Text


class ProviderStatusIndicator(Static):
    """Status indicator dot for a provider"""
    
    STATUS_COLORS = {
        'healthy': 'green',
        'degraded': 'yellow',
        'unhealthy': 'red',
        'offline': 'red',
        'unknown': 'gray',
    }
    
    def __init__(self, status: str = "unknown", **kwargs):
        super().__init__(**kwargs)
        self.status = status
    
    def render(self) -> str:
        color = self.STATUS_COLORS.get(self.status, 'gray')
        return f"[{color}]●[/{color}]"


class ProviderCard(Static):
    """Card displaying provider health information"""
    
    def __init__(
        self,
        provider_name: str,
        status: str = "unknown",
        latency_ms: float = 0.0,
        error_rate: float = 0.0,
        uptime: float = 0.0,
        last_checked: Optional[datetime] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.provider_name = provider_name
        self.status = status
        self.latency_ms = latency_ms
        self.error_rate = error_rate
        self.uptime = uptime
        self.last_checked = last_checked
    
    def render(self) -> Panel:
        # Status indicator
        status_colors = {
            'healthy': ('green', '✓'),
            'degraded': ('yellow', '⚠'),
            'unhealthy': ('red', '✗'),
            'offline': ('red', '○'),
            'unknown': ('gray', '?'),
        }
        color, icon = status_colors.get(self.status, ('gray', '?'))
        
        # Format latency
        if self.latency_ms > 0:
            if self.latency_ms < 500:
                latency_str = f"[green]{self.latency_ms:.0f}ms[/green]"
            elif self.latency_ms < 2000:
                latency_str = f"[yellow]{self.latency_ms:.0f}ms[/yellow]"
            else:
                latency_str = f"[red]{self.latency_ms:.0f}ms[/red]"
        else:
            latency_str = "[gray]N/A[/gray]"
        
        # Format error rate
        if self.error_rate > 5:
            error_str = f"[red]{self.error_rate:.1f}%[/red]"
        elif self.error_rate > 1:
            error_str = f"[yellow]{self.error_rate:.1f}%[/yellow]"
        else:
            error_str = f"[green]{self.error_rate:.1f}%[/green]"
        
        # Format uptime
        if self.uptime > 99:
            uptime_str = f"[green]{self.uptime:.1f}%[/green]"
        elif self.uptime > 95:
            uptime_str = f"[yellow]{self.uptime:.1f}%[/yellow]"
        else:
            uptime_str = f"[red]{self.uptime:.1f}%[/red]"
        
        # Last checked
        if self.last_checked:
            delta = datetime.now() - self.last_checked
            seconds = int(delta.total_seconds())
            if seconds < 60:
                checked_str = f"{seconds}s ago"
            elif seconds < 3600:
                checked_str = f"{seconds // 60}m ago"
            else:
                checked_str = f"{seconds // 3600}h ago"
        else:
            checked_str = "Never"
        
        content = (
            f"[bold]{self.provider_name.replace('_', ' ').title()}[/bold]\n\n"
            f"Latency:    {latency_str}\n"
            f"Error Rate: {error_str}\n"
            f"Uptime:     {uptime_str}\n"
            f"Checked:    {checked_str}"
        )
        
        return Panel(
            content,
            title=f"[{color}]{icon}[/{color}]",
            border_style=color,
        )


class ProviderHealthDashboard(Static):
    """
    Main provider health dashboard widget
    
    Displays real-time health status for all configured providers.
    """
    
    BINDINGS = [
        Binding("r", "refresh", "Refresh", show=True),
        Binding("d", "toggle_details", "Details", show=True),
    ]
    
    providers = reactive({})
    overall_status = reactive("unknown")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._health_data = {}
        self._show_details = False
    
    def compose(self) -> ComposeResult:
        with Container(id="provider-health-container"):
            yield Static("Provider Health", id="provider-health-title", classes="header")
            yield Static("", id="provider-health-grid", classes="provider-grid")
            yield Static("", id="provider-health-summary", classes="summary")
    
    def on_mount(self) -> None:
        """Start health monitoring on mount"""
        self._update_display()
    
    def action_refresh(self) -> None:
        """Refresh provider health data"""
        self._fetch_health_data()
    
    def action_toggle_details(self) -> None:
        """Toggle detailed view"""
        self._show_details = not self._show_details
        self._update_display()
    
    def _fetch_health_data(self):
        """Fetch latest health data from API"""
        # This would call the API in a real implementation
        # For now, we'll use the monitor directly
        try:
            from ..monitoring.provider_health import get_health_monitor, ProviderType
            import asyncio
            
            async def fetch():
                monitor = get_health_monitor()
                
                # Check all providers
                tasks = [
                    monitor.check_provider_health(provider)
                    for provider in ProviderType
                ]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                return monitor.get_health_summary()
            
            # Run async
            import threading
            result = {}
            
            def run():
                result['data'] = asyncio.run(fetch())
            
            thread = threading.Thread(target=run)
            thread.start()
            thread.join(timeout=10)
            
            if 'data' in result:
                self._health_data = result['data']
                self._update_display()
                
        except Exception as e:
            self.query_one("#provider-health-summary", Static).update(
                f"[red]Error fetching health data: {e}[/red]"
            )
    
    def _update_display(self):
        """Update the display with current health data"""
        if not self._health_data:
            self._fetch_health_data()
            return
        
        # Update grid
        grid = self.query_one("#provider-health-grid", Static)
        summary = self.query_one("#provider-health-summary", Static)
        
        # Build provider cards
        cards = []
        for provider_name, health in self._health_data.get('providers', {}).items():
            card = ProviderCard(
                provider_name=provider_name,
                status=health.get('status', 'unknown'),
                latency_ms=health.get('latency', {}).get('current_ms', 0),
                error_rate=health.get('errors', {}).get('error_rate', 0),
                uptime=health.get('uptime_percentage', 0),
                last_checked=datetime.fromisoformat(health['last_checked']) if health.get('last_checked') else None,
            )
            cards.append(card.render())
        
        # Display cards
        from rich.console import Group
        grid.update(Panel(
            Group(*cards),
            title="Providers",
            border_style="blue",
        ))
        
        # Update summary
        overall = self._health_data.get('overall_status', 'unknown')
        healthy = self._health_data.get('healthy_count', 0)
        degraded = self._health_data.get('degraded_count', 0)
        unhealthy = self._health_data.get('unhealthy_count', 0)
        
        status_colors = {
            'healthy': 'green',
            'degraded': 'yellow',
            'unhealthy': 'red',
            'unknown': 'gray',
        }
        color = status_colors.get(overall, 'gray')
        
        summary_text = (
            f"[bold]Overall Status:[/bold] [{color}]{overall.upper()}[/{color}]  "
            f"[green]✓ {healthy}[/green]  "
            f"[yellow]⚠ {degraded}[/yellow]  "
            f"[red]✗ {unhealthy}[/red]  "
            f"| [bold]Recommended:[/bold] {self._health_data.get('recommended_provider', 'N/A')}"
        )
        summary.update(summary_text)


class ProviderHealthScreen(ModalScreen):
    """Full-screen provider health dashboard"""
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("r", "refresh", "Refresh", show=True),
    ]
    
    def compose(self) -> ComposeResult:
        yield ProviderHealthDashboard(classes="full-screen")
    
    def action_refresh(self) -> None:
        """Refresh health data"""
        dashboard = self.query_one(ProviderHealthDashboard)
        dashboard._fetch_health_data()


def create_provider_health_widget() -> ProviderHealthDashboard:
    """Create and return a provider health dashboard widget"""
    return ProviderHealthDashboard()


if __name__ == "__main__":
    # Test the widget
    from textual.app import App
    
    class TestApp(App):
        def on_mount(self) -> None:
            self.push_screen(ProviderHealthScreen())
    
    TestApp().run()
