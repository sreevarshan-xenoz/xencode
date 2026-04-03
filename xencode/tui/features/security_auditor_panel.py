"""Security Auditor TUI panel."""

from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Button, Label, Static, DataTable
from textual.reactive import reactive
from typing import List, Dict, Any

from .base_feature_panel import BaseFeaturePanel


class VulnerabilityCard(Static):
    """Card for a security vulnerability."""
    
    DEFAULT_CSS = """
    VulnerabilityCard {
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        border: solid $error;
        background: $panel;
    }
    
    VulnerabilityCard.critical {
        border: solid $error;
        background: $error-darken-2;
    }
    
    VulnerabilityCard.high {
        border: solid $warning;
        background: $warning-darken-2;
    }
    
    VulnerabilityCard.medium {
        border: solid $accent;
    }
    
    VulnerabilityCard.low {
        border: solid $primary;
    }
    """
    
    def __init__(self, title: str, severity: str, file: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vuln_title = title
        self.severity = severity
        self.file = file
        self.add_class(severity.lower())
    
    def render(self) -> str:
        severity_icons = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
        }
        icon = severity_icons.get(self.severity.lower(), "⚪")
        return (
            f"{icon} [bold]{self.vuln_title}[/bold]\n"
            f"Severity: {self.severity.upper()} | File: {self.file}"
        )


class SecurityAuditorPanel(BaseFeaturePanel):
    """Panel for security auditing and vulnerability scanning."""
    
    DEFAULT_CSS = """
    SecurityAuditorPanel {
        height: 100%;
    }
    
    .security-controls {
        height: auto;
        padding: 1;
        background: $panel;
    }
    
    .security-content {
        height: 1fr;
        padding: 1;
    }
    
    .security-summary {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        border: solid $accent;
        background: $panel;
    }
    """
    
    scanning = reactive(False)
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            feature_name="security_auditor",
            title="🔒 Security Auditor",
            *args,
            **kwargs
        )
        self.vulnerabilities: List[Dict[str, Any]] = []
    
    def compose(self):
        """Compose the security auditor panel."""
        yield from super().compose()
    
    def on_mount(self) -> None:
        """Initialize panel on mount."""
        self.set_status("enabled")
        self._build_content()
    
    def _build_content(self) -> None:
        """Build the panel content."""
        if not self.content_container:
            return
        
        self.content_container.remove_children()
        
        with self.content_container:
            # Controls
            with Horizontal(classes="security-controls"):
                yield Button("Scan Code", id="btn-scan", variant="primary")
                yield Button("Check Dependencies", id="btn-deps")
                yield Button("Generate Report", id="btn-report")
            
            # Content area
            with ScrollableContainer(classes="security-content"):
                if self.vulnerabilities:
                    self._render_vulnerabilities()
                else:
                    yield Label(
                        "Click 'Scan Code' to check for security vulnerabilities.",
                        classes="feature-empty"
                    )
    
    def _render_vulnerabilities(self) -> None:
        """Render vulnerabilities list."""
        # Summary
        critical = sum(1 for v in self.vulnerabilities if v["severity"] == "critical")
        high = sum(1 for v in self.vulnerabilities if v["severity"] == "high")
        medium = sum(1 for v in self.vulnerabilities if v["severity"] == "medium")
        low = sum(1 for v in self.vulnerabilities if v["severity"] == "low")
        
        yield Static(
            f"Found {len(self.vulnerabilities)} issues: "
            f"🔴 {critical} Critical | 🟠 {high} High | 🟡 {medium} Medium | 🟢 {low} Low",
            classes="security-summary"
        )
        
        # Vulnerabilities
        for vuln in self.vulnerabilities:
            yield VulnerabilityCard(vuln["title"], vuln["severity"], vuln["file"])
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-scan":
            await self._scan_code()
        elif button_id == "btn-deps":
            await self._check_dependencies()
        elif button_id == "btn-report":
            await self._generate_report()
    
    async def _scan_code(self) -> None:
        """Scan code for vulnerabilities."""
        self.set_status("loading")
        self.scanning = True
        
        try:
            # TODO: Integrate with actual security auditor
            self.vulnerabilities = [
                {"title": "SQL Injection Risk", "severity": "critical", "file": "db.py:42"},
                {"title": "Hardcoded Secret", "severity": "high", "file": "config.py:15"},
                {"title": "Weak Crypto", "severity": "medium", "file": "auth.py:88"},
            ]
            self._build_content()
            self.set_status("enabled")
        except Exception as e:
            self.show_empty_state(f"Error scanning: {e}")
            self.set_status("disabled")
        finally:
            self.scanning = False
    
    async def _check_dependencies(self) -> None:
        """Check dependencies for vulnerabilities."""
        self.set_status("loading")
        # TODO: Implement dependency check
        self.set_status("enabled")
    
    async def _generate_report(self) -> None:
        """Generate security report."""
        # TODO: Implement report generation
        pass
