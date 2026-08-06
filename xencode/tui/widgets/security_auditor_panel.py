#!/usr/bin/env python3
"""
Security Auditor Panel Widget for Xencode TUI

Interactive security auditor interface with vulnerability dashboard, dependency tree,
risk indicators, fix suggestions, and audit history.
"""

from datetime import datetime
from typing import Any, Dict, List

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.message import Message
from textual.widgets import Button, DataTable, Label, ListItem


class VulnerabilityDashboard(Container):
    """Dashboard showing vulnerability summary and statistics"""

    DEFAULT_CSS = """
    VulnerabilityDashboard {
        height: 100%;
        border: solid $primary;
        background: $surface;
        padding: 1;
    }

    #vuln-header {
        dock: top;
        height: 3;
        background: $primary;
        padding: 0 2;
    }

    #vuln-content {
        height: 1fr;
        padding: 1;
    }

    .summary-row {
        height: auto;
        padding: 1;
        margin: 0 1;
    }

    .risk-card {
        height: 8;
        border: solid $primary;
        background: $panel;
        padding: 1;
        margin: 1;
    }

    .risk-card.critical {
        border: solid $error;
        background: $error-darken-3;
    }

    .risk-card.high {
        border: solid $warning;
        background: $warning-darken-3;
    }

    .risk-card.medium {
        border: solid yellow;
    }

    .risk-card.low {
        border: solid $success;
    }

    .risk-title {
        height: 1;
        text-style: bold;
    }

    .risk-value {
        height: 3;
        content-align: center middle;
        text-style: bold;
    }

    .risk-label {
        height: 1;
        content-align: center middle;
        text-style: dim;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.summary = {
            'total_vulnerabilities': 0,
            'code_vulnerabilities': 0,
            'dependency_vulnerabilities': 0,
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'info': 0
        }

    def compose(self) -> ComposeResult:
        """Compose the vulnerability dashboard"""
        yield Label("🔒 Vulnerability Dashboard", id="vuln-header")

        with ScrollableContainer(id="vuln-content"):
            # Total vulnerabilities
            with Horizontal(classes="summary-row"):
                yield Label("Total Vulnerabilities:", classes="bold")
                yield Label(str(self.summary['total_vulnerabilities']), id="total-vulns")

            with Horizontal(classes="summary-row"):
                yield Label("Code Vulnerabilities:", classes="dim")
                yield Label(str(self.summary['code_vulnerabilities']), id="code-vulns")

            with Horizontal(classes="summary-row"):
                yield Label("Dependency Vulnerabilities:", classes="dim")
                yield Label(str(self.summary['dependency_vulnerabilities']), id="dep-vulns")

            # Risk level cards
            with Container(classes="risk-card critical"):
                yield Label("🔴 Critical", classes="risk-title")
                yield Label(str(self.summary['critical']), id="critical-count", classes="risk-value")
                yield Label("Immediate Action Required", classes="risk-label")

            with Container(classes="risk-card high"):
                yield Label("🟠 High", classes="risk-title")
                yield Label(str(self.summary['high']), id="high-count", classes="risk-value")
                yield Label("High Priority", classes="risk-label")

            with Container(classes="risk-card medium"):
                yield Label("🟡 Medium", classes="risk-title")
                yield Label(str(self.summary['medium']), id="medium-count", classes="risk-value")
                yield Label("Medium Priority", classes="risk-label")

            with Container(classes="risk-card low"):
                yield Label("🟢 Low", classes="risk-title")
                yield Label(str(self.summary['low']), id="low-count", classes="risk-value")
                yield Label("Low Priority", classes="risk-label")

    def update_summary(self, summary: Dict[str, int]) -> None:
        """Update vulnerability summary"""
        self.summary.update(summary)

        try:
            self.query_one("#total-vulns", Label).update(str(summary.get('total_vulnerabilities', 0)))
            self.query_one("#code-vulns", Label).update(str(summary.get('code_vulnerabilities', 0)))
            self.query_one("#dep-vulns", Label).update(str(summary.get('dependency_vulnerabilities', 0)))
            self.query_one("#critical-count", Label).update(str(summary.get('critical', 0)))
            self.query_one("#high-count", Label).update(str(summary.get('high', 0)))
            self.query_one("#medium-count", Label).update(str(summary.get('medium', 0)))
            self.query_one("#low-count", Label).update(str(summary.get('low', 0)))
        except Exception:
                pass  # Silently ignore

class VulnerabilityListItem(ListItem):
    """A single vulnerability in the list"""

    DEFAULT_CSS = """
    VulnerabilityListItem {
        height: auto;
        padding: 1;
        margin: 0 1;
        background: $panel;
        border-left: thick $primary;
    }

    VulnerabilityListItem:hover {
        background: $boost;
    }

    VulnerabilityListItem.critical {
        border-left: thick $error;
        background: $error-darken-3;
    }

    VulnerabilityListItem.high {
        border-left: thick $warning;
        background: $warning-darken-3;
    }

    VulnerabilityListItem.medium {
        border-left: thick yellow;
    }

    VulnerabilityListItem.low {
        border-left: thick $success;
    }
    """

    def __init__(self, vuln_info: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.vuln_info = vuln_info
        risk_level = vuln_info.get('risk_level', 'low')
        self.add_class(risk_level)

    def compose(self) -> ComposeResult:
        """Compose the vulnerability item"""
        title = self.vuln_info.get('title', 'Unknown vulnerability')
        risk_level = self.vuln_info.get('risk_level', 'low').upper()
        file_path = self.vuln_info.get('file_path', '')
        line_number = self.vuln_info.get('line_number', 0)
        vuln_type = self.vuln_info.get('type', 'unknown')

        # Risk emoji
        risk_emoji = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢',
            'info': 'ℹ️'
        }.get(self.vuln_info.get('risk_level', 'low'), '❓')

        yield Label(f"{risk_emoji} [{risk_level}] {title}", classes="bold")
        yield Label(f"   Type: {vuln_type} | Location: {file_path}:{line_number}", classes="dim")


class VulnerabilityList(Container):
    """List of vulnerabilities"""

    DEFAULT_CSS = """
    VulnerabilityList {
        height: 100%;
        border: solid $accent;
        background: $surface;
    }

    #vuln-list-header {
        dock: top;
        height: 3;
        background: $accent;
        padding: 0 2;
    }

    #vuln-list-content {
        height: 1fr;
    }
    """

    class VulnerabilitySelected(Message):
        """Message sent when a vulnerability is selected"""
        def __init__(self, vuln_id: str):
            super().__init__()
            self.vuln_id = vuln_id

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vulnerabilities: List[Dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        """Compose the vulnerability list"""
        yield Label("🐛 Vulnerabilities", id="vuln-list-header")

        with ScrollableContainer(id="vuln-list-content"):
            yield Label("No vulnerabilities found", classes="dim")

    def update_vulnerabilities(self, vulnerabilities: List[Dict[str, Any]]) -> None:
        """Update the vulnerability list"""
        self.vulnerabilities = vulnerabilities

        list_content = self.query_one("#vuln-list-content", ScrollableContainer)
        list_content.remove_children()

        if vulnerabilities:
            for vuln in vulnerabilities:
                item = VulnerabilityListItem(vuln)
                list_content.mount(item)
        else:
            list_content.mount(Label("No vulnerabilities found", classes="dim"))


class DependencyTree(Container):
    """Tree view of dependencies with vulnerabilities"""

    DEFAULT_CSS = """
    DependencyTree {
        height: 100%;
        border: solid $primary;
        background: $surface;
        padding: 1;
    }

    #dep-tree-header {
        dock: top;
        height: 3;
        background: $primary;
        padding: 0 2;
    }

    #dep-tree-content {
        height: 1fr;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dependencies: List[Dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        """Compose the dependency tree"""
        yield Label("📦 Dependency Vulnerabilities", id="dep-tree-header")

        with ScrollableContainer(id="dep-tree-content"):
            table = DataTable(id="dep-table")
            table.add_columns("Package", "Current", "Fixed", "CVE", "Risk")
            yield table

    def update_dependencies(self, dependencies: List[Dict[str, Any]]) -> None:
        """Update the dependency tree"""
        self.dependencies = dependencies

        try:
            table = self.query_one("#dep-table", DataTable)
            table.clear()

            if dependencies:
                for dep in dependencies:
                    risk_emoji = {
                        'critical': '🔴',
                        'high': '🟠',
                        'medium': '🟡',
                        'low': '🟢'
                    }.get(dep.get('risk_level', 'low'), '❓')

                    table.add_row(
                        dep.get('package_name', 'unknown'),
                        dep.get('current_version', 'N/A'),
                        dep.get('fixed_version', 'N/A'),
                        dep.get('cve_id', 'N/A'),
                        f"{risk_emoji} {dep.get('risk_level', 'low').upper()}"
                    )
        except Exception:
                pass  # Silently ignore

class FixSuggestions(Container):
    """Panel showing fix suggestions for selected vulnerability"""

    DEFAULT_CSS = """
    FixSuggestions {
        height: 100%;
        border: solid $accent;
        background: $surface;
        padding: 1;
    }

    #fix-header {
        dock: top;
        height: 3;
        background: $accent;
        padding: 0 2;
    }

    #fix-content {
        height: 1fr;
        padding: 1;
    }

    .fix-section {
        height: auto;
        padding: 1;
        margin: 1 0;
    }

    .code-snippet {
        background: $panel;
        padding: 1;
        margin: 1 0;
        border: solid $primary;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_vuln = None

    def compose(self) -> ComposeResult:
        """Compose the fix suggestions"""
        yield Label("💡 Fix Suggestions", id="fix-header")

        with ScrollableContainer(id="fix-content"):
            yield Label("Select a vulnerability to see fix suggestions", classes="dim", id="fix-placeholder")

    def update_fix_suggestions(self, vuln: Dict[str, Any]) -> None:
        """Update fix suggestions for a vulnerability"""
        self.current_vuln = vuln

        fix_content = self.query_one("#fix-content", ScrollableContainer)
        fix_content.remove_children()

        if vuln:
            # Vulnerability details
            with Container(classes="fix-section"):
                fix_content.mount(Label(f"🐛 {vuln.get('title', 'Unknown')}", classes="bold"))
                fix_content.mount(Label(f"Type: {vuln.get('type', 'unknown')}", classes="dim"))
                fix_content.mount(Label(f"Risk: {vuln.get('risk_level', 'low').upper()}", classes="dim"))

            # Code snippet
            if vuln.get('code_snippet'):
                with Container(classes="fix-section"):
                    fix_content.mount(Label("Vulnerable Code:", classes="bold"))
                    with Container(classes="code-snippet"):
                        fix_content.mount(Label(vuln['code_snippet']))

            # Fix suggestion
            if vuln.get('fix_suggestion'):
                with Container(classes="fix-section"):
                    fix_content.mount(Label("Recommended Fix:", classes="bold"))
                    fix_content.mount(Label(vuln['fix_suggestion']))

            # References
            if vuln.get('references'):
                with Container(classes="fix-section"):
                    fix_content.mount(Label("References:", classes="bold"))
                    for ref in vuln['references']:
                        fix_content.mount(Label(f"• {ref}", classes="dim"))
        else:
            fix_content.mount(Label("Select a vulnerability to see fix suggestions", classes="dim"))


class AuditHistory(Container):
    """Panel showing audit history"""

    DEFAULT_CSS = """
    AuditHistory {
        height: 100%;
        border: solid $primary;
        background: $surface;
    }

    #history-header {
        dock: top;
        height: 3;
        background: $primary;
        padding: 0 2;
    }

    #history-content {
        height: 1fr;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history: List[Dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        """Compose the audit history"""
        yield Label("📜 Audit History", id="history-header")

        with ScrollableContainer(id="history-content"):
            table = DataTable(id="history-table")
            table.add_columns("Timestamp", "Path", "Total", "Critical", "High", "Status")
            yield table

    def update_history(self, history: List[Dict[str, Any]]) -> None:
        """Update the audit history"""
        self.history = history

        try:
            table = self.query_one("#history-table", DataTable)
            table.clear()

            if history:
                for entry in reversed(history[-20:]):  # Show last 20
                    timestamp = entry.get('timestamp', '')
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        time_str = timestamp[:16] if len(timestamp) > 16 else timestamp

                    summary = entry.get('summary', {})
                    status = '✅' if summary.get('critical', 0) == 0 else '⚠️'

                    table.add_row(
                        time_str,
                        entry.get('scan_path', 'N/A')[:30],
                        str(summary.get('total_vulnerabilities', 0)),
                        str(summary.get('critical', 0)),
                        str(summary.get('high', 0)),
                        status
                    )
        except Exception:
                pass  # Silently ignore

class SecurityAuditorPanel(Container):
    """Main security auditor panel with all components"""

    DEFAULT_CSS = """
    SecurityAuditorPanel {
        height: 100%;
        border: solid $primary;
        background: $surface;
    }

    SecurityAuditorPanel > #sa-header {
        dock: top;
        height: 3;
        background: $primary;
        padding: 0 2;
    }

    SecurityAuditorPanel > #sa-controls {
        dock: top;
        height: 3;
        background: $panel;
        padding: 0 2;
    }

    SecurityAuditorPanel > #sa-tabs {
        dock: top;
        height: 3;
        background: $panel;
    }

    SecurityAuditorPanel > #sa-content {
        height: 1fr;
    }

    .control-button {
        width: 1fr;
        margin: 0 1;
    }

    .tab-button {
        width: 1fr;
        margin: 0 1;
    }

    .tab-button.active {
        background: $primary;
    }
    """

    BINDINGS = [
        Binding("ctrl+s", "scan", "Scan"),
        Binding("ctrl+d", "dependencies", "Dependencies"),
        Binding("ctrl+r", "report", "Report"),
        Binding("1", "show_dashboard", "Dashboard"),
        Binding("2", "show_vulnerabilities", "Vulnerabilities"),
        Binding("3", "show_dependencies", "Dependencies"),
        Binding("4", "show_fixes", "Fixes"),
        Binding("5", "show_history", "History"),
    ]

    class SecurityAction(Message):
        """Message sent when a security action is requested"""
        def __init__(self, action: str, path: str = None, **kwargs):
            super().__init__()
            self.action = action
            self.path = path
            self.kwargs = kwargs

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_tab = "dashboard"
        self.scan_path = "."

    def compose(self) -> ComposeResult:
        """Compose the security auditor panel"""
        yield Label("🔒 Security Auditor", id="sa-header")

        with Horizontal(id="sa-controls"):
            yield Button("🔍 Scan", classes="control-button", id="btn-scan")
            yield Button("📦 Dependencies", classes="control-button", id="btn-dependencies")
            yield Button("📄 Report", classes="control-button", id="btn-report")
            yield Button("🔧 Audit", classes="control-button", id="btn-audit")

        with Horizontal(id="sa-tabs"):
            yield Button("Dashboard", classes="tab-button active", id="tab-dashboard")
            yield Button("Vulnerabilities", classes="tab-button", id="tab-vulnerabilities")
            yield Button("Dependencies", classes="tab-button", id="tab-dependencies")
            yield Button("Fixes", classes="tab-button", id="tab-fixes")
            yield Button("History", classes="tab-button", id="tab-history")

        with Container(id="sa-content"):
            yield VulnerabilityDashboard(id="dashboard-panel")
            yield VulnerabilityList(id="vulnerabilities-panel", classes="hidden")
            yield DependencyTree(id="dependencies-panel", classes="hidden")
            yield FixSuggestions(id="fixes-panel", classes="hidden")
            yield AuditHistory(id="history-panel", classes="hidden")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id

        if button_id == "btn-scan":
            self.action_scan()
        elif button_id == "btn-dependencies":
            self.action_dependencies()
        elif button_id == "btn-report":
            self.action_report()
        elif button_id == "btn-audit":
            self.action_audit()
        elif button_id.startswith("tab-"):
            tab_name = button_id.replace("tab-", "")
            self._show_tab(tab_name)

    def _show_tab(self, tab_name: str) -> None:
        """Show a specific tab"""
        self.current_tab = tab_name

        # Update button styles
        for button in self.query(".tab-button"):
            if button.id == f"tab-{tab_name}":
                button.add_class("active")
            else:
                button.remove_class("active")

        # Show/hide panels
        panels = {
            "dashboard": "dashboard-panel",
            "vulnerabilities": "vulnerabilities-panel",
            "dependencies": "dependencies-panel",
            "fixes": "fixes-panel",
            "history": "history-panel"
        }

        for name, panel_id in panels.items():
            panel = self.query_one(f"#{panel_id}")
            if name == tab_name:
                panel.remove_class("hidden")
            else:
                panel.add_class("hidden")

    def action_scan(self) -> None:
        """Scan for vulnerabilities"""
        self.post_message(self.SecurityAction("scan", self.scan_path))

    def action_dependencies(self) -> None:
        """Analyze dependencies"""
        self.post_message(self.SecurityAction("dependencies", self.scan_path))

    def action_report(self) -> None:
        """Generate security report"""
        self.post_message(self.SecurityAction("report", self.scan_path))

    def action_audit(self) -> None:
        """Run full security audit"""
        self.post_message(self.SecurityAction("audit", self.scan_path))

    def action_show_dashboard(self) -> None:
        """Show dashboard tab"""
        self._show_tab("dashboard")

    def action_show_vulnerabilities(self) -> None:
        """Show vulnerabilities tab"""
        self._show_tab("vulnerabilities")

    def action_show_dependencies(self) -> None:
        """Show dependencies tab"""
        self._show_tab("dependencies")

    def action_show_fixes(self) -> None:
        """Show fixes tab"""
        self._show_tab("fixes")

    def action_show_history(self) -> None:
        """Show history tab"""
        self._show_tab("history")

    def on_vulnerability_list_vulnerability_selected(self, message: VulnerabilityList.VulnerabilitySelected) -> None:
        """Handle vulnerability selection"""
        # Find the vulnerability and show fix suggestions
        vuln_list = self.query_one("#vulnerabilities-panel", VulnerabilityList)
        for vuln in vuln_list.vulnerabilities:
            if vuln.get('id') == message.vuln_id:
                fixes_panel = self.query_one("#fixes-panel", FixSuggestions)
                fixes_panel.update_fix_suggestions(vuln)
                self._show_tab("fixes")
                break

    # Convenience methods for updating components

    def update_dashboard(self, summary: Dict[str, int]) -> None:
        """Update vulnerability dashboard"""
        dashboard = self.query_one("#dashboard-panel", VulnerabilityDashboard)
        dashboard.update_summary(summary)

    def update_vulnerabilities(self, vulnerabilities: List[Dict[str, Any]]) -> None:
        """Update vulnerability list"""
        vuln_list = self.query_one("#vulnerabilities-panel", VulnerabilityList)
        vuln_list.update_vulnerabilities(vulnerabilities)

    def update_dependencies(self, dependencies: List[Dict[str, Any]]) -> None:
        """Update dependency tree"""
        dep_tree = self.query_one("#dependencies-panel", DependencyTree)
        dep_tree.update_dependencies(dependencies)

    def update_history(self, history: List[Dict[str, Any]]) -> None:
        """Update audit history"""
        history_panel = self.query_one("#history-panel", AuditHistory)
        history_panel.update_history(history)

    def set_scan_path(self, path: str) -> None:
        """Set the scan path"""
        self.scan_path = path
