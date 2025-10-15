#!/usr/bin/env python3
"""
Analytics Reporting System

Comprehensive reporting system for Xencode analytics that provides:
- Multi-format report generation (JSON, CSV, PDF, HTML)
- Scheduled reporting and automated delivery
- Analytics API for external integrations
- Customizable report templates and layouts
- Data export capabilities with filtering and aggregation

Key Features:
- Multiple output formats with professional styling
- Automated report scheduling and delivery
- RESTful API endpoints for external access
- Template-based report generation
- Data visualization and charting
- Export filtering and data aggregation
- Email delivery and webhook notifications
"""

import asyncio
import json
import csv
import io
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from enum import Enum
import base64
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Import for rich formatting and tables
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# Import existing analytics components
try:
    from .advanced_analytics_engine import AdvancedAnalyticsEngine
    from .analytics_integration import IntegratedAnalyticsOrchestrator
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    # Mock classes for standalone operation
    class AdvancedAnalyticsEngine:
        def __init__(self, *args, **kwargs): pass
        async def run_comprehensive_analysis(self, *args, **kwargs): return {}
    
    class IntegratedAnalyticsOrchestrator:
        def __init__(self, *args, **kwargs): pass
        async def run_comprehensive_analysis(self, *args, **kwargs): return {}

# Try to import optional dependencies for advanced features
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table as RLTable, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ReportFormat(str, Enum):
    """Supported report formats"""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    EXCEL = "excel"


class ReportType(str, Enum):
    """Types of reports that can be generated"""
    SUMMARY = "summary"
    DETAILED = "detailed"
    USAGE_PATTERNS = "usage_patterns"
    COST_ANALYSIS = "cost_analysis"
    PERFORMANCE = "performance"
    TRENDS = "trends"
    CUSTOM = "custom"


class DeliveryMethod(str, Enum):
    """Report delivery methods"""
    FILE = "file"
    EMAIL = "email"
    WEBHOOK = "webhook"
    API = "api"


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    report_type: ReportType
    format: ReportFormat
    title: str
    description: Optional[str] = None
    time_period_hours: int = 24
    include_charts: bool = True
    include_recommendations: bool = True
    custom_filters: Dict[str, Any] = field(default_factory=dict)
    template_path: Optional[Path] = None


@dataclass
class DeliveryConfig:
    """Configuration for report delivery"""
    method: DeliveryMethod
    destination: str  # file path, email address, webhook URL
    schedule: Optional[str] = None  # cron-like schedule
    subject: Optional[str] = None  # for email delivery
    webhook_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class GeneratedReport:
    """Container for a generated report"""
    report_id: str
    config: ReportConfig
    content: Union[str, bytes]
    format: ReportFormat
    generated_at: datetime
    file_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportTemplate:
    """Base class for report templates"""
    
    def __init__(self, template_name: str):
        self.template_name = template_name
        self.console = Console()
    
    def render_summary_section(self, data: Dict[str, Any]) -> str:
        """Render summary section of the report"""
        raise NotImplementedError
    
    def render_data_section(self, data: Dict[str, Any]) -> str:
        """Render main data section of the report"""
        raise NotImplementedError
    
    def render_recommendations_section(self, data: Dict[str, Any]) -> str:
        """Render recommendations section of the report"""
        raise NotImplementedError


class HTMLReportTemplate(ReportTemplate):
    """HTML report template with professional styling"""
    
    def __init__(self):
        super().__init__("html_template")
    
    def generate_report(self, data: Dict[str, Any], config: ReportConfig) -> str:
        """Generate complete HTML report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{config.title}</title>
            <style>
                {self._get_css_styles()}
            </style>
        </head>
        <body>
            <div class="container">
                <header class="report-header">
                    <h1>{config.title}</h1>
                    <p class="report-meta">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    {f'<p class="report-description">{config.description}</p>' if config.description else ''}
                </header>
                
                <main>
                    {self.render_summary_section(data)}
                    {self.render_data_section(data)}
                    {self.render_recommendations_section(data) if config.include_recommendations else ''}
                </main>
                
                <footer class="report-footer">
                    <p>Generated by Xencode Analytics System</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def render_summary_section(self, data: Dict[str, Any]) -> str:
        """Render HTML summary section"""
        summary = data.get("summary", {})
        
        return f"""
        <section class="summary-section">
            <h2>📊 Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Patterns Detected</h3>
                    <p class="metric-value">{summary.get('patterns_detected', 0)}</p>
                </div>
                <div class="summary-card">
                    <h3>Users Analyzed</h3>
                    <p class="metric-value">{summary.get('users_analyzed', 0)}</p>
                </div>
                <div class="summary-card">
                    <h3>Optimizations Found</h3>
                    <p class="metric-value">{summary.get('optimizations_found', 0)}</p>
                </div>
                <div class="summary-card">
                    <h3>Potential Savings</h3>
                    <p class="metric-value">${summary.get('total_potential_savings', 0):.2f}</p>
                </div>
            </div>
        </section>
        """
    
    def render_data_section(self, data: Dict[str, Any]) -> str:
        """Render HTML data section"""
        sections = []
        
        # Usage Patterns
        patterns = data.get("usage_patterns", [])
        if patterns:
            patterns_html = "<h3>🔍 Usage Patterns</h3><ul class='pattern-list'>"
            for pattern in patterns[:5]:
                patterns_html += f"""
                <li class="pattern-item">
                    <strong>{pattern.get('type', 'Unknown')}</strong>
                    <p>{pattern.get('description', 'No description')}</p>
                    <span class="confidence">Confidence: {pattern.get('confidence', 0):.1%}</span>
                </li>
                """
            patterns_html += "</ul>"
            sections.append(patterns_html)
        
        # Cost Optimizations
        optimizations = data.get("cost_optimizations", [])
        if optimizations:
            opt_html = "<h3>💰 Cost Optimizations</h3><div class='optimization-list'>"
            for opt in optimizations[:5]:
                opt_html += f"""
                <div class="optimization-item">
                    <h4>{opt.get('title', 'Optimization')}</h4>
                    <p>{opt.get('description', 'No description')}</p>
                    <div class="opt-details">
                        <span class="savings">Savings: ${opt.get('potential_savings', 0):.2f}</span>
                        <span class="effort">Effort: {opt.get('implementation_effort', 'Unknown')}</span>
                    </div>
                </div>
                """
            opt_html += "</div>"
            sections.append(opt_html)
        
        # ROI Projections
        roi = data.get("roi_projections", {})
        if roi:
            roi_html = f"""
            <h3>📈 ROI Projections</h3>
            <div class="roi-section">
                <div class="roi-metric">
                    <label>Monthly Savings:</label>
                    <span>${roi.get('potential_monthly_savings', 0):.2f}</span>
                </div>
                <div class="roi-metric">
                    <label>Annual Savings:</label>
                    <span>${roi.get('potential_annual_savings', 0):.2f}</span>
                </div>
                <div class="roi-metric">
                    <label>ROI Percentage:</label>
                    <span>{roi.get('roi_percentage', 0):.1f}%</span>
                </div>
                <div class="roi-metric">
                    <label>Payback Period:</label>
                    <span>{roi.get('payback_period_months', 0):.1f} months</span>
                </div>
            </div>
            """
            sections.append(roi_html)
        
        return f'<section class="data-section">{"".join(sections)}</section>'
    
    def render_recommendations_section(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations section"""
        recommendations = []
        
        # Collect recommendations from various sources
        for opt in data.get("cost_optimizations", [])[:3]:
            recommendations.extend(opt.get("recommended_actions", []))
        
        if not recommendations:
            recommendations = ["Continue monitoring system performance", "Review analytics regularly"]
        
        rec_html = "<h3>💡 Key Recommendations</h3><ol class='recommendations-list'>"
        for rec in recommendations[:5]:
            rec_html += f"<li>{rec}</li>"
        rec_html += "</ol>"
        
        return f'<section class="recommendations-section">{rec_html}</section>'
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for HTML report"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        
        .report-header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .report-header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .report-meta {
            color: #666;
            font-size: 0.9em;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .summary-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #3498db;
        }
        
        .summary-card h3 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #27ae60;
        }
        
        .data-section {
            margin: 30px 0;
        }
        
        .data-section h3 {
            color: #2c3e50;
            margin: 20px 0 15px 0;
            padding-bottom: 5px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .pattern-list {
            list-style: none;
        }
        
        .pattern-item {
            background: #f8f9fa;
            margin: 10px 0;
            padding: 15px;
            border-radius: 5px;
            border-left: 3px solid #3498db;
        }
        
        .confidence {
            background: #e74c3c;
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.8em;
        }
        
        .optimization-item {
            background: #fff3cd;
            margin: 15px 0;
            padding: 15px;
            border-radius: 5px;
            border-left: 3px solid #f39c12;
        }
        
        .opt-details {
            margin-top: 10px;
        }
        
        .savings {
            background: #27ae60;
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
            margin-right: 10px;
        }
        
        .effort {
            background: #95a5a6;
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
        }
        
        .roi-section {
            background: #e8f5e8;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #27ae60;
        }
        
        .roi-metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 5px 0;
            border-bottom: 1px solid #ddd;
        }
        
        .roi-metric label {
            font-weight: bold;
        }
        
        .recommendations-list {
            background: #f0f8ff;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        
        .recommendations-list li {
            margin: 10px 0;
            padding-left: 10px;
        }
        
        .report-footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #666;
            font-size: 0.9em;
        }
        """


class PDFReportTemplate(ReportTemplate):
    """PDF report template using ReportLab"""
    
    def __init__(self):
        super().__init__("pdf_template")
    
    def generate_report(self, data: Dict[str, Any], config: ReportConfig) -> bytes:
        """Generate PDF report"""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph(config.title, title_style))
        story.append(Spacer(1, 12))
        
        # Metadata
        meta_text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        if config.description:
            meta_text += f"<br/>{config.description}"
        story.append(Paragraph(meta_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Summary section
        story.extend(self._create_summary_section(data, styles))
        
        # Data sections
        story.extend(self._create_data_sections(data, styles))
        
        # Recommendations
        if config.include_recommendations:
            story.extend(self._create_recommendations_section(data, styles))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _create_summary_section(self, data: Dict[str, Any], styles) -> List:
        """Create PDF summary section"""
        story = []
        summary = data.get("summary", {})
        
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Create summary table
        summary_data = [
            ['Metric', 'Value'],
            ['Patterns Detected', str(summary.get('patterns_detected', 0))],
            ['Users Analyzed', str(summary.get('users_analyzed', 0))],
            ['Optimizations Found', str(summary.get('optimizations_found', 0))],
            ['Potential Savings', f"${summary.get('total_potential_savings', 0):.2f}"]
        ]
        
        table = RLTable(summary_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_data_sections(self, data: Dict[str, Any], styles) -> List:
        """Create PDF data sections"""
        story = []
        
        # Usage Patterns
        patterns = data.get("usage_patterns", [])
        if patterns:
            story.append(Paragraph("Usage Patterns", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            for i, pattern in enumerate(patterns[:5]):
                pattern_text = f"<b>{pattern.get('type', 'Unknown')}</b><br/>"
                pattern_text += f"{pattern.get('description', 'No description')}<br/>"
                pattern_text += f"<i>Confidence: {pattern.get('confidence', 0):.1%}</i>"
                story.append(Paragraph(pattern_text, styles['Normal']))
                story.append(Spacer(1, 8))
        
        # Cost Optimizations
        optimizations = data.get("cost_optimizations", [])
        if optimizations:
            story.append(Paragraph("Cost Optimizations", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            for opt in optimizations[:5]:
                opt_text = f"<b>{opt.get('title', 'Optimization')}</b><br/>"
                opt_text += f"{opt.get('description', 'No description')}<br/>"
                opt_text += f"<i>Savings: ${opt.get('potential_savings', 0):.2f} | "
                opt_text += f"Effort: {opt.get('implementation_effort', 'Unknown')}</i>"
                story.append(Paragraph(opt_text, styles['Normal']))
                story.append(Spacer(1, 8))
        
        return story
    
    def _create_recommendations_section(self, data: Dict[str, Any], styles) -> List:
        """Create PDF recommendations section"""
        story = []
        
        story.append(Paragraph("Key Recommendations", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        recommendations = []
        for opt in data.get("cost_optimizations", [])[:3]:
            recommendations.extend(opt.get("recommended_actions", []))
        
        if not recommendations:
            recommendations = ["Continue monitoring system performance", "Review analytics regularly"]
        
        for i, rec in enumerate(recommendations[:5], 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
            story.append(Spacer(1, 6))
        
        return story


class ReportGenerator:
    """Main report generator class"""
    
    def __init__(self, analytics_engine: Optional[AdvancedAnalyticsEngine] = None):
        self.analytics_engine = analytics_engine
        self.templates = {
            ReportFormat.HTML: HTMLReportTemplate(),
            ReportFormat.PDF: PDFReportTemplate() if REPORTLAB_AVAILABLE else None
        }
        self.console = Console()
    
    async def generate_report(self, config: ReportConfig) -> GeneratedReport:
        """Generate a report based on configuration"""
        
        # Collect analytics data
        if self.analytics_engine:
            data = await self.analytics_engine.run_comprehensive_analysis(config.time_period_hours)
        else:
            data = self._generate_mock_data()
        
        # Apply custom filters if specified
        if config.custom_filters:
            data = self._apply_filters(data, config.custom_filters)
        
        # Generate report content based on format
        content = await self._generate_content(data, config)
        
        # Create report object
        report = GeneratedReport(
            report_id=f"report_{int(time.time())}",
            config=config,
            content=content,
            format=config.format,
            generated_at=datetime.now(),
            metadata={
                "data_points": len(data.get("usage_patterns", [])),
                "analysis_period": config.time_period_hours,
                "generation_time": datetime.now().isoformat()
            }
        )
        
        return report
    
    async def _generate_content(self, data: Dict[str, Any], config: ReportConfig) -> Union[str, bytes]:
        """Generate report content in specified format"""
        
        if config.format == ReportFormat.JSON:
            return self._generate_json_report(data, config)
        elif config.format == ReportFormat.CSV:
            return self._generate_csv_report(data, config)
        elif config.format == ReportFormat.HTML:
            return self._generate_html_report(data, config)
        elif config.format == ReportFormat.PDF:
            return self._generate_pdf_report(data, config)
        elif config.format == ReportFormat.MARKDOWN:
            return self._generate_markdown_report(data, config)
        else:
            raise ValueError(f"Unsupported report format: {config.format}")
    
    def _generate_json_report(self, data: Dict[str, Any], config: ReportConfig) -> str:
        """Generate JSON format report"""
        report_data = {
            "report_metadata": {
                "title": config.title,
                "description": config.description,
                "generated_at": datetime.now().isoformat(),
                "time_period_hours": config.time_period_hours,
                "report_type": config.report_type.value
            },
            "analytics_data": data
        }
        
        return json.dumps(report_data, indent=2, default=str)
    
    def _generate_csv_report(self, data: Dict[str, Any], config: ReportConfig) -> str:
        """Generate CSV format report"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([f"# {config.title}"])
        writer.writerow([f"# Generated: {datetime.now().isoformat()}"])
        writer.writerow([])
        
        # Write summary data
        summary = data.get("summary", {})
        writer.writerow(["Summary Metrics"])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Patterns Detected", summary.get('patterns_detected', 0)])
        writer.writerow(["Users Analyzed", summary.get('users_analyzed', 0)])
        writer.writerow(["Optimizations Found", summary.get('optimizations_found', 0)])
        writer.writerow(["Potential Savings", f"${summary.get('total_potential_savings', 0):.2f}"])
        writer.writerow([])
        
        # Write usage patterns
        patterns = data.get("usage_patterns", [])
        if patterns:
            writer.writerow(["Usage Patterns"])
            writer.writerow(["Type", "Description", "Confidence", "Frequency"])
            for pattern in patterns:
                writer.writerow([
                    pattern.get('type', ''),
                    pattern.get('description', ''),
                    f"{pattern.get('confidence', 0):.1%}",
                    f"{pattern.get('frequency', 0):.1%}"
                ])
            writer.writerow([])
        
        # Write cost optimizations
        optimizations = data.get("cost_optimizations", [])
        if optimizations:
            writer.writerow(["Cost Optimizations"])
            writer.writerow(["Title", "Description", "Potential Savings", "Implementation Effort"])
            for opt in optimizations:
                writer.writerow([
                    opt.get('title', ''),
                    opt.get('description', ''),
                    f"${opt.get('potential_savings', 0):.2f}",
                    opt.get('implementation_effort', '')
                ])
        
        return output.getvalue()
    
    def _generate_html_report(self, data: Dict[str, Any], config: ReportConfig) -> str:
        """Generate HTML format report"""
        template = self.templates[ReportFormat.HTML]
        return template.generate_report(data, config)
    
    def _generate_pdf_report(self, data: Dict[str, Any], config: ReportConfig) -> bytes:
        """Generate PDF format report"""
        template = self.templates[ReportFormat.PDF]
        if not template:
            raise ImportError("PDF generation requires ReportLab. Install with: pip install reportlab")
        return template.generate_report(data, config)
    
    def _generate_markdown_report(self, data: Dict[str, Any], config: ReportConfig) -> str:
        """Generate Markdown format report"""
        md_content = f"# {config.title}\n\n"
        md_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if config.description:
            md_content += f"{config.description}\n\n"
        
        # Summary section
        summary = data.get("summary", {})
        md_content += "## 📊 Executive Summary\n\n"
        md_content += f"- **Patterns Detected:** {summary.get('patterns_detected', 0)}\n"
        md_content += f"- **Users Analyzed:** {summary.get('users_analyzed', 0)}\n"
        md_content += f"- **Optimizations Found:** {summary.get('optimizations_found', 0)}\n"
        md_content += f"- **Potential Savings:** ${summary.get('total_potential_savings', 0):.2f}\n\n"
        
        # Usage patterns
        patterns = data.get("usage_patterns", [])
        if patterns:
            md_content += "## 🔍 Usage Patterns\n\n"
            for i, pattern in enumerate(patterns[:5], 1):
                md_content += f"### {i}. {pattern.get('type', 'Unknown Pattern')}\n\n"
                md_content += f"{pattern.get('description', 'No description available')}\n\n"
                md_content += f"**Confidence:** {pattern.get('confidence', 0):.1%}\n\n"
        
        # Cost optimizations
        optimizations = data.get("cost_optimizations", [])
        if optimizations:
            md_content += "## 💰 Cost Optimizations\n\n"
            for i, opt in enumerate(optimizations[:5], 1):
                md_content += f"### {i}. {opt.get('title', 'Optimization')}\n\n"
                md_content += f"{opt.get('description', 'No description available')}\n\n"
                md_content += f"- **Potential Savings:** ${opt.get('potential_savings', 0):.2f}\n"
                md_content += f"- **Implementation Effort:** {opt.get('implementation_effort', 'Unknown')}\n\n"
        
        # Recommendations
        if config.include_recommendations:
            md_content += "## 💡 Key Recommendations\n\n"
            recommendations = []
            for opt in optimizations[:3]:
                recommendations.extend(opt.get("recommended_actions", []))
            
            if not recommendations:
                recommendations = ["Continue monitoring system performance", "Review analytics regularly"]
            
            for i, rec in enumerate(recommendations[:5], 1):
                md_content += f"{i}. {rec}\n"
        
        return md_content
    
    def _apply_filters(self, data: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply custom filters to analytics data"""
        filtered_data = data.copy()
        
        # Example filters
        if "min_confidence" in filters:
            min_conf = filters["min_confidence"]
            patterns = filtered_data.get("usage_patterns", [])
            filtered_data["usage_patterns"] = [
                p for p in patterns if p.get("confidence", 0) >= min_conf
            ]
        
        if "min_savings" in filters:
            min_savings = filters["min_savings"]
            optimizations = filtered_data.get("cost_optimizations", [])
            filtered_data["cost_optimizations"] = [
                opt for opt in optimizations if opt.get("potential_savings", 0) >= min_savings
            ]
        
        return filtered_data
    
    def _generate_mock_data(self) -> Dict[str, Any]:
        """Generate mock data for testing"""
        return {
            "summary": {
                "patterns_detected": 3,
                "users_analyzed": 15,
                "optimizations_found": 2,
                "total_potential_savings": 125.50
            },
            "usage_patterns": [
                {
                    "type": "peak_usage",
                    "description": "High usage detected during 2-4 PM",
                    "confidence": 0.85,
                    "frequency": 0.65
                }
            ],
            "cost_optimizations": [
                {
                    "title": "Switch to GPT-3.5 for routine tasks",
                    "description": "Use cheaper model for simple queries",
                    "potential_savings": 75.25,
                    "implementation_effort": "low",
                    "recommended_actions": ["Implement task classification", "Route simple tasks to GPT-3.5"]
                }
            ],
            "roi_projections": {
                "potential_monthly_savings": 125.50,
                "potential_annual_savings": 1506.00,
                "roi_percentage": 85.2,
                "payback_period_months": 3.2
            }
        }


class ReportScheduler:
    """Handles scheduled report generation and delivery"""
    
    def __init__(self, report_generator: ReportGenerator):
        self.report_generator = report_generator
        self.scheduled_reports: Dict[str, Dict[str, Any]] = {}
        self.console = Console()
        self._running = False
        self._scheduler_task = None
    
    def schedule_report(self, 
                       report_config: ReportConfig, 
                       delivery_config: DeliveryConfig,
                       schedule_id: Optional[str] = None) -> str:
        """Schedule a report for automatic generation"""
        
        if not schedule_id:
            schedule_id = f"scheduled_{int(time.time())}"
        
        self.scheduled_reports[schedule_id] = {
            "report_config": report_config,
            "delivery_config": delivery_config,
            "created_at": datetime.now(),
            "last_run": None,
            "next_run": self._calculate_next_run(delivery_config.schedule),
            "run_count": 0
        }
        
        return schedule_id
    
    def _calculate_next_run(self, schedule_str: Optional[str]) -> Optional[datetime]:
        """Calculate next run time from schedule string"""
        if not schedule_str:
            return None
        
        # Simple schedule parsing (extend as needed)
        if schedule_str == "daily":
            return datetime.now() + timedelta(days=1)
        elif schedule_str == "weekly":
            return datetime.now() + timedelta(weeks=1)
        elif schedule_str == "monthly":
            return datetime.now() + timedelta(days=30)
        elif schedule_str.startswith("every_"):
            # Format: "every_6_hours"
            parts = schedule_str.split("_")
            if len(parts) == 3:
                interval = int(parts[1])
                unit = parts[2]
                if unit == "hours":
                    return datetime.now() + timedelta(hours=interval)
                elif unit == "minutes":
                    return datetime.now() + timedelta(minutes=interval)
        
        return None
    
    async def start_scheduler(self) -> None:
        """Start the report scheduler"""
        if self._running:
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.console.print("📅 Report scheduler started")
    
    async def stop_scheduler(self) -> None:
        """Stop the report scheduler"""
        self._running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        self.console.print("📅 Report scheduler stopped")
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        while self._running:
            try:
                current_time = datetime.now()
                
                for schedule_id, schedule_info in self.scheduled_reports.items():
                    next_run = schedule_info.get("next_run")
                    
                    if next_run and current_time >= next_run:
                        await self._execute_scheduled_report(schedule_id, schedule_info)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.console.print(f"Error in scheduler loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _execute_scheduled_report(self, schedule_id: str, schedule_info: Dict[str, Any]) -> None:
        """Execute a scheduled report"""
        try:
            report_config = schedule_info["report_config"]
            delivery_config = schedule_info["delivery_config"]
            
            # Generate report
            report = await self.report_generator.generate_report(report_config)
            
            # Deliver report
            await self._deliver_report(report, delivery_config)
            
            # Update schedule info
            schedule_info["last_run"] = datetime.now()
            schedule_info["run_count"] += 1
            schedule_info["next_run"] = self._calculate_next_run(delivery_config.schedule)
            
            self.console.print(f"✅ Scheduled report {schedule_id} executed successfully")
            
        except Exception as e:
            self.console.print(f"❌ Error executing scheduled report {schedule_id}: {e}")
    
    async def _deliver_report(self, report: GeneratedReport, delivery_config: DeliveryConfig) -> None:
        """Deliver report using specified method"""
        
        if delivery_config.method == DeliveryMethod.FILE:
            await self._deliver_to_file(report, delivery_config)
        elif delivery_config.method == DeliveryMethod.EMAIL:
            await self._deliver_via_email(report, delivery_config)
        elif delivery_config.method == DeliveryMethod.WEBHOOK:
            await self._deliver_via_webhook(report, delivery_config)
        else:
            raise ValueError(f"Unsupported delivery method: {delivery_config.method}")
    
    async def _deliver_to_file(self, report: GeneratedReport, delivery_config: DeliveryConfig) -> None:
        """Deliver report to file system"""
        file_path = Path(delivery_config.destination)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to filename
        timestamp = report.generated_at.strftime("%Y%m%d_%H%M%S")
        file_stem = file_path.stem
        file_suffix = file_path.suffix
        timestamped_path = file_path.parent / f"{file_stem}_{timestamp}{file_suffix}"
        
        if isinstance(report.content, bytes):
            timestamped_path.write_bytes(report.content)
        else:
            timestamped_path.write_text(report.content, encoding='utf-8')
        
        report.file_path = timestamped_path
    
    async def _deliver_via_email(self, report: GeneratedReport, delivery_config: DeliveryConfig) -> None:
        """Deliver report via email (requires SMTP configuration)"""
        # This is a basic implementation - would need proper SMTP configuration
        self.console.print(f"📧 Email delivery to {delivery_config.destination} (not implemented)")
    
    async def _deliver_via_webhook(self, report: GeneratedReport, delivery_config: DeliveryConfig) -> None:
        """Deliver report via webhook"""
        # This would use aiohttp or similar to POST to webhook URL
        self.console.print(f"🔗 Webhook delivery to {delivery_config.destination} (not implemented)")


class AnalyticsReportingSystem:
    """Main analytics reporting system orchestrator"""
    
    def __init__(self, analytics_engine: Optional[AdvancedAnalyticsEngine] = None):
        self.analytics_engine = analytics_engine
        self.report_generator = ReportGenerator(analytics_engine)
        self.scheduler = ReportScheduler(self.report_generator)
        self.console = Console()
        
        # Report storage
        self.generated_reports: Dict[str, GeneratedReport] = {}
    
    async def start(self) -> None:
        """Start the reporting system"""
        await self.scheduler.start_scheduler()
        self.console.print("🚀 Analytics Reporting System started")
    
    async def stop(self) -> None:
        """Stop the reporting system"""
        await self.scheduler.stop_scheduler()
        self.console.print("🛑 Analytics Reporting System stopped")
    
    async def generate_report(self, config: ReportConfig) -> GeneratedReport:
        """Generate a report on-demand"""
        report = await self.report_generator.generate_report(config)
        self.generated_reports[report.report_id] = report
        return report
    
    def schedule_report(self, 
                       report_config: ReportConfig, 
                       delivery_config: DeliveryConfig) -> str:
        """Schedule a recurring report"""
        return self.scheduler.schedule_report(report_config, delivery_config)
    
    async def save_report(self, report: GeneratedReport, file_path: Path) -> None:
        """Save report to file"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(report.content, bytes):
            file_path.write_bytes(report.content)
        else:
            file_path.write_text(report.content, encoding='utf-8')
        
        report.file_path = file_path
    
    def get_report_history(self) -> List[GeneratedReport]:
        """Get history of generated reports"""
        return list(self.generated_reports.values())
    
    def get_scheduled_reports(self) -> Dict[str, Dict[str, Any]]:
        """Get list of scheduled reports"""
        return self.scheduler.scheduled_reports.copy()


# Demo function
async def run_reporting_system_demo():
    """Run analytics reporting system demo"""
    console = Console()
    console.print("🚀 [bold cyan]Analytics Reporting System Demo[/bold cyan]\n")
    
    try:
        # Create reporting system
        if ANALYTICS_AVAILABLE:
            from .advanced_analytics_engine import AdvancedAnalyticsEngine
            analytics_engine = AdvancedAnalyticsEngine()
            analytics_engine.generate_sample_data(days=3)
        else:
            analytics_engine = None
        
        reporting_system = AnalyticsReportingSystem(analytics_engine)
        
        console.print("📊 Generating sample reports in multiple formats...")
        
        # Generate reports in different formats
        formats_to_test = [ReportFormat.JSON, ReportFormat.CSV, ReportFormat.HTML, ReportFormat.MARKDOWN]
        
        if REPORTLAB_AVAILABLE:
            formats_to_test.append(ReportFormat.PDF)
        
        for report_format in formats_to_test:
            config = ReportConfig(
                report_type=ReportType.SUMMARY,
                format=report_format,
                title=f"Xencode Analytics Report - {report_format.value.upper()}",
                description="Comprehensive analytics report with usage patterns and cost optimizations",
                time_period_hours=72,
                include_charts=True,
                include_recommendations=True
            )
            
            report = await reporting_system.generate_report(config)
            
            # Save report to file
            output_dir = Path("reports")
            file_extension = report_format.value
            if report_format == ReportFormat.PDF:
                file_extension = "pdf"
            
            file_path = output_dir / f"sample_report.{file_extension}"
            await reporting_system.save_report(report, file_path)
            
            console.print(f"✅ {report_format.value.upper()} report generated: {file_path}")
        
        # Demo scheduled reporting
        console.print("\n📅 Setting up scheduled reporting...")
        
        schedule_config = ReportConfig(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.HTML,
            title="Daily Analytics Summary",
            time_period_hours=24
        )
        
        delivery_config = DeliveryConfig(
            method=DeliveryMethod.FILE,
            destination="reports/daily_summary.html",
            schedule="daily"
        )
        
        schedule_id = reporting_system.schedule_report(schedule_config, delivery_config)
        console.print(f"✅ Scheduled daily report: {schedule_id}")
        
        # Show report history
        console.print("\n📋 Report Generation Summary:")
        history = reporting_system.get_report_history()
        for report in history:
            console.print(f"   📄 {report.config.format.value.upper()}: {report.report_id}")
            console.print(f"      Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
            console.print(f"      Size: {len(report.content)} {'bytes' if isinstance(report.content, bytes) else 'characters'}")
        
        console.print("\n✨ [green]Analytics reporting system demo complete![/green]")
        console.print(f"📁 Reports saved to: {Path('reports').absolute()}")
        
    except Exception as e:
        console.print(f"❌ [red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_reporting_system_demo())