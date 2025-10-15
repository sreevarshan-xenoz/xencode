# Analytics Reporting System

## Overview

The Analytics Reporting System provides comprehensive report generation capabilities for Xencode analytics, supporting multiple output formats, scheduled reporting, and external API integrations. This system transforms analytics data into professional reports that can be automatically generated, delivered, and consumed by external systems.

## Key Features

### 📊 Multi-Format Report Generation
- **JSON Reports**: Structured data for programmatic consumption
- **CSV Reports**: Tabular data for spreadsheet analysis
- **HTML Reports**: Professional web-ready reports with styling
- **PDF Reports**: Print-ready documents with professional formatting
- **Markdown Reports**: Documentation-friendly format

### 📅 Scheduled Reporting
- **Automated Generation**: Schedule reports for daily, weekly, monthly intervals
- **Custom Schedules**: Flexible scheduling with custom intervals
- **Multiple Delivery Methods**: File system, email, webhook delivery
- **Report Management**: Track, monitor, and manage scheduled reports

### 🌐 Analytics API
- **RESTful Endpoints**: Access analytics data via HTTP API
- **Authentication**: Secure API access with token-based authentication
- **Rate Limiting**: Protect against API abuse
- **OpenAPI Documentation**: Interactive API documentation with Swagger

### 🎛️ Customization
- **Custom Filters**: Filter data by confidence, savings, time periods
- **Template System**: Customizable report templates and layouts
- **Branding**: Professional styling and corporate branding support
- **Data Aggregation**: Flexible data grouping and summarization

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                Analytics Reporting System                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Report          │  │ Report          │  │ Analytics    │ │
│  │ Generator       │  │ Scheduler       │  │ API          │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ HTML Template   │  │ PDF Template    │  │ Custom       │ │
│  │ Engine          │  │ Engine          │  │ Templates    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Advanced Analytics Engine                 │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Analytics Data Collection**: Raw analytics data from various sources
2. **Report Configuration**: Define report type, format, and filters
3. **Data Processing**: Apply filters and aggregations
4. **Template Rendering**: Generate formatted output using templates
5. **Delivery**: Save to file, send via email, or serve via API
6. **Scheduling**: Automated recurring report generation

## Installation and Setup

### Dependencies

```bash
# Core dependencies
pip install rich

# Optional dependencies for enhanced features
pip install reportlab  # For PDF generation
pip install fastapi uvicorn  # For API functionality
pip install matplotlib  # For chart generation
```

### Basic Usage

```python
from xencode.analytics_reporting_system import (
    AnalyticsReportingSystem, ReportConfig, ReportFormat, ReportType
)
import asyncio

async def main():
    # Create reporting system
    reporting_system = AnalyticsReportingSystem()
    
    # Start the system
    await reporting_system.start()
    
    # Configure report
    config = ReportConfig(
        report_type=ReportType.SUMMARY,
        format=ReportFormat.HTML,
        title="Monthly Analytics Report",
        description="Comprehensive monthly analytics summary",
        time_period_hours=720,  # 30 days
        include_charts=True,
        include_recommendations=True
    )
    
    # Generate report
    report = await reporting_system.generate_report(config)
    
    # Save to file
    await reporting_system.save_report(report, Path("monthly_report.html"))
    
    # Stop the system
    await reporting_system.stop()

asyncio.run(main())
```

## API Reference

### AnalyticsReportingSystem

```python
class AnalyticsReportingSystem:
    def __init__(self, analytics_engine: Optional[AdvancedAnalyticsEngine] = None)
    async def start(self) -> None
    async def stop(self) -> None
    async def generate_report(self, config: ReportConfig) -> GeneratedReport
    def schedule_report(self, report_config: ReportConfig, delivery_config: DeliveryConfig) -> str
    async def save_report(self, report: GeneratedReport, file_path: Path) -> None
    def get_report_history(self) -> List[GeneratedReport]
    def get_scheduled_reports(self) -> Dict[str, Dict[str, Any]]
```

### ReportConfig

```python
@dataclass
class ReportConfig:
    report_type: ReportType
    format: ReportFormat
    title: str
    description: Optional[str] = None
    time_period_hours: int = 24
    include_charts: bool = True
    include_recommendations: bool = True
    custom_filters: Dict[str, Any] = field(default_factory=dict)
    template_path: Optional[Path] = None
```

### DeliveryConfig

```python
@dataclass
class DeliveryConfig:
    method: DeliveryMethod
    destination: str  # file path, email address, webhook URL
    schedule: Optional[str] = None  # cron-like schedule
    subject: Optional[str] = None  # for email delivery
    webhook_headers: Dict[str, str] = field(default_factory=dict)
```

## Report Formats

### JSON Reports

Structured data format ideal for programmatic consumption:

```json
{
  "report_metadata": {
    "title": "Analytics Report",
    "generated_at": "2025-10-15T10:30:00",
    "time_period_hours": 24,
    "report_type": "summary"
  },
  "analytics_data": {
    "summary": {
      "patterns_detected": 5,
      "users_analyzed": 127,
      "optimizations_found": 8,
      "total_potential_savings": 342.50
    },
    "usage_patterns": [...],
    "cost_optimizations": [...],
    "roi_projections": {...}
  }
}
```

### CSV Reports

Tabular format for spreadsheet analysis:

```csv
# Monthly Analytics Report
# Generated: 2025-10-15T10:30:00

Summary Metrics
Metric,Value
Patterns Detected,5
Users Analyzed,127
Optimizations Found,8
Potential Savings,$342.50

Usage Patterns
Type,Description,Confidence,Frequency
peak_usage,High usage during business hours,85%,65%
model_dominance,GPT-4 dominates usage,92%,78%
```

### HTML Reports

Professional web-ready reports with CSS styling:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Analytics Report</title>
    <style>/* Professional CSS styling */</style>
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1>Analytics Report</h1>
            <p class="report-meta">Generated on 2025-10-15 10:30:00</p>
        </header>
        
        <section class="summary-section">
            <h2>📊 Executive Summary</h2>
            <div class="summary-grid">
                <!-- Summary cards -->
            </div>
        </section>
        
        <!-- Additional sections -->
    </div>
</body>
</html>
```

### PDF Reports

Print-ready documents using ReportLab:

- Professional formatting with headers and footers
- Tables and charts with proper styling
- Page breaks and section organization
- Corporate branding support

### Markdown Reports

Documentation-friendly format:

```markdown
# Analytics Report

**Generated:** 2025-10-15 10:30:00

## 📊 Executive Summary

- **Patterns Detected:** 5
- **Users Analyzed:** 127
- **Optimizations Found:** 8
- **Potential Savings:** $342.50

## 🔍 Usage Patterns

### 1. Peak Usage Pattern
High usage detected during business hours
**Confidence:** 85%

### 2. Model Dominance
GPT-4 dominates usage with 78% of requests
**Confidence:** 92%
```

## Scheduled Reporting

### Schedule Configuration

```python
from xencode.analytics_reporting_system import DeliveryConfig, DeliveryMethod

# Daily reports
daily_config = DeliveryConfig(
    method=DeliveryMethod.FILE,
    destination="reports/daily_summary.html",
    schedule="daily"
)

# Weekly reports
weekly_config = DeliveryConfig(
    method=DeliveryMethod.EMAIL,
    destination="team@company.com",
    schedule="weekly",
    subject="Weekly Analytics Report"
)

# Custom intervals
custom_config = DeliveryConfig(
    method=DeliveryMethod.WEBHOOK,
    destination="https://api.company.com/reports",
    schedule="every_6_hours",
    webhook_headers={"Authorization": "Bearer token"}
)
```

### Schedule Management

```python
# Schedule a report
schedule_id = reporting_system.schedule_report(report_config, delivery_config)

# List scheduled reports
scheduled_reports = reporting_system.get_scheduled_reports()

# Cancel scheduled report
del reporting_system.scheduler.scheduled_reports[schedule_id]
```

## Analytics API

### Starting the API Server

```python
from xencode.analytics_api import AnalyticsAPI
import asyncio

async def main():
    api = AnalyticsAPI()
    await api.start_server(host="0.0.0.0", port=8000)

asyncio.run(main())
```

### API Endpoints

#### GET /health
System health check

```json
{
  "status": "healthy",
  "components": {
    "analytics_engine": true,
    "reporting_system": true,
    "orchestrator": true
  },
  "active_alerts": 0,
  "uptime_seconds": 3600.5
}
```

#### GET /analytics/summary
Get analytics summary

**Parameters:**
- `hours` (int): Analysis time period in hours (default: 24)

**Response:**
```json
{
  "patterns_detected": 5,
  "users_analyzed": 127,
  "optimizations_found": 8,
  "total_potential_savings": 342.50,
  "analysis_period_hours": 24,
  "generated_at": "2025-10-15T10:30:00"
}
```

#### GET /analytics/patterns
Get usage patterns

**Parameters:**
- `hours` (int): Analysis time period
- `min_confidence` (float): Minimum confidence threshold (0-1)

**Response:**
```json
[
  {
    "pattern_id": "pattern_123",
    "pattern_type": "peak_usage",
    "description": "High usage during business hours",
    "confidence": 0.85,
    "frequency": 0.65,
    "metadata": {}
  }
]
```

#### POST /reports/generate
Generate analytics report

**Request Body:**
```json
{
  "report_type": "summary",
  "format": "html",
  "title": "Custom Analytics Report",
  "description": "Monthly analytics summary",
  "time_period_hours": 720,
  "include_charts": true,
  "include_recommendations": true,
  "custom_filters": {
    "min_confidence": 0.8,
    "min_savings": 50.0
  }
}
```

**Response:**
```json
{
  "report_id": "report_1760507143",
  "status": "generated",
  "format": "html",
  "generated_at": "2025-10-15T10:30:00",
  "download_url": "/reports/report_1760507143/download"
}
```

### Authentication

The API uses Bearer token authentication:

```bash
curl -H "Authorization: Bearer demo-token" \
     http://localhost:8000/analytics/summary
```

### Rate Limiting

- Default: 100 requests per hour per user
- Configurable limits based on user tiers
- 429 status code when limits exceeded

## Custom Templates

### Creating Custom Templates

```python
from xencode.analytics_reporting_system import ReportTemplate

class CustomReportTemplate(ReportTemplate):
    def __init__(self):
        super().__init__("custom_template")
    
    def render_summary_section(self, data: Dict[str, Any]) -> str:
        # Custom summary rendering logic
        return "<div>Custom summary content</div>"
    
    def render_data_section(self, data: Dict[str, Any]) -> str:
        # Custom data rendering logic
        return "<div>Custom data visualization</div>"
    
    def render_recommendations_section(self, data: Dict[str, Any]) -> str:
        # Custom recommendations rendering
        return "<div>Custom recommendations</div>"
```

### Template Registration

```python
# Register custom template
report_generator.templates[ReportFormat.HTML] = CustomReportTemplate()
```

## Data Filtering

### Available Filters

```python
custom_filters = {
    "min_confidence": 0.8,        # Minimum pattern confidence
    "min_savings": 100.0,         # Minimum cost savings
    "user_ids": ["user1", "user2"], # Specific users
    "date_range": {               # Custom date range
        "start": "2025-10-01",
        "end": "2025-10-15"
    },
    "models": ["gpt-4", "gpt-3.5"], # Specific models
    "categories": ["performance", "cost"] # Analysis categories
}
```

### Filter Implementation

```python
config = ReportConfig(
    report_type=ReportType.DETAILED,
    format=ReportFormat.JSON,
    title="Filtered Analytics Report",
    custom_filters=custom_filters
)

report = await reporting_system.generate_report(config)
```

## Performance Considerations

### Optimization Features

1. **Async Processing**: Non-blocking report generation
2. **Template Caching**: Reuse compiled templates
3. **Data Streaming**: Process large datasets efficiently
4. **Concurrent Generation**: Multiple reports in parallel

### Performance Metrics

- **Report Generation**: <2 seconds for standard reports
- **Large Reports**: <5 seconds for 30-day analysis
- **Memory Usage**: <50MB per report generation
- **Concurrent Reports**: Up to 10 simultaneous generations

### Scaling Considerations

```python
# Configure for high-volume usage
reporting_system = AnalyticsReportingSystem(
    max_concurrent_reports=20,
    cache_size=100,
    cleanup_interval=3600  # 1 hour
)
```

## Monitoring and Logging

### Report Generation Metrics

```python
# Track report generation
report_metrics = {
    "generation_time": 1.5,
    "content_size": 15000,
    "format": "html",
    "success": True
}
```

### Error Handling

```python
try:
    report = await reporting_system.generate_report(config)
except ReportGenerationError as e:
    logger.error(f"Report generation failed: {e}")
    # Handle error appropriately
```

### Health Monitoring

```python
# Check system health
status = reporting_system.get_system_status()
if status["status"] != "healthy":
    # Alert administrators
    send_alert(f"Reporting system unhealthy: {status}")
```

## Integration Examples

### Business Intelligence Integration

```python
# Generate data for BI tools
config = ReportConfig(
    report_type=ReportType.DETAILED,
    format=ReportFormat.CSV,
    title="BI Data Export",
    time_period_hours=168,  # Weekly data
    custom_filters={"format": "bi_export"}
)

report = await reporting_system.generate_report(config)
# Upload to BI system
upload_to_bi_system(report.content)
```

### Slack Integration

```python
# Send reports to Slack
async def send_to_slack(report: GeneratedReport):
    slack_message = {
        "text": f"📊 {report.config.title}",
        "attachments": [{
            "title": "Analytics Report Ready",
            "text": f"Report generated at {report.generated_at}",
            "color": "good"
        }]
    }
    
    await post_to_slack(slack_message)
```

### Email Delivery

```python
# Email report delivery
delivery_config = DeliveryConfig(
    method=DeliveryMethod.EMAIL,
    destination="analytics-team@company.com",
    schedule="weekly",
    subject="Weekly Analytics Report - {{date}}"
)
```

## Troubleshooting

### Common Issues

1. **PDF Generation Fails**
   ```bash
   pip install reportlab
   ```

2. **Large Report Timeouts**
   ```python
   config.time_period_hours = 168  # Reduce time period
   ```

3. **Memory Issues**
   ```python
   # Process in chunks
   config.custom_filters["batch_size"] = 1000
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
reporting_system.debug_mode = True
```

### Performance Debugging

```python
# Profile report generation
import time

start_time = time.time()
report = await reporting_system.generate_report(config)
generation_time = time.time() - start_time

print(f"Report generated in {generation_time:.2f} seconds")
```

## Future Enhancements

### Planned Features

1. **Advanced Visualizations**
   - Interactive charts with Plotly
   - Custom dashboard layouts
   - Real-time data updates

2. **Enhanced Templates**
   - Drag-and-drop template builder
   - Corporate branding themes
   - Multi-language support

3. **Advanced Scheduling**
   - Conditional report generation
   - Dynamic scheduling based on data
   - Report dependencies and workflows

4. **Integration Enhancements**
   - Native BI tool connectors
   - Cloud storage integration
   - Advanced webhook features

### Roadmap

- **Phase 1**: Enhanced visualizations and interactive reports
- **Phase 2**: Advanced template system and branding
- **Phase 3**: Workflow automation and conditional reporting
- **Phase 4**: Enterprise integrations and compliance features

## Contributing

### Development Guidelines

1. **Code Style**: Follow PEP 8 guidelines
2. **Testing**: Maintain >90% test coverage
3. **Documentation**: Update docs for new features
4. **Templates**: Follow template design patterns

### Adding New Report Formats

```python
# Example: Adding XML format
class XMLReportTemplate(ReportTemplate):
    def generate_report(self, data: Dict[str, Any], config: ReportConfig) -> str:
        # XML generation logic
        return xml_content

# Register new format
ReportFormat.XML = "xml"
report_generator.templates[ReportFormat.XML] = XMLReportTemplate()
```

## License

This analytics reporting system is part of the Xencode project and is licensed under the MIT License.