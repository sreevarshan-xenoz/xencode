#!/usr/bin/env python3
"""
Tests for Analytics Reporting System

Comprehensive test suite for the analytics reporting system including
report generation, scheduling, and API functionality.
"""

import pytest
import asyncio
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Import the components to test
try:
    from xencode.analytics_reporting_system import (
        AnalyticsReportingSystem, ReportGenerator, ReportScheduler,
        ReportConfig, DeliveryConfig, GeneratedReport,
        ReportFormat, ReportType, DeliveryMethod,
        HTMLReportTemplate, PDFReportTemplate
    )
    REPORTING_AVAILABLE = True
except ImportError:
    REPORTING_AVAILABLE = False
    pytest.skip("Analytics reporting system not available", allow_module_level=True)


class TestReportGenerator:
    """Test the ReportGenerator class"""
    
    def setup_method(self):
        """Set up test environment"""
        self.generator = ReportGenerator()
    
    @pytest.mark.asyncio
    async def test_generate_json_report(self):
        """Test JSON report generation"""
        config = ReportConfig(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.JSON,
            title="Test JSON Report",
            description="Test report for JSON format",
            time_period_hours=24
        )
        
        report = await self.generator.generate_report(config)
        
        assert isinstance(report, GeneratedReport)
        assert report.format == ReportFormat.JSON
        assert report.config.title == "Test JSON Report"
        assert isinstance(report.content, str)
        
        # Validate JSON content
        json_data = json.loads(report.content)
        assert "report_metadata" in json_data
        assert "analytics_data" in json_data
        assert json_data["report_metadata"]["title"] == "Test JSON Report"
    
    @pytest.mark.asyncio
    async def test_generate_csv_report(self):
        """Test CSV report generation"""
        config = ReportConfig(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.CSV,
            title="Test CSV Report",
            time_period_hours=24
        )
        
        report = await self.generator.generate_report(config)
        
        assert isinstance(report, GeneratedReport)
        assert report.format == ReportFormat.CSV
        assert isinstance(report.content, str)
        
        # Check CSV structure
        lines = report.content.strip().split('\n')
        assert len(lines) > 5  # Should have header and data
        assert "Test CSV Report" in lines[0]
    
    @pytest.mark.asyncio
    async def test_generate_html_report(self):
        """Test HTML report generation"""
        config = ReportConfig(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.HTML,
            title="Test HTML Report",
            time_period_hours=24,
            include_recommendations=True
        )
        
        report = await self.generator.generate_report(config)
        
        assert isinstance(report, GeneratedReport)
        assert report.format == ReportFormat.HTML
        assert isinstance(report.content, str)
        
        # Check HTML structure
        assert "<!DOCTYPE html>" in report.content
        assert "Test HTML Report" in report.content
        assert "<title>" in report.content
        assert "</html>" in report.content
    
    @pytest.mark.asyncio
    async def test_generate_markdown_report(self):
        """Test Markdown report generation"""
        config = ReportConfig(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.MARKDOWN,
            title="Test Markdown Report",
            time_period_hours=24
        )
        
        report = await self.generator.generate_report(config)
        
        assert isinstance(report, GeneratedReport)
        assert report.format == ReportFormat.MARKDOWN
        assert isinstance(report.content, str)
        
        # Check Markdown structure
        assert "# Test Markdown Report" in report.content
        assert "## " in report.content  # Should have sections
        assert "**" in report.content   # Should have bold text
    
    @pytest.mark.asyncio
    async def test_custom_filters(self):
        """Test custom filters in report generation"""
        config = ReportConfig(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.JSON,
            title="Filtered Report",
            time_period_hours=24,
            custom_filters={
                "min_confidence": 0.8,
                "min_savings": 50.0
            }
        )
        
        report = await self.generator.generate_report(config)
        
        assert isinstance(report, GeneratedReport)
        json_data = json.loads(report.content)
        
        # Filters should be applied to the data
        patterns = json_data["analytics_data"].get("usage_patterns", [])
        optimizations = json_data["analytics_data"].get("cost_optimizations", [])
        
        # All patterns should meet confidence threshold
        for pattern in patterns:
            assert pattern.get("confidence", 0) >= 0.8
        
        # All optimizations should meet savings threshold
        for opt in optimizations:
            assert opt.get("potential_savings", 0) >= 50.0
    
    @pytest.mark.asyncio
    async def test_unsupported_format(self):
        """Test handling of unsupported report format"""
        config = ReportConfig(
            report_type=ReportType.SUMMARY,
            format="unsupported_format",  # Invalid format
            title="Test Report",
            time_period_hours=24
        )
        
        with pytest.raises(ValueError, match="Unsupported report format"):
            await self.generator.generate_report(config)


class TestHTMLReportTemplate:
    """Test the HTMLReportTemplate class"""
    
    def setup_method(self):
        """Set up test environment"""
        self.template = HTMLReportTemplate()
    
    def test_generate_html_report(self):
        """Test HTML report generation"""
        config = ReportConfig(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.HTML,
            title="Test HTML Template",
            description="Testing HTML template generation",
            include_recommendations=True
        )
        
        mock_data = {
            "summary": {
                "patterns_detected": 5,
                "users_analyzed": 25,
                "optimizations_found": 3,
                "total_potential_savings": 150.75
            },
            "usage_patterns": [
                {
                    "type": "peak_usage",
                    "description": "High usage during business hours",
                    "confidence": 0.85
                }
            ],
            "cost_optimizations": [
                {
                    "title": "Switch to cheaper model",
                    "description": "Use GPT-3.5 for simple tasks",
                    "potential_savings": 75.50,
                    "implementation_effort": "low",
                    "recommended_actions": ["Implement task classification"]
                }
            ],
            "roi_projections": {
                "potential_monthly_savings": 150.75,
                "potential_annual_savings": 1809.00,
                "roi_percentage": 85.2,
                "payback_period_months": 2.5
            }
        }
        
        html_content = self.template.generate_report(mock_data, config)
        
        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content
        assert "Test HTML Template" in html_content
        assert "Testing HTML template generation" in html_content
        assert "Executive Summary" in html_content
        assert "Usage Patterns" in html_content
        assert "Cost Optimizations" in html_content
        assert "ROI Projections" in html_content
        assert "Key Recommendations" in html_content
    
    def test_render_summary_section(self):
        """Test summary section rendering"""
        mock_data = {
            "summary": {
                "patterns_detected": 3,
                "users_analyzed": 15,
                "optimizations_found": 2,
                "total_potential_savings": 100.25
            }
        }
        
        summary_html = self.template.render_summary_section(mock_data)
        
        assert "Executive Summary" in summary_html
        assert "3" in summary_html  # patterns_detected
        assert "15" in summary_html  # users_analyzed
        assert "2" in summary_html   # optimizations_found
        assert "$100.25" in summary_html  # total_potential_savings
    
    def test_render_data_section(self):
        """Test data section rendering"""
        mock_data = {
            "usage_patterns": [
                {
                    "type": "temporal_peak",
                    "description": "Peak usage at 2 PM",
                    "confidence": 0.9
                }
            ],
            "cost_optimizations": [
                {
                    "title": "Model optimization",
                    "description": "Switch models for efficiency",
                    "potential_savings": 50.0,
                    "implementation_effort": "medium"
                }
            ]
        }
        
        data_html = self.template.render_data_section(mock_data)
        
        assert "Usage Patterns" in data_html
        assert "temporal_peak" in data_html
        assert "Peak usage at 2 PM" in data_html
        assert "Cost Optimizations" in data_html
        assert "Model optimization" in data_html
        assert "$50.00" in data_html


class TestReportScheduler:
    """Test the ReportScheduler class"""
    
    def setup_method(self):
        """Set up test environment"""
        self.generator = ReportGenerator()
        self.scheduler = ReportScheduler(self.generator)
    
    def test_schedule_report(self):
        """Test report scheduling"""
        report_config = ReportConfig(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.JSON,
            title="Scheduled Report",
            time_period_hours=24
        )
        
        delivery_config = DeliveryConfig(
            method=DeliveryMethod.FILE,
            destination="reports/scheduled.json",
            schedule="daily"
        )
        
        schedule_id = self.scheduler.schedule_report(report_config, delivery_config)
        
        assert schedule_id is not None
        assert schedule_id in self.scheduler.scheduled_reports
        
        schedule_info = self.scheduler.scheduled_reports[schedule_id]
        assert schedule_info["report_config"] == report_config
        assert schedule_info["delivery_config"] == delivery_config
        assert schedule_info["run_count"] == 0
    
    def test_calculate_next_run(self):
        """Test next run time calculation"""
        # Test daily schedule
        next_run = self.scheduler._calculate_next_run("daily")
        assert next_run is not None
        assert next_run > datetime.now()
        
        # Test weekly schedule
        next_run = self.scheduler._calculate_next_run("weekly")
        assert next_run is not None
        assert next_run > datetime.now()
        
        # Test custom interval
        next_run = self.scheduler._calculate_next_run("every_6_hours")
        assert next_run is not None
        assert next_run > datetime.now()
        
        # Test invalid schedule
        next_run = self.scheduler._calculate_next_run("invalid_schedule")
        assert next_run is None
    
    @pytest.mark.asyncio
    async def test_scheduler_lifecycle(self):
        """Test scheduler start and stop"""
        assert not self.scheduler._running
        
        # Start scheduler
        await self.scheduler.start_scheduler()
        assert self.scheduler._running
        assert self.scheduler._scheduler_task is not None
        
        # Stop scheduler
        await self.scheduler.stop_scheduler()
        assert not self.scheduler._running
    
    @pytest.mark.asyncio
    async def test_file_delivery(self):
        """Test file delivery method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock report
            report = GeneratedReport(
                report_id="test_report",
                config=ReportConfig(
                    report_type=ReportType.SUMMARY,
                    format=ReportFormat.JSON,
                    title="Test Report"
                ),
                content='{"test": "data"}',
                format=ReportFormat.JSON,
                generated_at=datetime.now()
            )
            
            delivery_config = DeliveryConfig(
                method=DeliveryMethod.FILE,
                destination=str(Path(temp_dir) / "test_report.json")
            )
            
            # Test delivery
            await self.scheduler._deliver_to_file(report, delivery_config)
            
            # Check file was created
            assert report.file_path is not None
            assert report.file_path.exists()
            assert report.file_path.read_text() == '{"test": "data"}'


class TestAnalyticsReportingSystem:
    """Test the main AnalyticsReportingSystem class"""
    
    def setup_method(self):
        """Set up test environment"""
        self.reporting_system = AnalyticsReportingSystem()
    
    @pytest.mark.asyncio
    async def test_system_lifecycle(self):
        """Test system start and stop"""
        # Start system
        await self.reporting_system.start()
        
        # Stop system
        await self.reporting_system.stop()
    
    @pytest.mark.asyncio
    async def test_generate_report(self):
        """Test on-demand report generation"""
        config = ReportConfig(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.JSON,
            title="On-Demand Report",
            time_period_hours=24
        )
        
        report = await self.reporting_system.generate_report(config)
        
        assert isinstance(report, GeneratedReport)
        assert report.report_id in self.reporting_system.generated_reports
        assert report.config.title == "On-Demand Report"
    
    def test_schedule_report(self):
        """Test report scheduling"""
        report_config = ReportConfig(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.HTML,
            title="Scheduled Report"
        )
        
        delivery_config = DeliveryConfig(
            method=DeliveryMethod.FILE,
            destination="reports/scheduled.html",
            schedule="weekly"
        )
        
        schedule_id = self.reporting_system.schedule_report(report_config, delivery_config)
        
        assert schedule_id is not None
        scheduled_reports = self.reporting_system.get_scheduled_reports()
        assert schedule_id in scheduled_reports
    
    @pytest.mark.asyncio
    async def test_save_report(self):
        """Test saving report to file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate report
            config = ReportConfig(
                report_type=ReportType.SUMMARY,
                format=ReportFormat.JSON,
                title="Save Test Report"
            )
            
            report = await self.reporting_system.generate_report(config)
            
            # Save report
            file_path = Path(temp_dir) / "saved_report.json"
            await self.reporting_system.save_report(report, file_path)
            
            # Verify file was saved
            assert file_path.exists()
            assert report.file_path == file_path
            
            # Verify content
            saved_content = file_path.read_text()
            assert saved_content == report.content
    
    def test_get_report_history(self):
        """Test getting report history"""
        # Initially empty
        history = self.reporting_system.get_report_history()
        assert len(history) == 0
        
        # Add some reports to history
        report1 = GeneratedReport(
            report_id="report1",
            config=ReportConfig(ReportType.SUMMARY, ReportFormat.JSON, "Report 1"),
            content="{}",
            format=ReportFormat.JSON,
            generated_at=datetime.now()
        )
        
        report2 = GeneratedReport(
            report_id="report2",
            config=ReportConfig(ReportType.SUMMARY, ReportFormat.CSV, "Report 2"),
            content="",
            format=ReportFormat.CSV,
            generated_at=datetime.now()
        )
        
        self.reporting_system.generated_reports["report1"] = report1
        self.reporting_system.generated_reports["report2"] = report2
        
        # Check history
        history = self.reporting_system.get_report_history()
        assert len(history) == 2
        assert report1 in history
        assert report2 in history


class TestIntegration:
    """Integration tests for the reporting system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_report_generation(self):
        """Test complete report generation workflow"""
        reporting_system = AnalyticsReportingSystem()
        
        # Test multiple formats
        formats_to_test = [
            ReportFormat.JSON,
            ReportFormat.CSV,
            ReportFormat.HTML,
            ReportFormat.MARKDOWN
        ]
        
        for report_format in formats_to_test:
            config = ReportConfig(
                report_type=ReportType.SUMMARY,
                format=report_format,
                title=f"E2E Test Report - {report_format.value}",
                description="End-to-end test report",
                time_period_hours=48,
                include_charts=True,
                include_recommendations=True
            )
            
            # Generate report
            report = await reporting_system.generate_report(config)
            
            # Verify report
            assert isinstance(report, GeneratedReport)
            assert report.format == report_format
            assert report.content is not None
            assert len(report.content) > 0
            assert report.generated_at is not None
    
    @pytest.mark.asyncio
    async def test_scheduled_report_workflow(self):
        """Test scheduled report workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            reporting_system = AnalyticsReportingSystem()
            
            # Schedule a report
            report_config = ReportConfig(
                report_type=ReportType.SUMMARY,
                format=ReportFormat.JSON,
                title="Scheduled Integration Test",
                time_period_hours=24
            )
            
            delivery_config = DeliveryConfig(
                method=DeliveryMethod.FILE,
                destination=str(Path(temp_dir) / "scheduled_report.json"),
                schedule="every_1_minutes"  # Fast schedule for testing
            )
            
            # Schedule report
            schedule_id = reporting_system.schedule_report(report_config, delivery_config)
            
            # Start system
            await reporting_system.start()
            
            try:
                # Wait a bit for potential execution (in real scenario)
                await asyncio.sleep(0.1)
                
                # Verify scheduling
                scheduled_reports = reporting_system.get_scheduled_reports()
                assert schedule_id in scheduled_reports
                
                schedule_info = scheduled_reports[schedule_id]
                assert schedule_info["report_config"].title == "Scheduled Integration Test"
                assert schedule_info["delivery_config"].schedule == "every_1_minutes"
                
            finally:
                await reporting_system.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_report_generation(self):
        """Test concurrent report generation"""
        reporting_system = AnalyticsReportingSystem()
        
        # Create multiple report configs
        configs = [
            ReportConfig(
                report_type=ReportType.SUMMARY,
                format=ReportFormat.JSON,
                title=f"Concurrent Report {i}",
                time_period_hours=24
            )
            for i in range(5)
        ]
        
        # Generate reports concurrently
        tasks = [
            reporting_system.generate_report(config)
            for config in configs
        ]
        
        reports = await asyncio.gather(*tasks)
        
        # Verify all reports were generated
        assert len(reports) == 5
        for i, report in enumerate(reports):
            assert isinstance(report, GeneratedReport)
            assert report.config.title == f"Concurrent Report {i}"
            assert report.report_id in reporting_system.generated_reports


class TestPerformance:
    """Performance tests for the reporting system"""
    
    @pytest.mark.asyncio
    async def test_large_report_generation(self):
        """Test generation of large reports"""
        reporting_system = AnalyticsReportingSystem()
        
        # Generate report with large time period
        config = ReportConfig(
            report_type=ReportType.DETAILED,
            format=ReportFormat.JSON,
            title="Large Report Test",
            time_period_hours=720,  # 30 days
            include_charts=True,
            include_recommendations=True
        )
        
        start_time = asyncio.get_event_loop().time()
        report = await reporting_system.generate_report(config)
        end_time = asyncio.get_event_loop().time()
        
        generation_time = end_time - start_time
        
        # Should complete in reasonable time (less than 5 seconds)
        assert generation_time < 5.0
        assert isinstance(report, GeneratedReport)
        assert len(report.content) > 1000  # Should have substantial content
    
    @pytest.mark.asyncio
    async def test_multiple_format_performance(self):
        """Test performance of generating multiple formats"""
        reporting_system = AnalyticsReportingSystem()
        
        formats = [ReportFormat.JSON, ReportFormat.CSV, ReportFormat.HTML, ReportFormat.MARKDOWN]
        
        start_time = asyncio.get_event_loop().time()
        
        # Generate all formats
        for report_format in formats:
            config = ReportConfig(
                report_type=ReportType.SUMMARY,
                format=report_format,
                title=f"Performance Test - {report_format.value}",
                time_period_hours=24
            )
            
            report = await reporting_system.generate_report(config)
            assert isinstance(report, GeneratedReport)
        
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        
        # Should complete all formats quickly (less than 3 seconds)
        assert total_time < 3.0
    
    def test_memory_usage_with_many_reports(self):
        """Test memory usage with many generated reports"""
        reporting_system = AnalyticsReportingSystem()
        
        # Generate many small reports
        for i in range(100):
            report = GeneratedReport(
                report_id=f"memory_test_{i}",
                config=ReportConfig(ReportType.SUMMARY, ReportFormat.JSON, f"Report {i}"),
                content=f'{{"report_number": {i}}}',
                format=ReportFormat.JSON,
                generated_at=datetime.now()
            )
            reporting_system.generated_reports[report.report_id] = report
        
        # Verify all reports are stored
        history = reporting_system.get_report_history()
        assert len(history) == 100
        
        # Memory usage should be reasonable (this is a basic check)
        total_content_size = sum(len(report.content) for report in history)
        assert total_content_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])