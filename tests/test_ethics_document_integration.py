#!/usr/bin/env python3
"""
Tests for Ethics Framework Integration with Document Processing
"""

import asyncio
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from xencode.ethics_document_integration import (
    DocumentEthicsProcessor,
    HarmfulContentDetector,
    ContentCategory,
    ContentSeverity,
    ContentIssue,
    EthicsComplianceReport
)


class TestHarmfulContentDetector:
    """Test harmful content detection"""
    
    def setup_method(self):
        self.detector = HarmfulContentDetector()
    
    @pytest.mark.asyncio
    async def test_detect_hate_speech(self):
        """Test hate speech detection"""
        text = "These people are inferior and should be eliminated from society"
        issues = await self.detector.detect_harmful_content(text)
        
        assert len(issues) > 0
        hate_speech_issues = [i for i in issues if i.category == ContentCategory.HATE_SPEECH]
        assert len(hate_speech_issues) > 0
        assert hate_speech_issues[0].confidence > 0.4
    
    @pytest.mark.asyncio
    async def test_detect_violence(self):
        """Test violence detection"""
        text = "We should kill these people who disagree with us"
        issues = await self.detector.detect_harmful_content(text)
        
        violence_issues = [i for i in issues if i.category == ContentCategory.VIOLENCE]
        assert len(violence_issues) > 0
        assert violence_issues[0].severity in [ContentSeverity.HIGH, ContentSeverity.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_detect_toxicity(self):
        """Test general toxicity detection"""
        text = "This is toxic harmful offensive abusive content that threatens people"
        issues = await self.detector.detect_harmful_content(text)
        
        toxicity_issues = [i for i in issues if i.category == ContentCategory.TOXICITY]
        assert len(toxicity_issues) > 0
    
    @pytest.mark.asyncio
    async def test_clean_content(self):
        """Test that clean content doesn't trigger false positives"""
        text = "Welcome to our professional workplace. We value diversity and inclusion."
        issues = await self.detector.detect_harmful_content(text)
        
        # Should have no high-confidence issues
        high_confidence_issues = [i for i in issues if i.confidence > 0.7]
        assert len(high_confidence_issues) == 0
    
    @pytest.mark.asyncio
    async def test_context_adjustment(self):
        """Test that context affects confidence scoring"""
        text = "These people are inferior in this example of what not to say"
        
        # Test with educational context (should lower confidence)
        educational_context = {"purpose": "educational"}
        issues_edu = await self.detector.detect_harmful_content(text, educational_context)
        
        # Test with public context (should raise confidence)
        public_context = {"visibility": "public"}
        issues_public = await self.detector.detect_harmful_content(text, public_context)
        
        if issues_edu and issues_public:
            # Educational context should have lower confidence
            assert issues_edu[0].confidence < issues_public[0].confidence


class TestDocumentEthicsProcessor:
    """Test document ethics processing"""
    
    def setup_method(self):
        self.processor = DocumentEthicsProcessor()
    
    @pytest.mark.asyncio
    async def test_process_clean_document(self):
        """Test processing of clean document"""
        content = "This is a professional document with appropriate content."
        metadata = {"id": "test1", "name": "Clean Document"}
        
        report = await self.processor.process_document_with_ethics(content, metadata)
        
        assert isinstance(report, EthicsComplianceReport)
        assert report.document_id == "test1"
        assert report.document_name == "Clean Document"
        assert report.compliance_score >= 0.7  # Should be compliant
        assert len(report.issues_found) == 0 or all(i.confidence < 0.5 for i in report.issues_found)
    
    @pytest.mark.asyncio
    async def test_process_problematic_document(self):
        """Test processing of document with issues"""
        content = "Men are naturally better at technical work. Women should stick to administrative roles."
        metadata = {"id": "test2", "name": "Biased Document"}
        
        report = await self.processor.process_document_with_ethics(content, metadata)
        
        assert isinstance(report, EthicsComplianceReport)
        assert report.compliance_score < 0.7  # Should not be compliant
        assert len(report.issues_found) > 0
        assert len(report.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_empty_document(self):
        """Test processing of empty document"""
        content = ""
        metadata = {"id": "test3", "name": "Empty Document"}
        
        report = await self.processor.process_document_with_ethics(content, metadata)
        
        assert report.compliance_score == 0.0
        assert "no content" in report.recommendations[0].lower()
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing of multiple documents"""
        documents = [
            ("Clean professional content", {"id": "doc1", "name": "Doc 1"}),
            ("Toxic harmful offensive content", {"id": "doc2", "name": "Doc 2"}),
            ("Normal business document", {"id": "doc3", "name": "Doc 3"})
        ]
        
        reports = await self.processor.batch_process_documents(documents)
        
        assert len(reports) == 3
        assert all(isinstance(r, EthicsComplianceReport) for r in reports)
        
        # Check that different documents have different compliance scores
        scores = [r.compliance_score for r in reports]
        assert len(set(scores)) > 1  # Should have different scores
    
    def test_compliance_summary(self):
        """Test compliance summary generation"""
        # Create mock reports
        reports = [
            EthicsComplianceReport(
                document_id="1", document_name="Doc 1", processed_at=None,
                compliance_score=0.9, issues_found=[], bias_detections=[],
                privacy_violations=[], recommendations=[]
            ),
            EthicsComplianceReport(
                document_id="2", document_name="Doc 2", processed_at=None,
                compliance_score=0.3, issues_found=[
                    ContentIssue("1", ContentCategory.BIAS, ContentSeverity.HIGH, 
                               "Test issue", "location", 0.8, "action")
                ], bias_detections=[], privacy_violations=[], recommendations=[]
            )
        ]
        
        summary = self.processor.get_compliance_summary(reports)
        
        assert summary["total_documents"] == 2
        assert summary["compliant_documents"] == 1  # Only first doc is compliant
        assert summary["compliance_rate"] == 0.5
        assert summary["total_issues"] == 1
        assert "bias" in summary["issues_by_category"]


class TestIntegrationWithDocumentProcessor:
    """Test integration with document processor"""
    
    @pytest.mark.asyncio
    async def test_document_processor_integration(self):
        """Test that document processor can use ethics integration"""
        from xencode.document_processor import document_processor
        
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("This is a test document with professional content.")
            temp_path = Path(temp_file.name)
        
        try:
            # Test the integrated processing
            result = await document_processor.process_document_with_ethics(temp_path)
            
            assert 'processing_result' in result
            assert 'ethics_report' in result
            assert 'compliance_score' in result
            
            if result['ethics_report']:
                assert isinstance(result['ethics_report'], EthicsComplianceReport)
                assert result['compliance_score'] >= 0.0
                assert result['compliance_score'] <= 1.0
        
        finally:
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_integration_error_handling(self):
        """Test error handling in integration"""
        from xencode.document_processor import document_processor
        
        # Test with non-existent file
        result = await document_processor.process_document_with_ethics("nonexistent.txt")
        
        assert 'error' in result
        assert not result.get('processing_result', {}).get('success', True)


class TestContentIssue:
    """Test ContentIssue dataclass"""
    
    def test_content_issue_creation(self):
        """Test creating content issues"""
        issue = ContentIssue(
            issue_id="test_1",
            category=ContentCategory.BIAS,
            severity=ContentSeverity.HIGH,
            description="Test bias issue",
            location="line 5",
            confidence=0.85,
            suggested_action="Review content"
        )
        
        assert issue.issue_id == "test_1"
        assert issue.category == ContentCategory.BIAS
        assert issue.severity == ContentSeverity.HIGH
        assert issue.confidence == 0.85
        assert isinstance(issue.context, dict)
        assert issue.detected_at is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])