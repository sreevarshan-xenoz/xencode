#!/usr/bin/env python3
"""
Ethics Framework Integration with Document Processing

Integrates the AI ethics framework with document processing to provide:
- Bias detection for processed documents
- Content filtering for harmful material
- Ethics compliance reporting
- Automated content moderation
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib

# Import existing components
try:
    from .ai_ethics_framework import (
        EthicsFramework, BiasDetector, PrivacyAnalyzer, FairnessAnalyzer,
        EthicsViolation, EthicsViolationType, BiasType, EthicsSeverity,
        get_ethics_framework
    )
    ETHICS_AVAILABLE = True
except ImportError:
    ETHICS_AVAILABLE = False
    # Mock classes for development
    class EthicsFramework:
        def __init__(self, *args, **kwargs): pass
        async def analyze_interaction(self, *args, **kwargs): return []
    class BiasDetector:
        def __init__(self, *args, **kwargs): pass
        async def detect_bias(self, *args, **kwargs): return []
    class PrivacyAnalyzer:
        def __init__(self, *args, **kwargs): pass
        async def detect_privacy_violations(self, *args, **kwargs): return []
    class FairnessAnalyzer:
        def __init__(self, *args, **kwargs): pass
        async def analyze_fairness(self, *args, **kwargs): return []
    def get_ethics_framework(): return EthicsFramework()


class ContentSeverity(str, Enum):
    """Severity levels for content issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ContentCategory(str, Enum):
    """Categories of content issues"""
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    HARASSMENT = "harassment"
    DISCRIMINATION = "discrimination"
    MISINFORMATION = "misinformation"
    PRIVACY_VIOLATION = "privacy_violation"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    BIAS = "bias"
    TOXICITY = "toxicity"


@dataclass
class ContentIssue:
    """Represents a content issue found in a document"""
    issue_id: str
    category: ContentCategory
    severity: ContentSeverity
    description: str
    location: str  # Where in the document
    confidence: float
    suggested_action: str
    context: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class EthicsComplianceReport:
    """Ethics compliance report for processed documents"""
    document_id: str
    document_name: str
    processed_at: datetime
    compliance_score: float  # 0-1 scale
    issues_found: List[ContentIssue]
    bias_detections: List[Dict[str, Any]]
    privacy_violations: List[Dict[str, Any]]
    fairness_issues: List[Dict[str, Any]]
    ethics_violations: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class HarmfulContentDetector:
    """Detects harmful content in documents"""
    
    def __init__(self):
        self.harmful_patterns = self._load_harmful_patterns()
        self.toxicity_keywords = self._load_toxicity_keywords()
    
    def _load_harmful_patterns(self) -> Dict[ContentCategory, List[str]]:
        """Load patterns for detecting harmful content"""
        return {
            ContentCategory.HATE_SPEECH: [
                r'\b(hate|despise|loathe)\b.*\b(group|people|race|religion)\b',
                r'\b(inferior|superior)\b.*\b(race|ethnicity|gender)\b',
                r'\b(should be|deserve to be)\b.*\b(eliminated|removed|banned)\b'
            ],
            ContentCategory.VIOLENCE: [
                r'\b(kill|murder|assault|attack|harm)\b.*\b(people|person|group)\b',
                r'\b(violence|violent)\b.*\b(solution|answer|way)\b',
                r'\b(threat|threaten)\b.*\b(with|of)\b.*\b(violence|harm)\b'
            ],
            ContentCategory.HARASSMENT: [
                r'\b(harass|bully|intimidate|stalk)\b',
                r'\b(repeatedly|constantly)\b.*\b(contact|message|follow)\b',
                r'\b(unwanted|unwelcome)\b.*\b(attention|contact|advances)\b'
            ],
            ContentCategory.DISCRIMINATION: [
                r'\b(discriminate|exclude|reject)\b.*\b(based on|because of)\b',
                r'\b(not hire|not accept|not allow)\b.*\b(due to|because)\b',
                r'\b(preference for|only want)\b.*\b(certain|specific)\b.*\b(type|kind)\b'
            ]
        }
    
    def _load_toxicity_keywords(self) -> Set[str]:
        """Load keywords that indicate toxic content"""
        return {
            'toxic', 'harmful', 'offensive', 'inappropriate', 'abusive',
            'threatening', 'hostile', 'aggressive', 'malicious', 'hateful'
        }
    
    async def detect_harmful_content(self, text: str, context: Dict[str, Any] = None) -> List[ContentIssue]:
        """Detect harmful content in text"""
        issues = []
        
        if not text:
            return issues
        
        text_lower = text.lower()
        
        # Check for harmful patterns
        for category, patterns in self.harmful_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    confidence = self._calculate_harm_confidence(match, text, context)
                    if confidence > 0.4:  # Threshold for reporting
                        issue = ContentIssue(
                            issue_id=f"harm_{int(time.time())}_{hash(match.group()) % 10000}",
                            category=category,
                            severity=self._determine_severity(confidence),
                            description=f"Potential {category.value} detected: {match.group()}",
                            location=f"Position {match.start()}-{match.end()}",
                            confidence=confidence,
                            suggested_action=self._get_suggested_action(category),
                            context=context or {}
                        )
                        issues.append(issue)
        
        # Check for toxicity keywords
        toxicity_issues = await self._detect_toxicity(text, context)
        issues.extend(toxicity_issues)
        
        return issues
    
    def _calculate_harm_confidence(self, match, text: str, context: Dict[str, Any] = None) -> float:
        """Calculate confidence score for harmful content detection"""
        base_confidence = 0.6
        
        # Adjust based on context
        if context:
            # Higher confidence in public-facing content
            if context.get("visibility") == "public":
                base_confidence += 0.2
            
            # Lower confidence in educational/research context
            if context.get("purpose") in ["educational", "research", "academic"]:
                base_confidence -= 0.3
        
        # Check surrounding context
        surrounding_text = text[max(0, match.start()-100):match.end()+100].lower()
        
        # Negative indicators (decrease confidence)
        if any(word in surrounding_text for word in ["example", "not", "avoid", "don't", "never"]):
            base_confidence -= 0.2
        
        # Positive indicators (increase confidence)
        if any(word in surrounding_text for word in ["should", "must", "always", "definitely"]):
            base_confidence += 0.1
        
        return max(0.0, min(1.0, base_confidence))
    
    async def _detect_toxicity(self, text: str, context: Dict[str, Any] = None) -> List[ContentIssue]:
        """Detect general toxicity in text"""
        issues = []
        words = text.lower().split()
        
        toxic_count = sum(1 for word in words if word in self.toxicity_keywords)
        
        if toxic_count > 0:
            toxicity_ratio = toxic_count / len(words)
            if toxicity_ratio > 0.02:  # More than 2% toxic words
                confidence = min(0.9, toxicity_ratio * 10)
                
                issue = ContentIssue(
                    issue_id=f"toxicity_{int(time.time())}",
                    category=ContentCategory.TOXICITY,
                    severity=self._determine_severity(confidence),
                    description=f"High toxicity detected: {toxic_count} toxic words in {len(words)} total words",
                    location="Throughout document",
                    confidence=confidence,
                    suggested_action="Review and moderate content for toxic language",
                    context=context or {}
                )
                issues.append(issue)
        
        return issues
    
    def _determine_severity(self, confidence: float) -> ContentSeverity:
        """Determine severity based on confidence score"""
        if confidence >= 0.9:
            return ContentSeverity.CRITICAL
        elif confidence >= 0.7:
            return ContentSeverity.HIGH
        elif confidence >= 0.5:
            return ContentSeverity.MEDIUM
        elif confidence >= 0.3:
            return ContentSeverity.LOW
        else:
            return ContentSeverity.INFO
    
    def _get_suggested_action(self, category: ContentCategory) -> str:
        """Get suggested action for content category"""
        actions = {
            ContentCategory.HATE_SPEECH: "Remove or flag content for review",
            ContentCategory.VIOLENCE: "Remove violent content immediately",
            ContentCategory.HARASSMENT: "Review and potentially remove harassing content",
            ContentCategory.DISCRIMINATION: "Review for discriminatory language and bias",
            ContentCategory.MISINFORMATION: "Fact-check and add disclaimers if needed",
            ContentCategory.PRIVACY_VIOLATION: "Remove or redact personal information",
            ContentCategory.INAPPROPRIATE_CONTENT: "Review content appropriateness",
            ContentCategory.BIAS: "Review for bias and consider alternative phrasing",
            ContentCategory.TOXICITY: "Moderate toxic language and tone"
        }
        return actions.get(category, "Review content for compliance")


class DocumentEthicsProcessor:
    """Main processor that integrates ethics checking with document processing"""
    
    def __init__(self):
        if ETHICS_AVAILABLE:
            self.ethics_framework = get_ethics_framework()
            self.bias_detector = BiasDetector()
            self.privacy_analyzer = PrivacyAnalyzer()
            self.fairness_analyzer = FairnessAnalyzer()
        else:
            self.ethics_framework = EthicsFramework()
            self.bias_detector = BiasDetector()
            self.privacy_analyzer = PrivacyAnalyzer()
            self.fairness_analyzer = FairnessAnalyzer()
        
        self.harmful_content_detector = HarmfulContentDetector()
        self.compliance_threshold = 0.7  # Minimum compliance score
    
    async def process_document_with_ethics(self, document_content: str, 
                                         document_metadata: Dict[str, Any] = None) -> EthicsComplianceReport:
        """Process document content with comprehensive ethics checking"""
        
        document_id = document_metadata.get("id", f"doc_{int(time.time())}")
        document_name = document_metadata.get("name", "Unknown Document")
        
        # Initialize report
        report = EthicsComplianceReport(
            document_id=document_id,
            document_name=document_name,
            processed_at=datetime.now(),
            compliance_score=1.0,  # Start with perfect score
            issues_found=[],
            bias_detections=[],
            privacy_violations=[],
            fairness_issues=[],
            ethics_violations=[],
            recommendations=[],
            metadata=document_metadata or {}
        )
        
        if not document_content:
            report.compliance_score = 0.0
            report.recommendations.append("Document has no content to analyze")
            return report
        
        # Detect harmful content using our custom detector
        harmful_issues = await self.harmful_content_detector.detect_harmful_content(
            document_content, document_metadata
        )
        report.issues_found.extend(harmful_issues)
        
        # Use the full ethics framework for comprehensive analysis
        if ETHICS_AVAILABLE:
            try:
                # Analyze the document as if it were an AI interaction
                # (treating the document content as an AI response to analyze)
                ethics_violations = await self.ethics_framework.analyze_interaction(
                    user_input="Document content analysis",
                    ai_response=document_content,
                    context=document_metadata
                )
                
                # Convert ethics violations to our format
                for violation in ethics_violations:
                    violation_dict = {
                        "id": violation.id,
                        "type": violation.violation_type.value,
                        "bias_type": violation.bias_type.value if violation.bias_type else None,
                        "severity": violation.severity.value,
                        "description": violation.description,
                        "confidence": violation.confidence_score,
                        "detected_at": violation.detected_at.isoformat()
                    }
                    report.ethics_violations.append(violation_dict)
                    
                    # Also create a ContentIssue for consistency
                    content_issue = ContentIssue(
                        issue_id=violation.id,
                        category=self._map_violation_to_category(violation.violation_type),
                        severity=self._map_ethics_severity(violation.severity),
                        description=violation.description,
                        location="Ethics framework analysis",
                        confidence=violation.confidence_score,
                        suggested_action=self._get_ethics_action(violation.violation_type)
                    )
                    report.issues_found.append(content_issue)
                
            except Exception as e:
                # If ethics framework fails, continue with basic analysis
                report.recommendations.append(f"Advanced ethics analysis failed: {str(e)}")
        
        # Detect bias using the bias detector directly
        bias_results = await self.bias_detector.detect_bias(document_content, document_metadata)
        for bias_type, confidence, description in bias_results:
            bias_detection = {
                "bias_type": bias_type.value,
                "confidence": confidence,
                "description": description,
                "detected_at": datetime.now().isoformat()
            }
            report.bias_detections.append(bias_detection)
            
            # Create content issue for bias
            if confidence > 0.5:
                bias_issue = ContentIssue(
                    issue_id=f"bias_{int(time.time())}_{hash(description) % 10000}",
                    category=ContentCategory.BIAS,
                    severity=ContentSeverity.MEDIUM if confidence > 0.7 else ContentSeverity.LOW,
                    description=f"Bias detected: {description}",
                    location="Content analysis",
                    confidence=confidence,
                    suggested_action="Review content for potential bias and consider revision"
                )
                report.issues_found.append(bias_issue)
        
        # Detect privacy violations
        privacy_results = await self.privacy_analyzer.detect_privacy_violations(
            document_content, document_metadata
        )
        for pii_type, matched_text, confidence in privacy_results:
            privacy_violation = {
                "pii_type": pii_type,
                "matched_text": matched_text[:20] + "..." if len(matched_text) > 20 else matched_text,
                "confidence": confidence,
                "detected_at": datetime.now().isoformat()
            }
            report.privacy_violations.append(privacy_violation)
            
            # Create content issue for privacy violation
            if confidence > 0.6:
                privacy_issue = ContentIssue(
                    issue_id=f"privacy_{int(time.time())}_{hash(matched_text) % 10000}",
                    category=ContentCategory.PRIVACY_VIOLATION,
                    severity=ContentSeverity.HIGH,
                    description=f"Privacy violation: {pii_type} detected",
                    location="Content analysis",
                    confidence=confidence,
                    suggested_action="Remove or redact personal information"
                )
                report.issues_found.append(privacy_issue)
        
        # Analyze fairness
        fairness_results = await self.fairness_analyzer.analyze_fairness(
            query="Document content analysis",
            response=document_content,
            context=document_metadata
        )
        for issue_type, confidence, description in fairness_results:
            fairness_issue = {
                "issue_type": issue_type,
                "confidence": confidence,
                "description": description,
                "detected_at": datetime.now().isoformat()
            }
            report.fairness_issues.append(fairness_issue)
            
            # Create content issue for fairness
            if confidence > 0.5:
                fairness_content_issue = ContentIssue(
                    issue_id=f"fairness_{int(time.time())}_{hash(description) % 10000}",
                    category=ContentCategory.DISCRIMINATION,  # Map fairness to discrimination
                    severity=ContentSeverity.MEDIUM if confidence > 0.7 else ContentSeverity.LOW,
                    description=f"Fairness issue: {description}",
                    location="Content analysis",
                    confidence=confidence,
                    suggested_action="Review content for fair representation and treatment"
                )
                report.issues_found.append(fairness_content_issue)
        
        # Calculate compliance score
        report.compliance_score = self._calculate_compliance_score(report)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        return report
    
    def _map_violation_to_category(self, violation_type) -> ContentCategory:
        """Map ethics violation type to content category"""
        mapping = {
            "bias_detected": ContentCategory.BIAS,
            "privacy_violation": ContentCategory.PRIVACY_VIOLATION,
            "fairness_issue": ContentCategory.DISCRIMINATION,
            "harmful_content": ContentCategory.INAPPROPRIATE_CONTENT,
            "discrimination": ContentCategory.DISCRIMINATION,
            "misinformation": ContentCategory.MISINFORMATION
        }
        return mapping.get(violation_type.value if hasattr(violation_type, 'value') else str(violation_type), 
                          ContentCategory.INAPPROPRIATE_CONTENT)
    
    def _map_ethics_severity(self, ethics_severity) -> ContentSeverity:
        """Map ethics severity to content severity"""
        mapping = {
            "critical": ContentSeverity.CRITICAL,
            "high": ContentSeverity.HIGH,
            "medium": ContentSeverity.MEDIUM,
            "low": ContentSeverity.LOW,
            "info": ContentSeverity.INFO
        }
        return mapping.get(ethics_severity.value if hasattr(ethics_severity, 'value') else str(ethics_severity),
                          ContentSeverity.MEDIUM)
    
    def _get_ethics_action(self, violation_type) -> str:
        """Get suggested action for ethics violation type"""
        actions = {
            "bias_detected": "Review content for bias and consider inclusive alternatives",
            "privacy_violation": "Remove or redact personal information immediately",
            "fairness_issue": "Ensure fair representation and treatment of all groups",
            "harmful_content": "Remove or moderate harmful content",
            "discrimination": "Review for discriminatory language and practices",
            "misinformation": "Fact-check content and add appropriate disclaimers"
        }
        return actions.get(violation_type.value if hasattr(violation_type, 'value') else str(violation_type),
                          "Review content for ethics compliance")
    
    def _calculate_compliance_score(self, report: EthicsComplianceReport) -> float:
        """Calculate overall compliance score based on issues found"""
        if not report.issues_found:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {
            ContentSeverity.CRITICAL: 0.4,
            ContentSeverity.HIGH: 0.3,
            ContentSeverity.MEDIUM: 0.2,
            ContentSeverity.LOW: 0.1,
            ContentSeverity.INFO: 0.05
        }
        
        total_penalty = 0.0
        for issue in report.issues_found:
            penalty = severity_weights.get(issue.severity, 0.1) * issue.confidence
            total_penalty += penalty
        
        # Cap penalty at 1.0 (worst possible score is 0.0)
        total_penalty = min(1.0, total_penalty)
        
        return max(0.0, 1.0 - total_penalty)
    
    def _generate_recommendations(self, report: EthicsComplianceReport) -> List[str]:
        """Generate recommendations based on compliance report"""
        recommendations = []
        
        if report.compliance_score < self.compliance_threshold:
            recommendations.append(f"Document compliance score ({report.compliance_score:.2f}) is below threshold ({self.compliance_threshold})")
        
        # Group issues by category
        issues_by_category = {}
        for issue in report.issues_found:
            if issue.category not in issues_by_category:
                issues_by_category[issue.category] = []
            issues_by_category[issue.category].append(issue)
        
        # Generate category-specific recommendations
        for category, issues in issues_by_category.items():
            high_severity_count = sum(1 for issue in issues if issue.severity in [ContentSeverity.CRITICAL, ContentSeverity.HIGH])
            
            if high_severity_count > 0:
                recommendations.append(f"Address {high_severity_count} high-severity {category.value} issues immediately")
            
            if len(issues) > 3:
                recommendations.append(f"Multiple {category.value} issues detected - consider comprehensive content review")
        
        # Specific recommendations based on detection results
        if report.bias_detections:
            high_confidence_bias = [b for b in report.bias_detections if b["confidence"] > 0.7]
            if high_confidence_bias:
                recommendations.append("Review content for bias and consider inclusive language alternatives")
        
        if report.privacy_violations:
            recommendations.append("Remove or redact personal information to comply with privacy regulations")
        
        if report.fairness_issues:
            recommendations.append("Ensure fair representation and treatment of all groups in content")
        
        if report.ethics_violations:
            critical_violations = [v for v in report.ethics_violations if v["severity"] == "critical"]
            if critical_violations:
                recommendations.append("Address critical ethics violations immediately")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Document meets ethics compliance standards")
        else:
            recommendations.append("Consider implementing content review processes for future documents")
        
        return recommendations
    
    async def batch_process_documents(self, documents: List[Tuple[str, Dict[str, Any]]]) -> List[EthicsComplianceReport]:
        """Process multiple documents for ethics compliance"""
        reports = []
        
        for document_content, metadata in documents:
            try:
                report = await self.process_document_with_ethics(document_content, metadata)
                reports.append(report)
            except Exception as e:
                # Create error report
                error_report = EthicsComplianceReport(
                    document_id=metadata.get("id", "error_doc"),
                    document_name=metadata.get("name", "Error Document"),
                    processed_at=datetime.now(),
                    compliance_score=0.0,
                    issues_found=[],
                    bias_detections=[],
                    privacy_violations=[],
                    fairness_issues=[],
                    ethics_violations=[],
                    recommendations=[f"Processing failed: {str(e)}"],
                    metadata=metadata
                )
                reports.append(error_report)
        
        return reports
    
    def get_compliance_summary(self, reports: List[EthicsComplianceReport]) -> Dict[str, Any]:
        """Generate summary statistics from compliance reports"""
        if not reports:
            return {"total_documents": 0}
        
        total_documents = len(reports)
        compliant_documents = sum(1 for r in reports if r.compliance_score >= self.compliance_threshold)
        
        # Calculate average scores
        avg_compliance_score = sum(r.compliance_score for r in reports) / total_documents
        
        # Count issues by category
        issues_by_category = {}
        total_issues = 0
        
        for report in reports:
            for issue in report.issues_found:
                category = issue.category.value
                if category not in issues_by_category:
                    issues_by_category[category] = 0
                issues_by_category[category] += 1
                total_issues += 1
        
        # Count different types of detections
        total_bias_detections = sum(len(r.bias_detections) for r in reports)
        total_privacy_violations = sum(len(r.privacy_violations) for r in reports)
        total_fairness_issues = sum(len(r.fairness_issues) for r in reports)
        total_ethics_violations = sum(len(r.ethics_violations) for r in reports)
        
        return {
            "total_documents": total_documents,
            "compliant_documents": compliant_documents,
            "compliance_rate": compliant_documents / total_documents,
            "average_compliance_score": avg_compliance_score,
            "total_issues": total_issues,
            "issues_by_category": issues_by_category,
            "total_bias_detections": total_bias_detections,
            "total_privacy_violations": total_privacy_violations,
            "total_fairness_issues": total_fairness_issues,
            "total_ethics_violations": total_ethics_violations,
            "processed_at": datetime.now().isoformat()
        }


# Global instance for easy access
document_ethics_processor = DocumentEthicsProcessor()


# Demo function
async def run_ethics_integration_demo():
    """Run ethics document integration demo"""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        
        console = Console()
        console.print("🛡️ [bold cyan]Ethics Framework Document Integration Demo[/bold cyan]\n")
        
        # Sample documents with various content issues
        test_documents = [
            (
                "This is a great product that works well for everyone. Our team is diverse and inclusive.",
                {"id": "doc1", "name": "Good Document", "type": "marketing"}
            ),
            (
                "Men are naturally better at technical tasks than women. This is just a biological fact.",
                {"id": "doc2", "name": "Biased Document", "type": "article"}
            ),
            (
                "Contact John Doe at john.doe@email.com or call 555-123-4567 for more information.",
                {"id": "doc3", "name": "Privacy Issue Document", "type": "contact"}
            ),
            (
                "These people are inferior and should not be allowed in our community. They are a threat.",
                {"id": "doc4", "name": "Harmful Content Document", "type": "comment"}
            )
        ]
        
        processor = DocumentEthicsProcessor()
        
        console.print("🔍 Processing documents for ethics compliance...\n")
        
        reports = await processor.batch_process_documents(test_documents)
        
        # Display results
        for report in reports:
            # Create compliance panel
            compliance_color = "green" if report.compliance_score >= 0.7 else "yellow" if report.compliance_score >= 0.4 else "red"
            
            panel_content = f"Compliance Score: {report.compliance_score:.2f}\n"
            panel_content += f"Issues Found: {len(report.issues_found)}\n"
            panel_content += f"Bias Detections: {len(report.bias_detections)}\n"
            panel_content += f"Privacy Violations: {len(report.privacy_violations)}\n"
            panel_content += f"Fairness Issues: {len(report.fairness_issues)}\n"
            panel_content += f"Ethics Violations: {len(report.ethics_violations)}"
            
            console.print(Panel(
                panel_content,
                title=f"📄 {report.document_name}",
                border_style=compliance_color
            ))
            
            # Show issues if any
            if report.issues_found:
                issues_table = Table(show_header=True, header_style="bold magenta")
                issues_table.add_column("Category", style="cyan")
                issues_table.add_column("Severity", style="yellow")
                issues_table.add_column("Description", style="white")
                issues_table.add_column("Confidence", style="green")
                
                for issue in report.issues_found[:3]:  # Show first 3 issues
                    issues_table.add_row(
                        issue.category.value,
                        issue.severity.value,
                        issue.description[:50] + "..." if len(issue.description) > 50 else issue.description,
                        f"{issue.confidence:.2f}"
                    )
                
                console.print(issues_table)
            
            console.print()
        
        # Show summary
        summary = processor.get_compliance_summary(reports)
        
        console.print("📊 [bold yellow]Compliance Summary[/bold yellow]")
        summary_table = Table(show_header=False)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Documents", str(summary["total_documents"]))
        summary_table.add_row("Compliant Documents", str(summary["compliant_documents"]))
        summary_table.add_row("Compliance Rate", f"{summary['compliance_rate']:.1%}")
        summary_table.add_row("Average Score", f"{summary['average_compliance_score']:.2f}")
        summary_table.add_row("Total Issues", str(summary["total_issues"]))
        
        console.print(summary_table)
        
        console.print("\n✨ [green]Ethics integration demo complete![/green]")
        
    except ImportError:
        print("Rich library not available. Running basic demo...")
        
        # Basic demo without rich formatting
        processor = DocumentEthicsProcessor()
        
        test_content = "This is a test document with some content that may contain bias."
        metadata = {"id": "test1", "name": "Test Document"}
        
        report = await processor.process_document_with_ethics(test_content, metadata)
        
        print(f"Document: {report.document_name}")
        print(f"Compliance Score: {report.compliance_score:.2f}")
        print(f"Issues Found: {len(report.issues_found)}")
        print(f"Recommendations: {len(report.recommendations)}")
        print("Ethics integration working!")


if __name__ == "__main__":
    asyncio.run(run_ethics_integration_demo())