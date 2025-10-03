#!/usr/bin/env python3
"""
AI Ethics and Bias Detection Framework for Xencode

Implements ethical AI guidelines, bias detection, and transparency mechanisms
to ensure responsible AI development and deployment.
"""

import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import hashlib
import re

logger = logging.getLogger(__name__)


class EthicsViolationType(Enum):
    """Types of ethics violations"""
    BIAS_DETECTED = "bias_detected"
    PRIVACY_VIOLATION = "privacy_violation"
    FAIRNESS_ISSUE = "fairness_issue"
    TRANSPARENCY_LACK = "transparency_lack"
    HARMFUL_CONTENT = "harmful_content"
    DISCRIMINATION = "discrimination"
    MISINFORMATION = "misinformation"
    CONSENT_VIOLATION = "consent_violation"


class BiasType(Enum):
    """Types of AI bias"""
    GENDER_BIAS = "gender_bias"
    RACIAL_BIAS = "racial_bias"
    AGE_BIAS = "age_bias"
    CULTURAL_BIAS = "cultural_bias"
    SOCIOECONOMIC_BIAS = "socioeconomic_bias"
    LINGUISTIC_BIAS = "linguistic_bias"
    TECHNICAL_BIAS = "technical_bias"
    CONFIRMATION_BIAS = "confirmation_bias"


class EthicsSeverity(Enum):
    """Severity levels for ethics violations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class EthicsViolation:
    """Ethics violation data structure"""
    id: str
    violation_type: EthicsViolationType
    bias_type: Optional[BiasType]
    severity: EthicsSeverity
    description: str
    context: Dict[str, Any]
    detected_at: datetime
    user_input: Optional[str]
    ai_response: Optional[str]
    confidence_score: float
    resolved: bool = False
    resolution_action: Optional[str] = None
    resolution_date: Optional[datetime] = None


@dataclass
class EthicsMetrics:
    """Ethics monitoring metrics"""
    total_violations: int
    violations_by_type: Dict[str, int]
    violations_by_severity: Dict[str, int]
    bias_detection_rate: float
    false_positive_rate: float
    resolution_rate: float
    avg_response_time_hours: float


class BiasDetector:
    """Detects various types of bias in AI responses"""
    
    def __init__(self):
        self.bias_patterns = self._load_bias_patterns()
        self.protected_attributes = [
            "gender", "race", "ethnicity", "age", "religion", "nationality",
            "sexual_orientation", "disability", "socioeconomic_status"
        ]
    
    def _load_bias_patterns(self) -> Dict[BiasType, List[str]]:
        """Load bias detection patterns"""
        return {
            BiasType.GENDER_BIAS: [
                r"\b(he|she|his|her|him)\b.*\b(better|worse|superior|inferior)\b",
                r"\b(men|women|male|female)\b.*\b(can't|cannot|shouldn't|should not)\b",
                r"\b(boys|girls)\b.*\b(naturally|inherently)\b",
            ],
            BiasType.RACIAL_BIAS: [
                r"\b(white|black|asian|hispanic|latino)\b.*\b(more|less)\b.*\b(intelligent|capable|skilled)\b",
                r"\b(race|ethnicity)\b.*\b(determines|influences)\b.*\b(ability|performance)\b",
            ],
            BiasType.AGE_BIAS: [
                r"\b(young|old|elderly|senior)\b.*\b(can't|cannot|unable)\b",
                r"\b(millennials|boomers|gen z)\b.*\b(lazy|entitled|stubborn)\b",
            ],
            BiasType.CULTURAL_BIAS: [
                r"\b(western|eastern|american|european)\b.*\b(superior|better|advanced)\b",
                r"\b(culture|tradition)\b.*\b(primitive|backward|outdated)\b",
            ],
            BiasType.SOCIOECONOMIC_BIAS: [
                r"\b(poor|rich|wealthy|low-income)\b.*\b(deserve|fault|blame)\b",
                r"\b(class|status)\b.*\b(determines|defines)\b.*\b(worth|value)\b",
            ],
        }
    
    async def detect_bias(self, text: str, context: Dict[str, Any] = None) -> List[Tuple[BiasType, float, str]]:
        """Detect bias in text with confidence scores"""
        detected_biases = []
        
        if not text:
            return detected_biases
        
        text_lower = text.lower()
        
        for bias_type, patterns in self.bias_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    confidence = self._calculate_confidence(match, text, context)
                    if confidence > 0.3:  # Threshold for reporting
                        detected_biases.append((
                            bias_type,
                            confidence,
                            f"Pattern match: {match.group()}"
                        ))
        
        # Additional semantic analysis could be added here
        # For now, we use pattern matching as a baseline
        
        return detected_biases
    
    def _calculate_confidence(self, match, text: str, context: Dict[str, Any] = None) -> float:
        """Calculate confidence score for bias detection"""
        base_confidence = 0.5
        
        # Adjust based on context
        if context:
            # Higher confidence if in sensitive context
            if context.get("topic") in ["hiring", "evaluation", "recommendation"]:
                base_confidence += 0.2
            
            # Lower confidence if in educational/informational context
            if context.get("intent") in ["educational", "informational", "academic"]:
                base_confidence -= 0.1
        
        # Adjust based on surrounding text
        surrounding_text = text[max(0, match.start()-50):match.end()+50].lower()
        
        # Positive indicators (increase confidence)
        if any(word in surrounding_text for word in ["always", "never", "all", "none", "inherently"]):
            base_confidence += 0.2
        
        # Negative indicators (decrease confidence)
        if any(word in surrounding_text for word in ["some", "might", "could", "possibly", "example"]):
            base_confidence -= 0.1
        
        return max(0.0, min(1.0, base_confidence))


class PrivacyAnalyzer:
    """Analyzes content for privacy violations"""
    
    def __init__(self):
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        }
    
    async def detect_privacy_violations(self, text: str, context: Dict[str, Any] = None) -> List[Tuple[str, str, float]]:
        """Detect potential privacy violations in text"""
        violations = []
        
        if not text:
            return violations
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                # Check if this is intentional (e.g., in examples)
                confidence = self._assess_privacy_risk(match, text, context)
                if confidence > 0.5:
                    violations.append((
                        pii_type,
                        match.group(),
                        confidence
                    ))
        
        return violations
    
    def _assess_privacy_risk(self, match, text: str, context: Dict[str, Any] = None) -> float:
        """Assess the privacy risk of detected PII"""
        base_risk = 0.8
        
        # Lower risk if in example context
        surrounding_text = text[max(0, match.start()-30):match.end()+30].lower()
        if any(word in surrounding_text for word in ["example", "sample", "demo", "test", "placeholder"]):
            base_risk -= 0.4
        
        # Higher risk if in user data context
        if context and context.get("source") == "user_input":
            base_risk += 0.2
        
        return max(0.0, min(1.0, base_risk))


class FairnessAnalyzer:
    """Analyzes AI responses for fairness issues"""
    
    async def analyze_fairness(self, query: str, response: str, context: Dict[str, Any] = None) -> List[Tuple[str, float, str]]:
        """Analyze response for fairness issues"""
        fairness_issues = []
        
        # Check for differential treatment
        differential_treatment = await self._check_differential_treatment(query, response)
        if differential_treatment:
            fairness_issues.extend(differential_treatment)
        
        # Check for representation bias
        representation_bias = await self._check_representation_bias(response)
        if representation_bias:
            fairness_issues.extend(representation_bias)
        
        return fairness_issues
    
    async def _check_differential_treatment(self, query: str, response: str) -> List[Tuple[str, float, str]]:
        """Check for differential treatment based on protected attributes"""
        issues = []
        
        # Simple heuristic: check if response quality varies based on mentioned groups
        response_quality_indicators = ["detailed", "comprehensive", "thorough", "brief", "simple", "basic"]
        
        for indicator in response_quality_indicators:
            if indicator in response.lower():
                # This is a simplified check - in practice, you'd want more sophisticated analysis
                pass
        
        return issues
    
    async def _check_representation_bias(self, response: str) -> List[Tuple[str, float, str]]:
        """Check for representation bias in examples and references"""
        issues = []
        
        # Check gender representation in examples
        male_pronouns = len(re.findall(r'\b(he|his|him)\b', response.lower()))
        female_pronouns = len(re.findall(r'\b(she|her|hers)\b', response.lower()))
        
        total_pronouns = male_pronouns + female_pronouns
        if total_pronouns > 3:  # Only check if there are enough pronouns
            male_ratio = male_pronouns / total_pronouns
            if male_ratio > 0.8 or male_ratio < 0.2:
                issues.append((
                    "gender_representation_bias",
                    0.6,
                    f"Unbalanced gender representation: {male_ratio:.1%} male pronouns"
                ))
        
        return issues


class EthicsFramework:
    """Main ethics framework for monitoring and enforcement"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path.home() / ".xencode" / "ethics.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.bias_detector = BiasDetector()
        self.privacy_analyzer = PrivacyAnalyzer()
        self.fairness_analyzer = FairnessAnalyzer()
        
        self._init_database()
        self._load_ethics_guidelines()
    
    def _init_database(self):
        """Initialize the ethics database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ethics_violations (
                    id TEXT PRIMARY KEY,
                    violation_type TEXT NOT NULL,
                    bias_type TEXT,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    context TEXT,
                    detected_at TEXT NOT NULL,
                    user_input TEXT,
                    ai_response TEXT,
                    confidence_score REAL NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_action TEXT,
                    resolution_date TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ethics_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_date TEXT NOT NULL,
                    total_interactions INTEGER NOT NULL,
                    violations_detected INTEGER NOT NULL,
                    false_positives INTEGER NOT NULL,
                    review_notes TEXT
                )
            """)
            
            conn.commit()
    
    def _load_ethics_guidelines(self):
        """Load ethics guidelines and policies"""
        self.guidelines = {
            "fairness": "Ensure equal treatment regardless of protected attributes",
            "transparency": "Provide clear explanations for AI decisions",
            "privacy": "Protect user data and respect privacy rights",
            "accountability": "Maintain responsibility for AI system outcomes",
            "non_maleficence": "Do no harm through AI system operation",
            "beneficence": "Actively promote user and societal well-being",
            "autonomy": "Respect user agency and decision-making",
            "justice": "Ensure fair distribution of AI benefits and risks"
        }
    
    async def analyze_interaction(
        self,
        user_input: str,
        ai_response: str,
        context: Dict[str, Any] = None
    ) -> List[EthicsViolation]:
        """Analyze a user-AI interaction for ethics violations"""
        violations = []
        interaction_id = hashlib.md5(f"{user_input}{ai_response}{datetime.now()}".encode()).hexdigest()
        
        # Bias detection
        bias_results = await self.bias_detector.detect_bias(ai_response, context)
        for bias_type, confidence, description in bias_results:
            violation = EthicsViolation(
                id=f"bias_{interaction_id}_{bias_type.value}",
                violation_type=EthicsViolationType.BIAS_DETECTED,
                bias_type=bias_type,
                severity=self._get_severity_from_confidence(confidence),
                description=f"Bias detected: {description}",
                context=context or {},
                detected_at=datetime.now(),
                user_input=user_input,
                ai_response=ai_response,
                confidence_score=confidence
            )
            violations.append(violation)
        
        # Privacy analysis
        privacy_results = await self.privacy_analyzer.detect_privacy_violations(ai_response, context)
        for pii_type, pii_value, confidence in privacy_results:
            violation = EthicsViolation(
                id=f"privacy_{interaction_id}_{pii_type}",
                violation_type=EthicsViolationType.PRIVACY_VIOLATION,
                bias_type=None,
                severity=EthicsSeverity.HIGH,  # Privacy violations are always high severity
                description=f"Privacy violation: {pii_type} detected in response",
                context=context or {},
                detected_at=datetime.now(),
                user_input=user_input,
                ai_response=ai_response,
                confidence_score=confidence
            )
            violations.append(violation)
        
        # Fairness analysis
        fairness_results = await self.fairness_analyzer.analyze_fairness(user_input, ai_response, context)
        for issue_type, confidence, description in fairness_results:
            violation = EthicsViolation(
                id=f"fairness_{interaction_id}_{issue_type}",
                violation_type=EthicsViolationType.FAIRNESS_ISSUE,
                bias_type=None,
                severity=self._get_severity_from_confidence(confidence),
                description=f"Fairness issue: {description}",
                context=context or {},
                detected_at=datetime.now(),
                user_input=user_input,
                ai_response=ai_response,
                confidence_score=confidence
            )
            violations.append(violation)
        
        # Store violations
        if violations:
            await self._store_violations(violations)
        
        return violations
    
    def _get_severity_from_confidence(self, confidence: float) -> EthicsSeverity:
        """Convert confidence score to severity level"""
        if confidence >= 0.9:
            return EthicsSeverity.CRITICAL
        elif confidence >= 0.7:
            return EthicsSeverity.HIGH
        elif confidence >= 0.5:
            return EthicsSeverity.MEDIUM
        elif confidence >= 0.3:
            return EthicsSeverity.LOW
        else:
            return EthicsSeverity.INFO
    
    async def _store_violations(self, violations: List[EthicsViolation]):
        """Store ethics violations in database"""
        with sqlite3.connect(self.db_path) as conn:
            for violation in violations:
                conn.execute("""
                    INSERT OR REPLACE INTO ethics_violations 
                    (id, violation_type, bias_type, severity, description, context,
                     detected_at, user_input, ai_response, confidence_score, resolved,
                     resolution_action, resolution_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    violation.id,
                    violation.violation_type.value,
                    violation.bias_type.value if violation.bias_type else None,
                    violation.severity.value,
                    violation.description,
                    json.dumps(violation.context),
                    violation.detected_at.isoformat(),
                    violation.user_input,
                    violation.ai_response,
                    violation.confidence_score,
                    violation.resolved,
                    violation.resolution_action,
                    violation.resolution_date.isoformat() if violation.resolution_date else None
                ))
            conn.commit()
    
    async def get_ethics_metrics(self, days: int = 30) -> EthicsMetrics:
        """Get ethics metrics for the specified period"""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Total violations
            total_violations = conn.execute("""
                SELECT COUNT(*) FROM ethics_violations WHERE detected_at > ?
            """, (cutoff_date,)).fetchone()[0]
            
            # Violations by type
            type_data = conn.execute("""
                SELECT violation_type, COUNT(*) FROM ethics_violations 
                WHERE detected_at > ? GROUP BY violation_type
            """, (cutoff_date,)).fetchall()
            violations_by_type = {row[0]: row[1] for row in type_data}
            
            # Violations by severity
            severity_data = conn.execute("""
                SELECT severity, COUNT(*) FROM ethics_violations 
                WHERE detected_at > ? GROUP BY severity
            """, (cutoff_date,)).fetchall()
            violations_by_severity = {row[0]: row[1] for row in severity_data}
            
            # Resolution rate
            resolved_count = conn.execute("""
                SELECT COUNT(*) FROM ethics_violations 
                WHERE detected_at > ? AND resolved = TRUE
            """, (cutoff_date,)).fetchone()[0]
            
            resolution_rate = (resolved_count / max(total_violations, 1)) * 100
            
            # Average response time (simplified)
            avg_response_time = conn.execute("""
                SELECT AVG(
                    (julianday(resolution_date) - julianday(detected_at)) * 24
                ) FROM ethics_violations 
                WHERE detected_at > ? AND resolved = TRUE
            """, (cutoff_date,)).fetchone()[0] or 0
            
            return EthicsMetrics(
                total_violations=total_violations,
                violations_by_type=violations_by_type,
                violations_by_severity=violations_by_severity,
                bias_detection_rate=0.0,  # Would need more data to calculate
                false_positive_rate=0.0,  # Would need manual review data
                resolution_rate=resolution_rate,
                avg_response_time_hours=avg_response_time
            )
    
    async def resolve_violation(self, violation_id: str, resolution_action: str):
        """Mark an ethics violation as resolved"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE ethics_violations 
                SET resolved = TRUE, resolution_action = ?, resolution_date = ?
                WHERE id = ?
            """, (resolution_action, datetime.now().isoformat(), violation_id))
            conn.commit()
    
    async def get_ethics_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive ethics report"""
        metrics = await self.get_ethics_metrics(days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Recent violations
            recent_violations = conn.execute("""
                SELECT violation_type, severity, description, detected_at 
                FROM ethics_violations 
                WHERE detected_at > ? AND resolved = FALSE
                ORDER BY detected_at DESC LIMIT 10
            """, ((datetime.now() - timedelta(days=days)).isoformat(),)).fetchall()
            
            # Trends
            daily_violations = conn.execute("""
                SELECT DATE(detected_at) as day, COUNT(*) 
                FROM ethics_violations 
                WHERE detected_at > ?
                GROUP BY DATE(detected_at)
                ORDER BY day
            """, ((datetime.now() - timedelta(days=days)).isoformat(),)).fetchall()
        
        return {
            "report_period_days": days,
            "metrics": asdict(metrics),
            "recent_violations": [
                {
                    "type": v[0],
                    "severity": v[1],
                    "description": v[2],
                    "detected_at": v[3]
                } for v in recent_violations
            ],
            "daily_trends": [
                {"date": trend[0], "violations": trend[1]} 
                for trend in daily_violations
            ],
            "guidelines": self.guidelines,
            "recommendations": await self._generate_recommendations(metrics)
        }
    
    async def _generate_recommendations(self, metrics: EthicsMetrics) -> List[str]:
        """Generate recommendations based on ethics metrics"""
        recommendations = []
        
        if metrics.total_violations > 10:
            recommendations.append("Consider implementing additional bias detection measures")
        
        if metrics.resolution_rate < 80:
            recommendations.append("Improve violation resolution processes and response times")
        
        if "bias_detected" in metrics.violations_by_type and metrics.violations_by_type["bias_detected"] > 5:
            recommendations.append("Review and update bias detection patterns and training data")
        
        if "privacy_violation" in metrics.violations_by_type:
            recommendations.append("Implement stronger privacy protection measures")
        
        return recommendations


# Global ethics framework instance
_ethics_framework: Optional[EthicsFramework] = None


def get_ethics_framework() -> EthicsFramework:
    """Get the global ethics framework instance"""
    global _ethics_framework
    if _ethics_framework is None:
        _ethics_framework = EthicsFramework()
    return _ethics_framework


async def analyze_ai_interaction(
    user_input: str,
    ai_response: str,
    context: Dict[str, Any] = None
) -> List[EthicsViolation]:
    """Convenience function to analyze AI interaction for ethics violations"""
    framework = get_ethics_framework()
    return await framework.analyze_interaction(user_input, ai_response, context)