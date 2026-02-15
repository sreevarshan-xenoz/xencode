"""
Adversarial Attack Detection System
Implements AdversarialDefenseManager for threat detection, attack pattern recognition,
real-time threat mitigation, and attack response and recovery systems.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import json
import secrets
import hashlib
from datetime import datetime, timedelta
import numpy as np
from collections import deque, defaultdict
import re
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Types of adversarial attacks."""
    FGSM = "fgsm"  # Fast Gradient Sign Method
    PGD = "pgd"    # Projected Gradient Descent
    CW = "cw"      # Carlini & Wagner
    JSMA = "jsma"  # Jacobian-based Saliency Map Attack
    DEEP_FOOL = "deepfool"
    UNTARGETED = "untargeted"
    TARGETED = "targeted"
    POISONING = "poisoning"
    EVASION = "evasion"
    MODEL_EXTRACTION = "model_extraction"
    ADVERSARIAL_PATCH = "adversarial_patch"
    BACKDOOR_ATTACK = "backdoor_attack"


class ThreatSeverity(Enum):
    """Severity levels for threats."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DefenseStrategy(Enum):
    """Defense strategies against adversarial attacks."""
    INPUT_VALIDATION = "input_validation"
    ADVERSARIAL_TRAINING = "adversarial_training"
    DEFENSIVE_DISTILLATION = "defensive_distillation"
    DETECTION_AND_REJECTION = "detection_and_rejection"
    RANDOMIZATION = "randomization"
    FEATURE_SQUEEZING = "feature_squeezing"
    DIVERSITY_ENSEMBLE = "diversity_ensemble"


@dataclass
class AttackPattern:
    """Represents a known attack pattern."""
    pattern_id: str
    attack_type: AttackType
    signature: str  # Regex or hash pattern
    severity: ThreatSeverity
    description: str
    mitigation_strategies: List[DefenseStrategy]
    detection_rules: List[Dict[str, Any]]
    last_seen: datetime
    frequency: int


@dataclass
class ThreatDetection:
    """Result of threat detection."""
    detection_id: str
    timestamp: datetime
    attack_type: Optional[AttackType]
    severity: ThreatSeverity
    confidence_score: float
    input_sample: Any  # The suspicious input
    detected_patterns: List[str]
    mitigation_applied: List[DefenseStrategy]
    metadata: Dict[str, Any]


@dataclass
class DefenseMechanism:
    """A defense mechanism against adversarial attacks."""
    mechanism_id: str
    strategy: DefenseStrategy
    implementation: Callable
    parameters: Dict[str, Any]
    effectiveness_score: float
    last_updated: datetime
    is_active: bool
    metadata: Dict[str, Any]


class PatternMatcher:
    """Matches input patterns against known attack signatures."""
    
    def __init__(self):
        self.attack_patterns: Dict[str, AttackPattern] = {}
        
    def add_attack_pattern(self, pattern: AttackPattern):
        """Add a known attack pattern."""
        self.attack_patterns[pattern.pattern_id] = pattern
        
    def detect_patterns(self, input_data: Any) -> List[Tuple[str, float]]:
        """
        Detect known attack patterns in input data.
        
        Returns:
            List of tuples (pattern_id, confidence_score)
        """
        detections = []
        
        # Convert input to string for pattern matching
        input_str = str(input_data)
        
        for pattern_id, pattern in self.attack_patterns.items():
            # Simple string matching for demonstration
            # In a real system, this would use more sophisticated pattern matching
            if pattern.signature.lower() in input_str.lower():
                # Calculate a simple confidence based on match length and input length
                confidence = min(len(pattern.signature) / len(input_str) if input_str else 0, 1.0)
                detections.append((pattern_id, confidence))
                
        return detections


class AnomalyDetector:
    """Detects anomalous inputs that may indicate adversarial attacks."""
    
    def __init__(self, anomaly_threshold: float = 0.7):
        self.anomaly_threshold = anomaly_threshold
        self.input_history: deque = deque(maxlen=1000)  # Keep last 1000 inputs
        self.feature_statistics = {}
        
    def update_baseline(self, input_features: np.ndarray):
        """Update the baseline statistics with new input features."""
        if len(input_features) == 0:
            return
            
        # Store input for history
        self.input_history.append(input_features)
        
        # Update feature statistics
        if len(self.feature_statistics) == 0:
            # Initialize statistics
            self.feature_statistics = {
                'mean': np.mean(input_features, axis=0),
                'std': np.std(input_features, axis=0),
                'min': np.min(input_features, axis=0),
                'max': np.max(input_features, axis=0)
            }
        else:
            # Update running statistics
            n = len(self.input_history)
            new_mean = np.mean(input_features, axis=0)
            self.feature_statistics['mean'] = ((n-1) * self.feature_statistics['mean'] + new_mean) / n
            
            # Update std (simplified)
            self.feature_statistics['std'] = np.std([inp for inp in self.input_history], axis=0)
            
    def detect_anomaly(self, input_features: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if the input is anomalous.
        
        Returns:
            Tuple of (is_anomalous, anomaly_score)
        """
        if len(self.feature_statistics) == 0:
            # No baseline, assume normal
            return False, 0.0
            
        # Calculate z-score for each feature
        z_scores = np.abs((input_features - self.feature_statistics['mean']) / 
                         (self.feature_statistics['std'] + 1e-8))  # Add small value to prevent division by zero
        
        # Calculate overall anomaly score (mean of z-scores)
        anomaly_score = np.mean(z_scores)
        
        is_anomalous = anomaly_score > self.anomaly_threshold
        
        return is_anomalous, float(anomaly_score)


class AdversarialInputValidator:
    """Validates inputs for adversarial characteristics."""
    
    def __init__(self):
        self.validation_rules = [
            self._check_input_bounds,
            self._check_gradient_patterns,
            self._check_statistical_properties,
            self._check_frequency_domain
        ]
        
    def validate_input(self, input_data: Any) -> Tuple[bool, List[str]]:
        """
        Validate input for adversarial characteristics.
        
        Returns:
            Tuple of (is_valid, list_of_issues_found)
        """
        issues = []
        
        for rule in self.validation_rules:
            try:
                is_valid, issue = rule(input_data)
                if not is_valid:
                    issues.append(issue)
            except Exception as e:
                logger.warning(f"Validation rule failed: {str(e)}")
                issues.append(f"Validation error: {str(e)}")
                
        is_valid = len(issues) == 0
        return is_valid, issues
        
    def _check_input_bounds(self, input_data: Any) -> Tuple[bool, str]:
        """Check if input values are within expected bounds."""
        if isinstance(input_data, (list, np.ndarray)):
            data_array = np.array(input_data)
            if np.any(data_array < -10) or np.any(data_array > 10):  # Example bounds
                return False, "Input values exceed expected bounds"
        return True, ""
        
    def _check_gradient_patterns(self, input_data: Any) -> Tuple[bool, str]:
        """Check for unusual gradient patterns (indicative of FGSM attacks)."""
        # This is a simplified check
        # In a real system, this would analyze gradients of the model w.r.t. input
        if isinstance(input_data, np.ndarray) and input_data.size > 1:
            # Check for high-frequency patterns that might indicate adversarial perturbations
            grad = np.gradient(input_data.astype(float))
            if np.std(grad) > 5.0:  # Threshold is arbitrary for demonstration
                return False, "High gradient variance detected (possible adversarial perturbation)"
        return True, ""
        
    def _check_statistical_properties(self, input_data: Any) -> Tuple[bool, str]:
        """Check statistical properties for anomalies."""
        if isinstance(input_data, (list, np.ndarray)):
            data_array = np.array(input_data)
            if data_array.size > 1:
                # Check for unusual statistical properties
                mean_val = np.mean(data_array)
                std_val = np.std(data_array)
                
                # Example: if std is unusually high compared to mean
                if std_val > abs(mean_val) * 2 and std_val > 1.0:
                    return False, "Unusual statistical properties detected"
        return True, ""
        
    def _check_frequency_domain(self, input_data: Any) -> Tuple[bool, str]:
        """Check frequency domain for adversarial patterns."""
        if isinstance(input_data, np.ndarray) and input_data.size > 1:
            # Apply FFT to detect high-frequency adversarial patterns
            fft_result = np.fft.fft(input_data.astype(float))
            high_freq_energy = np.sum(np.abs(fft_result[len(fft_result)//2:]))
            total_energy = np.sum(np.abs(fft_result))
            
            if total_energy > 0 and high_freq_energy / total_energy > 0.7:  # Arbitrary threshold
                return False, "High frequency energy detected (possible adversarial pattern)"
        return True, ""


class ThreatMitigator:
    """Applies mitigation strategies to detected threats."""
    
    def __init__(self):
        self.mitigation_strategies = {
            DefenseStrategy.INPUT_VALIDATION: self._apply_input_validation,
            DefenseStrategy.ADVERSARIAL_TRAINING: self._apply_adversarial_training,
            DefenseStrategy.DEFENSIVE_DISTILLATION: self._apply_defensive_distillation,
            DefenseStrategy.DETECTION_AND_REJECTION: self._apply_detection_and_rejection,
            DefenseStrategy.RANDOMIZATION: self._apply_randomization,
            DefenseStrategy.FEATURE_SQUEEZING: self._apply_feature_squeezing,
            DefenseStrategy.DIVERSITY_ENSEMBLE: self._apply_diversity_ensemble
        }
        
    def apply_mitigation(
        self, 
        input_data: Any, 
        strategies: List[DefenseStrategy]
    ) -> Tuple[Any, List[DefenseStrategy]]:
        """
        Apply mitigation strategies to input data.
        
        Returns:
            Tuple of (mitigated_input, list_of_strategies_applied)
        """
        mitigated_data = input_data
        applied_strategies = []
        
        for strategy in strategies:
            if strategy in self.mitigation_strategies:
                try:
                    mitigated_data = self.mitigation_strategies[strategy](mitigated_data)
                    applied_strategies.append(strategy)
                except Exception as e:
                    logger.error(f"Failed to apply mitigation {strategy.value}: {str(e)}")
            else:
                logger.warning(f"Unknown mitigation strategy: {strategy}")
                
        return mitigated_data, applied_strategies
        
    def _apply_input_validation(self, input_data: Any) -> Any:
        """Apply input validation."""
        # Input validation is typically a pass/fail check
        # For mitigation, we might clip values to bounds
        if isinstance(input_data, np.ndarray):
            # Clip to reasonable bounds
            return np.clip(input_data, -5.0, 5.0)
        return input_data
        
    def _apply_adversarial_training(self, input_data: Any) -> Any:
        """Apply adversarial training mitigation (simulated)."""
        # In a real system, this would involve using a model trained on adversarial examples
        # For simulation, we'll add slight randomization
        if isinstance(input_data, np.ndarray):
            noise = np.random.normal(0, 0.01, input_data.shape)
            return input_data + noise
        return input_data
        
    def _apply_defensive_distillation(self, input_data: Any) -> Any:
        """Apply defensive distillation (simulated)."""
        # Defensive distillation affects the model, not the input directly
        # For simulation, we'll smooth the input slightly
        if isinstance(input_data, np.ndarray):
            # Apply a simple smoothing filter
            if input_data.ndim == 1:
                # Simple moving average
                smoothed = np.convolve(input_data, np.ones(3)/3, mode='same')
                return smoothed
        return input_data
        
    def _apply_detection_and_rejection(self, input_data: Any) -> Any:
        """Apply detection and rejection."""
        # This strategy would reject inputs, but for mitigation we'll return as-is
        # The rejection happens at the detection stage
        return input_data
        
    def _apply_randomization(self, input_data: Any) -> Any:
        """Apply randomization."""
        if isinstance(input_data, np.ndarray):
            # Add random noise
            noise = np.random.uniform(-0.05, 0.05, input_data.shape)
            return input_data + noise
        return input_data
        
    def _apply_feature_squeezing(self, input_data: Any) -> Any:
        """Apply feature squeezing."""
        if isinstance(input_data, np.ndarray):
            # Reduce precision to squeeze out small adversarial perturbations
            return np.round(input_data, decimals=1)
        return input_data
        
    def _apply_diversity_ensemble(self, input_data: Any) -> Any:
        """Apply diversity ensemble (simulated)."""
        # Ensemble methods work at the model level
        # For input mitigation, we'll return as-is
        return input_data


class AdversarialDefenseManager:
    """
    Adversarial defense manager for threat detection with attack pattern recognition,
    real-time threat mitigation, and attack response systems.
    """
    
    def __init__(self, detection_threshold: float = 0.5):
        self.pattern_matcher = PatternMatcher()
        self.anomaly_detector = AnomalyDetector()
        self.input_validator = AdversarialInputValidator()
        self.threat_mitigator = ThreatMitigator()
        self.detection_threshold = detection_threshold
        self.threat_history: List[ThreatDetection] = []
        self.defense_mechanisms: Dict[str, DefenseMechanism] = {}
        self.attack_countermeasures: Dict[AttackType, List[DefenseStrategy]] = {}
        self.response_actions: Dict[ThreatSeverity, Callable] = {}
        
        # Initialize with common attack patterns
        self._initialize_known_attack_patterns()
        self._initialize_countermeasures()
        self._initialize_response_actions()
        
    def _initialize_known_attack_patterns(self):
        """Initialize with known attack patterns."""
        # Example attack patterns (these would be more sophisticated in practice)
        patterns = [
            AttackPattern(
                pattern_id="fgsm_basic",
                attack_type=AttackType.FGSM,
                signature="gradient_sign",
                severity=ThreatSeverity.HIGH,
                description="Basic FGSM attack pattern",
                mitigation_strategies=[
                    DefenseStrategy.FEATURE_SQUEEZING,
                    DefenseStrategy.RANDOMIZATION,
                    DefenseStrategy.INPUT_VALIDATION
                ],
                detection_rules=[{"type": "gradient_analysis", "threshold": 0.5}],
                last_seen=datetime.min,
                frequency=0
            ),
            AttackPattern(
                pattern_id="pgd_iterative",
                attack_type=AttackType.PGD,
                signature="iterative_perturbation",
                severity=ThreatSeverity.CRITICAL,
                description="PGD iterative attack pattern",
                mitigation_strategies=[
                    DefenseStrategy.ADVERSARIAL_TRAINING,
                    DefenseStrategy.DIVERSITY_ENSEMBLE
                ],
                detection_rules=[{"type": "perturbation_analysis", "threshold": 0.3}],
                last_seen=datetime.min,
                frequency=0
            ),
            AttackPattern(
                pattern_id="high_freq_noise",
                attack_type=AttackType.UNTARGETED,
                signature="high_frequency",
                severity=ThreatSeverity.MEDIUM,
                description="High frequency noise pattern",
                mitigation_strategies=[
                    DefenseStrategy.FEATURE_SQUEEZING,
                    DefenseStrategy.DEFENSIVE_DISTILLATION
                ],
                detection_rules=[{"type": "frequency_analysis", "threshold": 0.7}],
                last_seen=datetime.min,
                frequency=0
            )
        ]
        
        for pattern in patterns:
            self.pattern_matcher.add_attack_pattern(pattern)
            
    def _initialize_countermeasures(self):
        """Initialize countermeasures for different attack types."""
        self.attack_countermeasures = {
            AttackType.FGSM: [
                DefenseStrategy.FEATURE_SQUEEZING,
                DefenseStrategy.RANDOMIZATION,
                DefenseStrategy.INPUT_VALIDATION
            ],
            AttackType.PGD: [
                DefenseStrategy.ADVERSARIAL_TRAINING,
                DefenseStrategy.DIVERSITY_ENSEMBLE,
                DefenseStrategy.DEFENSIVE_DISTILLATION
            ],
            AttackType.CW: [
                DefenseStrategy.DIVERSITY_ENSEMBLE,
                DefenseStrategy.DEFENSIVE_DISTILLATION,
                DefenseStrategy.FEATURE_SQUEEZING
            ],
            AttackType.JSMA: [
                DefenseStrategy.INPUT_VALIDATION,
                DefenseStrategy.FEATURE_SQUEEZING,
                DefenseStrategy.RANDOMIZATION
            ],
            AttackType.EVASION: [
                DefenseStrategy.DETECTION_AND_REJECTION,
                DefenseStrategy.ADVERSARIAL_TRAINING,
                DefenseStrategy.DIVERSITY_ENSEMBLE
            ],
            AttackType.POISONING: [
                DefenseStrategy.INPUT_VALIDATION,
                DefenseStrategy.DETECTION_AND_REJECTION,
                DefenseStrategy.ADVERSARIAL_TRAINING
            ]
        }
        
    def _initialize_response_actions(self):
        """Initialize response actions for different threat severities."""
        self.response_actions = {
            ThreatSeverity.LOW: self._respond_low_severity,
            ThreatSeverity.MEDIUM: self._respond_medium_severity,
            ThreatSeverity.HIGH: self._respond_high_severity,
            ThreatSeverity.CRITICAL: self._respond_critical_severity
        }
        
    def detect_threat(self, input_data: Any) -> ThreatDetection:
        """Detect threats in the input data."""
        detection_id = f"detect_{secrets.token_hex(8)}"
        timestamp = datetime.now()
        
        # Initialize severity and confidence
        max_severity = ThreatSeverity.LOW
        max_confidence = 0.0
        detected_attack_type = None
        detected_patterns = []
        mitigation_strategies = []
        
        # 1. Pattern matching
        matched_patterns = self.pattern_matcher.detect_patterns(input_data)
        for pattern_id, confidence in matched_patterns:
            if confidence > max_confidence:
                max_confidence = confidence
                pattern = self.pattern_matcher.attack_patterns[pattern_id]
                if pattern.severity.value > max_severity.value:
                    max_severity = pattern.severity
                    detected_attack_type = pattern.attack_type
                detected_patterns.append(pattern_id)
                mitigation_strategies.extend(pattern.mitigation_strategies)
        
        # 2. Anomaly detection
        if isinstance(input_data, (list, np.ndarray)):
            features = np.array(input_data).flatten()
            is_anomalous, anomaly_score = self.anomaly_detector.detect_anomaly(features)
            if is_anomalous and anomaly_score > max_confidence:
                max_confidence = min(anomaly_score, 1.0)
                # Set to medium severity for anomalies
                if ThreatSeverity.MEDIUM.value > max_severity.value:
                    max_severity = ThreatSeverity.MEDIUM
                detected_patterns.append("anomaly_detected")
        
        # 3. Input validation
        is_valid, validation_issues = self.input_validator.validate_input(input_data)
        if not is_valid and len(validation_issues) > 0:
            # Increase confidence based on number of validation issues
            validation_confidence = min(len(validation_issues) * 0.2, 1.0)
            if validation_confidence > max_confidence:
                max_confidence = validation_confidence
            if ThreatSeverity.MEDIUM.value > max_severity.value:
                max_severity = ThreatSeverity.MEDIUM
            detected_patterns.extend([f"validation_issue: {issue}" for issue in validation_issues])
        
        # Update baseline for anomaly detector
        if isinstance(input_data, (list, np.ndarray)):
            features = np.array(input_data).flatten()
            self.anomaly_detector.update_baseline(features)
        
        # Determine mitigation strategies based on detected attack type
        if detected_attack_type and detected_attack_type in self.attack_countermeasures:
            mitigation_strategies.extend(self.attack_countermeasures[detected_attack_type])
        
        # Apply threshold for threat classification
        if max_confidence < self.detection_threshold:
            max_severity = ThreatSeverity.LOW
            detected_attack_type = None
            detected_patterns = []
            mitigation_strategies = []
        
        # Apply mitigation strategies
        mitigated_input, applied_strategies = self.threat_mitigator.apply_mitigation(
            input_data, list(set(mitigation_strategies))
        )
        
        # Create threat detection result
        detection = ThreatDetection(
            detection_id=detection_id,
            timestamp=timestamp,
            attack_type=detected_attack_type,
            severity=max_severity,
            confidence_score=max_confidence,
            input_sample=input_data,
            detected_patterns=detected_patterns,
            mitigation_applied=applied_strategies,
            metadata={
                "original_input_type": type(input_data).__name__,
                "mitigation_applied_count": len(applied_strategies),
                "detection_method": "multi_layer_analysis"
            }
        )
        
        # Store in history
        self.threat_history.append(detection)
        
        # Trigger response based on severity
        response_action = self.response_actions.get(max_severity, self._respond_low_severity)
        response_action(detection)
        
        logger.info(f"Threat detection completed: {max_severity.value} severity, "
                   f"confidence {max_confidence:.2f}, patterns: {detected_patterns}")
        
        return detection
        
    def add_defense_mechanism(
        self, 
        strategy: DefenseStrategy, 
        implementation: Callable,
        parameters: Dict[str, Any] = None,
        effectiveness_score: float = 0.5
    ) -> str:
        """Add a custom defense mechanism."""
        mechanism_id = f"defense_{secrets.token_hex(8)}"
        
        mechanism = DefenseMechanism(
            mechanism_id=mechanism_id,
            strategy=strategy,
            implementation=implementation,
            parameters=parameters or {},
            effectiveness_score=effectiveness_score,
            last_updated=datetime.now(),
            is_active=True,
            metadata={"added_by": "defense_manager"}
        )
        
        self.defense_mechanisms[mechanism_id] = mechanism
        
        logger.info(f"Added defense mechanism: {mechanism_id} for {strategy.value}")
        return mechanism_id
        
    def get_threat_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected threats."""
        if not self.threat_history:
            return {
                "total_detections": 0,
                "threats_by_severity": {},
                "threats_by_type": {},
                "average_confidence": 0.0,
                "recent_detections": []
            }
        
        # Count threats by severity
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        total_confidence = 0.0
        
        for detection in self.threat_history:
            severity_counts[detection.severity.value] += 1
            if detection.attack_type:
                type_counts[detection.attack_type.value] += 1
            total_confidence += detection.confidence_score
        
        avg_confidence = total_confidence / len(self.threat_history)
        
        return {
            "total_detections": len(self.threat_history),
            "threats_by_severity": dict(severity_counts),
            "threats_by_type": dict(type_counts),
            "average_confidence": avg_confidence,
            "recent_detections": [
                {
                    "severity": det.severity.value,
                    "confidence": det.confidence_score,
                    "timestamp": det.timestamp.isoformat(),
                    "attack_type": det.attack_type.value if det.attack_type else None
                }
                for det in self.threat_history[-10:]  # Last 10 detections
            ]
        }
        
    def _respond_low_severity(self, detection: ThreatDetection):
        """Response for low severity threats."""
        logger.debug(f"Low severity threat detected: {detection.detection_id}")
        
    def _respond_medium_severity(self, detection: ThreatDetection):
        """Response for medium severity threats."""
        logger.info(f"Medium severity threat detected: {detection.detection_id}")
        logger.info(f"Patterns detected: {detection.detected_patterns}")
        
    def _respond_high_severity(self, detection: ThreatDetection):
        """Response for high severity threats."""
        logger.warning(f"High severity threat detected: {detection.detection_id}")
        logger.warning(f"Applying mitigation strategies: {[s.value for s in detection.mitigation_applied]}")
        
    def _respond_critical_severity(self, detection: ThreatDetection):
        """Response for critical severity threats."""
        logger.error(f"CRITICAL threat detected: {detection.detection_id}")
        logger.error(f"Attack type: {detection.attack_type.value if detection.attack_type else 'unknown'}")
        logger.error(f"Full detection details: {detection}")
        
        # In a real system, this might trigger additional security measures
        # like alerting security teams, temporarily blocking the source, etc.
        
    def register_attack_pattern(
        self, 
        attack_type: AttackType, 
        signature: str, 
        severity: ThreatSeverity,
        description: str,
        mitigation_strategies: List[DefenseStrategy]
    ) -> str:
        """Register a new attack pattern."""
        pattern_id = f"pattern_{secrets.token_hex(8)}"
        
        pattern = AttackPattern(
            pattern_id=pattern_id,
            attack_type=attack_type,
            signature=signature,
            severity=severity,
            description=description,
            mitigation_strategies=mitigation_strategies,
            detection_rules=[{"type": "signature_match", "threshold": 0.5}],
            last_seen=datetime.min,
            frequency=0
        )
        
        self.pattern_matcher.add_attack_pattern(pattern)
        
        logger.info(f"Registered new attack pattern: {pattern_id} for {attack_type.value}")
        return pattern_id
        
    def update_model_features(self, model_predictions: List[Tuple[Any, Any]]) -> bool:
        """
        Update the defense system with new model predictions to improve detection.
        
        Args:
            model_predictions: List of (input, prediction) tuples
            
        Returns:
            True if update was successful
        """
        try:
            for input_data, prediction in model_predictions:
                if isinstance(input_data, (list, np.ndarray)):
                    features = np.array(input_data).flatten()
                    self.anomaly_detector.update_baseline(features)
                    
            logger.info(f"Updated defense system with {len(model_predictions)} predictions")
            return True
        except Exception as e:
            logger.error(f"Failed to update model features: {str(e)}")
            return False
            
    def reset_statistics(self):
        """Reset threat detection statistics."""
        self.threat_history.clear()
        logger.info("Reset threat detection statistics")


# Convenience function for easy use
def create_adversarial_defense_manager(
    detection_threshold: float = 0.5
) -> AdversarialDefenseManager:
    """
    Convenience function to create an adversarial defense manager.
    
    Args:
        detection_threshold: Threshold for classifying inputs as threats
        
    Returns:
        AdversarialDefenseManager instance
    """
    return AdversarialDefenseManager(detection_threshold)