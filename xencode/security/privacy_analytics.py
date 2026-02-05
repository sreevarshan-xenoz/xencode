"""
Privacy-Preserving Analytics System
Implements PrivacyAnalyticsEngine for secure metrics, differential privacy mechanisms,
anonymization and pseudonymization, and privacy-aware data processing.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import json
import secrets
import hashlib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict, Counter
import math
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


logger = logging.getLogger(__name__)


class PrivacyLevel(Enum):
    """Levels of privacy protection."""
    NONE = "none"
    ANONYMIZATION = "anonymization"
    PSEUDONYMIZATION = "pseudonymization"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    SECURE_MULTI_PARTY_COMPUTATION = "secure_multi_party_computation"


class DataSensitivity(Enum):
    """Sensitivity levels of data."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    PERSONAL = "personal"
    MEDICAL = "medical"
    FINANCIAL = "financial"


@dataclass
class PrivacyMetric:
    """A privacy-preserving metric."""
    metric_id: str
    name: str
    value: Any
    privacy_level: PrivacyLevel
    sensitivity: DataSensitivity
    epsilon: Optional[float]  # For differential privacy
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class AnonymizationRule:
    """Rule for anonymizing data."""
    rule_id: str
    field_name: str
    anonymization_method: str  # e.g., "remove", "generalize", "suppress", "swap"
    parameters: Dict[str, Any]
    sensitivity: DataSensitivity
    applies_to: List[str]  # List of data types this rule applies to


@dataclass
class DifferentialPrivacyParams:
    """Parameters for differential privacy."""
    epsilon: float  # Privacy budget
    delta: float    # Approximate DP parameter
    sensitivity: float  # Global sensitivity of the function
    noise_scale: float  # Scale of noise to add


class AnonymizationEngine:
    """Engine for anonymizing data."""
    
    def __init__(self):
        self.anonymization_rules: List[AnonymizationRule] = []
        self._setup_default_rules()
        
    def _setup_default_rules(self):
        """Setup default anonymization rules."""
        default_rules = [
            AnonymizationRule(
                rule_id="rule_remove_names",
                field_name="name",
                anonymization_method="remove",
                parameters={},
                sensitivity=DataSensitivity.PERSONAL,
                applies_to=["user_data", "profile"]
            ),
            AnonymizationRule(
                rule_id="rule_generalize_age",
                field_name="age",
                anonymization_method="generalize",
                parameters={"bins": [0, 18, 35, 50, 65, 100]},
                sensitivity=DataSensitivity.PERSONAL,
                applies_to=["user_data", "demographics"]
            ),
            AnonymizationRule(
                rule_id="rule_mask_email",
                field_name="email",
                anonymization_method="mask",
                parameters={"mask_char": "*", "visible_chars": 2},
                sensitivity=DataSensitivity.PERSONAL,
                applies_to=["user_data", "contact"]
            ),
            AnonymizationRule(
                rule_id="rule_remove_location",
                field_name="location",
                anonymization_method="remove",
                parameters={},
                sensitivity=DataSensitivity.CONFIDENTIAL,
                applies_to=["user_data", "tracking"]
            )
        ]
        
        for rule in default_rules:
            self.anonymization_rules.append(rule)
            
    def anonymize_data(self, data: Dict[str, Any], data_type: str = "general") -> Dict[str, Any]:
        """Apply anonymization rules to data."""
        anonymized_data = data.copy()
        
        for rule in self.anonymization_rules:
            if data_type in rule.applies_to and rule.field_name in anonymized_data:
                field_value = anonymized_data[rule.field_name]
                
                if rule.anonymization_method == "remove":
                    anonymized_data.pop(rule.field_name, None)
                elif rule.anonymization_method == "generalize":
                    anonymized_data[rule.field_name] = self._generalize_value(
                        field_value, rule.parameters
                    )
                elif rule.anonymization_method == "mask":
                    anonymized_data[rule.field_name] = self._mask_value(
                        field_value, rule.parameters
                    )
                elif rule.anonymization_method == "suppress":
                    anonymized_data[rule.field_name] = self._suppress_value(
                        field_value, rule.parameters
                    )
                    
        return anonymized_data
        
    def _generalize_value(self, value: Any, params: Dict[str, Any]) -> Any:
        """Generalize a value according to specified bins."""
        if not isinstance(value, (int, float)) or "bins" not in params:
            return value
            
        bins = params["bins"]
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i + 1]:
                return f"{bins[i]}-{bins[i + 1]}"
                
        return f">={bins[-1]}" if value >= bins[-1] else f"<{bins[0]}"
        
    def _mask_value(self, value: str, params: Dict[str, Any]) -> str:
        """Mask a string value."""
        if not isinstance(value, str):
            return value
            
        mask_char = params.get("mask_char", "*")
        visible_chars = params.get("visible_chars", 0)
        
        if len(value) <= visible_chars:
            return value
        else:
            return value[:visible_chars] + mask_char * (len(value) - visible_chars)
            
    def _suppress_value(self, value: Any, params: Dict[str, Any]) -> Any:
        """Suppress a value (replace with None or a default)."""
        return params.get("replacement_value", None)


class DifferentialPrivacyEngine:
    """Engine for implementing differential privacy."""
    
    def __init__(self, default_epsilon: float = 1.0, default_delta: float = 1e-5):
        self.default_epsilon = default_epsilon
        self.default_delta = default_delta
        self.used_budget: float = 0.0
        self.max_budget: float = 10.0  # Total privacy budget
        
    def calculate_noise_scale(self, sensitivity: float, epsilon: float, delta: float = None) -> float:
        """Calculate the scale of noise to add for differential privacy."""
        if delta is None:
            delta = self.default_delta
            
        # For pure differential privacy (ε-differential privacy)
        # Noise scale for Laplace mechanism: sensitivity / epsilon
        # For approximate DP (ε, δ)-differential privacy
        # Noise scale for Gaussian mechanism: sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        if delta == 0:
            # Pure DP - use Laplace mechanism
            return sensitivity / epsilon
        else:
            # Approximate DP - use Gaussian mechanism
            return (sensitivity * math.sqrt(2 * math.log(1.25 / delta))) / epsilon
            
    def add_laplace_noise(self, value: Union[float, np.ndarray], sensitivity: float, epsilon: float) -> Union[float, np.ndarray]:
        """Add Laplace noise to a value."""
        if self.used_budget + epsilon > self.max_budget:
            raise ValueError(f"Insufficient privacy budget. Requested: {epsilon}, Available: {self.max_budget - self.used_budget}")
            
        noise_scale = sensitivity / epsilon
        noise = np.random.laplace(0, noise_scale)
        
        self.used_budget += epsilon
        
        if isinstance(value, np.ndarray):
            return value + noise
        else:
            return value + noise
            
    def add_gaussian_noise(self, value: Union[float, np.ndarray], sensitivity: float, epsilon: float, delta: float = None) -> Union[float, np.ndarray]:
        """Add Gaussian noise to a value."""
        if delta is None:
            delta = self.default_delta
            
        if self.used_budget + epsilon > self.max_budget:
            raise ValueError(f"Insufficient privacy budget. Requested: {epsilon}, Available: {self.max_budget - self.used_budget}")
            
        noise_scale = (sensitivity * math.sqrt(2 * math.log(1.25 / delta))) / epsilon
        noise = np.random.normal(0, noise_scale)
        
        self.used_budget += epsilon
        
        if isinstance(value, np.ndarray):
            return value + noise
        else:
            return value + noise
            
    def compute_private_sum(self, values: List[float], epsilon: float = None) -> float:
        """Compute a differentially private sum."""
        if epsilon is None:
            epsilon = self.default_epsilon
            
        # Calculate sensitivity (L1 sensitivity for sum is 1 if values are bounded in [0,1])
        # For unbounded values, we need to clamp them
        clamped_values = np.clip(values, -1, 1)  # Assuming values are roughly in [-1, 1]
        raw_sum = float(np.sum(clamped_values))
        
        # Sensitivity of sum function is the maximum possible difference when one record changes
        # If values are in [-1, 1], sensitivity is 2 (changing from -1 to 1 or vice versa)
        sensitivity = 2.0
        
        return self.add_laplace_noise(raw_sum, sensitivity, epsilon)
        
    def compute_private_mean(self, values: List[float], epsilon: float = None) -> float:
        """Compute a differentially private mean."""
        if epsilon is None:
            epsilon = self.default_epsilon
            
        # Mean = sum / n, so we compute private sum and divide by true n
        n = len(values)
        if n == 0:
            return 0.0
            
        private_sum = self.compute_private_sum(values, epsilon)
        return private_sum / n
        
    def compute_private_histogram(self, values: List[Any], bins: int, epsilon: float = None) -> List[int]:
        """Compute a differentially private histogram."""
        if epsilon is None:
            epsilon = self.default_epsilon
            
        # Divide epsilon among bins (using sequential composition)
        bin_epsilon = epsilon / bins
        
        # Create histogram
        hist, _ = np.histogram(values, bins=bins)
        
        # Add noise to each bin
        sensitivity = 1  # Changing one record affects one bin by at most 1
        private_hist = []
        
        for count in hist:
            noisy_count = self.add_laplace_noise(float(count), sensitivity, bin_epsilon)
            # Ensure non-negative counts
            private_hist.append(max(0, round(noisy_count)))
            
        return private_hist
        
    def reset_budget(self):
        """Reset the privacy budget."""
        self.used_budget = 0.0


class PseudonymizationEngine:
    """Engine for pseudonymizing data."""
    
    def __init__(self):
        self.pseudonym_map: Dict[str, str] = {}  # Original -> Pseudonym
        self.reverse_map: Dict[str, str] = {}   # Pseudonym -> Original
        self.salt = secrets.token_hex(32)
        
    def pseudonymize_value(self, value: str, context: str = "") -> str:
        """Pseudonymize a value."""
        # Create a deterministic pseudonym based on the value and context
        key = f"{value}_{context}_{self.salt}"
        pseudonym = hashlib.sha256(key.encode()).hexdigest()[:16]
        
        # Store the mapping
        self.pseudonym_map[value] = pseudonym
        self.reverse_map[pseudonym] = value
        
        return pseudonym
        
    def pseudonymize_data(self, data: Dict[str, Any], fields_to_pseudonymize: List[str], context: str = "") -> Dict[str, Any]:
        """Pseudonymize specified fields in the data."""
        pseudonymized_data = data.copy()
        
        for field in fields_to_pseudonymize:
            if field in pseudonymized_data and isinstance(pseudonymized_data[field], str):
                original_value = pseudonymized_data[field]
                pseudonymized_data[field] = self.pseudonymize_value(original_value, context)
                
        return pseudonymized_data
        
    def get_original_value(self, pseudonym: str) -> Optional[str]:
        """Get the original value from a pseudonym."""
        return self.reverse_map.get(pseudonym)


class PrivacyAnalyticsEngine:
    """
    Privacy analytics engine for secure metrics with differential privacy mechanisms,
    anonymization, pseudonymization, and privacy-aware data processing.
    """
    
    def __init__(self, default_epsilon: float = 1.0):
        self.anonymization_engine = AnonymizationEngine()
        self.differential_privacy_engine = DifferentialPrivacyEngine(default_epsilon)
        self.pseudonymization_engine = PseudonymizationEngine()
        self.collected_metrics: List[PrivacyMetric] = []
        self.privacy_levels = {
            DataSensitivity.PUBLIC: PrivacyLevel.NONE,
            DataSensitivity.INTERNAL: PrivacyLevel.ANONYMIZATION,
            DataSensitivity.CONFIDENTIAL: PrivacyLevel.DIFFERENTIAL_PRIVACY,
            DataSensitivity.SECRET: PrivacyLevel.HOMOMORPHIC_ENCRYPTION,
            DataSensitivity.PERSONAL: PrivacyLevel.PSEUDONYMIZATION,
            DataSensitivity.MEDICAL: PrivacyLevel.DIFFERENTIAL_PRIVACY,
            DataSensitivity.FINANCIAL: PrivacyLevel.DIFFERENTIAL_PRIVACY
        }
        
    def collect_metric(
        self, 
        name: str, 
        value: Any, 
        sensitivity: DataSensitivity,
        epsilon: Optional[float] = None,
        apply_privacy: bool = True
    ) -> str:
        """Collect a privacy-preserving metric."""
        metric_id = f"metric_{secrets.token_hex(8)}"
        timestamp = datetime.now()
        
        # Determine privacy level based on sensitivity
        privacy_level = self.privacy_levels.get(sensitivity, PrivacyLevel.ANONYMIZATION)
        
        # Apply privacy protection if requested
        protected_value = value
        if apply_privacy:
            if privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY and isinstance(value, (int, float)):
                # Apply differential privacy to numeric values
                if epsilon is None:
                    epsilon = self.differential_privacy_engine.default_epsilon
                    
                # For this example, we'll add noise to the value
                # In practice, you might not directly add noise to stored values
                # but rather to computed statistics
                sensitivity_param = 1.0  # This would be calculated based on the function
                protected_value = self.differential_privacy_engine.add_laplace_noise(
                    value, sensitivity_param, epsilon
                )
            elif privacy_level == PrivacyLevel.ANONYMIZATION:
                # For anonymization, we might process the data differently
                # depending on the context
                pass
            elif privacy_level == PrivacyLevel.PSEUDONYMIZATION:
                # For pseudonymization, we might replace identifiers
                if isinstance(value, str):
                    protected_value = self.pseudonymization_engine.pseudonymize_value(value, name)
        
        metric = PrivacyMetric(
            metric_id=metric_id,
            name=name,
            value=protected_value,
            privacy_level=privacy_level,
            sensitivity=sensitivity,
            epsilon=epsilon,
            timestamp=timestamp,
            metadata={
                "original_type": type(value).__name__,
                "privacy_applied": apply_privacy,
                "collection_context": "privacy_analytics_engine"
            }
        )
        
        self.collected_metrics.append(metric)
        
        logger.info(f"Collected privacy metric: {name} with sensitivity {sensitivity.value}")
        return metric_id
        
    def compute_private_statistics(self, data: List[float], statistic_type: str = "mean", epsilon: float = None) -> Any:
        """Compute differentially private statistics."""
        if epsilon is None:
            epsilon = self.differential_privacy_engine.default_epsilon
            
        if statistic_type == "mean":
            return self.differential_privacy_engine.compute_private_mean(data, epsilon)
        elif statistic_type == "sum":
            return self.differential_privacy_engine.compute_private_sum(data, epsilon)
        elif statistic_type == "histogram":
            # For histogram, we need to specify number of bins
            bins = min(10, len(set(data)))  # At most 10 bins or number of unique values
            return self.differential_privacy_engine.compute_private_histogram(data, bins, epsilon)
        else:
            raise ValueError(f"Unsupported statistic type: {statistic_type}")
            
    def anonymize_dataset(self, dataset: List[Dict[str, Any]], data_type: str = "general") -> List[Dict[str, Any]]:
        """Anonymize a dataset."""
        anonymized_dataset = []
        for record in dataset:
            anonymized_record = self.anonymization_engine.anonymize_data(record, data_type)
            anonymized_dataset.append(anonymized_record)
            
        logger.info(f"Anonymized dataset with {len(dataset)} records")
        return anonymized_dataset
    
    def pseudonymize_dataset(
        self, 
        dataset: List[Dict[str, Any]], 
        fields_to_pseudonymize: List[str], 
        context: str = ""
    ) -> List[Dict[str, Any]]:
        """Pseudonymize specified fields in a dataset."""
        pseudonymized_dataset = []
        for record in dataset:
            pseudonymized_record = self.pseudonymization_engine.pseudonymize_data(
                record, fields_to_pseudonymize, context
            )
            pseudonymized_dataset.append(pseudonymized_record)
            
        logger.info(f"Pseudonymized dataset with {len(dataset)} records, fields: {fields_to_pseudonymize}")
        return pseudonymized_dataset
        
    def apply_differential_privacy_to_dataframe(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        epsilon: float = None
    ) -> pd.DataFrame:
        """Apply differential privacy to specific columns of a DataFrame."""
        if epsilon is None:
            epsilon = self.differential_privacy_engine.default_epsilon
            
        private_df = df.copy()
        
        for col in columns:
            if col in private_df.columns:
                # Get the data and apply differential privacy
                data = private_df[col].dropna()  # Remove NaN values
                
                if pd.api.types.is_numeric_dtype(data):
                    # Apply noise to numeric data
                    sensitivity = data.max() - data.min() if len(data) > 0 else 1.0
                    # Normalize the data to [0, 1] range for consistent sensitivity
                    if sensitivity != 0:
                        normalized_data = (data - data.min()) / sensitivity
                    else:
                        normalized_data = data
                        
                    # Add noise
                    noisy_data = []
                    for val in normalized_data:
                        noisy_val = self.differential_privacy_engine.add_laplace_noise(
                            float(val), 1.0, epsilon / len(columns)  # Divide epsilon among columns
                        )
                        # Denormalize
                        denoised_val = data.min() + noisy_val * sensitivity
                        noisy_data.append(denoised_val)
                        
                    # Update the column with noisy data
                    private_df.loc[data.index, col] = noisy_data
                    
        return private_df
        
    def get_privacy_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of privacy metrics collected."""
        if not self.collected_metrics:
            return {
                "total_metrics": 0,
                "metrics_by_sensitivity": {},
                "metrics_by_privacy_level": {},
                "privacy_budget_used": self.differential_privacy_engine.used_budget,
                "recent_metrics": []
            }
        
        # Count metrics by sensitivity
        sensitivity_counts = defaultdict(int)
        privacy_level_counts = defaultdict(int)
        
        for metric in self.collected_metrics:
            sensitivity_counts[metric.sensitivity.value] += 1
            privacy_level_counts[metric.privacy_level.value] += 1
            
        return {
            "total_metrics": len(self.collected_metrics),
            "metrics_by_sensitivity": dict(sensitivity_counts),
            "metrics_by_privacy_level": dict(privacy_level_counts),
            "privacy_budget_used": self.differential_privacy_engine.used_budget,
            "recent_metrics": [
                {
                    "name": m.name,
                    "sensitivity": m.sensitivity.value,
                    "privacy_level": m.privacy_level.value,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in self.collected_metrics[-10:]  # Last 10 metrics
            ]
        }
        
    def reset_privacy_budget(self):
        """Reset the privacy budget."""
        self.differential_privacy_engine.reset_budget()
        logger.info("Reset privacy budget")
        
    def create_privacy_preserving_report(
        self, 
        data: List[Dict[str, Any]], 
        report_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a privacy-preserving report based on configuration.
        
        Args:
            data: Input data for the report
            report_config: Configuration specifying privacy requirements
            
        Returns:
            Privacy-preserving report
        """
        report = {
            "report_id": f"report_{secrets.token_hex(8)}",
            "generated_at": datetime.now().isoformat(),
            "config": report_config,
            "sections": []
        }
        
        # Process each section of the report
        for section_config in report_config.get("sections", []):
            section_name = section_config["name"]
            data_fields = section_config["fields"]
            privacy_level = section_config.get("privacy_level", "differential_privacy")
            epsilon = section_config.get("epsilon", 1.0)
            
            section_data = {}
            
            if privacy_level == "differential_privacy":
                # Apply differential privacy to numeric fields
                for field in data_fields:
                    if field in data[0] if data else []:
                        field_values = [row[field] for row in data if field in row and isinstance(row[field], (int, float))]
                        if field_values:
                            private_stat = self.compute_private_statistics(field_values, "mean", epsilon)
                            section_data[field] = {
                                "private_mean": private_stat,
                                "count": len(field_values)
                            }
                            
            elif privacy_level == "anonymization":
                # Apply anonymization
                anonymized_data = self.anonymize_dataset(data, section_name)
                section_data["records"] = anonymized_data
                section_data["count"] = len(anonymized_data)
                
            elif privacy_level == "pseudonymization":
                # Apply pseudonymization
                pseudonymized_data = self.pseudonymize_dataset(data, data_fields, section_name)
                section_data["records"] = pseudonymized_data
                section_data["count"] = len(pseudonymized_data)
                
            report["sections"].append({
                "name": section_name,
                "data": section_data,
                "privacy_level": privacy_level
            })
            
        return report
        
    def measure_privacy_loss(self, operations: List[Dict[str, Any]]) -> float:
        """
        Measure cumulative privacy loss from a series of operations.
        
        Args:
            operations: List of operations with epsilon values
            
        Returns:
            Total privacy budget consumed
        """
        total_epsilon = 0.0
        total_delta = 0.0
        
        for op in operations:
            epsilon = op.get("epsilon", 0.0)
            delta = op.get("delta", 0.0)
            
            # For sequential composition, we add up epsilons and deltas
            total_epsilon += epsilon
            total_delta += delta
            
        return total_epsilon


# Convenience function for easy use
def create_privacy_analytics_engine(
    default_epsilon: float = 1.0
) -> PrivacyAnalyticsEngine:
    """
    Convenience function to create a privacy analytics engine.
    
    Args:
        default_epsilon: Default privacy budget parameter
        
    Returns:
        PrivacyAnalyticsEngine instance
    """
    return PrivacyAnalyticsEngine(default_epsilon)