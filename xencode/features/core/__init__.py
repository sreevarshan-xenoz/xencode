"""
Xencode Features Core Module

Core infrastructure for feature CLI and TUI integration.
"""

from .cli import FeatureCommandGroup
from .tui import FeatureTUIManager
from .config import FeatureSystemConfig, FeatureConfigManager
from .schema import (
    FeatureSchema,
    SchemaField,
    SchemaType,
    SchemaValidator,
    schema_validator
)

__all__ = [
    "FeatureCommandGroup",
    "FeatureTUIManager",
    "FeatureSystemConfig",
    "FeatureConfigManager",
    "FeatureSchema",
    "SchemaField",
    "SchemaType",
    "SchemaValidator",
    "schema_validator"
]
