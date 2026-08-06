"""
Xencode Features Core Module

Core infrastructure for feature CLI and TUI integration.
"""

from .cli import FeatureCommandGroup
from .config import FeatureConfigManager, FeatureSystemConfig
from .schema import (
    FeatureSchema,
    SchemaField,
    SchemaType,
    SchemaValidator,
    schema_validator,
)
from .tui import FeatureTUIManager

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
