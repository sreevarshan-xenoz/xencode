"""
Feature Configuration Schema System

Provides YAML/JSON schema validation for feature configurations.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class SchemaType(str, Enum):
    """Configuration schema types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ENUM = "enum"


@dataclass
class SchemaField:
    """Schema field definition"""
    name: str
    type: SchemaType
    required: bool = False
    default: Any = None
    description: str = ""
    enum_values: list = field(default_factory=list)
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    items_type: Optional[SchemaType] = None
    properties: Dict[str, 'SchemaField'] = field(default_factory=dict)
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a value against this schema field"""
        # Check required
        if value is None:
            if self.required:
                return False, f"Field '{self.name}' is required"
            return True, None
        
        # Type validation
        if self.type == SchemaType.STRING:
            if not isinstance(value, str):
                return False, f"Field '{self.name}' must be a string"
            if self.pattern:
                import re
                if not re.match(self.pattern, value):
                    return False, f"Field '{self.name}' does not match pattern '{self.pattern}'"
        
        elif self.type == SchemaType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                return False, f"Field '{self.name}' must be an integer"
            if self.min_value is not None and value < self.min_value:
                return False, f"Field '{self.name}' must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Field '{self.name}' must be <= {self.max_value}"
        
        elif self.type == SchemaType.FLOAT:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False, f"Field '{self.name}' must be a number"
            if self.min_value is not None and value < self.min_value:
                return False, f"Field '{self.name}' must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Field '{self.name}' must be <= {self.max_value}"
        
        elif self.type == SchemaType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"Field '{self.name}' must be a boolean"
        
        elif self.type == SchemaType.ARRAY:
            if not isinstance(value, list):
                return False, f"Field '{self.name}' must be an array"
            if self.items_type:
                for i, item in enumerate(value):
                    item_field = SchemaField(
                        name=f"{self.name}[{i}]",
                        type=self.items_type
                    )
                    valid, error = item_field.validate(item)
                    if not valid:
                        return False, error
        
        elif self.type == SchemaType.OBJECT:
            if not isinstance(value, dict):
                return False, f"Field '{self.name}' must be an object"
            if self.properties:
                for prop_name, prop_field in self.properties.items():
                    prop_value = value.get(prop_name)
                    valid, error = prop_field.validate(prop_value)
                    if not valid:
                        return False, error
        
        elif self.type == SchemaType.ENUM:
            if value not in self.enum_values:
                return False, f"Field '{self.name}' must be one of {self.enum_values}"
        
        return True, None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema field to dictionary"""
        result = {
            'type': self.type.value,
            'required': self.required,
            'description': self.description
        }
        
        if self.default is not None:
            result['default'] = self.default
        
        if self.enum_values:
            result['enum'] = self.enum_values
        
        if self.min_value is not None:
            result['minimum'] = self.min_value
        
        if self.max_value is not None:
            result['maximum'] = self.max_value
        
        if self.pattern:
            result['pattern'] = self.pattern
        
        if self.items_type:
            result['items'] = {'type': self.items_type.value}
        
        if self.properties:
            result['properties'] = {
                name: field.to_dict() 
                for name, field in self.properties.items()
            }
        
        return result


@dataclass
class FeatureSchema:
    """Complete schema for a feature configuration"""
    name: str
    version: str = "1.0.0"
    description: str = ""
    fields: Dict[str, SchemaField] = field(default_factory=dict)
    
    def validate(self, config: Dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate a configuration against this schema"""
        errors = []
        
        for field_name, field_schema in self.fields.items():
            value = config.get(field_name)
            valid, error = field_schema.validate(value)
            if not valid:
                errors.append(error)
        
        return len(errors) == 0, errors
    
    def apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values to configuration"""
        result = config.copy()
        
        for field_name, field_schema in self.fields.items():
            if field_name not in result and field_schema.default is not None:
                result[field_name] = field_schema.default
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary (JSON Schema format)"""
        return {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'title': self.name,
            'version': self.version,
            'description': self.description,
            'type': 'object',
            'properties': {
                name: field.to_dict() 
                for name, field in self.fields.items()
            },
            'required': [
                name for name, field in self.fields.items() 
                if field.required
            ]
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export schema as JSON"""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_yaml(self) -> str:
        """Export schema as YAML"""
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureSchema':
        """Create schema from dictionary"""
        fields = {}
        
        properties = data.get('properties', {})
        required_fields = data.get('required', [])
        
        for field_name, field_data in properties.items():
            field_type = SchemaType(field_data.get('type', 'string'))
            
            fields[field_name] = SchemaField(
                name=field_name,
                type=field_type,
                required=field_name in required_fields,
                default=field_data.get('default'),
                description=field_data.get('description', ''),
                enum_values=field_data.get('enum', []),
                min_value=field_data.get('minimum'),
                max_value=field_data.get('maximum'),
                pattern=field_data.get('pattern'),
                items_type=SchemaType(field_data['items']['type']) if 'items' in field_data else None
            )
        
        return cls(
            name=data.get('title', ''),
            version=data.get('version', '1.0.0'),
            description=data.get('description', ''),
            fields=fields
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'FeatureSchema':
        """Load schema from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'FeatureSchema':
        """Load schema from YAML string"""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_file(cls, path: Path) -> 'FeatureSchema':
        """Load schema from file (JSON or YAML)"""
        content = path.read_text()
        
        if path.suffix in ['.json']:
            return cls.from_json(content)
        elif path.suffix in ['.yaml', '.yml']:
            return cls.from_yaml(content)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


class SchemaValidator:
    """Validator for feature configurations using schemas"""
    
    def __init__(self, schema_dir: Path = None):
        self.schema_dir = schema_dir or Path(__file__).parent.parent / "schemas"
        self.schemas: Dict[str, FeatureSchema] = {}
        self._load_schemas()
    
    def _load_schemas(self) -> None:
        """Load all schemas from schema directory"""
        if not self.schema_dir.exists():
            return
        
        for schema_file in self.schema_dir.glob("*.json"):
            try:
                schema = FeatureSchema.from_file(schema_file)
                self.schemas[schema.name] = schema
            except Exception as e:
                print(f"Error loading schema {schema_file}: {e}")
        
        for schema_file in self.schema_dir.glob("*.yaml"):
            try:
                schema = FeatureSchema.from_file(schema_file)
                self.schemas[schema.name] = schema
            except Exception as e:
                print(f"Error loading schema {schema_file}: {e}")
    
    def get_schema(self, feature_name: str) -> Optional[FeatureSchema]:
        """Get schema for a feature"""
        return self.schemas.get(feature_name)
    
    def validate_config(self, feature_name: str, config: Dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate a feature configuration"""
        schema = self.get_schema(feature_name)
        
        if not schema:
            # No schema available, allow any config
            return True, []
        
        return schema.validate(config)
    
    def apply_defaults(self, feature_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values from schema"""
        schema = self.get_schema(feature_name)
        
        if not schema:
            return config
        
        return schema.apply_defaults(config)
    
    def register_schema(self, schema: FeatureSchema) -> None:
        """Register a schema programmatically"""
        self.schemas[schema.name] = schema
    
    def save_schema(self, feature_name: str, format: str = 'yaml') -> Path:
        """Save a schema to file"""
        schema = self.get_schema(feature_name)
        
        if not schema:
            raise ValueError(f"Schema for '{feature_name}' not found")
        
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            schema_file = self.schema_dir / f"{feature_name}.json"
            schema_file.write_text(schema.to_json())
        elif format == 'yaml':
            schema_file = self.schema_dir / f"{feature_name}.yaml"
            schema_file.write_text(schema.to_yaml())
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return schema_file


# Global schema validator instance
schema_validator = SchemaValidator()
