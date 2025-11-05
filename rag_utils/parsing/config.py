"""
parsing/config.py

Serializable config utilities for configuration management.
"""
from __future__ import annotations
from pydantic import BaseModel
from typing import Dict, Any
import json


# ---- Serializable config helpers ----
_CLASS_PATH_KEY = "class_path"


def serializable_config_to_yaml(obj: BaseModel) -> str:
    """Convert a BaseModel to YAML string."""
    try:
        import yaml
        data = obj.model_dump()
        return yaml.dump(data)
    except ImportError:
        raise ImportError("PyYAML is required for YAML serialization. Install it with: pip install pyyaml")


class SerializableConfig(BaseModel):
    """Base class for serializable configuration objects."""
    
    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        return serializable_config_to_yaml(self)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Dump model to dictionary, including class path for deserialization."""
        d = super().model_dump(**kwargs)
        d[_CLASS_PATH_KEY] = f"{self.__module__}.{self.__class__.__name__}"
        return d

    def pretty_print(self) -> None:
        """Pretty print the configuration as JSON."""
        print(json.dumps(self.model_dump(), indent=2))

