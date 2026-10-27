"""Input schema of the assembly dictionary (pydantic models).

See ``docs/guide/input_format.md`` for the keys and their meaning. JSON schema:
``otaf.schema.SystemData.model_json_schema()``.
"""

from __future__ import annotations

__author__ = "Kramer84"

from ._models import (
    ExtentLocal,
    Loops,
    Part,
    Surface,
    SystemData,
    validate_point_dict,
    validate_system_data,
)

__all__ = [
    "ExtentLocal",
    "Loops",
    "Part",
    "Surface",
    "SystemData",
    "validate_point_dict",
    "validate_system_data",
]
