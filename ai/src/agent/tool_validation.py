from __future__ import annotations

from typing import Any

from .base import Tool


class ToolValidationError(ValueError):
    pass


def validate_tool_arguments(tool: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
    parameters = getattr(tool, "parameters", None)
    if not isinstance(parameters, dict):
        return dict(arguments)

    if parameters.get("type") != "object":
        raise ToolValidationError(
            f"Tool '{tool.name}' parameters must be an object schema."
        )

    properties = parameters.get("properties", {})
    if not isinstance(properties, dict):
        raise ToolValidationError(
            f"Tool '{tool.name}' parameters.properties must be an object."
        )

    required = parameters.get("required", [])
    if not isinstance(required, list):
        raise ToolValidationError(
            f"Tool '{tool.name}' parameters.required must be a list."
        )

    normalized = dict(arguments)

    for name in required:
        if not isinstance(name, str):
            raise ToolValidationError(
                f"Tool '{tool.name}' parameters.required contains a non-string key."
            )
        if name not in normalized:
            raise ToolValidationError(
                f"Tool '{tool.name}' missing required argument '{name}'."
            )

    if parameters.get("additionalProperties") is False:
        allowed_names = set(properties.keys())
        unknown_names = set(normalized.keys()) - allowed_names
        if unknown_names:
            unknown = ", ".join(sorted(unknown_names))
            raise ToolValidationError(
                f"Tool '{tool.name}' got unknown argument(s): {unknown}."
            )

    for name, schema in properties.items():
        if name not in normalized:
            if isinstance(schema, dict) and "default" in schema:
                normalized[name] = schema["default"]
            continue

        if isinstance(schema, dict):
            _validate_value(tool.name, name, normalized[name], schema)

    return normalized


def _validate_value(
    tool_name: str,
    argument_name: str,
    value: Any,
    schema: dict[str, Any],
) -> None:
    expected_type = schema.get("type")
    if expected_type is None:
        return

    if isinstance(expected_type, list):
        valid = any(_matches_json_type(value, item) for item in expected_type)
    elif isinstance(expected_type, str):
        valid = _matches_json_type(value, expected_type)
    else:
        raise ToolValidationError(
            f"Tool '{tool_name}' argument '{argument_name}' has invalid schema type."
        )

    if not valid:
        actual_type = _json_type_name(value)
        raise ToolValidationError(
            f"Tool '{tool_name}' argument '{argument_name}' expected "
            f"{expected_type}, got {actual_type}."
        )


def _matches_json_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (isinstance(value, int | float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "null":
        return value is None
    return False


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__
