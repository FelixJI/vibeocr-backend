"""Lint OpenAPI documents and reject backward-incompatible API changes.

The Runtime API's ``openapi.yaml`` is intentionally encoded as JSON, which is a
YAML 1.2 subset.  This keeps the release gate deterministic and dependency-free.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FORMAL_OPENAPI = (
    ROOT
    / "packages"
    / "vibeocr-contracts-py"
    / "src"
    / "vibeocr"
    / "protocol"
    / "v2"
    / "openapi.yaml"
)
HTTP_METHODS = frozenset(
    {"get", "put", "post", "delete", "patch", "head", "options", "trace"}
)
JSON_SCHEMA_TYPES = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)
PATH_PARAMETER = re.compile(r"{([^{}]+)}")

Document = dict[str, Any]


def load_document(path: Path) -> Document:
    """Load a JSON-encoded OpenAPI document."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot load JSON-compatible YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: document root must be an object")
    return value


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _resolve_pointer(document: Document, reference: str) -> object:
    if not reference.startswith("#/"):
        raise ValueError("only local JSON Pointer references are allowed")
    value: object = document
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError("target does not exist")
        value = value[part]
    return value


def _resolve(document: Document, value: object) -> object:
    seen: set[str] = set()
    while isinstance(value, Mapping) and isinstance(value.get("$ref"), str):
        reference = value["$ref"]
        if reference in seen:
            raise ValueError(f"cyclic reference chain at {reference}")
        seen.add(reference)
        value = _resolve_pointer(document, reference)
    return value


def _walk(value: object, location: str = "$") -> list[tuple[str, object]]:
    values = [(location, value)]
    if isinstance(value, Mapping):
        for key in sorted(value):
            values.extend(_walk(value[key], f"{location}/{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            values.extend(_walk(item, f"{location}/{index}"))
    return values


def _lint_schema(schema: object, location: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(schema, Mapping):
        return [f"{location}: schema must be an object"]

    schema_type = schema.get("type")
    types: list[object]
    if isinstance(schema_type, list):
        types = schema_type
    elif schema_type is None:
        types = []
    else:
        types = [schema_type]
    if any(item not in JSON_SCHEMA_TYPES for item in types):
        errors.append(f"{location}: schema has an invalid type")

    properties = schema.get("properties")
    if properties is not None and not isinstance(properties, Mapping):
        errors.append(f"{location}/properties: must be an object")
        properties = {}
    required = schema.get("required")
    if required is not None:
        if (
            not isinstance(required, list)
            or any(not isinstance(item, str) for item in required)
            or len(required) != len(set(required))
        ):
            errors.append(f"{location}/required: must contain unique property names")
        elif isinstance(properties, Mapping):
            unknown = sorted(set(required) - set(properties))
            if unknown:
                errors.append(
                    f"{location}/required: unknown properties {', '.join(unknown)}"
                )

    if "enum" in schema and not isinstance(schema["enum"], list):
        errors.append(f"{location}/enum: must be an array")
    if "items" in schema and not isinstance(schema["items"], Mapping):
        errors.append(f"{location}/items: must be a schema object")
    if "additionalProperties" in schema and not isinstance(
        schema["additionalProperties"], (bool, Mapping)
    ):
        errors.append(
            f"{location}/additionalProperties: must be a boolean or schema object"
        )
    for keyword in ("allOf", "anyOf", "oneOf"):
        value = schema.get(keyword)
        if value is not None and (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, Mapping) for item in value)
        ):
            errors.append(f"{location}/{keyword}: must be a non-empty schema array")
    if isinstance(properties, Mapping):
        for name in sorted(properties):
            errors.extend(
                _lint_schema(properties[name], f"{location}/properties/{name}")
            )
    if isinstance(schema.get("items"), Mapping):
        errors.extend(_lint_schema(schema["items"], f"{location}/items"))
    if isinstance(schema.get("additionalProperties"), Mapping):
        errors.extend(
            _lint_schema(
                schema["additionalProperties"],
                f"{location}/additionalProperties",
            )
        )
    for keyword in ("allOf", "anyOf", "oneOf"):
        value = schema.get(keyword)
        if isinstance(value, list):
            for index, item in enumerate(value):
                errors.extend(_lint_schema(item, f"{location}/{keyword}/{index}"))
    return errors


def lint_document(document: Document) -> list[str]:
    """Return deterministic structural lint errors for an OpenAPI document."""

    errors: list[str] = []
    if not str(document.get("openapi", "")).startswith("3.1."):
        errors.append("$.openapi: expected an OpenAPI 3.1.x version")
    info = document.get("info")
    if not isinstance(info, Mapping):
        errors.append("$.info: must be an object")
    else:
        for field in ("title", "version"):
            if not isinstance(info.get(field), str) or not info[field].strip():
                errors.append(f"$.info/{field}: must be a non-empty string")

    paths = document.get("paths")
    if not isinstance(paths, Mapping) or not paths:
        errors.append("$.paths: must be a non-empty object")
        paths = {}
    components = document.get("components")
    if not isinstance(components, Mapping):
        errors.append("$.components: must be an object")

    operation_ids: dict[str, str] = {}
    for path in sorted(paths):
        path_item = paths[path]
        path_location = f"$.paths/{path}"
        if not isinstance(path, str) or not path.startswith("/"):
            errors.append(f"{path_location}: path must start with '/'")
        if not isinstance(path_item, Mapping):
            errors.append(f"{path_location}: path item must be an object")
            continue
        methods = [method for method in path_item if method.lower() in HTTP_METHODS]
        if not methods:
            errors.append(f"{path_location}: path item has no HTTP operation")
        for method in sorted(methods):
            operation = path_item[method]
            location = f"{path_location}/{method.lower()}"
            if not isinstance(operation, Mapping):
                errors.append(f"{location}: operation must be an object")
                continue
            operation_id = operation.get("operationId")
            if not isinstance(operation_id, str) or not operation_id.strip():
                errors.append(f"{location}/operationId: must be a non-empty string")
            elif operation_id in operation_ids:
                errors.append(
                    f"{location}/operationId: duplicate {operation_id!r}; "
                    f"first used at {operation_ids[operation_id]}"
                )
            else:
                operation_ids[operation_id] = location
            responses = operation.get("responses")
            if not isinstance(responses, Mapping) or not responses:
                errors.append(f"{location}/responses: must be a non-empty object")

            declared_parameters: set[str] = set()
            for parameter in list(path_item.get("parameters", [])) + list(
                operation.get("parameters", [])
            ):
                try:
                    parameter = _resolve(document, parameter)
                except ValueError:
                    continue
                if (
                    isinstance(parameter, Mapping)
                    and parameter.get("in") == "path"
                    and parameter.get("required") is True
                    and isinstance(parameter.get("name"), str)
                ):
                    declared_parameters.add(parameter["name"])
            missing_parameters = sorted(
                set(PATH_PARAMETER.findall(path)) - declared_parameters
            )
            if missing_parameters:
                errors.append(
                    f"{location}/parameters: undeclared required path parameters "
                    f"{', '.join(missing_parameters)}"
                )

    for location, value in _walk(document):
        if isinstance(value, Mapping) and "$ref" in value:
            reference = value["$ref"]
            if not isinstance(reference, str):
                errors.append(f"{location}/$ref: must be a string")
            else:
                try:
                    _resolve_pointer(document, reference)
                except ValueError as exc:
                    errors.append(f"{location}/$ref: {reference!r} {exc}")

    schemas = _mapping(_mapping(document.get("components")).get("schemas"))
    for name in sorted(schemas):
        errors.extend(_lint_schema(schemas[name], f"$.components/schemas/{name}"))
    for location, value in _walk(document):
        if isinstance(value, Mapping) and isinstance(value.get("schema"), Mapping):
            errors.extend(_lint_schema(value["schema"], f"{location}/schema"))
    return sorted(set(errors))


def _operations(document: Document) -> dict[tuple[str, str], Mapping[str, Any]]:
    operations: dict[tuple[str, str], Mapping[str, Any]] = {}
    for path, path_item in _mapping(document.get("paths")).items():
        for method, operation in _mapping(path_item).items():
            if method.lower() in HTTP_METHODS and isinstance(operation, Mapping):
                operations[(path, method.lower())] = operation
    return operations


def _schema_types(schema: Mapping[str, Any]) -> frozenset[str]:
    value = schema.get("type")
    if isinstance(value, str):
        return frozenset({value})
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return frozenset(value)
    if isinstance(schema.get("properties"), Mapping):
        return frozenset({"object"})
    if "items" in schema:
        return frozenset({"array"})
    return frozenset()


def _required(schema: Mapping[str, Any]) -> set[str]:
    value = schema.get("required")
    return set(value) if isinstance(value, list) else set()


def _compare_schema(
    baseline_document: Document,
    current_document: Document,
    baseline_schema: object,
    current_schema: object,
    location: str,
    direction: str,
    issues: list[str],
    visited: set[tuple[int, int, str]],
) -> None:
    try:
        baseline_schema = _resolve(baseline_document, baseline_schema)
        current_schema = _resolve(current_document, current_schema)
    except ValueError as exc:
        issues.append(f"{location}: cannot compare schema: {exc}")
        return
    if not isinstance(baseline_schema, Mapping) or not isinstance(
        current_schema, Mapping
    ):
        if type(baseline_schema) is not type(current_schema):
            issues.append(f"{location}: schema kind changed")
        return

    visit_key = (id(baseline_schema), id(current_schema), direction)
    if visit_key in visited:
        return
    visited.add(visit_key)

    baseline_types = _schema_types(baseline_schema)
    current_types = _schema_types(current_schema)
    if baseline_types != current_types:
        issues.append(
            f"{location}: type changed from "
            f"{sorted(baseline_types) or ['any']} to "
            f"{sorted(current_types) or ['any']}"
        )

    baseline_properties = _mapping(baseline_schema.get("properties"))
    current_properties = _mapping(current_schema.get("properties"))
    removed_properties = sorted(set(baseline_properties) - set(current_properties))
    for name in removed_properties:
        issues.append(f"{location}.{name}: {direction} field was removed")

    baseline_required = _required(baseline_schema)
    current_required = _required(current_schema)
    if direction == "request":
        for name in sorted(current_required - baseline_required):
            issues.append(f"{location}.{name}: request field became required")
    else:
        for name in sorted(baseline_required - current_required):
            issues.append(f"{location}.{name}: response field is no longer required")

    for name in sorted(set(baseline_properties) & set(current_properties)):
        _compare_schema(
            baseline_document,
            current_document,
            baseline_properties[name],
            current_properties[name],
            f"{location}.{name}",
            direction,
            issues,
            visited,
        )

    if "items" in baseline_schema and "items" in current_schema:
        _compare_schema(
            baseline_document,
            current_document,
            baseline_schema["items"],
            current_schema["items"],
            f"{location}[]",
            direction,
            issues,
            visited,
        )

    baseline_enum = baseline_schema.get("enum")
    current_enum = current_schema.get("enum")
    if isinstance(baseline_enum, list) and isinstance(current_enum, list):
        if direction == "request":
            removed_values = [
                item for item in baseline_enum if item not in current_enum
            ]
            if removed_values:
                issues.append(
                    f"{location}: request enum no longer accepts {removed_values!r}"
                )
        else:
            added_values = [item for item in current_enum if item not in baseline_enum]
            if added_values:
                issues.append(
                    f"{location}: response enum added values {added_values!r}"
                )


def _content_schemas(document: Document, container: object) -> dict[str, object]:
    try:
        container = _resolve(document, container)
    except ValueError:
        return {}
    content = _mapping(_mapping(container).get("content"))
    return {
        media_type: media["schema"]
        for media_type, media in content.items()
        if isinstance(media, Mapping) and "schema" in media
    }


def detect_breaking_changes(
    baseline_document: Document, current_document: Document
) -> list[str]:
    """Return backward-incompatible changes from baseline to current."""

    issues: list[str] = []
    baseline_operations = _operations(baseline_document)
    current_operations = _operations(current_document)
    for path, method in sorted(set(baseline_operations) - set(current_operations)):
        issues.append(f"{method.upper()} {path}: operation was removed")

    for key in sorted(set(baseline_operations) & set(current_operations)):
        path, method = key
        prefix = f"{method.upper()} {path}"
        baseline_operation = baseline_operations[key]
        current_operation = current_operations[key]
        if baseline_operation.get("operationId") != current_operation.get(
            "operationId"
        ):
            issues.append(f"{prefix}: operationId changed")

        baseline_request = baseline_operation.get("requestBody")
        current_request = current_operation.get("requestBody")
        if baseline_request is not None and current_request is None:
            issues.append(f"{prefix}: request body was removed")
        elif baseline_request is not None and current_request is not None:
            old_request = _mapping(_resolve(baseline_document, baseline_request))
            new_request = _mapping(_resolve(current_document, current_request))
            if (
                old_request.get("required") is not True
                and new_request.get("required") is True
            ):
                issues.append(f"{prefix}: request body became required")
            old_content = _content_schemas(baseline_document, baseline_request)
            new_content = _content_schemas(current_document, current_request)
            for media_type in sorted(set(old_content) - set(new_content)):
                issues.append(
                    f"{prefix} request: media type {media_type!r} was removed"
                )
            for media_type in sorted(set(old_content) & set(new_content)):
                _compare_schema(
                    baseline_document,
                    current_document,
                    old_content[media_type],
                    new_content[media_type],
                    f"{prefix} request {media_type}",
                    "request",
                    issues,
                    set(),
                )

        baseline_responses = _mapping(baseline_operation.get("responses"))
        current_responses = _mapping(current_operation.get("responses"))
        for status in sorted(set(baseline_responses) - set(current_responses)):
            issues.append(f"{prefix}: response status {status} was removed")
        for status in sorted(set(baseline_responses) & set(current_responses)):
            old_content = _content_schemas(
                baseline_document, baseline_responses[status]
            )
            new_content = _content_schemas(current_document, current_responses[status])
            for media_type in sorted(set(old_content) - set(new_content)):
                issues.append(
                    f"{prefix} response {status}: media type {media_type!r} was removed"
                )
            for media_type in sorted(set(old_content) & set(new_content)):
                _compare_schema(
                    baseline_document,
                    current_document,
                    old_content[media_type],
                    new_content[media_type],
                    f"{prefix} response {status} {media_type}",
                    "response",
                    issues,
                    set(),
                )
    return sorted(set(issues))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dependency-free OpenAPI lint and compatibility gate."
    )
    parser.add_argument(
        "--lint",
        action="append",
        type=Path,
        metavar="FILE",
        help="lint FILE; repeat for multiple documents (defaults to formal spec)",
    )
    parser.add_argument("--baseline", type=Path, help="released baseline OpenAPI")
    parser.add_argument("--current", type=Path, help="candidate OpenAPI")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (args.baseline is None) != (args.current is None):
        parser.error("--baseline and --current must be provided together")

    paths = list(args.lint or [])
    if args.baseline is not None:
        paths.extend((args.baseline, args.current))
    if not paths:
        paths.append(FORMAL_OPENAPI)

    documents: dict[Path, Document] = {}
    errors: list[str] = []
    for path in dict.fromkeys(path.resolve() for path in paths):
        try:
            document = load_document(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        documents[path] = document
        errors.extend(f"{path}: {error}" for error in lint_document(document))

    if not errors and args.baseline is not None:
        baseline_path = args.baseline.resolve()
        current_path = args.current.resolve()
        errors.extend(
            f"BREAKING: {issue}"
            for issue in detect_breaking_changes(
                documents[baseline_path], documents[current_path]
            )
        )

    if errors:
        for error in sorted(errors):
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("OpenAPI quality gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
