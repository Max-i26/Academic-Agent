"""Safe filter translation from FilterQuery to LanceDB SQL/DataFusion expressions."""

import re
from typing import Any
from academic_agent.storage.models import FilterQuery

# Explicit allowlist of fields supported by the LanceDB schema
ALLOWED_FILTER_FIELDS = {
    "source_file": str,
    "chunk_id": str,
    "record_id": str,
    "section_title": str,
    "content_type": str,
    "fragment_index": int,
    "total_fragments": int,
    "source_pages": list,
}


def _escape_sql_string(val: str) -> str:
    """Safely escape a string literal for LanceDB SQL.

    Replaces single quotes with doubled single quotes ('') and rejects null bytes.
    """
    if "\0" in val:
        raise ValueError("Null bytes are not permitted in filter strings")
    escaped = val.replace("'", "''")
    return f"'{escaped}'"


def _format_integer(val: Any, field_name: str) -> int:
    """Validate and format an integer value, disallowing booleans."""
    if isinstance(val, bool) or not isinstance(val, int):
        raise TypeError(f"Field '{field_name}' must be an integer, got {type(val).__name__}")
    return val


def build_lance_filter(filter_input: FilterQuery | dict[str, Any] | None) -> str | None:
    """Translate a FilterQuery or dict into a safe, injection-proof LanceDB SQL predicate string.

    Args:
        filter_input: FilterQuery model instance or plain dictionary of key-value constraints.

    Returns:
        SQL predicate string (e.g. "(source_file = 'lec1.pdf') AND (array_has_any(source_pages, [1, 2]))")
        or None if no filter constraints are specified.

    Raises:
        ValueError: If an unsupported field is requested or malicious input is detected.
        TypeError: If a field value violates expected types.
    """
    if filter_input is None:
        return None

    predicates: list[str] = []

    if isinstance(filter_input, FilterQuery):
        # 1. source_file
        if filter_input.source_file is not None:
            if not isinstance(filter_input.source_file, str):
                raise TypeError("source_file must be a string")
            sf = filter_input.source_file.strip()
            if sf:
                predicates.append(f"source_file = {_escape_sql_string(sf)}")

        # 2. chunk_id
        if filter_input.chunk_id is not None:
            if not isinstance(filter_input.chunk_id, str):
                raise TypeError("chunk_id must be a string")
            cid = filter_input.chunk_id.strip()
            if cid:
                predicates.append(f"chunk_id = {_escape_sql_string(cid)}")

        # 3. section_title
        if filter_input.section_title is not None:
            if not isinstance(filter_input.section_title, str):
                raise TypeError("section_title must be a string")
            st = filter_input.section_title.strip()
            if st:
                predicates.append(f"section_title = {_escape_sql_string(st)}")

        # 4. content_type
        if filter_input.content_type is not None:
            if not isinstance(filter_input.content_type, str):
                raise TypeError("content_type must be a string")
            ct = filter_input.content_type.strip()
            if ct:
                predicates.append(f"content_type = {_escape_sql_string(ct)}")

        # 5. source_pages
        if filter_input.source_pages:
            if not isinstance(filter_input.source_pages, (list, tuple)):
                raise TypeError("source_pages must be a list of integers")
            cleaned_pages = [_format_integer(p, "source_pages") for p in filter_input.source_pages]
            if cleaned_pages:
                page_csv = ", ".join(str(p) for p in sorted(set(cleaned_pages)))
                predicates.append(f"array_has_any(source_pages, [{page_csv}])")

        # 6. custom_filters
        if filter_input.custom_filters:
            for k, v in filter_input.custom_filters.items():
                pred = _translate_key_value(k, v)
                if pred:
                    predicates.append(pred)

    elif isinstance(filter_input, dict):
        for k, v in filter_input.items():
            pred = _translate_key_value(k, v)
            if pred:
                predicates.append(pred)
    else:
        raise TypeError(f"Expected FilterQuery or dict, got {type(filter_input).__name__}")

    if not predicates:
        return None

    return " AND ".join(f"({p})" for p in predicates)


def _translate_key_value(field: str, val: Any) -> str | None:
    """Validate field against allowlist and format predicate clause."""
    if field not in ALLOWED_FILTER_FIELDS:
        raise ValueError(
            f"Unsupported filter field '{field}'. Allowed fields: {sorted(ALLOWED_FILTER_FIELDS.keys())}"
        )

    expected_type = ALLOWED_FILTER_FIELDS[field]

    if val is None:
        return f"{field} IS NULL"

    if field == "source_pages":
        if not isinstance(val, (list, tuple)):
            raise TypeError("source_pages must be a list of integers")
        cleaned_pages = [_format_integer(p, "source_pages") for p in val]
        if not cleaned_pages:
            return None
        page_csv = ", ".join(str(p) for p in sorted(set(cleaned_pages)))
        return f"array_has_any(source_pages, [{page_csv}])"

    if expected_type is str:
        if not isinstance(val, str):
            raise TypeError(f"Field '{field}' must be a string, got {type(val).__name__}")
        cleaned = val.strip()
        if not cleaned:
            return None
        return f"{field} = {_escape_sql_string(cleaned)}"

    if expected_type is int:
        int_val = _format_integer(val, field)
        return f"{field} = {int_val}"

    raise ValueError(f"Unhandled filter type for field '{field}'")
