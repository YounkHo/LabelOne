from __future__ import annotations

import re
import shlex
from typing import Callable, Literal

from labelone.errors import InvalidPathError

from .models import DatasetAsset


SearchMode = Literal["text", "regex", "condition", "smart"]


def _searchable(asset: DatasetAsset) -> str:
    values = [asset.display_path, asset.status.value, *asset.labels, *asset.shape_types]
    return " ".join(values)


def _safe_regex(pattern: str) -> re.Pattern[str]:
    if len(pattern) > 256:
        raise InvalidPathError("Regular expression is too long", details={"maximum": 256})
    unsafe = [
        (r"\(\?", "lookarounds and inline extensions are not supported"),
        (r"\\[1-9]", "backreferences are not supported"),
        (r"\([^)]*[+*{][^)]*\)[+*{]", "nested quantifiers are not supported"),
        (r"\([^)]*\|[^)]*\)[+*{]", "quantified alternation groups are not supported"),
        (r"\.\*.*\.\*", "multiple unbounded wildcards are not supported"),
    ]
    for expression, reason in unsafe:
        if re.search(expression, pattern):
            raise InvalidPathError("Unsafe regular expression", details={"reason": reason})
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise InvalidPathError("Invalid regular expression", details={"error": str(exc)}) from exc


def _truth(value: str) -> bool:
    folded = value.casefold()
    if folded in {"1", "true", "yes", "是"}:
        return True
    if folded in {"0", "false", "no", "否"}:
        return False
    raise InvalidPathError("Boolean search value is invalid", details={"value": value})


def _numeric_match(asset: DatasetAsset, field: str, operator: str, raw: str) -> bool:
    fields = {
        "width": asset.width,
        "height": asset.height,
        "pixels": (asset.width * asset.height) if asset.width is not None and asset.height is not None else None,
        "annotations": asset.annotation_count,
    }
    if field not in fields:
        raise InvalidPathError("Unknown numeric search field", details={"field": field})
    try:
        target = float(raw)
    except ValueError as exc:
        raise InvalidPathError("Numeric search value is invalid", details={"value": raw}) from exc
    actual = fields[field]
    if actual is None:
        return False
    return {
        ">": actual > target,
        ">=": actual >= target,
        "<": actual < target,
        "<=": actual <= target,
        "=": actual == target,
    }[operator]


def _condition_token(token: str) -> Callable[[DatasetAsset], bool]:
    negate = token.casefold().startswith("not:")
    value = token[4:] if negate else token
    numeric = re.fullmatch(r"(width|height|pixels|annotations)(>=|<=|>|<|=)(\d+(?:\.\d+)?)", value, re.IGNORECASE)
    if numeric:
        matcher = lambda asset: _numeric_match(asset, numeric.group(1).casefold(), numeric.group(2), numeric.group(3))
    elif ":" in value:
        field, raw = value.split(":", 1)
        field, folded = field.casefold(), raw.casefold()
        if field in {"class", "label"}:
            matcher = lambda asset: any(folded in label.casefold() for label in asset.labels)
        elif field in {"type", "shape"}:
            matcher = lambda asset: any(folded == shape.casefold() for shape in asset.shape_types)
        elif field == "status":
            matcher = (lambda asset: asset.status.value != "valid") if folded in {"error", "异常"} else (lambda asset: asset.status.value == folded)
        elif field == "path":
            matcher = lambda asset: folded in asset.display_path.casefold()
        elif field == "annotated":
            expected = _truth(raw)
            matcher = lambda asset: bool(asset.annotation_count) is expected
        elif field == "selectable":
            expected = _truth(raw)
            matcher = lambda asset: asset.selectable is expected
        elif field == "has" and folded in {"annotation", "annotations", "标注"}:
            matcher = lambda asset: bool(asset.annotation_count)
        else:
            raise InvalidPathError("Unknown condition search field", details={"field": field})
    elif value.casefold() == "size>8k":
        matcher = lambda asset: max(asset.width or 0, asset.height or 0) > 8_000
    else:
        folded = value.casefold()
        matcher = lambda asset: folded in _searchable(asset).casefold()
    return (lambda asset: not matcher(asset)) if negate else matcher


def _sql_condition_token(token: str) -> tuple[str, list[object]]:
    negate = token.casefold().startswith("not:")
    value = token[4:] if negate else token
    parameters: list[object] = []
    numeric = re.fullmatch(r"(width|height|pixels|annotations)(>=|<=|>|<|=)(\d+(?:\.\d+)?)", value, re.IGNORECASE)
    if numeric:
        columns = {
            "width": "width",
            "height": "height",
            "pixels": "(width * height)",
            "annotations": "annotation_count",
        }
        expression = f"COALESCE({columns[numeric.group(1).casefold()]}, 0) {numeric.group(2)} ?"
        parameters.append(float(numeric.group(3)))
    elif ":" in value:
        field, raw = value.split(":", 1)
        field, folded = field.casefold(), raw.casefold()
        if field in {"class", "label"}:
            expression = "EXISTS (SELECT 1 FROM json_each(assets.labels_json) WHERE instr(lower(CAST(value AS TEXT)), ?) > 0)"
            parameters.append(folded)
        elif field in {"type", "shape"}:
            expression = "EXISTS (SELECT 1 FROM json_each(assets.shape_types_json) WHERE lower(CAST(value AS TEXT)) = ?)"
            parameters.append(folded)
        elif field == "status":
            expression = "status != 'valid'" if folded in {"error", "异常"} else "status = ?"
            if "?" in expression:
                parameters.append(folded)
        elif field == "path":
            expression = "instr(lower(display_path), ?) > 0"
            parameters.append(folded)
        elif field == "annotated":
            expression = "COALESCE(annotation_count, 0) > 0" if _truth(raw) else "COALESCE(annotation_count, 0) = 0"
        elif field == "selectable":
            expression = "selectable = ?"
            parameters.append(int(_truth(raw)))
        elif field == "has" and folded in {"annotation", "annotations", "标注"}:
            expression = "COALESCE(annotation_count, 0) > 0"
        else:
            raise InvalidPathError("Unknown condition search field", details={"field": field})
    elif value.casefold() == "size>8k":
        expression = "MAX(COALESCE(width, 0), COALESCE(height, 0)) > 8000"
    else:
        expression = "instr(lower(display_path || ' ' || status || ' ' || labels_json || ' ' || shape_types_json), ?) > 0"
        parameters.append(value.casefold())
    return (f"NOT ({expression})" if negate else expression), parameters


def _condition_tokens(query: str) -> list[list[str]]:
    try:
        tokens = shlex.split(query)
    except ValueError as exc:
        raise InvalidPathError("Invalid condition query", details={"error": str(exc)}) from exc
    groups: list[list[str]] = [[]]
    for token in tokens:
        if token.casefold() == "or":
            if not groups[-1]:
                raise InvalidPathError("OR requires a condition on both sides")
            groups.append([])
        elif token.casefold() != "and":
            groups[-1].append(token)
    if not groups[-1]:
        raise InvalidPathError("Search query cannot end with OR")
    return groups


def compile_asset_sql(query: str, mode: SearchMode) -> tuple[str, list[object], re.Pattern[str] | None]:
    query = query.strip()
    if not query:
        return "1=1", [], None
    if len(query) > 1_024:
        raise InvalidPathError("Search query is too long", details={"maximum": 1024})
    if mode == "text":
        return (
            "instr(lower(display_path), ?) > 0",
            [query.casefold()],
            None,
        )
    if mode == "regex":
        return "LABELONE_REGEX(display_path) = 1", [], _safe_regex(query)
    if mode not in {"condition", "smart"}:
        raise InvalidPathError("Unknown search mode", details={"mode": mode})
    clauses: list[str] = []
    parameters: list[object] = []
    for group in _condition_tokens(query):
        parts: list[str] = []
        for token in group:
            expression, values = _sql_condition_token(token)
            parts.append(expression)
            parameters.extend(values)
        clauses.append(f"({' AND '.join(parts)})")
    return " OR ".join(clauses), parameters, None


def compile_asset_predicate(query: str, mode: SearchMode) -> Callable[[DatasetAsset], bool]:
    query = query.strip()
    if not query:
        return lambda _: True
    if len(query) > 1_024:
        raise InvalidPathError("Search query is too long", details={"maximum": 1024})
    if mode == "text":
        folded = query.casefold()
        return lambda asset: folded in _searchable(asset).casefold()
    if mode == "regex":
        expression = _safe_regex(query)
        return lambda asset: expression.search(_searchable(asset)[:4_096]) is not None
    if mode not in {"condition", "smart"}:
        raise InvalidPathError("Unknown search mode", details={"mode": mode})
    groups = [[_condition_token(token) for token in group] for group in _condition_tokens(query)]
    return lambda asset: any(all(matcher(asset) for matcher in group) for group in groups)
