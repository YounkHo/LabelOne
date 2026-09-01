from __future__ import annotations

from math import isfinite
from typing import Any

from labelone.errors import AnnotationValidationError


_POINT_COUNTS: dict[str, tuple[int, int | None]] = {
    "point": (1, 1),
    "rectangle": (2, 4),
    "rotation": (4, 4),
    "quadrilateral": (4, 4),
    "line": (2, 2),
    "circle": (2, 2),
    "polygon": (3, None),
    "linestrip": (2, None),
    "cuboid": (8, 8),
}
_ALLOWED_TYPES = frozenset(_POINT_COUNTS)


def _fail(message: str, *, shape_index: int | None = None, field: str | None = None) -> None:
    details: dict[str, object] = {}
    if shape_index is not None:
        details["shape_index"] = shape_index
    if field:
        details["field"] = field
    raise AnnotationValidationError(message, details=details)


def validate_annotation_document(document: dict[str, Any]) -> None:
    shapes = document.get("shapes", [])
    if not isinstance(shapes, list):
        _fail("Annotation field 'shapes' must be a list", field="shapes")
    for dimension in ("imageWidth", "imageHeight"):
        value = document.get(dimension)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            _fail(f"{dimension} must be a positive integer", field=dimension)
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            _fail("Each shape must be an object", shape_index=index)
        label = shape.get("label")
        if not isinstance(label, str) or not label.strip():
            _fail("Shape label must be a non-empty string", shape_index=index, field="label")
        shape_type = shape.get("shape_type")
        if not isinstance(shape_type, str) or not shape_type:
            _fail("Shape type must be a non-empty string", shape_index=index, field="shape_type")
        if shape_type not in _ALLOWED_TYPES:
            _fail(f"Unsupported shape type '{shape_type}'", shape_index=index, field="shape_type")
        points = shape.get("points")
        if not isinstance(points, list):
            _fail("Shape points must be a list", shape_index=index, field="points")
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                _fail("Every point must contain exactly x and y", shape_index=index, field="points")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(float(value)) for value in point):
                _fail("Point coordinates must be finite numbers", shape_index=index, field="points")
        minimum, maximum = _POINT_COUNTS.get(shape_type, (1, None))
        invalid_count = len(points) < minimum or (maximum is not None and len(points) != maximum)
        if shape_type == "rectangle":
            invalid_count = len(points) not in {2, 4}
        if invalid_count:
            _fail(
                f"Shape type '{shape_type}' expects " + (f"{minimum} points" if maximum == minimum else f"at least {minimum} points"),
                shape_index=index,
                field="points",
            )
        if shape_type == "rotation":
            direction = shape.get("direction")
            if not isinstance(direction, (int, float)) or isinstance(direction, bool) or not isfinite(float(direction)):
                _fail("Rotation shape requires a finite direction in radians", shape_index=index, field="direction")
