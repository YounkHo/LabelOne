from __future__ import annotations

from copy import deepcopy
from math import atan2, isfinite, tau
from typing import Any

from labelone.errors import AnnotationValidationError


def normalize_annotation_document(document: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(document)
    shapes = normalized.get("shapes", [])
    if not isinstance(shapes, list):
        return normalized
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            continue
        points = shape.get("points")
        shape_type = shape.get("shape_type")
        if shape_type == "rectangle" and isinstance(points, list) and len(points) == 2:
            first, second = points
            if isinstance(first, list) and isinstance(second, list) and len(first) == len(second) == 2:
                left, right = sorted((first[0], second[0]))
                top, bottom = sorted((first[1], second[1]))
                shape["points"] = [[left, top], [right, top], [right, bottom], [left, bottom]]
        if shape_type == "rotation" and isinstance(points, list) and len(points) == 4:
            first, second = points[0], points[1]
            if isinstance(first, list) and isinstance(second, list) and len(first) == len(second) == 2:
                dx = float(second[0]) - float(first[0])
                dy = float(second[1]) - float(first[1])
                if not isfinite(dx) or not isfinite(dy) or (dx == 0 and dy == 0):
                    raise AnnotationValidationError(
                        "Rotation shape has a degenerate first edge",
                        details={"shape_index": index, "field": "points"},
                    )
                shape["direction"] = atan2(dy, dx) % tau
    return normalized
