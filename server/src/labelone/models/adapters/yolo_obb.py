from __future__ import annotations

from typing import Literal

import numpy as np

from labelone.errors import ModelRuntimeError

from ..types import AnnotationResult
from .onnx import OnnxRuntimeAdapter, _ImageTransform


_Layout = Literal["raw", "raw_objectness", "end_to_end", "end_to_end_angle_last"]
_MatrixOrientation = Literal["channels_first", "channels_last"]
_EPSILON = 1e-9


def _cross(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) * 0.5


def _line_intersection(start: np.ndarray, end: np.ndarray, clip_start: np.ndarray, clip_end: np.ndarray) -> np.ndarray:
    direction = end - start
    clip_direction = clip_end - clip_start
    denominator = _cross(direction, clip_direction)
    if abs(denominator) <= _EPSILON:
        return (start + end) * 0.5
    distance = _cross(clip_start - start, clip_direction) / denominator
    return start + distance * direction


def _convex_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = [point.astype(np.float64, copy=True) for point in subject]
    signed_twice_area = float(
        np.dot(clip[:, 0], np.roll(clip[:, 1], -1))
        - np.dot(clip[:, 1], np.roll(clip[:, 0], -1))
    )
    orientation = 1.0 if signed_twice_area >= 0 else -1.0

    for index, clip_end in enumerate(clip):
        clip_start = clip[index - 1]
        input_points = output
        output = []
        if not input_points:
            break
        start = input_points[-1]
        start_inside = orientation * _cross(clip_end - clip_start, start - clip_start) >= -_EPSILON
        for end in input_points:
            end_inside = orientation * _cross(clip_end - clip_start, end - clip_start) >= -_EPSILON
            if end_inside:
                if not start_inside:
                    output.append(_line_intersection(start, end, clip_start, clip_end))
                output.append(end)
            elif start_inside:
                output.append(_line_intersection(start, end, clip_start, clip_end))
            start = end
            start_inside = end_inside
    if not output:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(output, dtype=np.float64)


def _xywhr_to_corners(boxes: np.ndarray) -> np.ndarray:
    centers = boxes[:, :2].astype(np.float64, copy=False)
    half_width = boxes[:, 2].astype(np.float64, copy=False) * 0.5
    half_height = boxes[:, 3].astype(np.float64, copy=False) * 0.5
    angles = boxes[:, 4].astype(np.float64, copy=False)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    width_vector = np.stack((cosine * half_width, sine * half_width), axis=1)
    height_vector = np.stack((-sine * half_height, cosine * half_height), axis=1)
    return np.stack(
        (
            centers - width_vector - height_vector,
            centers + width_vector - height_vector,
            centers + width_vector + height_vector,
            centers - width_vector + height_vector,
        ),
        axis=1,
    )


def _rotated_iou(
    current: int,
    remaining: np.ndarray,
    corners: np.ndarray,
    areas: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    overlaps = np.zeros(len(remaining), dtype=np.float64)
    if not len(remaining):
        return overlaps
    current_bounds = bounds[current]
    coarse = (
        (bounds[remaining, 0] < current_bounds[2])
        & (bounds[remaining, 2] > current_bounds[0])
        & (bounds[remaining, 1] < current_bounds[3])
        & (bounds[remaining, 3] > current_bounds[1])
    )
    for output_index in np.flatnonzero(coarse):
        candidate = int(remaining[output_index])
        intersection = _polygon_area(_convex_intersection(corners[current], corners[candidate]))
        union = float(areas[current] + areas[candidate] - intersection)
        overlaps[output_index] = intersection / max(union, _EPSILON)
    return overlaps


def _rotated_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, threshold: float) -> list[int]:
    corners = _xywhr_to_corners(boxes)
    areas = boxes[:, 2].astype(np.float64) * boxes[:, 3].astype(np.float64)
    bounds = np.column_stack(
        (
            corners[:, :, 0].min(axis=1),
            corners[:, :, 1].min(axis=1),
            corners[:, :, 0].max(axis=1),
            corners[:, :, 1].max(axis=1),
        )
    )
    keep: list[int] = []
    for class_id in np.unique(classes):
        class_indices = np.flatnonzero(classes == class_id)
        order = class_indices[np.argsort(scores[class_indices], kind="stable")[::-1]]
        while order.size:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            order = remaining[_rotated_iou(current, remaining, corners, areas, bounds) <= threshold]
    return sorted(keep, key=lambda index: (-float(scores[index]), index))


def _number_parameter(
    parameters: dict[str, object],
    config: dict[str, object],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = parameters.get(name, config.get(name, default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError(f"YOLO OBB {name} must be numeric", details={"value": raw}) from exc
    if not np.isfinite(value) or not minimum <= value <= maximum:
        raise ModelRuntimeError(
            f"YOLO OBB {name} is outside the supported range",
            details={"value": value, "minimum": minimum, "maximum": maximum},
        )
    return value


def _requested_layout(parameters: dict[str, object], config: dict[str, object]) -> str:
    value = str(parameters.get("output_layout", config.get("output_layout", "auto"))).strip().casefold()
    aliases = {
        "end2end": "end_to_end",
        "end2end_angle_last": "end_to_end_angle_last",
        "ultralytics_raw": "raw",
    }
    value = aliases.get(value, value)
    supported = {"auto", "raw", "raw_objectness", "end_to_end", "end_to_end_angle_last"}
    if value not in supported:
        raise ModelRuntimeError(
            "Unsupported YOLO OBB output layout",
            details={"layout": value, "supported": sorted(supported)},
        )
    return value


def _supported_widths(class_count: int, requested_layout: str) -> set[int]:
    widths: dict[str, int] = {
        "raw": 5 + class_count,
        "raw_objectness": 6 + class_count,
        "end_to_end": 7,
        "end_to_end_angle_last": 7,
    }
    if requested_layout in {"raw", "raw_objectness"} and class_count <= 0:
        raise ModelRuntimeError(
            "Raw YOLO OBB output requires configured classes",
            details={"layout": requested_layout, "class_count": class_count},
        )
    if requested_layout == "auto":
        return {7, *(width for name, width in widths.items() if name.startswith("raw") and class_count > 0)}
    return {widths[requested_layout]}


def _prediction_matrix(
    prediction: np.ndarray,
    *,
    class_count: int,
    requested_layout: str,
) -> tuple[np.ndarray, _MatrixOrientation]:
    array = np.asarray(prediction)
    original_shape = list(array.shape)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ModelRuntimeError(
                "YOLO OBB output must have a single batch",
                details={"shape": original_shape, "batch": int(array.shape[0])},
            )
        array = array[0]
    if array.ndim != 2:
        raise ModelRuntimeError(
            "Unsupported YOLO OBB output rank",
            details={"shape": original_shape, "expected": "[1,C,N], [1,N,C], [C,N], or [N,C]"},
        )

    widths = _supported_widths(class_count, requested_layout)
    rows_are_features = int(array.shape[0]) in widths
    columns_are_features = int(array.shape[1]) in widths
    if rows_are_features and columns_are_features:
        raise ModelRuntimeError(
            "Ambiguous YOLO OBB output axes",
            details={"shape": original_shape, "supported_feature_widths": sorted(widths)},
        )
    if rows_are_features:
        return np.asarray(array.T), "channels_first"
    if columns_are_features:
        return np.asarray(array), "channels_last"
    raise ModelRuntimeError(
        "Unsupported YOLO OBB output shape",
        details={
            "shape": original_shape,
            "class_count": class_count,
            "supported_feature_widths": sorted(widths),
        },
    )


def _looks_like_class_ids(values: np.ndarray, class_count: int) -> bool:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return False
    rounded = np.rint(finite)
    if not np.all(np.abs(finite - rounded) <= 1e-4) or np.any(rounded < 0):
        return False
    return class_count <= 0 or bool(np.all(rounded < class_count))


def _looks_like_scores(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return bool(finite.size and np.all((finite >= 0) & (finite <= 1)))


def _resolve_layout(
    prediction: np.ndarray,
    *,
    class_count: int,
    requested_layout: str,
    orientation: _MatrixOrientation,
) -> _Layout:
    width = int(prediction.shape[1])
    if requested_layout != "auto":
        return requested_layout  # type: ignore[return-value]

    raw_layouts: list[_Layout] = []
    if class_count > 0 and width == 5 + class_count:
        raw_layouts.append("raw")
    if class_count > 0 and width == 6 + class_count:
        raw_layouts.append("raw_objectness")
    if orientation == "channels_first" and raw_layouts:
        return raw_layouts[0]

    standard_end_to_end = width == 7 and _looks_like_scores(prediction[:, 5]) and _looks_like_class_ids(prediction[:, 6], class_count)
    angle_last_end_to_end = width == 7 and _looks_like_scores(prediction[:, 4]) and _looks_like_class_ids(prediction[:, 5], class_count)
    if standard_end_to_end:
        return "end_to_end"
    if angle_last_end_to_end:
        return "end_to_end_angle_last"
    if raw_layouts:
        return raw_layouts[0]
    raise ModelRuntimeError(
        "Could not infer YOLO OBB output layout",
        details={
            "shape": list(prediction.shape),
            "class_count": class_count,
            "hint": "Set output_layout to raw, raw_objectness, end_to_end, or end_to_end_angle_last",
        },
    )


def _decode_prediction(
    prediction: np.ndarray,
    *,
    layout: _Layout,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if layout == "end_to_end":
        boxes = np.column_stack((prediction[:, :4], prediction[:, 4])).astype(np.float64, copy=False)
        scores = prediction[:, 5].astype(np.float64, copy=False)
        class_values = prediction[:, 6]
        classes = np.where(np.isfinite(class_values), np.rint(class_values), -1).astype(np.int64)
        return boxes, scores, classes
    if layout == "end_to_end_angle_last":
        boxes = np.column_stack((prediction[:, :4], prediction[:, 6])).astype(np.float64, copy=False)
        scores = prediction[:, 4].astype(np.float64, copy=False)
        class_values = prediction[:, 5]
        classes = np.where(np.isfinite(class_values), np.rint(class_values), -1).astype(np.int64)
        return boxes, scores, classes
    if class_count <= 0:
        raise ModelRuntimeError("Raw YOLO OBB output requires configured classes")

    class_start = 5 if layout == "raw_objectness" else 4
    class_scores = prediction[:, class_start : class_start + class_count].astype(np.float64, copy=False)
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]
    if layout == "raw_objectness":
        scores = scores * prediction[:, 4]
    boxes = np.column_stack((prediction[:, :4], prediction[:, -1])).astype(np.float64, copy=False)
    return boxes, scores.astype(np.float64, copy=False), class_ids.astype(np.int64, copy=False)


class YoloObbOnnxAdapter(OnnxRuntimeAdapter):
    """Clean-room Ultralytics-style ONNX OBB post-processing adapter."""

    def _annotations(
        self,
        outputs: dict[str, np.ndarray],
        transform: _ImageTransform,
        parameters: dict[str, object],
    ) -> list[AnnotationResult]:
        if not outputs:
            return []
        classes = list(self.record.config.get("classes") or [])
        requested_layout = _requested_layout(parameters, self.record.config)
        output_name_value = parameters.get("output_name", self.record.config.get("output_name"))
        if output_name_value is not None and not isinstance(output_name_value, str):
            raise ModelRuntimeError("YOLO OBB output_name must be a string", details={"value": output_name_value})

        candidates: list[tuple[str, np.ndarray, _MatrixOrientation]] = []
        failures: dict[str, dict[str, object]] = {}
        selected_outputs = outputs.items()
        if output_name_value is not None:
            if output_name_value not in outputs:
                raise ModelRuntimeError(
                    "Configured YOLO OBB output was not exported",
                    details={"output_name": output_name_value, "available": sorted(outputs)},
                )
            selected_outputs = [(output_name_value, outputs[output_name_value])]
        for name, value in selected_outputs:
            try:
                matrix, orientation = _prediction_matrix(
                    value,
                    class_count=len(classes),
                    requested_layout=requested_layout,
                )
            except ModelRuntimeError as exc:
                failures[name] = {"message": exc.message, **exc.details}
                continue
            candidates.append((name, matrix, orientation))
        if not candidates:
            raise ModelRuntimeError(
                "No supported YOLO OBB detection output was found",
                details={"outputs": {name: list(np.asarray(value).shape) for name, value in outputs.items()}, "failures": failures},
            )
        if len(candidates) > 1:
            raise ModelRuntimeError(
                "Multiple YOLO OBB detection outputs are ambiguous",
                details={"candidates": [name for name, _, _ in candidates], "hint": "Set output_name explicitly"},
            )
        _, prediction, orientation = candidates[0]
        if prediction.shape[0] == 0:
            return []
        layout = _resolve_layout(
            prediction,
            class_count=len(classes),
            requested_layout=requested_layout,
            orientation=orientation,
        )
        boxes, scores, class_ids = _decode_prediction(prediction, layout=layout, class_count=len(classes))

        angle_unit = str(parameters.get("angle_unit", self.record.config.get("angle_unit", "radians"))).strip().casefold()
        if angle_unit in {"degree", "degrees", "deg"}:
            boxes[:, 4] = np.deg2rad(boxes[:, 4])
        elif angle_unit not in {"radian", "radians", "rad"}:
            raise ModelRuntimeError(
                "Unsupported YOLO OBB angle unit",
                details={"angle_unit": angle_unit, "supported": ["radians", "degrees"]},
            )

        confidence = _number_parameter(
            parameters,
            self.record.config,
            "conf_threshold",
            0.25,
            minimum=0.0,
            maximum=1.0,
        )
        iou_threshold = _number_parameter(
            parameters,
            self.record.config,
            "iou_threshold",
            0.45,
            minimum=0.0,
            maximum=1.0,
        )
        raw_max_det = parameters.get("max_det", self.record.config.get("max_det", 300))
        try:
            max_detections = int(raw_max_det)
        except (TypeError, ValueError) as exc:
            raise ModelRuntimeError("YOLO OBB max_det must be an integer", details={"value": raw_max_det}) from exc
        if max_detections <= 0:
            raise ModelRuntimeError("YOLO OBB max_det must be positive", details={"value": max_detections})

        finite = np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
        valid_classes = class_ids >= 0
        if classes:
            valid_classes &= class_ids < len(classes)
        selected = finite & valid_classes & (boxes[:, 2] > 0) & (boxes[:, 3] > 0) & (scores >= confidence)
        boxes, scores, class_ids = boxes[selected], scores[selected], class_ids[selected]
        if not len(boxes):
            return []

        if transform.scale <= 0:
            raise ModelRuntimeError("Invalid YOLO OBB letterbox scale", details={"scale": transform.scale})
        boxes[:, 0] = (boxes[:, 0] - transform.pad_x) / transform.scale
        boxes[:, 1] = (boxes[:, 1] - transform.pad_y) / transform.scale
        boxes[:, 2:4] /= transform.scale

        keep = _rotated_nms(boxes, scores, class_ids, iou_threshold)[:max_detections]
        corners = _xywhr_to_corners(boxes[keep])
        results: list[AnnotationResult] = []
        for index, points in zip(keep, corners, strict=True):
            class_id = int(class_ids[index])
            label = classes[class_id] if 0 <= class_id < len(classes) else str(class_id)
            results.append(
                AnnotationResult(
                    label=label,
                    score=float(scores[index]),
                    shape_type="rotation",
                    points=[[float(x), float(y)] for x, y in points],
                )
            )
        return results
