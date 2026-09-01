from __future__ import annotations

from collections import defaultdict
import heapq
from math import ceil, floor, sqrt

import numpy as np
from PIL import Image

from labelone.errors import ModelRuntimeError

from ..types import AnnotationResult
from .onnx import OnnxRuntimeAdapter, _ImageTransform


_Vertex = tuple[int, int]
_Edge = tuple[_Vertex, _Vertex]
_EPSILON = 1e-9


def _numeric_parameter(
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
        raise ModelRuntimeError(f"YOLO segmentation {name} must be numeric", details={"value": raw}) from exc
    if not np.isfinite(value) or not minimum <= value <= maximum:
        raise ModelRuntimeError(
            f"YOLO segmentation {name} is outside the supported range",
            details={"value": value, "minimum": minimum, "maximum": maximum},
        )
    return value


def _integer_parameter(
    parameters: dict[str, object],
    config: dict[str, object],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = parameters.get(name, config.get(name, default))
    if isinstance(raw, bool):
        raise ModelRuntimeError(f"YOLO segmentation {name} must be an integer", details={"value": raw})
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeError(f"YOLO segmentation {name} must be an integer", details={"value": raw}) from exc
    try:
        exact = float(raw) == value
    except (TypeError, ValueError):
        exact = False
    if not exact or not minimum <= value <= maximum:
        raise ModelRuntimeError(
            f"YOLO segmentation {name} is outside the supported range",
            details={"value": raw, "minimum": minimum, "maximum": maximum},
        )
    return value


def _prototype_tensor(value: np.ndarray, *, output_name: str) -> np.ndarray:
    prototype = np.asarray(value)
    original_shape = list(prototype.shape)
    if prototype.ndim != 4 or prototype.shape[0] != 1 or any(int(dimension) <= 0 for dimension in prototype.shape):
        raise ModelRuntimeError(
            "Unsupported YOLO segmentation prototype shape",
            details={"output_name": output_name, "shape": original_shape, "expected": "[1,M,H,W]"},
        )
    prototype = prototype[0]
    if not np.issubdtype(prototype.dtype, np.number):
        raise ModelRuntimeError(
            "YOLO segmentation prototypes must be numeric",
            details={"output_name": output_name, "dtype": str(prototype.dtype)},
        )
    return prototype.astype(np.float32, copy=False)


def _detection_matrix(value: np.ndarray, *, expected_width: int, output_name: str) -> np.ndarray:
    prediction = np.asarray(value)
    original_shape = list(prediction.shape)
    if prediction.ndim == 3:
        if prediction.shape[0] != 1:
            raise ModelRuntimeError(
                "YOLO segmentation detections must have one batch",
                details={"output_name": output_name, "shape": original_shape},
            )
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ModelRuntimeError(
            "Unsupported YOLO segmentation detection rank",
            details={"output_name": output_name, "shape": original_shape, "expected": "[1,C,N], [1,N,C], [C,N], or [N,C]"},
        )
    rows_are_features = prediction.shape[0] == expected_width
    columns_are_features = prediction.shape[1] == expected_width
    if rows_are_features and columns_are_features:
        raise ModelRuntimeError(
            "Ambiguous YOLO segmentation detection axes",
            details={"output_name": output_name, "shape": original_shape, "feature_width": expected_width},
        )
    if rows_are_features:
        return np.asarray(prediction.T)
    if columns_are_features:
        return np.asarray(prediction)
    raise ModelRuntimeError(
        "YOLO segmentation detection width does not match classes and prototypes",
        details={"output_name": output_name, "shape": original_shape, "expected_feature_width": expected_width},
    )


def _select_outputs(
    outputs: dict[str, np.ndarray],
    *,
    class_count: int,
    parameters: dict[str, object],
    config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    if class_count <= 0:
        raise ModelRuntimeError("YOLO segmentation requires configured classes", details={"class_count": class_count})
    prototype_name = parameters.get("prototype_output_name", config.get("prototype_output_name"))
    detection_name = parameters.get("detection_output_name", config.get("detection_output_name"))
    for name, value in (("prototype_output_name", prototype_name), ("detection_output_name", detection_name)):
        if value is not None and not isinstance(value, str):
            raise ModelRuntimeError(f"YOLO segmentation {name} must be a string", details={"value": value})
        if isinstance(value, str) and value not in outputs:
            raise ModelRuntimeError(
                f"Configured YOLO segmentation {name} was not exported",
                details={"output_name": value, "available": sorted(outputs)},
            )

    prototype_candidates: list[tuple[str, np.ndarray]] = []
    prototype_failures: dict[str, dict[str, object]] = {}
    prototype_items = [(prototype_name, outputs[prototype_name])] if isinstance(prototype_name, str) else list(outputs.items())
    for name, value in prototype_items:
        if np.asarray(value).ndim != 4:
            continue
        try:
            prototype_candidates.append((name, _prototype_tensor(value, output_name=name)))
        except ModelRuntimeError as exc:
            prototype_failures[name] = {"message": exc.message, **exc.details}
    if not prototype_candidates:
        raise ModelRuntimeError(
            "No supported YOLO segmentation prototype output was found",
            details={"outputs": {name: list(np.asarray(value).shape) for name, value in outputs.items()}, "failures": prototype_failures},
        )
    if len(prototype_candidates) > 1:
        raise ModelRuntimeError(
            "Multiple YOLO segmentation prototype outputs are ambiguous",
            details={"candidates": [name for name, _ in prototype_candidates], "hint": "Set prototype_output_name explicitly"},
        )
    selected_prototype_name, prototype = prototype_candidates[0]
    mask_dimension = int(prototype.shape[0])
    expected_width = 4 + class_count + mask_dimension

    detection_candidates: list[tuple[str, np.ndarray]] = []
    detection_failures: dict[str, dict[str, object]] = {}
    detection_items = [(detection_name, outputs[detection_name])] if isinstance(detection_name, str) else list(outputs.items())
    for name, value in detection_items:
        if name == selected_prototype_name or np.asarray(value).ndim not in {2, 3}:
            continue
        try:
            detection_candidates.append(
                (name, _detection_matrix(value, expected_width=expected_width, output_name=name))
            )
        except ModelRuntimeError as exc:
            detection_failures[name] = {"message": exc.message, **exc.details}
    if not detection_candidates:
        raise ModelRuntimeError(
            "No supported YOLO segmentation detection output was found",
            details={
                "outputs": {name: list(np.asarray(value).shape) for name, value in outputs.items()},
                "expected_feature_width": expected_width,
                "failures": detection_failures,
            },
        )
    if len(detection_candidates) > 1:
        raise ModelRuntimeError(
            "Multiple YOLO segmentation detection outputs are ambiguous",
            details={"candidates": [name for name, _ in detection_candidates], "hint": "Set detection_output_name explicitly"},
        )
    return detection_candidates[0][1], prototype


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    result = boxes.astype(np.float64, copy=True)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
    result[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
    result[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
    result[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
    return result


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    other_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area + other_area - intersection, 1e-7)


def _class_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, threshold: float) -> list[int]:
    keep: list[int] = []
    for class_id in np.unique(classes):
        indices = np.flatnonzero(classes == class_id)
        order = indices[np.argsort(scores[indices], kind="stable")[::-1]]
        while order.size:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            order = remaining[_iou(boxes[current], boxes[remaining]) <= threshold]
    return sorted(keep, key=lambda index: (-float(scores[index]), index))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _crop_probability_to_model_box(
    probability: np.ndarray,
    box: np.ndarray,
    transform: _ImageTransform,
) -> np.ndarray:
    height, width = probability.shape
    x1 = max(0, min(width, floor(float(box[0]) * width / transform.input_width)))
    y1 = max(0, min(height, floor(float(box[1]) * height / transform.input_height)))
    x2 = max(0, min(width, ceil(float(box[2]) * width / transform.input_width)))
    y2 = max(0, min(height, ceil(float(box[3]) * height / transform.input_height)))
    cropped = np.zeros_like(probability, dtype=np.float32)
    if x2 > x1 and y2 > y1:
        cropped[y1:y2, x1:x2] = probability[y1:y2, x1:x2]
    return cropped


def _target_dimensions(transform: _ImageTransform, pixel_budget: int) -> tuple[int, int]:
    source_pixels = transform.original_width * transform.original_height
    scale = min(1.0, sqrt(pixel_budget / max(1, source_pixels)))
    return max(1, round(transform.original_width * scale)), max(1, round(transform.original_height * scale))


def _render_binary_mask(
    probability: np.ndarray,
    restored_box: np.ndarray,
    transform: _ImageTransform,
    *,
    target_width: int,
    target_height: int,
    threshold: float,
) -> tuple[np.ndarray, int, int]:
    target_x1 = max(0, min(target_width, floor(float(restored_box[0]) * target_width / transform.original_width)))
    target_y1 = max(0, min(target_height, floor(float(restored_box[1]) * target_height / transform.original_height)))
    target_x2 = max(0, min(target_width, ceil(float(restored_box[2]) * target_width / transform.original_width)))
    target_y2 = max(0, min(target_height, ceil(float(restored_box[3]) * target_height / transform.original_height)))
    if target_x2 <= target_x1 or target_y2 <= target_y1:
        return np.zeros((0, 0), dtype=bool), target_x1, target_y1

    prototype_height, prototype_width = probability.shape
    x_scale = (
        transform.original_width
        / target_width
        * transform.scale
        * prototype_width
        / transform.input_width
    )
    y_scale = (
        transform.original_height
        / target_height
        * transform.scale
        * prototype_height
        / transform.input_height
    )
    x_offset = (
        (target_x1 * transform.original_width / target_width * transform.scale + transform.pad_x)
        * prototype_width
        / transform.input_width
    )
    y_offset = (
        (target_y1 * transform.original_height / target_height * transform.scale + transform.pad_y)
        * prototype_height
        / transform.input_height
    )
    image = Image.fromarray(probability.astype(np.float32, copy=False))
    rendered = image.transform(
        (target_x2 - target_x1, target_y2 - target_y1),
        Image.Transform.AFFINE,
        (x_scale, 0.0, x_offset, 0.0, y_scale, y_offset),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(rendered, dtype=np.float32) >= threshold, target_x1, target_y1


def _component_candidates(mask: np.ndarray, *, limit: int, minimum_pixels: float) -> list[list[tuple[int, int]]]:
    if not mask.size or not np.any(mask):
        return []
    height, width = mask.shape
    visited = np.zeros(mask.size, dtype=bool)
    flat_mask = mask.ravel()
    selected: list[tuple[int, int, list[tuple[int, int]]]] = []
    sequence = 0
    for start_value in np.flatnonzero(flat_mask):
        start = int(start_value)
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        component: list[tuple[int, int]] = []
        while stack:
            current = stack.pop()
            y, x = divmod(current, width)
            component.append((y, x))
            if x > 0:
                neighbor = current - 1
                if flat_mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
            if x + 1 < width:
                neighbor = current + 1
                if flat_mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
            if y > 0:
                neighbor = current - width
                if flat_mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
            if y + 1 < height:
                neighbor = current + width
                if flat_mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        if len(component) < minimum_pixels:
            continue
        entry = (len(component), -sequence, component)
        sequence += 1
        if len(selected) < limit:
            heapq.heappush(selected, entry)
        elif len(component) > selected[0][0]:
            heapq.heapreplace(selected, entry)
    return [entry[2] for entry in sorted(selected, key=lambda item: (-item[0], -item[1]))]


def _direction(edge: _Edge) -> int:
    (x1, y1), (x2, y2) = edge
    return {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[(x2 - x1, y2 - y1)]


def _boundary_loops(component: list[tuple[int, int]], mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    edges: set[_Edge] = set()
    for y, x in component:
        if y == 0 or not mask[y - 1, x]:
            edges.add(((x, y), (x + 1, y)))
        if x + 1 == width or not mask[y, x + 1]:
            edges.add(((x + 1, y), (x + 1, y + 1)))
        if y + 1 == height or not mask[y + 1, x]:
            edges.add(((x + 1, y + 1), (x, y + 1)))
        if x == 0 or not mask[y, x - 1]:
            edges.add(((x, y + 1), (x, y)))

    adjacency: dict[_Vertex, list[_Edge]] = defaultdict(list)
    for edge in edges:
        adjacency[edge[0]].append(edge)
    unused = set(edges)
    loops: list[np.ndarray] = []
    turn_priority = {1: 0, 0: 1, 3: 2, 2: 3}
    while unused:
        first = min(unused)
        unused.remove(first)
        start = first[0]
        current_edge = first
        vertices = [start, first[1]]
        while vertices[-1] != start:
            candidates = [edge for edge in adjacency.get(vertices[-1], []) if edge in unused]
            if not candidates:
                break
            incoming = _direction(current_edge)
            current_edge = min(
                candidates,
                key=lambda edge: (turn_priority[(_direction(edge) - incoming) % 4], edge),
            )
            unused.remove(current_edge)
            vertices.append(current_edge[1])
        if vertices[-1] == start and len(vertices) >= 4:
            loops.append(np.asarray(vertices[:-1], dtype=np.float64))
    return loops


def _signed_area(points: np.ndarray) -> float:
    return float(
        np.dot(points[:, 0], np.roll(points[:, 1], -1))
        - np.dot(points[:, 1], np.roll(points[:, 0], -1))
    ) * 0.5


def _remove_collinear(points: np.ndarray) -> np.ndarray:
    current = points
    while len(current) > 3:
        previous = np.roll(current, 1, axis=0)
        following = np.roll(current, -1, axis=0)
        first = current - previous
        second = following - current
        cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        keep = np.abs(cross) > _EPSILON
        if bool(np.all(keep)) or int(np.count_nonzero(keep)) < 3:
            break
        current = current[keep]
    return current


def _rdp_open(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) <= 2:
        return points
    start, end = points[0], points[-1]
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared <= _EPSILON:
        distances = np.linalg.norm(points[1:-1] - start, axis=1)
    else:
        relative = points[1:-1] - start
        projection = np.clip(relative @ segment / length_squared, 0.0, 1.0)
        nearest = start + projection[:, None] * segment
        distances = np.linalg.norm(points[1:-1] - nearest, axis=1)
    if not len(distances):
        return points[[0, -1]]
    split = int(np.argmax(distances)) + 1
    if float(distances[split - 1]) <= epsilon:
        return points[[0, -1]]
    left = _rdp_open(points[: split + 1], epsilon)
    right = _rdp_open(points[split:], epsilon)
    return np.vstack((left[:-1], right))


def _simplify_closed_polygon(points: np.ndarray, *, epsilon: float, max_points: int) -> np.ndarray:
    reduced = _remove_collinear(points)
    if epsilon > 0 and len(reduced) > 3:
        simplified = _rdp_open(np.vstack((reduced, reduced[0])), epsilon)[:-1]
        if len(simplified) >= 3:
            reduced = simplified
    if len(reduced) > max_points:
        indices = np.floor(np.arange(max_points) * len(reduced) / max_points).astype(int)
        reduced = reduced[indices]
    return reduced


def _polygons_from_mask(
    mask: np.ndarray,
    *,
    origin_x: int,
    origin_y: int,
    target_width: int,
    target_height: int,
    original_width: int,
    original_height: int,
    max_components: int,
    minimum_original_area: float,
    simplify_epsilon: float,
    max_points: int,
) -> list[np.ndarray]:
    pixel_original_area = original_width / target_width * original_height / target_height
    minimum_pixels = minimum_original_area / max(pixel_original_area, _EPSILON)
    components = _component_candidates(mask, limit=max_components, minimum_pixels=minimum_pixels)
    polygons: list[np.ndarray] = []
    for component in components:
        loops = _boundary_loops(component, mask)
        if not loops:
            continue
        outer = max(loops, key=lambda loop: abs(_signed_area(loop)))
        polygon = _simplify_closed_polygon(outer, epsilon=simplify_epsilon, max_points=max_points)
        if len(polygon) < 3:
            continue
        polygon[:, 0] = (polygon[:, 0] + origin_x) * original_width / target_width
        polygon[:, 1] = (polygon[:, 1] + origin_y) * original_height / target_height
        polygon[:, 0] = np.clip(polygon[:, 0], 0, original_width)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, original_height)
        polygons.append(polygon)
    return polygons


class YoloSegmentationOnnxAdapter(OnnxRuntimeAdapter):
    """Clean-room Ultralytics-style detection/prototype segmentation adapter."""

    def _annotations(
        self,
        outputs: dict[str, np.ndarray],
        transform: _ImageTransform,
        parameters: dict[str, object],
    ) -> list[AnnotationResult]:
        if not outputs:
            return []
        classes = list(self.record.config.get("classes") or [])
        prediction, prototypes = _select_outputs(
            outputs,
            class_count=len(classes),
            parameters=parameters,
            config=self.record.config,
        )
        prototype_values = int(np.prod(prototypes.shape, dtype=np.int64))
        maximum_prototype_values = _integer_parameter(
            parameters,
            self.record.config,
            "prototype_max_values",
            64_000_000,
            minimum=1,
            maximum=256_000_000,
        )
        if prototype_values > maximum_prototype_values:
            raise ModelRuntimeError(
                "YOLO segmentation prototype output exceeds the value budget",
                details={"values": prototype_values, "maximum": maximum_prototype_values, "shape": list(prototypes.shape)},
            )
        if not prediction.shape[0]:
            return []

        confidence = _numeric_parameter(
            parameters, self.record.config, "conf_threshold", 0.25, minimum=0.0, maximum=1.0
        )
        iou_threshold = _numeric_parameter(
            parameters, self.record.config, "iou_threshold", 0.45, minimum=0.0, maximum=1.0
        )
        mask_threshold = _numeric_parameter(
            parameters, self.record.config, "mask_threshold", 0.5, minimum=0.0, maximum=1.0
        )
        max_detections = _integer_parameter(
            parameters, self.record.config, "max_det", 100, minimum=1, maximum=10_000
        )
        mask_pixel_budget = _integer_parameter(
            parameters, self.record.config, "mask_max_pixels", 1_048_576, minimum=1_024, maximum=16_777_216
        )
        max_components = _integer_parameter(
            parameters, self.record.config, "max_mask_components", 16, minimum=1, maximum=1_024
        )
        max_polygon_points = _integer_parameter(
            parameters, self.record.config, "max_polygon_points", 512, minimum=3, maximum=100_000
        )
        max_total_points = _integer_parameter(
            parameters, self.record.config, "max_total_polygon_points", 10_000, minimum=3, maximum=1_000_000
        )
        minimum_area = _numeric_parameter(
            parameters, self.record.config, "min_mask_area", 4.0, minimum=0.0, maximum=1e15
        )
        simplify_epsilon = _numeric_parameter(
            parameters, self.record.config, "polygon_simplify", 1.0, minimum=0.0, maximum=1_000.0
        )

        class_count = len(classes)
        mask_dimension = int(prototypes.shape[0])
        boxes_xywh = prediction[:, :4].astype(np.float64, copy=False)
        class_scores = prediction[:, 4 : 4 + class_count].astype(np.float64, copy=False)
        coefficients = prediction[:, 4 + class_count : 4 + class_count + mask_dimension].astype(np.float32, copy=False)
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_scores)), class_ids]
        boxes = _xywh_to_xyxy(boxes_xywh)
        finite = (
            np.isfinite(boxes).all(axis=1)
            & np.isfinite(scores)
            & np.isfinite(coefficients).all(axis=1)
        )
        selected = finite & (boxes_xywh[:, 2] > 0) & (boxes_xywh[:, 3] > 0) & (scores >= confidence)
        boxes, scores, class_ids, coefficients = (
            boxes[selected],
            scores[selected],
            class_ids[selected],
            coefficients[selected],
        )
        if not len(boxes):
            return []
        keep = _class_nms(boxes, scores, class_ids, iou_threshold)[:max_detections]
        boxes, scores, class_ids, coefficients = (
            boxes[keep],
            scores[keep],
            class_ids[keep],
            coefficients[keep],
        )

        if transform.scale <= 0 or transform.input_width <= 0 or transform.input_height <= 0:
            raise ModelRuntimeError(
                "Invalid YOLO segmentation letterbox transform",
                details={"scale": transform.scale, "input_size": [transform.input_width, transform.input_height]},
            )
        restored_boxes = boxes.copy()
        restored_boxes[:, [0, 2]] = (restored_boxes[:, [0, 2]] - transform.pad_x) / transform.scale
        restored_boxes[:, [1, 3]] = (restored_boxes[:, [1, 3]] - transform.pad_y) / transform.scale
        restored_boxes[:, [0, 2]] = restored_boxes[:, [0, 2]].clip(0, transform.original_width)
        restored_boxes[:, [1, 3]] = restored_boxes[:, [1, 3]].clip(0, transform.original_height)

        target_width, target_height = _target_dimensions(transform, mask_pixel_budget)
        flattened_prototypes = prototypes.reshape(mask_dimension, -1)
        results: list[AnnotationResult] = []
        remaining_points = max_total_points
        for box, restored_box, score, class_id, coefficient in zip(
            boxes, restored_boxes, scores, class_ids, coefficients, strict=True
        ):
            if remaining_points < 3 or restored_box[2] <= restored_box[0] or restored_box[3] <= restored_box[1]:
                continue
            probability = _sigmoid(coefficient @ flattened_prototypes).reshape(prototypes.shape[1:])
            probability = _crop_probability_to_model_box(probability, box, transform)
            binary, origin_x, origin_y = _render_binary_mask(
                probability,
                restored_box,
                transform,
                target_width=target_width,
                target_height=target_height,
                threshold=mask_threshold,
            )
            polygon_budget = min(max_polygon_points, remaining_points)
            polygons = _polygons_from_mask(
                binary,
                origin_x=origin_x,
                origin_y=origin_y,
                target_width=target_width,
                target_height=target_height,
                original_width=transform.original_width,
                original_height=transform.original_height,
                max_components=max_components,
                minimum_original_area=minimum_area,
                simplify_epsilon=simplify_epsilon,
                max_points=polygon_budget,
            )
            label = classes[int(class_id)] if 0 <= int(class_id) < len(classes) else str(int(class_id))
            for polygon in polygons:
                if len(polygon) > remaining_points:
                    indices = np.floor(np.arange(remaining_points) * len(polygon) / remaining_points).astype(int)
                    polygon = polygon[indices]
                if len(polygon) < 3:
                    continue
                results.append(
                    AnnotationResult(
                        label=label,
                        score=float(score),
                        shape_type="polygon",
                        points=[[float(x), float(y)] for x, y in polygon],
                    )
                )
                remaining_points -= len(polygon)
                if remaining_points < 3:
                    break
        return results
